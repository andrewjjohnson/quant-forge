"""Outcome wrappers cannot rewrite a frozen candidate's labeling contract."""

from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._json import mapping
from quantforge.experiments._prediction_sessions import recorded_session_indexes
from quantforge.prediction import PredictionStudy, run_prediction_study
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.experiments.test_standalone_provenance import (
    context_result,
    rehash_prediction,
)
from tests.unit.experiments.test_window_session_integrity import refresh_window_manifest
from tests.unit.helpers import make_dataset
from tests.unit.prediction.test_prediction_window import (
    WindowProvider,
    _rewrite_window_checksums,  # pyright: ignore[reportPrivateUsage]
    grid,
)
from tests.unit.prediction.test_study import (
    FutureCloseChangeEvaluator,
    FutureCloseChangeValues,
    FutureCloseOutcomeLabeler,
    FutureCloseValues,
    NumericPrediction,
    RecordingPredictionStrategy,
)


def change_wrapper(configuration: PrimitiveMapping, field: str) -> None:
    labeler = mapping(configuration["outcome_labeler"])
    if field == "required_future_sessions":
        labeler[field] = 2
    elif field == "required_market_fields":
        labeler[field] = ["high", "low"]
    else:
        mapping(
            configuration[
                "evaluator" if field == "evaluation_schema" else "outcome_labeler"
            ]
        )["result_schema_version"] = "foreign"


def change_prediction(snapshot: PrimitiveMapping, field: str) -> None:
    manifest = mapping(snapshot["manifest"])
    configuration = mapping(manifest["configuration"])
    change_wrapper(configuration, field)
    indexes = recorded_session_indexes(mapping(manifest["market_data"]))
    sessions = list(indexes)
    rows: list[PrimitiveMapping] = []
    for row in cast(list[PrimitiveMapping], snapshot["rows"]):
        outcome = mapping(row["outcome"])
        if field == "required_future_sessions":
            index = indexes[cast(str, mapping(row["prediction"])["signal_session"])] + 2
            if index >= len(sessions):
                continue
            outcome["outcome_session"] = sessions[index]
        outcome["outcome_result_schema_version"] = mapping(
            configuration["outcome_labeler"]
        )["result_schema_version"]
        outcome["outcome_id"] = configuration_identity(
            {
                "record_type": "prediction_outcome",
                **{key: value for key, value in outcome.items() if key != "outcome_id"},
            }
        )
        evaluation = mapping(row["evaluation"])
        evaluation["outcome_id"] = outcome["outcome_id"]
        evaluation["evaluation_result_schema_version"] = mapping(
            configuration["evaluator"]
        )["result_schema_version"]
        evaluation["evaluation_id"] = configuration_identity(
            {
                "record_type": "prediction_evaluation",
                "prediction": mapping(row["prediction"])["values"],
                **{
                    key: value
                    for key, value in evaluation.items()
                    if key != "evaluation_id"
                },
            }
        )
        rows.append(row)
    snapshot["rows"] = cast(list[Primitive], rows)
    counts = mapping(manifest["record_counts"])
    counts["labeled_rows"] = len(rows)
    counts["unavailable_outcomes"] = cast(int, counts["generated_predictions"]) - len(
        rows
    )
    rehash_prediction(snapshot)


@pytest.mark.parametrize("nested", [False, True], ids=["standalone", "grid"])
@pytest.mark.parametrize("window", [False, True], ids=["plain", "window"])
@pytest.mark.parametrize(
    "field",
    [
        "required_future_sessions",
        "required_market_fields",
        "outcome_schema",
        "evaluation_schema",
    ],
)
def test_rehashed_outcome_wrapper_remains_bound_to_component_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
    window: bool,
    field: str,
) -> None:
    if window:
        study = grid(tmp_path / "grid", WindowProvider())
        study.run()
        root = tmp_path / "grid" / study.study_id
        block_research(monkeypatch)
    else:
        root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    record_path = trial_path(root)
    trial = read_record(record_path)
    original_definition = trial["trial_definition"]
    artifact_path = root / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    snapshot = mapping(artifact["prediction_window" if window else "prediction_study"])
    if window:
        change_wrapper(mapping(mapping(snapshot["manifest"])["configuration"]), field)
        for decision in cast(list[PrimitiveMapping], snapshot["decisions"]):
            prediction = mapping(decision["prediction_study"])
            change_prediction(prediction, field)
            decision["prediction_study_id"] = mapping(prediction["manifest"])[
                "study_id"
            ]
        refresh_window_manifest(snapshot)
        _rewrite_window_checksums(artifact_path, record_path, artifact)
    else:
        change_prediction(snapshot, field)
        artifact["prediction_study_id"] = mapping(snapshot["manifest"])["study_id"]
        artifact["artifact_fingerprint"] = trial["artifact_fingerprint"] = (
            configuration_identity(
                {
                    key: value
                    for key, value in artifact.items()
                    if key != "artifact_fingerprint"
                }
            )
        )
        write_json(artifact_path, artifact)
        write_json(record_path, trial)
    assert read_record(record_path)["trial_definition"] == original_definition
    if nested:
        source, kind = root, StudyType.PARAMETER_STUDY
    else:
        source = tmp_path / "prediction.json"
        kind = StudyType.PREDICTION_WINDOW if window else StudyType.PREDICTION
        write_json(source, snapshot)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ManifestError, match=r"(outcome|evaluation).*contract"):
        inspect_study(kind, source, artifact_root=tmp_path)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    ("component", "field", "invalid"),
    [
        ("outcome_labeler", "required_future_sessions", value)
        for value in (None, True, 0, -1, 1.0, "1")
    ]
    + [
        ("outcome_labeler", "required_market_fields", value)
        for value in (
            None,
            [],
            [""],
            ["close", "close"],
            ["open", "close"],
            [1],
            [["close"]],
        )
    ]
    + [
        (component, "result_schema_version", value)
        for component in ("outcome_labeler", "evaluator")
        for value in (None, "", 1)
    ],
)
def test_manifest_only_checks_outcome_contract_without_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    field: str,
    invalid: Primitive,
) -> None:
    result = context_result()
    manifest = mapping(result["manifest"])
    mapping(mapping(manifest["configuration"])[component])[field] = invalid
    rehash_prediction(result)
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=r"(outcome|evaluation).*contract"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


class LabelerWithoutDuplicateDeclarations(FutureCloseOutcomeLabeler):
    def configuration(self) -> PrimitiveMapping:
        return {
            key: value
            for key, value in super().configuration().items()
            if key
            not in {"parameters", "required_market_fields", "result_schema_version"}
        }


class EvaluatorWithoutDuplicateDeclarations(FutureCloseChangeEvaluator):
    def configuration(self) -> PrimitiveMapping:
        return {
            key: value
            for key, value in super().configuration().items()
            if key != "result_schema_version"
        }


@pytest.mark.parametrize("horizon", [1, 2])
def test_generic_component_wrappers_preserved_without_duplicate_declarations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    horizon: int,
) -> None:
    events: list[str] = []
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        RecordingPredictionStrategy(events),
        LabelerWithoutDuplicateDeclarations(events, horizon),
        EvaluatorWithoutDuplicateDeclarations(events),
    )
    result = run_prediction_study(make_dataset(("100", "101", "102", "103")), study)
    path = tmp_path / "prediction.json"
    write_json(path, result.to_primitive())
    block_research(monkeypatch)
    bundle = inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    labeler = mapping(
        mapping(bundle.provenance.configuration.to_primitive()["configuration"])[
            "outcome_labeler"
        ]
    )
    assert labeler["required_future_sessions"] == horizon
    assert labeler["required_market_fields"] == ["close"]
    assert labeler["result_schema_version"] == "1"
