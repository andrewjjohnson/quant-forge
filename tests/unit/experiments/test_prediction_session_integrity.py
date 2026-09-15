from datetime import date
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import (
    ForwardReturnValues,
    PredictionStudy,
    SignalDisposition,
    SignalFeatureCandidate,
    forward_return_outcome,
    run_prediction_study,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record, trial_path
from tests.unit.experiments.test_prediction_row_integrity import (
    prediction_pair as prediction_pair,
)
from tests.unit.experiments.test_window_decision_integrity import rehash_window
from tests.unit.helpers import make_dataset
from tests.unit.prediction.test_feature_dataset import FixtureCandidateRule
from tests.unit.prediction.test_prediction_window import WindowProvider, grid


def rehash_row(row: PrimitiveMapping) -> None:
    outcome = cast(PrimitiveMapping, row["outcome"])
    evaluation = cast(PrimitiveMapping, row["evaluation"])
    outcome["outcome_id"] = configuration_identity(
        {
            **{key: value for key, value in outcome.items() if key != "outcome_id"},
            "record_type": "prediction_outcome",
        }
    )
    evaluation["outcome_id"] = outcome["outcome_id"]
    evaluation["evaluation_id"] = configuration_identity(
        {
            **{
                key: value
                for key, value in evaluation.items()
                if key != "evaluation_id"
            },
            "prediction": cast(PrimitiveMapping, row["prediction"])["values"],
            "record_type": "prediction_evaluation",
        }
    )
    row["row_id"] = configuration_identity(
        {
            "record_type": "prediction_study_row",
            "study_id": row["study_id"],
            "outcome_id": outcome["outcome_id"],
            "evaluation_id": evaluation["evaluation_id"],
            "signal": {"features": row["features"], "prediction": row["prediction"]},
        }
    )


@pytest.mark.parametrize(
    "component", ["prediction_rule", "outcome_labeler", "evaluator"]
)
@pytest.mark.parametrize("field", ["configuration", "name", "implementation_version"])
def test_prediction_component_identity_survives_rehashing_outer_study(
    tmp_path: Path,
    prediction_pair: tuple[PrimitiveMapping, PrimitiveMapping],
    component: str,
    field: str,
) -> None:
    result, _ = prediction_pair
    manifest = cast(PrimitiveMapping, result["manifest"])
    configured = cast(
        PrimitiveMapping, cast(PrimitiveMapping, manifest["configuration"])[component]
    )
    if field == "configuration":
        cast(PrimitiveMapping, configured[field])["material_option"] = "changed"
    else:
        configured[field] = "changed"
    manifest["study_id"] = configuration_identity(
        {
            "component": "quantforge_prediction_study",
            "engine_version": manifest["engine_version"],
            "market_data": manifest["market_data"],
            "study_configuration": manifest["configuration"],
            **(
                {"prediction_context": manifest["prediction_context"]}
                if "prediction_context" in manifest
                else {}
            ),
        }
    )
    for row in cast(list[PrimitiveMapping], result["rows"]):
        row["study_id"] = manifest["study_id"]
        rehash_row(row)
    path = tmp_path / "prediction.json"
    write_json(path, result)
    with pytest.raises(ManifestError, match="prediction study component identity"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize(
    "outcome_session",
    [None, "2024-07-03", "2024-07-04", "2024-07-06", "2024-07-09", "2024-08-01"],
)
def test_outcome_sessions_follow_recorded_exchange_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    horizon: int,
    outcome_session: str | None,
) -> None:
    composition = forward_return_outcome(horizon)
    dataset = make_dataset(
        ("100", "101", "102", "103"),
        sessions=(
            date(2024, 7, 3),
            date(2024, 7, 5),
            date(2024, 7, 8),
            date(2024, 7, 9),
        ),
    )
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(
        FixtureCandidateRule((SignalDisposition.ACCEPTED,) * 4),
        composition.labeler,
        composition.evaluator,
    )
    result = run_prediction_study(dataset, study).to_primitive()
    block_research(monkeypatch)
    row = cast(list[PrimitiveMapping], result["rows"])[0]
    outcome = cast(PrimitiveMapping, row["outcome"])
    original_session = outcome["outcome_session"]
    if outcome_session is not None:
        outcome["outcome_session"] = outcome_session
        rehash_row(row)
    path = tmp_path / "prediction.json"
    write_json(path, result)
    if outcome_session is None or outcome_session == original_session:
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match=r"outcome session.*horizon"):
            inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        None,
        "symbol",
        "strategy_id",
        "strategy_configuration_id",
        "strategy_parameters",
        "signal_session",
        "missing_row",
    ],
)
def test_nested_window_checks_unlabeled_signals_without_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str | None
) -> None:
    # July 11 is the last daily bar: every real window prediction is unlabeled.
    dataset = make_dataset(
        ("100", "101"), sessions=(date(2024, 7, 10), date(2024, 7, 11))
    )
    study = grid(
        tmp_path / "grid",
        WindowProvider(),
        dataset=dataset if change != "missing_row" else None,
    )
    result = study.run()
    root = tmp_path / "grid" / result.study_id
    record_path = trial_path(root)
    trial = read_record(record_path)
    artifact_path = root / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    window = cast(PrimitiveMapping, artifact["prediction_window"])
    decision = cast(list[PrimitiveMapping], window["decisions"])[0]
    prediction_study = cast(PrimitiveMapping, decision["prediction_study"])
    if change == "missing_row":
        prediction_study["rows"] = []
        cast(PrimitiveMapping, prediction_study["manifest"])["record_counts"] = {
            "generated_predictions": 1,
            "labeled_rows": 0,
            "unavailable_outcomes": 1,
        }
    else:
        assert prediction_study["rows"] == []
        signal = cast(list[PrimitiveMapping], decision["generated_signals"])[0]
        prediction = cast(PrimitiveMapping, signal["prediction"])
        if change is not None:
            prediction[change] = (
                {"foreign": True}
                if change == "strategy_parameters"
                else "2024-07-12"
                if change == "signal_session"
                else "foreign"
            )
    rehash_window(window)
    artifact["prediction_window_id"] = cast(PrimitiveMapping, window["manifest"])[
        "window_result_id"
    ]
    artifact["artifact_fingerprint"] = configuration_identity(
        {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
    )
    trial["artifact_fingerprint"] = artifact["artifact_fingerprint"]
    write_json(artifact_path, artifact)
    write_json(record_path, trial)
    block_research(monkeypatch)
    if change is None:
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    else:
        with pytest.raises(
            ManifestError,
            match=r"provenance|signal session|missing an available outcome",
        ):
            inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
