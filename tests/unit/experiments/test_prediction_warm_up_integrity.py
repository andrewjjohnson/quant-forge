"""Persisted rule wrappers must agree with captured warm-up declarations."""

from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._json import mapping
from quantforge.prediction import run_prediction_study
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
from tests.unit.prediction import test_multi_timeframe_study as fixtures
from tests.unit.prediction.test_prediction_window import (
    WindowProvider,
    _rewrite_window_checksums,  # pyright: ignore[reportPrivateUsage]
    grid,
)


class RuleWithoutWarmUpDeclaration(fixtures.FixtureMultiTimeframeRule):
    def configuration(self) -> PrimitiveMapping:
        return {
            key: value
            for key, value in super().configuration().items()
            if key != "warm_up_observations"
        }


def test_result_preserves_wrapper_when_rule_configuration_omits_warm_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rule = RuleWithoutWarmUpDeclaration(fixtures._requirements())  # pyright: ignore[reportPrivateUsage]
    result = run_prediction_study(
        fixtures._prediction_dataset(),  # pyright: ignore[reportPrivateUsage]
        fixtures._study(rule),  # pyright: ignore[reportPrivateUsage]
        context_provider=fixtures.FixtureContextProvider(
            fixtures._prediction_context()  # pyright: ignore[reportPrivateUsage]
        ),
    ).to_primitive()
    path = tmp_path / "prediction.json"
    write_json(path, result)
    block_research(monkeypatch)
    inspected = inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    assert (
        mapping(
            mapping(inspected.provenance.configuration.to_primitive()["configuration"])[
                "prediction_rule"
            ]
        )["warm_up_observations"]
        == 2
    )


@pytest.mark.parametrize("warm_up", [1, None, True, 2.0, "missing"])
def test_manifest_only_input_validates_warm_up_without_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    warm_up: Primitive,
) -> None:
    result = context_result()
    manifest = mapping(result["manifest"])
    rule = mapping(mapping(manifest["configuration"])["prediction_rule"])
    if warm_up == "missing":
        del rule["warm_up_observations"]
    else:
        rule["warm_up_observations"] = warm_up
    rehash_prediction(result)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="warm-up"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("nested", [False, True], ids=["standalone", "grid"])
@pytest.mark.parametrize("window", [False, True], ids=["plain", "window"])
@pytest.mark.parametrize("warm_up", [1, True, 2.0, None])
def test_rehashed_prediction_cannot_contradict_captured_warm_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
    window: bool,
    warm_up: Primitive,
) -> None:
    if window:
        study = grid(tmp_path / "grid", WindowProvider())
        study.run()
        root = tmp_path / "grid" / study.study_id
        block_research(monkeypatch)
    else:
        root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    record_path = trial_path(root)
    trial = read_record(record_path)
    artifact_path = root / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    snapshot = mapping(artifact["prediction_window" if window else "prediction_study"])
    manifest = mapping(snapshot["manifest"])
    rule = mapping(mapping(manifest["configuration"])["prediction_rule"])
    assert (
        rule["warm_up_observations"]
        == mapping(rule["configuration"])["warm_up_observations"]
        == 2
    )
    original_definition = trial["trial_definition"]
    rule["warm_up_observations"] = warm_up
    if window:
        for decision in cast(list[PrimitiveMapping], snapshot["decisions"]):
            prediction = mapping(decision["prediction_study"])
            mapping(
                mapping(mapping(prediction["manifest"])["configuration"])[
                    "prediction_rule"
                ]
            )["warm_up_observations"] = warm_up
            rehash_prediction(prediction)
            decision["prediction_study_id"] = mapping(prediction["manifest"])[
                "study_id"
            ]
        refresh_window_manifest(snapshot)
        _rewrite_window_checksums(artifact_path, record_path, artifact)
    else:
        rehash_prediction(snapshot)
        artifact["prediction_study_id"] = manifest["study_id"]
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
        source, study_type = root, StudyType.PARAMETER_STUDY
    else:
        source = tmp_path / "prediction.json"
        study_type = StudyType.PREDICTION_WINDOW if window else StudyType.PREDICTION
        write_json(source, snapshot)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ManifestError, match="warm-up"):
        inspect_study(study_type, source, artifact_root=tmp_path)
    assert {path: path.read_bytes() for path in before} == before
