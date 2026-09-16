from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, StudyType, inspect_study
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.prediction.test_prediction_window import WindowProvider, grid


@pytest.mark.parametrize("status", ["succeeded", "failed", "excluded"])
@pytest.mark.parametrize(
    "change",
    [
        "parameters",
        "combination_id",
        "combination_index",
        "dataset_family_fingerprint",
        "indicator_backend",
        "trial_definition",
        "indicator_configuration_ids",
    ],
)
def test_prediction_trial_matches_its_grid_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, change: str
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    path = trial_path(root, status)
    trial = read_record(path)
    if change == "parameters":
        cast(PrimitiveMapping, trial[change])["window"] = 99
    elif change == "combination_index":
        trial[change] = cast(int, trial[change]) + 1
    elif change in {"trial_definition", "indicator_backend"}:
        trial[change] = {"changed": True}
    elif change == "indicator_configuration_ids":
        trial[change] = ["changed"]
    else:
        trial[change] = "0" * 64
    write_json(path, trial)
    with pytest.raises(
        ManifestError,
        match=r"trial.*(coordinates|identity|metadata)",
    ):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)


@pytest.mark.parametrize("status", ["failed", "excluded"])
def test_prediction_trial_cannot_be_renamed_with_stale_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    path = trial_path(root, status)
    trial = read_record(path)
    trial["trial_id"] = "0" * 64
    path.unlink()
    write_json(path.with_name(f"{trial['trial_id']}.json"), trial)
    with pytest.raises(ManifestError, match="prediction trial identity"):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)


@pytest.mark.parametrize("invalid", [-1, 4, True, 0.0, "0", None])
def test_prediction_trial_position_requires_a_bounded_integer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: Primitive
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    path = trial_path(root, "excluded")
    trial = read_record(path)
    trial["combination_index"] = invalid
    write_json(path, trial)
    with pytest.raises(ManifestError, match="coordinates"):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)


@pytest.mark.parametrize("window", [99, 2])
def test_rehashed_prediction_combination_still_matches_its_grid_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: int
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    path = trial_path(root, "excluded")
    trial = read_record(path)
    parameters = cast(PrimitiveMapping, trial["parameters"])
    parameters["window"] = window
    manifest = read_record(root / "manifest.json")
    factory = cast(PrimitiveMapping, manifest["study_factory"])
    trial["combination_id"] = configuration_identity(
        {
            "component": "quantforge_prediction_grid_combination",
            "schema_version": manifest["schema_version"],
            "factory_name": factory["name"],
            "factory_version": factory["version"],
            "factory_configuration": factory["configuration"],
            "parameters": parameters,
        }
    )
    write_json(path, trial)
    with pytest.raises(
        ManifestError, match="trial parameters do not match grid coordinates"
    ):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)


@pytest.mark.parametrize("window", [False, True])
def test_rehashed_success_artifact_cannot_be_replaced_by_another_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: bool
) -> None:
    if window:
        study = grid(tmp_path / "prediction", WindowProvider())
        result = study.run()
        root = tmp_path / "prediction" / result.study_id
        block_research(monkeypatch)
    else:
        root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    successes = [
        path
        for path in sorted((root / "trials").glob("*.json"))
        if read_record(path)["status"] == "succeeded"
    ]
    assert len(successes) >= 2
    target = read_record(successes[0])
    foreign = read_record(successes[1])
    artifact = read_record(root / cast(str, foreign["artifact_location"]))
    artifact["grid_study_id"] = target["study_id"]
    artifact["trial_id"] = target["trial_id"]
    target["analysis"] = artifact["analysis"]
    artifact["artifact_fingerprint"] = configuration_identity(
        {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
    )
    target["artifact_fingerprint"] = artifact["artifact_fingerprint"]
    write_json(root / cast(str, target["artifact_location"]), artifact)
    write_json(successes[0], target)
    with pytest.raises(ManifestError, match="result differs from its trial definition"):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
