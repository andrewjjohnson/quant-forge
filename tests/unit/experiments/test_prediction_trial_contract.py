from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, configuration_identity
from quantforge.experiments import ArtifactType, ManifestError, StudyType, inspect_study
from quantforge.experiments._json import mapping
from quantforge.prediction.grid import PredictionTrialAnalysis
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    read_record,
    trial_path,
)


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("status", ["failed", "excluded", "succeeded"])
@pytest.mark.parametrize("version", ["2", None, 1, True, "missing"])
def test_prediction_trial_schema_must_match_its_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
    status: str,
    version: Primitive,
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    if not complete:
        (root / "summary.json").unlink()
    path = trial_path(root, status)
    trial = read_record(path)
    if version == "missing":
        trial.pop("schema_version")
    else:
        trial["schema_version"] = version
    write_json(path, trial)
    with pytest.raises(ManifestError):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "null",
        "missing",
        "empty",
        "scalar",
        "count",
        "metric",
        "comparisons",
        "artifacts",
        "extra",
    ],
)
def test_successful_prediction_requires_complete_analysis_without_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    (root / "summary.json").unlink()
    path = trial_path(root)
    trial = read_record(path)
    artifact_path = root / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    analysis = mapping(trial["analysis"])
    if change == "null":
        trial["analysis"] = None
    elif change == "missing":
        trial.pop("analysis")
    elif change == "empty":
        trial["analysis"] = {}
    elif change == "scalar":
        trial["analysis"] = "unavailable"
    elif change == "count":
        analysis["prediction_count"] = True
    elif change == "metric":
        mapping(analysis["metrics"])["accuracy"] = "not-a-number"
    elif change == "comparisons":
        analysis["period_comparisons"] = ["not-a-record"]
    elif change == "artifacts":
        analysis["artifacts"] = None
    else:
        analysis["unexpected"] = "unsupported-field"
    if "analysis" in trial:
        artifact["analysis"] = deepcopy(trial["analysis"])
    else:
        artifact.pop("analysis")
    artifact["artifact_fingerprint"] = trial["artifact_fingerprint"] = (
        configuration_identity(
            {
                key: value
                for key, value in artifact.items()
                if key != "artifact_fingerprint"
            }
        )
    )
    write_json(path, trial)
    write_json(artifact_path, artifact)
    before = {saved: saved.read_bytes() for saved in (path, artifact_path)}
    with pytest.raises(ManifestError, match="analysis"):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert {saved: saved.read_bytes() for saved in before} == before


@pytest.mark.parametrize("empty_analysis", [False, True])
def test_resumable_prediction_keeps_valid_analysis_and_trial_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_analysis: bool
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    (root / "summary.json").unlink()
    if empty_analysis:
        path = trial_path(root)
        trial = read_record(path)
        artifact_path = root / cast(str, trial["artifact_location"])
        artifact = read_record(artifact_path)
        trial["analysis"] = artifact["analysis"] = PredictionTrialAnalysis.create(
            prediction_count=0, metrics={}
        ).to_primitive()
        trial["artifact_fingerprint"] = artifact["artifact_fingerprint"] = (
            configuration_identity(
                {
                    key: value
                    for key, value in artifact.items()
                    if key != "artifact_fingerprint"
                }
            )
        )
        write_json(path, trial)
        write_json(artifact_path, artifact)
    original = {path: path.read_bytes() for path in (root / "trials").glob("*.json")}
    inspected = inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert "trial_counts" not in inspected.provenance.observations.to_primitive()
    for entry in inspected.index.entries:
        if entry.artifact_type is ArtifactType.TRIAL_RESULT:
            trial = read_record(tmp_path / entry.path)
            assert entry.schema_version == trial["schema_version"]
            if trial["status"] == "succeeded":
                assert isinstance(trial["analysis"], dict)
            else:
                assert trial["analysis"] is None
    assert {path: path.read_bytes() for path in original} == original
