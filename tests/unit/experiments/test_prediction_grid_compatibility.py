"""Inspect only supported QF-32 schemas and authoritative completion exports."""

from pathlib import Path
from typing import Any

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import (
    ArtifactType,
    ManifestError,
    StudyType,
    _grid_integrity,
    inspect_study,
    verify_artifacts,
)
from quantforge.prediction import grid as grid_module
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    FixtureTrialAnalyzer,
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.prediction.test_prediction_grid import (
    _grid,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize("layout", ["directory", "manifest", "snapshot"])
@pytest.mark.parametrize("schema", ["1", "2", "1.0", "", None, 1, True, "missing"])
def test_prediction_study_schema_is_checked_before_indexing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    schema: Primitive,
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    manifest = read_record(root / "manifest.json")
    if schema == "missing":
        manifest.pop("schema_version")
    else:
        manifest["schema_version"] = schema
    manifest["study_id"] = configuration_identity(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"study_id", "execution", "cache_policy", "interpretation"}
        }
    )
    export = tmp_path / "metadata"
    export.mkdir()
    path = export / "manifest.json"
    write_json(path, {"manifest": manifest} if layout == "snapshot" else manifest)
    source = export if layout == "directory" else path
    if schema == "1":
        inspected = inspect_study(
            StudyType.PARAMETER_STUDY, source, artifact_root=tmp_path
        )
        assert (
            inspected.provenance.configuration.to_primitive()["schema_version"] == "1"
        )
        assert "trial_counts" not in inspected.provenance.observations.to_primitive()
    else:
        with pytest.raises(ManifestError, match="unsupported QF-32 study schema"):
            inspect_study(StudyType.PARAMETER_STUDY, source, artifact_root=tmp_path)


def interrupt_prediction(*args: Any, **kwargs: Any) -> None:
    raise KeyboardInterrupt("interrupted after persisting the running retry")


@pytest.mark.parametrize("summary_content", ["original", "malformed", "foreign"])
def test_interrupted_prediction_retry_omits_stale_summary_and_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, summary_content: str
) -> None:
    grid = _grid(
        tmp_path / "prediction", analyzer=FixtureTrialAnalyzer(), retry_failed=True
    )
    result = grid.run()
    root = tmp_path / "prediction" / result.study_id
    path = trial_path(root, "failed")
    summary_path = root / "summary.json"
    completed_summary = summary_path.read_bytes()
    with monkeypatch.context() as interruption:
        interruption.setattr(
            grid_module, "run_prediction_study_in_session", interrupt_prediction
        )
        with pytest.raises(KeyboardInterrupt, match="running retry"):
            grid.resume()
    assert read_record(path)["status"] == "running"
    assert summary_path.read_bytes() == completed_summary
    if summary_content == "malformed":
        summary_path.write_bytes(b"unfinished old summary")
    elif summary_content == "foreign":
        write_json(summary_path, {"study_id": "another study"})
    before = {item: item.read_bytes() for item in root.rglob("*") if item.is_file()}
    with monkeypatch.context() as inspection:
        block_research(inspection)
        inspected = inspect_study(
            StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path
        )
        assert "trial_counts" not in inspected.provenance.observations.to_primitive()
        assert sum(
            entry.artifact_type is ArtifactType.TRIAL_RESULT
            for entry in inspected.index.entries
        ) == len(result.trials)
        assert not any(
            entry.artifact_type is ArtifactType.PARAMETER_SUMMARY
            for entry in inspected.index.entries
        )
        assert verify_artifacts(inspected.index, tmp_path).valid
    assert {item: item.read_bytes() for item in before} == before
    resumed = grid.resume()
    assert read_record(path)["status"] == "succeeded"
    assert summary_path.read_bytes() != completed_summary
    block_research(monkeypatch)
    inspected = inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert (
        inspected.provenance.observations.to_primitive()["trial_counts"]
        == (resumed.summary_primitive()["counts"])
    )
    assert any(
        entry.artifact_type is ArtifactType.PARAMETER_SUMMARY
        for entry in inspected.index.entries
    )


@pytest.mark.parametrize("status", ["pending", "running"])
@pytest.mark.parametrize("change", ["none", "coordinates", "completion"])
def test_stale_summary_decision_keeps_trial_validation_and_snapshot_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, change: str
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    path = trial_path(root, "failed")
    completed = path.read_bytes()
    trial = read_record(path)
    trial.update(
        status=status, failure_type=None, failure_message=None, finished_at=None
    )
    if change == "coordinates":
        trial["combination_index"] = 99
    write_json(path, trial)
    original = _grid_integrity.validate_trial_status

    def finish_trial(kind: StudyType, record: PrimitiveMapping) -> None:
        original(kind, record)
        if record["trial_id"] == trial["trial_id"]:
            path.write_bytes(completed)

    if change == "completion":
        monkeypatch.setattr(_grid_integrity, "validate_trial_status", finish_trial)
    if change == "none":
        inspected = inspect_study(
            StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path
        )
        assert "trial_counts" not in inspected.provenance.observations.to_primitive()
        assert not any(
            entry.artifact_type is ArtifactType.PARAMETER_SUMMARY
            for entry in inspected.index.entries
        )
    else:
        message = (
            "coordinates" if change == "coordinates" else "changed during indexing"
        )
        with pytest.raises(ManifestError, match=message):
            inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
