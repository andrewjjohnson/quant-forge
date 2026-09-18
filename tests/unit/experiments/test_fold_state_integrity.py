import shutil
from pathlib import Path

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import ArtifactType, ManifestError, inspect_validation
from quantforge.oos import OOSSource, load_oos_source
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.study_fixtures import CapturedStudy, copy_study
from tests.unit.experiments.study_fixtures import study_baselines as study_baselines
from tests.unit.experiments.test_adapters import block_research


@pytest.fixture(params=[True, False], ids=["prediction", "backtest"])
def captured_study(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    study_baselines: dict[bool, CapturedStudy],
) -> CapturedStudy:
    block_research(monkeypatch)
    return copy_study(study_baselines[bool(request.param)], tmp_path)


@pytest.mark.parametrize(
    "field",
    ["selection_id", "artifact_id", "failures", "status", "fold_id", "study_id"],
)
def test_rehashed_fold_state_must_match_the_complete_captured_record(
    tmp_path: Path, captured_study: CapturedStudy, field: str
) -> None:
    source = captured_study.source
    path = captured_study.study_path / "folds" / source.folds[0].fold_id / "state.json"
    state = read_record(path)
    state[field] = (
        [{"stage": "test", "error_type": "ChangedFailure"}]
        if field == "failures"
        else "changed"
    )
    write_record(path, state)
    assert read_record(path) == state  # The replacement envelope has a valid hash.
    with pytest.raises(ManifestError, match="fold state differs from captured source"):
        inspect_validation(source, captured_study.study_path, artifact_root=tmp_path)


def incomplete_source(completed: CapturedStudy) -> tuple[OOSSource, Path, Path]:
    root = completed.study_path / "folds"
    failed_path = root / completed.source.folds[0].fold_id / "state.json"
    state = read_record(failed_path)
    state.update(
        status="failed",
        artifact_id=None,
        failures=[{"stage": "test", "error_type": "FixtureFailure"}],
    )
    write_record(failed_path, state)
    pending_root = root / completed.source.folds[1].fold_id
    shutil.rmtree(pending_root)
    source = load_oos_source(completed.source.plan, completed.study_path)
    return source, failed_path, pending_root / "state.json"


def test_unchanged_failed_and_missing_folds_remain_indexable(
    tmp_path: Path, captured_study: CapturedStudy
) -> None:
    source, _, _ = incomplete_source(captured_study)
    bundle = inspect_validation(
        source, captured_study.study_path, artifact_root=tmp_path
    )
    assert (
        sum(
            entry.artifact_type is ArtifactType.FOLD_STATE
            for entry in bundle.index.entries
        )
        == 1
    )
    assert all(
        entry.artifact_type is not ArtifactType.WALK_FORWARD_WINDOW
        for entry in bundle.index.entries
    )


def test_failure_history_cannot_disappear_after_source_capture(
    tmp_path: Path, captured_study: CapturedStudy
) -> None:
    source, path, _ = incomplete_source(captured_study)
    state = read_record(path)
    state["failures"] = []
    write_record(path, state)
    with pytest.raises(ManifestError, match="fold state differs from captured source"):
        inspect_validation(source, captured_study.study_path, artifact_root=tmp_path)


@pytest.mark.parametrize("delete", [False, True])
def test_pending_state_presence_cannot_change_after_source_capture(
    tmp_path: Path, captured_study: CapturedStudy, delete: bool
) -> None:
    source, _, path = incomplete_source(captured_study)
    state: PrimitiveMapping = {
        "study_id": source.study_id,
        "fold_id": source.folds[1].fold_id,
        "status": "pending",
        "selection_id": None,
        "artifact_id": None,
        "failures": [],
    }
    write_record(path, state)
    if delete:
        source = load_oos_source(source.plan, captured_study.study_path)
        # A captured, intact pending record is also a valid input.
        inspect_validation(source, captured_study.study_path, artifact_root=tmp_path)
        path.unlink()
    with pytest.raises(ManifestError, match="fold state"):
        inspect_validation(source, captured_study.study_path, artifact_root=tmp_path)
