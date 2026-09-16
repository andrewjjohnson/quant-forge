"""A grid index cannot silently omit trials committed after its initial listing."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.experiments._producer_snapshot import ProducerReadSet
from quantforge.experiments.artifacts import ArtifactIndex
from tests.unit.experiments.test_grid_integrity import build_grid_export, trial_path
from tests.unit.experiments.test_inspection_races import replace_bytes


@pytest.mark.parametrize(
    "study_type", [StudyType.PARAMETER_STUDY, StudyType.OPTIMIZATION]
)
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("phase", ["after_listing", "after_verification"])
def test_new_trial_requires_a_fresh_grid_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_type: StudyType,
    empty: bool,
    phase: str,
) -> None:
    root = build_grid_export(tmp_path, study_type, monkeypatch)
    for name in ("summary.json", "ranking.json", "stability.json"):
        (root / name).unlink(missing_ok=True)
    target = trial_path(root, "excluded")
    content = target.read_bytes()
    for path in (root / "trials").glob("*.json"):
        if empty or path == target:
            path.unlink()
    original_index = inspect_study(study_type, root, artifact_root=tmp_path).index
    assert verify_artifacts(original_index, tmp_path).valid
    added = False
    original_glob = Path.glob
    original_verify = ProducerReadSet.verify

    def add() -> None:
        nonlocal added
        replace_bytes(target, content)
        added = True

    def glob_then_add(path: Path, pattern: str, **kwargs: Any) -> Iterator[Path]:
        found = list(original_glob(path, pattern, **kwargs))
        if path == root / "trials" and pattern == "*.json" and not added:
            add()
        return iter(found)

    def verify_then_add(
        self: ProducerReadSet, index: ArtifactIndex, artifact_root: Path
    ) -> None:
        original_verify(self, index, artifact_root)
        if not added:
            add()

    with monkeypatch.context() as race:
        if phase == "after_listing":
            race.setattr(Path, "glob", glob_then_add)
        else:
            race.setattr(ProducerReadSet, "verify", verify_then_add)
        with pytest.raises(ManifestError, match="changed during indexing"):
            inspect_study(study_type, root, artifact_root=tmp_path)
    assert added
    assert target.read_bytes() == content
    fresh = inspect_study(study_type, root, artifact_root=tmp_path)
    assert verify_artifacts(fresh.index, tmp_path).valid
    assert any(
        entry.path == target.relative_to(tmp_path).as_posix()
        for entry in fresh.index.entries
    )


@pytest.mark.parametrize(
    "study_type", [StudyType.PARAMETER_STUDY, StudyType.OPTIMIZATION]
)
def test_unowned_temporary_trial_file_does_not_change_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, study_type: StudyType
) -> None:
    root = build_grid_export(tmp_path, study_type, monkeypatch)
    original = inspect_study(study_type, root, artifact_root=tmp_path)
    (root / "trials" / "writing.tmp").write_text("partial temporary output")
    observed = inspect_study(study_type, root, artifact_root=tmp_path)
    assert observed.index == original.index
