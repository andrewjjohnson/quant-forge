"""Grid summaries must belong to the same snapshot as their validation inputs."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from quantforge.experiments import (
    ManifestError,
    StudyType,
    adapters,
    inspect_study,
    verify_artifacts,
)
from quantforge.experiments._producer_snapshot import ProducerReadSet
from quantforge.experiments.artifacts import ArtifactEntry, ArtifactIndex
from tests.unit.experiments.test_grid_integrity import grid_export as grid_export
from tests.unit.experiments.test_inspection_races import replace_bytes


def files(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize(
    "phase",
    [
        "after_absence_check",
        "after_directory_listing",
        "after_verification",
        "after_trial_recheck",
    ],
)
@pytest.mark.parametrize("valid", [False, True], ids=["invalid", "native"])
def test_new_summary_requires_a_fresh_grid_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grid_export: tuple[StudyType, Path],
    phase: str,
    valid: bool,
) -> None:
    study_type, root = grid_export
    target = root / "summary.json"
    native = target.read_bytes()
    content = native if valid else b'{"not":"a validated summary"}'
    target.unlink()
    before = files(root)
    added = indexed_summary = False
    original_is_file = Path.is_file
    original_iterdir = Path.iterdir
    original_glob = Path.glob
    trial_listings = 0
    original_verify = ProducerReadSet.verify
    original_index = adapters.index_artifact

    def add_summary() -> None:
        nonlocal added
        replace_bytes(target, content)
        added = True

    def is_file_then_add(path: Path) -> bool:
        present = original_is_file(path)
        if path == target and not added:
            assert not present
            add_summary()
        return present

    def iterdir_then_add(path: Path) -> Iterator[Path]:
        found = list(original_iterdir(path))
        if path == root and not added:
            add_summary()
        return iter(found)

    def verify_then_add(
        self: ProducerReadSet, index: ArtifactIndex, artifact_root: Path
    ) -> None:
        original_verify(self, index, artifact_root)
        if not added:
            add_summary()

    def glob_then_add(path: Path, pattern: str, **kwargs: Any) -> Iterator[Path]:
        nonlocal trial_listings
        found = list(original_glob(path, pattern, **kwargs))
        if path == root / "trials" and pattern == "*.json":
            trial_listings += 1
            if trial_listings == 2 and not added:
                add_summary()
        return iter(found)

    def record_index(artifact_root: Path, **kwargs: Any) -> ArtifactEntry:
        nonlocal indexed_summary
        if kwargs["path"] == target.relative_to(tmp_path).as_posix():
            indexed_summary = True
        return original_index(artifact_root, **kwargs)

    with monkeypatch.context() as race:
        race.setattr(adapters, "index_artifact", record_index)
        if phase == "after_absence_check":
            race.setattr(Path, "is_file", is_file_then_add)
        elif phase == "after_directory_listing":
            race.setattr(Path, "iterdir", iterdir_then_add)
        elif phase == "after_verification":
            race.setattr(ProducerReadSet, "verify", verify_then_add)
        else:
            race.setattr(Path, "glob", glob_then_add)
        with pytest.raises(ManifestError, match="changed during indexing"):
            inspect_study(study_type, root, artifact_root=tmp_path)
    assert added
    assert not indexed_summary
    assert files(root) == {**before, target: content}
    if not valid:
        with pytest.raises(ManifestError):
            inspect_study(study_type, root, artifact_root=tmp_path)
        replace_bytes(target, native)
    fresh = inspect_study(study_type, root, artifact_root=tmp_path)
    assert verify_artifacts(fresh.index, tmp_path).valid
    assert any(
        entry.path == target.relative_to(tmp_path).as_posix()
        for entry in fresh.index.entries
    )


def test_summary_removed_after_byte_verification_requires_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grid_export: tuple[StudyType, Path],
) -> None:
    study_type, root = grid_export
    target = root / "summary.json"
    before = files(root)
    original_verify = ProducerReadSet.verify

    def verify_then_remove(
        self: ProducerReadSet, index: ArtifactIndex, artifact_root: Path
    ) -> None:
        original_verify(self, index, artifact_root)
        target.unlink()

    with monkeypatch.context() as race:
        race.setattr(ProducerReadSet, "verify", verify_then_remove)
        with pytest.raises(ManifestError, match="changed during indexing"):
            inspect_study(study_type, root, artifact_root=tmp_path)
    assert files(root) == {
        path: content for path, content in before.items() if path != target
    }
    replace_bytes(target, before[target])
    fresh = inspect_study(study_type, root, artifact_root=tmp_path)
    assert verify_artifacts(fresh.index, tmp_path).valid


@pytest.mark.parametrize("replacement", [False, True], ids=["temporary", "identical"])
def test_unchanged_membership_preserves_the_original_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grid_export: tuple[StudyType, Path],
    replacement: bool,
) -> None:
    study_type, root = grid_export
    target = root / "summary.json"
    content = target.read_bytes()
    if not replacement:
        target.unlink()
    before = files(root)
    original = inspect_study(study_type, root, artifact_root=tmp_path)
    original_verify = ProducerReadSet.verify
    temporary = root / "summary.json.tmp"

    def verify_then_write(
        self: ProducerReadSet, index: ArtifactIndex, artifact_root: Path
    ) -> None:
        original_verify(self, index, artifact_root)
        if replacement:
            replace_bytes(target, content)
        else:
            temporary.write_bytes(content)

    with monkeypatch.context() as race:
        race.setattr(ProducerReadSet, "verify", verify_then_write)
        observed = inspect_study(study_type, root, artifact_root=tmp_path)
    assert observed == original
    assert verify_artifacts(observed.index, tmp_path).valid
    assert files(root) == (before if replacement else {**before, temporary: content})
