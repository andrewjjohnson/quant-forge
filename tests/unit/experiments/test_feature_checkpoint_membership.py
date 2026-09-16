"""Feature indexing must retain one stable checkpoint membership snapshot."""

from collections.abc import Callable, Iterator
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
from quantforge.prediction import SignalFeatureDatasetResult
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_feature_integrity import (
    feature_result as feature_result,
)
from tests.unit.experiments.test_inspection_races import replace_bytes
from tests.unit.helpers import make_dataset
from tests.unit.prediction.test_feature_dataset import (
    FixtureCandidateRule,
    _build_fixture,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture(params=["after_listing", "after_verification"])
def phase(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def inspect_during_change(
    root: Path,
    artifact_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    change: Callable[[], None],
    *,
    reject: bool,
) -> ArtifactIndex:
    original = inspect_study(
        StudyType.FEATURE_DATASET, root, artifact_root=artifact_root
    ).index
    assert verify_artifacts(original, artifact_root).valid
    original_glob = Path.glob
    original_verify = ProducerReadSet.verify
    changed = False
    expected_files: dict[Path, bytes] = {}

    def change_once() -> None:
        nonlocal changed, expected_files
        if changed:
            return
        changed = True
        change()
        expected_files = {
            path: path.read_bytes()
            for path in artifact_root.rglob("*")
            if path.is_file()
        }

    def glob_then_change(path: Path, pattern: str, **kwargs: Any) -> Iterator[Path]:
        paths = list(original_glob(path, pattern, **kwargs))
        if path == root / "rows" and pattern == "*.json":
            change_once()
        return iter(paths)

    def verify_then_change(
        self: ProducerReadSet, index: ArtifactIndex, artifact_root: Path
    ) -> None:
        original_verify(self, index, artifact_root)
        change_once()

    with monkeypatch.context() as race:
        if phase == "after_listing":
            race.setattr(Path, "glob", glob_then_change)
        else:
            race.setattr(ProducerReadSet, "verify", verify_then_change)
        if reject:
            with pytest.raises(
                ManifestError,
                match=r"changed during indexing|cannot read producer metadata",
            ):
                inspect_study(
                    StudyType.FEATURE_DATASET, root, artifact_root=artifact_root
                )
        else:
            observed = inspect_study(
                StudyType.FEATURE_DATASET, root, artifact_root=artifact_root
            )
            assert observed.index == original
            assert verify_artifacts(observed.index, artifact_root).valid
    assert changed
    assert {
        path: path.read_bytes() for path in artifact_root.rglob("*") if path.is_file()
    } == expected_files
    return original


@pytest.mark.parametrize("change", ["add", "remove", "rename", "remove_directory"])
def test_checkpoint_membership_changes_require_a_fresh_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feature_result: SignalFeatureDatasetResult,
    phase: str,
    change: str,
) -> None:
    root = tmp_path / "features" / feature_result.dataset_id
    target = next((root / "rows").glob("*.json"))
    content = target.read_bytes()
    added_path = target.with_name("new.json")

    def mutate() -> None:
        if change == "add":
            replace_bytes(added_path, content)
        elif change == "remove":
            target.unlink()
        elif change == "rename":
            target.rename(added_path)
        else:
            (root / "rows").rename(root / "moved_rows")

    original = inspect_during_change(
        root, tmp_path, monkeypatch, phase, mutate, reject=True
    )
    if change == "remove_directory":
        (root / "moved_rows").rename(root / "rows")
    else:
        added_path.unlink(missing_ok=True)
        replace_bytes(target, content)
    fresh = inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)
    assert fresh.index == original
    assert verify_artifacts(fresh.index, tmp_path).valid


@pytest.mark.parametrize("remove_directory", [False, True])
def test_empty_checkpoint_directory_is_rechecked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    remove_directory: bool,
) -> None:
    result = _build_fixture(
        make_dataset(("100", "102")), FixtureCandidateRule(()), tmp_path / "features"
    )
    block_research(monkeypatch)
    root = tmp_path / "features" / result.dataset_id

    def mutate() -> None:
        if remove_directory:
            (root / "rows").rmdir()
        else:
            replace_bytes(root / "rows" / "new.json", b"{}")

    inspect_during_change(root, tmp_path, monkeypatch, phase, mutate, reject=True)


@pytest.mark.parametrize("temporary", [False, True], ids=["identical", "temporary"])
def test_identical_checkpoints_and_unowned_temporary_files_preserve_the_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feature_result: SignalFeatureDatasetResult,
    phase: str,
    temporary: bool,
) -> None:
    root = tmp_path / "features" / feature_result.dataset_id
    target = next((root / "rows").glob("*.json"))

    def mutate() -> None:
        if temporary:
            (root / "rows" / "writing.tmp").write_text("incomplete checkpoint")
        else:
            replace_bytes(target, target.read_bytes())

    inspect_during_change(root, tmp_path, monkeypatch, phase, mutate, reject=False)
