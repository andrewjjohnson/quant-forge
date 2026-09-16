import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from quantforge.experiments import ManifestError, capture_code_provenance
from tests.unit.experiments.test_contracts import manifest


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=QuantForge Test",
            "-c",
            "user.email=quantforge@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init")
    (root / "strategy.py").write_text("threshold = 1\n")
    (root / "uv.lock").write_text("version = 1\n")
    (root / ".gitignore").write_text(".env\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "Initial fixture")
    return root


@pytest.mark.parametrize("change", ["unstaged", "staged", "untracked", "deleted"])
def test_dirty_code_cannot_be_captured_as_identical_provenance(
    repository: Path, change: str
) -> None:
    clean = capture_code_provenance(repository)
    assert clean.git_dirty is False
    source = repository / "strategy.py"
    if change == "untracked":
        # A user's status preference must not hide executable untracked code.
        git(repository, "config", "status.showUntrackedFiles", "no")
        source = repository / "untracked_strategy.py"
    for threshold in (2, 3):
        source.write_text(f"threshold = {threshold}\n")
        if change == "staged":
            git(repository, "add", source.name)
        elif change == "deleted":
            source.unlink()
        with pytest.raises(ManifestError, match="dirty working tree"):
            capture_code_provenance(repository)


def test_clean_capture_is_stable_and_new_commit_changes_study_identity(
    repository: Path, tmp_path: Path
) -> None:
    original = capture_code_provenance(repository)
    assert original.git_commit == git(repository, "rev-parse", "HEAD")
    assert original.git_dirty is False
    assert original.dependency_lock_sha256 is not None
    (repository / ".env").write_text("TEST_SECRET=local-only-marker\n")
    assert capture_code_provenance(repository) == original
    (repository / "strategy.py").write_text("threshold = 2\n")
    git(repository, "add", "strategy.py")
    git(repository, "commit", "-m", "Change strategy")
    changed = capture_code_provenance(repository)
    assert changed.git_commit != original.git_commit
    assert changed.dependency_lock_sha256 == original.dependency_lock_sha256
    record = manifest(tmp_path)
    first = replace(record, execution=replace(record.execution, code=original))
    second = replace(record, execution=replace(record.execution, code=changed))
    assert first.study_id != second.study_id
    assert b"local-only-marker" not in first.serialize()


def test_unavailable_repository_stays_explicitly_unknown(tmp_path: Path) -> None:
    code = capture_code_provenance(tmp_path / "unavailable")
    assert code.git_commit is None
    assert code.git_dirty is None
    assert code.dependency_lock_sha256 is None
