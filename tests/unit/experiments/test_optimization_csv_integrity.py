"""Indexed optimization tables must agree with the validated structured results."""

import csv
import io
from pathlib import Path
from typing import Any

import pytest

from quantforge.experiments import ManifestError, StudyType, adapters, inspect_study
from quantforge.experiments._producer_snapshot import ProducerReadSet
from quantforge.experiments.artifacts import ArtifactEntry
from tests.unit.experiments.test_grid_integrity import build_grid_export

TABLES = (
    "trials.csv",
    "eligible_rankings.csv",
    "ineligible_trials.csv",
    "failures.csv",
    "exclusions.csv",
    "stability.csv",
    "parameter_summary.csv",
)


@pytest.mark.parametrize("filename", TABLES)
def test_optimization_csv_must_match_saved_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    path = root / filename
    rows = list(csv.reader(io.StringIO(path.read_text())))
    if len(rows) == 1:
        rows.append([""] * len(rows[0]))
    rows[1][0] = "foreign"
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    path.write_text(stream.getvalue())
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ManifestError, match="optimization CSV"):
        inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("filename", TABLES)
def test_completed_optimization_requires_every_native_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    (root / filename).unlink()
    with pytest.raises(ManifestError, match="optimization CSV"):
        inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)


@pytest.mark.parametrize("complete", [False, True])
def test_only_completed_native_optimization_tables_are_indexed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    (root / "unrelated.csv").write_text("foreign\n")
    if not complete:
        for name in ("summary.json", "ranking.json", "stability.json"):
            (root / name).unlink()
        for name in TABLES:
            (root / name).write_text("stale\n")
    bundle = inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    indexed = {
        path.name
        for entry in bundle.index.entries
        if (path := tmp_path / entry.path).parent == root and path.suffix == ".csv"
    }
    assert indexed == (set(TABLES) if complete else set())


@pytest.mark.parametrize("changed", [False, True])
def test_csv_replacement_after_comparison_is_checked_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: bool,
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    original = ProducerReadSet.expect
    path = root / "trials.csv"
    replacement = path.read_bytes() + (b"\n" if changed else b"")
    updated = False

    def replace_csv(reads: ProducerReadSet, target: Path, content: bytes) -> None:
        nonlocal updated
        original(reads, target, content)
        if target == path:
            temporary = path.with_suffix(".replacement")
            temporary.write_bytes(replacement)
            temporary.replace(path)
            updated = True

    monkeypatch.setattr(ProducerReadSet, "expect", replace_csv)
    if changed:
        with pytest.raises(ManifestError, match="changed during indexing"):
            inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    else:
        inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    assert updated
    assert path.read_bytes() == replacement


def test_summary_disappearing_after_csv_indexing_cannot_skip_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    original = adapters.index_artifact
    summary_path = root / "summary.json"

    def remove_summary(*args: Any, **kwargs: Any) -> ArtifactEntry:
        entry = original(*args, **kwargs)
        if Path(entry.path).name == "trials.csv":
            summary_path.unlink()
        return entry

    monkeypatch.setattr(adapters, "index_artifact", remove_summary)
    with pytest.raises(ManifestError, match="changed during indexing"):
        inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    assert not summary_path.exists()
