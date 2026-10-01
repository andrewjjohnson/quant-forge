"""QF-73: only a regular Finder `.DS_Store` file is tolerated in QF-39 `folds/`."""

import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from quantforge.backtesting.errors import ResultExportError
from quantforge.experiments import (
    StudyType,
    create_manifest,
    inspect_validation,
    verify_artifacts,
    write_manifest,
)
from quantforge.oos import (
    BacktestOOSAggregate,
    HoldoutLedger,
    OOSIntegrityError,
    OOSSource,
    PredictionOOSAggregate,
    aggregate_backtest,
    aggregate_prediction,
    export_oos_aggregate,
    load_oos_source,
)
from quantforge.oos._records import mapping, records
from quantforge.oos.source import FINDER_METADATA_FILENAME
from quantforge.reporting import build_research_report, export_research_report
from quantforge.walk_forward.models import WalkForwardPersistenceError
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import execution
from tests.unit.reporting.test_research_validation import block_holdout

from .conftest import CompletedStudy, complete_study
from .test_source import rewrite_oos

# The header of a real Finder file; the loader never reads these bytes.
FINDER_BYTES = b"\x00\x00\x00\x01Bud1" + bytes(26)
POLICY = "; only planned fold IDs and a regular Finder .DS_Store file are permitted"


def folds_path(study: CompletedStudy) -> Path:
    return study.study.study_path / "folds"


def fold_root(study: CompletedStudy, index: int = 0) -> Path:
    return folds_path(study) / study.source.folds[index].fold_id


def add_finder_metadata(directory: Path) -> Path:
    path = directory / FINDER_METADATA_FILENAME
    path.write_bytes(FINDER_BYTES)
    return path


def file_digests(root: Path) -> dict[str, str]:
    """Every scientific file's bytes, excluding Finder metadata itself."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != FINDER_METADATA_FILENAME
    }


def aggregate(source: OOSSource) -> PredictionOOSAggregate | BacktestOOSAggregate:
    return (
        aggregate_prediction(source)
        if source.plan.environment.study_type.value == "prediction"
        else aggregate_backtest(source)
    )


def reload(study: CompletedStudy) -> OOSSource:
    return load_oos_source(study.source.plan, study.study.study_path)


def test_finder_metadata_file_loads_identically(
    completed_study: CompletedStudy, tmp_path: Path
) -> None:
    study_path = completed_study.study.study_path
    before = file_digests(study_path)
    reference = aggregate(completed_study.source)
    finder = add_finder_metadata(folds_path(completed_study))
    source = reload(completed_study)
    assert source == completed_study.source
    assert source.study_id == completed_study.source.study_id
    assert source.lineage_id == completed_study.source.lineage_id
    assert [item.to_primitive() for item in source.references] == [
        item.to_primitive() for item in completed_study.source.references
    ]
    assert all(
        FINDER_METADATA_FILENAME not in str(item.to_primitive())
        for item in source.references
    )
    tolerated = aggregate(source)
    assert tolerated.aggregate_id == reference.aggregate_id
    assert tolerated.to_primitive() == reference.to_primitive()
    plain = export_oos_aggregate(reference, tmp_path / "plain")
    with_finder = export_oos_aggregate(tolerated, tmp_path / "finder")
    assert plain.name == with_finder.name
    assert plain.read_bytes() == with_finder.read_bytes()
    # Reading writes nothing, and the Finder file is neither removed nor changed.
    assert file_digests(study_path) == before
    assert finder.read_bytes() == FINDER_BYTES


def test_finder_metadata_inside_fold_directories_is_not_listed(
    prediction_study: CompletedStudy,
) -> None:
    # Finder writes into every browsed folder; the loader reads fold files by
    # exact name, so nested metadata never needs (or receives) an exception.
    root = fold_root(prediction_study)
    add_finder_metadata(folds_path(prediction_study))
    for directory in (root, *(path for path in root.rglob("*") if path.is_dir())):
        add_finder_metadata(directory)
    assert reload(prediction_study) == prediction_study.source


def test_unexpected_entries_fail_with_sorted_diagnostics(
    prediction_study: CompletedStudy,
) -> None:
    folds = folds_path(prediction_study)
    add_finder_metadata(folds)
    (folds / "zz-notes.txt").write_text("notes")
    (folds / "aa-extra").mkdir()
    (folds / ".hidden").write_bytes(b"")
    (folds / "._.DS_Store").write_bytes(FINDER_BYTES)
    expected = (
        "duplicate or unexpected fold artifact directory entries: "
        "'folds/._.DS_Store' (file), 'folds/.hidden' (file), "
        "'folds/aa-extra' (directory), 'folds/zz-notes.txt' (file)" + POLICY
    )
    for _ in range(2):
        with pytest.raises(OOSIntegrityError) as raised:
            reload(prediction_study)
        assert str(raised.value) == expected


def test_diagnostics_are_bounded_and_deterministic(
    prediction_study: CompletedStudy,
) -> None:
    folds = folds_path(prediction_study)
    for index in reversed(range(12)):
        (folds / f"extra-{index:02d}").write_bytes(b"")
    with pytest.raises(OOSIntegrityError) as raised:
        reload(prediction_study)
    reported = ", ".join(f"'folds/extra-{index:02d}' (file)" for index in range(10))
    assert str(raised.value) == (
        "duplicate or unexpected fold artifact directory entries: "
        f"{reported} and 2 more" + POLICY
    )


@pytest.mark.parametrize(
    "name",
    [
        "notes.txt",
        ".ds_store",
        "DS_Store",
        ".DS_Store.tmp",
        "._.DS_Store",
        ".hidden",
        ".localized",
        "Thumbs.db",
        "desktop.ini",
    ],
)
@pytest.mark.parametrize("kind", ["file", "directory"])
def test_other_names_are_not_tolerated(
    completed_study: CompletedStudy, name: str, kind: str
) -> None:
    path = folds_path(completed_study) / name
    if kind == "file":
        path.write_bytes(FINDER_BYTES)
    else:
        path.mkdir()
    with pytest.raises(OOSIntegrityError, match="unexpected") as raised:
        reload(completed_study)
    assert f"'folds/{name}' ({kind})" in str(raised.value)


def _directory(path: Path, outside: Path) -> None:
    del outside
    path.mkdir()


def _directory_with_content(path: Path, outside: Path) -> None:
    del outside
    path.mkdir()
    (path / "state.json").write_bytes(b"{}")


def _symlink_to_file(path: Path, outside: Path) -> None:
    outside.write_bytes(FINDER_BYTES)
    path.symlink_to(outside)


def _symlink_to_fold(path: Path, outside: Path) -> None:
    del outside
    fold = next(entry for entry in path.parent.iterdir() if entry.is_dir())
    path.symlink_to(fold, target_is_directory=True)


def _dangling_symlink(path: Path, outside: Path) -> None:
    path.symlink_to(outside / "missing")


def _fifo(path: Path, outside: Path) -> None:
    del outside
    os.mkfifo(path)


@pytest.mark.parametrize(
    ("create", "kind"),
    [
        (_directory, "directory"),
        (_directory_with_content, "directory"),
        (_symlink_to_file, "symlink"),
        (_symlink_to_fold, "symlink"),
        (_dangling_symlink, "symlink"),
        (_fifo, "special file"),
    ],
    ids=[
        "directory",
        "nonempty-directory",
        "symlink",
        "fold-symlink",
        "dangling",
        "fifo",
    ],
)
def test_finder_name_that_is_not_a_regular_file_fails_closed(
    completed_study: CompletedStudy,
    tmp_path: Path,
    create: Callable[[Path, Path], None],
    kind: str,
) -> None:
    create(folds_path(completed_study) / FINDER_METADATA_FILENAME, tmp_path / "x")
    with pytest.raises(OOSIntegrityError) as raised:
        reload(completed_study)
    assert str(raised.value) == (
        "duplicate or unexpected fold artifact directory entries: "
        f"'folds/.DS_Store' ({kind})" + POLICY
    )


def test_duplicate_fold_still_fails_and_is_named(
    completed_study: CompletedStudy,
) -> None:
    add_finder_metadata(folds_path(completed_study))
    shutil.copytree(fold_root(completed_study), folds_path(completed_study) / "copy")
    with pytest.raises(OOSIntegrityError, match="duplicate") as raised:
        reload(completed_study)
    assert "'folds/copy' (directory)" in str(raised.value)
    assert "'folds/.DS_Store'" not in str(raised.value)


def _remove_state(root: Path) -> None:
    (root / "state.json").unlink()


def _remove_oos(root: Path) -> None:
    (root / "oos.json").unlink()


def _corrupt_oos(root: Path) -> None:
    path = root / "oos.json"
    path.write_text(path.read_text().replace('"fingerprint":"', '"fingerprint":"0', 1))


def _wrong_artifact_id(root: Path) -> None:
    state = read_record(root / "state.json")
    state["artifact_id"] = "0" * 64
    write_record(root / "state.json", state)


def _remove_selection(root: Path) -> None:
    (root / "selection.json").unlink()


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (_remove_state, OOSIntegrityError, "orphaned fold has no state"),
        (_remove_oos, WalkForwardPersistenceError, "corrupt"),
        (_corrupt_oos, WalkForwardPersistenceError, "corrupt"),
        (_wrong_artifact_id, OOSIntegrityError, "incompatible OOS artifact"),
        (_remove_selection, OOSIntegrityError, "frozen selection"),
    ],
    ids=[
        "missing-state",
        "missing-oos",
        "corrupt-oos",
        "artifact-id",
        "missing-selection",
    ],
)
def test_missing_or_corrupt_evidence_still_fails_with_finder_metadata(
    completed_study: CompletedStudy,
    mutate: Callable[[Path], None],
    error: type[Exception],
    match: str,
) -> None:
    root = fold_root(completed_study)
    add_finder_metadata(folds_path(completed_study))
    add_finder_metadata(root)
    mutate(root)
    with pytest.raises(error, match=match):
        reload(completed_study)


def test_altered_prediction_window_still_fails_with_finder_metadata(
    prediction_study: CompletedStudy,
) -> None:
    root = fold_root(prediction_study)
    add_finder_metadata(folds_path(prediction_study))
    artifact = read_record(root / "oos.json")
    payload = mapping(artifact["result"])
    records(payload["decisions"])[0]["decision_timestamp"] = "2024-07-08T17:30:00+00:00"
    rewrite_oos(root, artifact)
    with pytest.raises(
        ValueError, match=r"identity|timestamp|backend|provenance|context"
    ):
        reload(prediction_study)


def test_tampered_backtest_export_still_fails_with_finder_metadata(
    backtest_study: CompletedStudy,
) -> None:
    add_finder_metadata(folds_path(backtest_study))
    trades = next((fold_root(backtest_study) / "test").rglob("trades.csv"))
    trades.write_text(trades.read_text() + "\n")
    with pytest.raises(ResultExportError, match="integrity validation failed"):
        reload(backtest_study)


def test_stateless_fold_with_only_finder_metadata_is_orphaned(
    completed_study: CompletedStudy,
) -> None:
    # QF-39 writes state.json before anything else in a fold directory, so
    # Finder can only add metadata after state exists. A stateless directory
    # holding only metadata means its state was removed: fail closed.
    root = fold_root(completed_study, 1)
    shutil.rmtree(root)
    root.mkdir()
    add_finder_metadata(root)
    with pytest.raises(OOSIntegrityError, match="orphaned fold has no state"):
        reload(completed_study)


def publish(
    study: CompletedStudy, root: Path, ledger: HoldoutLedger
) -> dict[str, bytes]:
    """The QF-40 -> QF-9 -> QF-41 consumer chain over existing artifacts only."""
    source = reload(study)
    aggregate_path = export_oos_aggregate(aggregate(source), root / "oos")
    with pytest.MonkeyPatch.context() as patch:
        block_research(patch)
        block_holdout(patch)
        manifest = create_manifest(
            inspect_validation(
                source,
                study.study.study_path,
                artifact_root=root,
                aggregate_path=aggregate_path,
                ledger=ledger,
                study_type=StudyType.OOS_VALIDATION,
            ),
            execution(),
        )
        assert all(
            FINDER_METADATA_FILENAME not in entry.path
            for entry in manifest.artifacts.entries
        )
        manifest_path = write_manifest(
            manifest, root / "experiments", artifact_root=root
        )
        verify_artifacts(manifest.artifacts, root).require_valid()
        report = build_research_report(
            manifest_path,
            artifact_root=root,
            holdout_source=source,
            holdout_ledger=ledger,
        )
        html = export_research_report(report, root / "html")
    assert report.header.to_primitive()["holdout_state"] == "reserved_unconsumed"
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in (aggregate_path, manifest_path, html)
    }


@pytest.mark.parametrize("prediction", [True, False], ids=["prediction", "backtest"])
def test_downstream_consumers_read_finder_metadata_study_without_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction: bool
) -> None:
    completed = complete_study(tmp_path, prediction=prediction)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("consumers must not select, evaluate or configure strategies")

    for name in ("select", "evaluate", "configuration"):
        monkeypatch.setattr(type(completed.evaluator), name, forbidden)
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(completed.source)
    reference = publish(completed, tmp_path, ledger)
    study_bytes = file_digests(completed.study.study_path)
    add_finder_metadata(folds_path(completed))
    add_finder_metadata(fold_root(completed))
    # Identical content-addressed outputs: exact re-publication verifies bytes.
    assert publish(completed, tmp_path, ledger) == reference
    assert file_digests(completed.study.study_path) == study_bytes
    assert ledger.state(completed.source).state.value == "reserved_unconsumed"
    assert not list((ledger.root / "exposures").iterdir())


# Audited listings outside the `folds/` membership check keep their behavior.


def test_finder_metadata_inside_a_backtest_export_still_fails_closed(
    backtest_study: CompletedStudy,
) -> None:
    # QF-5 immutable exports require their exact file set (not changed here).
    add_finder_metadata(folds_path(backtest_study))
    export = next(path for path in (fold_root(backtest_study) / "test").iterdir())
    add_finder_metadata(export)
    with pytest.raises(ResultExportError, match="invalid immutable backtest artifact"):
        reload(backtest_study)


def test_holdout_ledger_audit_is_unchanged(
    prediction_study: CompletedStudy, tmp_path: Path
) -> None:
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(prediction_study.source)
    add_finder_metadata(ledger.root)
    add_finder_metadata(ledger.root / "exposures")
    assert ledger.state(prediction_study.source).state.value == "reserved_unconsumed"
    add_finder_metadata(ledger.root / "lineages")
    with pytest.raises(OOSIntegrityError, match="invalid holdout lineage directory"):
        ledger.state(prediction_study.source)
