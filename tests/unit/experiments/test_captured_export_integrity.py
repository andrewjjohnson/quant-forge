from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import (
    ResultExportError,
    export_backtest_result,
    validate_backtest_result_artifact,
)
from quantforge.backtesting.export import BACKTEST_ARTIFACT_FILENAMES
from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import (
    ArtifactRelationship,
    ArtifactType,
    ManifestError,
    RelationshipType,
    StudyType,
    inspect_study,
    inspect_validation,
)
from quantforge.experiments.persistence import read_producer_record
from quantforge.oos import HoldoutEvaluation, HoldoutLedger
from quantforge.walk_forward.models import BacktestOOSArtifact
from tests.unit.backtesting.test_runner import configured_result
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import build_grid_export, read_record
from tests.unit.oos.conftest import complete_study


def test_backtest_export_manifest_and_directory_have_identical_complete_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = export_backtest_result(configured_result(), tmp_path)
    block_research(monkeypatch)
    directory = inspect_study(StudyType.BACKTEST, export, artifact_root=tmp_path)
    manifest = inspect_study(
        StudyType.BACKTEST, export / "manifest.json", artifact_root=tmp_path
    )
    assert manifest == directory
    assert {Path(entry.path).name for entry in manifest.index.entries} == set(
        BACKTEST_ARTIFACT_FILENAMES
    )


@pytest.mark.parametrize(
    "filename",
    ["orders.csv", "fills.csv", "trades.csv", "equity.csv", "integrity.json"],
)
@pytest.mark.parametrize("missing", [False, True], ids=["modified", "missing"])
def test_backtest_manifest_input_requires_intact_sibling_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    missing: bool,
) -> None:
    export = export_backtest_result(configured_result(), tmp_path)
    if missing:
        (export / filename).unlink()
    else:
        (export / filename).write_text("tampered\n")
    block_research(monkeypatch)
    with pytest.raises(ResultExportError):
        inspect_study(
            StudyType.BACKTEST, export / "manifest.json", artifact_root=tmp_path
        )


@pytest.mark.parametrize("holdout", [False, True], ids=["fold", "holdout"])
@pytest.mark.parametrize("filename", ["orders.csv", "fills.csv", "trades.csv"])
def test_rehashed_backtest_tables_must_match_captured_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, holdout: bool, filename: str
) -> None:
    completed = complete_study(tmp_path, prediction=False)
    source = completed.source
    ledger = None
    if holdout:
        ledger = HoldoutLedger.create(tmp_path / "ledger")
        ledger.reserve(source)
        evaluation = HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        )
        consumed = ledger.consume(evaluation, run_id="captured-holdout")
        assert consumed.result_reference is not None
        result_path = ledger.root / cast(
            str, consumed.result_reference.to_primitive()["path"]
        )
        result, _ = read_producer_record(result_path)
        artifact = cast(PrimitiveMapping, result["artifact"])
        export = (
            result_path.parent / "evaluation" / cast(str, artifact["export_location"])
        )
    else:
        fold = source.folds[0]
        assert isinstance(fold.artifact, BacktestOOSArtifact)
        export = (
            completed.study.study_path
            / "folds"
            / fold.fold_id
            / "test"
            / fold.artifact.export_location
        )
    # Change bytes while retaining valid CSV records and counts, so this still
    # exercises the captured-export fingerprint rather than row validation.
    changed = (export / filename).read_bytes().replace(b"\n", b"\r\n")
    (export / filename).write_bytes(changed)
    integrity = read_record(export / "integrity.json")
    cast(PrimitiveMapping, integrity["files"])[filename] = sha256(changed).hexdigest()
    write_json(export / "integrity.json", integrity)
    assert validate_backtest_result_artifact(export) == export
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=r"export.*captured fingerprint"):
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )


@pytest.mark.parametrize("holdout", [False, True], ids=["fold", "holdout"])
def test_nested_backtest_files_retain_the_standalone_schema_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, holdout: bool
) -> None:
    completed = complete_study(tmp_path, prediction=False)
    source = completed.source
    ledger = None
    if holdout:
        ledger = HoldoutLedger.create(tmp_path / "ledger")
        ledger.reserve(source)
        consumed = ledger.consume(
            HoldoutEvaluation.prepare(
                source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
            ),
            run_id="schema-holdout",
        )
        assert consumed.result_reference is not None
        result_path = ledger.root / cast(
            str, consumed.result_reference.to_primitive()["path"]
        )
        result, _ = read_producer_record(result_path)
        export = (
            result_path.parent
            / "evaluation"
            / cast(str, cast(PrimitiveMapping, result["artifact"])["export_location"])
        )
    else:
        fold = source.folds[0]
        assert isinstance(fold.artifact, BacktestOOSArtifact)
        export = (
            completed.study.study_path
            / "folds"
            / fold.fold_id
            / "test"
            / fold.artifact.export_location
        )
    block_research(monkeypatch)
    standalone = inspect_study(StudyType.BACKTEST, export, artifact_root=tmp_path)
    nested = inspect_validation(
        source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
    )
    backtest = read_record(export / "manifest.json")
    expected_schema = cast(str, backtest["result_schema_version"])
    expected_run = cast(str, backtest["run_id"])
    files = {
        entry.path: (
            entry.schema_version,
            entry.producer_study_id,
            entry.producer_run_id,
        )
        for entry in standalone.index.entries
        if not entry.json_pointer
    }
    assert files
    assert set(files.values()) == {(expected_schema, expected_run, expected_run)}
    assert {
        entry.path: (
            entry.schema_version,
            entry.producer_study_id,
            entry.producer_run_id,
        )
        for entry in nested.index.entries
        if entry.path in files
    } == files
    entries = {entry.artifact_id: entry for entry in nested.index.entries}
    for entry in nested.index.entries:
        if entry.path in files:
            parents = [
                entries[edge.target_id]
                for edge in nested.index.relationships
                if edge.source_id == entry.artifact_id
                and edge.relationship is RelationshipType.DERIVED_FROM
            ]
            assert len(parents) == 1
            assert parents[0].artifact_type is (
                ArtifactType.HOLDOUT_RESULT
                if holdout
                else ArtifactType.WALK_FORWARD_WINDOW
            )


def test_optimization_backtests_retain_each_run_schema_and_trial_relationship(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    bundle = inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    run_ids: set[str] = set()
    for trial_entry in bundle.index.entries:
        if trial_entry.artifact_type is not ArtifactType.TRIAL_RESULT:
            continue
        trial = read_record(tmp_path / trial_entry.path)
        assert trial_entry.producer_study_id == bundle.provenance.producer_study_id
        assert trial_entry.producer_run_id is None
        if trial["status"] != "succeeded":
            continue
        export = root / cast(str, trial["artifact_location"])
        backtest = read_record(export / "manifest.json")
        run_id = cast(str, backtest["run_id"])
        run_ids.add(run_id)
        standalone = inspect_study(StudyType.BACKTEST, export, artifact_root=tmp_path)
        files = {
            entry.path: entry
            for entry in standalone.index.entries
            if not entry.json_pointer
        }
        nested = {
            entry.path: entry for entry in bundle.index.entries if entry.path in files
        }
        assert files
        assert set(nested) == set(files)
        for path, entry in nested.items():
            assert (
                entry.schema_version
                == files[path].schema_version
                == backtest["result_schema_version"]
            )
            assert entry.producer_study_id == files[path].producer_study_id == run_id
            assert entry.producer_run_id == files[path].producer_run_id == run_id
            assert entry.sha256 == files[path].sha256
            assert (
                ArtifactRelationship(
                    entry.artifact_id,
                    RelationshipType.DERIVED_FROM,
                    trial_entry.artifact_id,
                )
                in bundle.index.relationships
            )
    assert len(run_ids) >= 2
