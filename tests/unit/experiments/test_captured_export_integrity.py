from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import validate_backtest_result_artifact
from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    inspect_validation,
)
from quantforge.experiments.persistence import read_producer_record
from quantforge.oos import HoldoutEvaluation, HoldoutLedger
from quantforge.walk_forward.models import BacktestOOSArtifact
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record
from tests.unit.oos.conftest import complete_study


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
    changed = (export / filename).read_bytes() + b"\n"
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
def test_nested_backtest_files_retain_the_standalone_result_schema(
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
    expected_schema = cast(
        str, read_record(export / "manifest.json")["result_schema_version"]
    )
    files = {
        entry.path: entry.schema_version
        for entry in standalone.index.entries
        if not entry.json_pointer
    }
    assert files
    assert set(files.values()) == {expected_schema}
    assert {
        entry.path: entry.schema_version
        for entry in nested.index.entries
        if entry.path in files
    } == files
