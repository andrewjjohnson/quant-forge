"""Nested exports must retain the same benchmark contract as standalone QF-5."""

from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import validate_backtest_result_artifact
from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    inspect_validation,
)
from quantforge.experiments._json import mapping
from quantforge.oos import HoldoutEvaluation, HoldoutLedger, load_oos_source
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    trial_path,
)
from tests.unit.experiments.test_grid_integrity import (
    read_record as read_json,
)
from tests.unit.oos.conftest import complete_study


def rewrite_benchmark(export: Path, change: str) -> PrimitiveMapping:
    manifest = read_json(export / "manifest.json")
    benchmark = mapping(manifest["benchmark"])
    if change == "missing_benchmark":
        del manifest["benchmark"]
    elif change == "missing_configuration":
        del benchmark["configuration"]
    elif change == "null_configuration":
        benchmark["configuration"] = None
    elif change == "benchmark_id":
        benchmark["benchmark_id"] = "0" * 64
    else:
        configuration = mapping(benchmark["configuration"])
        configuration[change] = (
            {"model": "fixed_per_order", "amount": "999"}
            if change == "commission"
            else "999"
        )
        benchmark["benchmark_id"] = configuration_identity(
            {
                "run_id": manifest["run_id"],
                "record_type": "benchmark",
                "configuration": configuration,
            }
        )
    write_json(export / "manifest.json", manifest)
    integrity = read_json(export / "integrity.json")
    mapping(integrity["files"])["manifest.json"] = sha256(
        (export / "manifest.json").read_bytes()
    ).hexdigest()
    write_json(export / "integrity.json", integrity)
    assert validate_backtest_result_artifact(export) == export
    return manifest


@pytest.mark.parametrize("complete", [False, True], ids=["resumable", "complete"])
@pytest.mark.parametrize(
    "change",
    [
        "initial_capital",
        "commission",
        "slippage",
        "evaluation_interval",
        "benchmark_id",
        "missing_benchmark",
        "missing_configuration",
        "null_configuration",
    ],
)
def test_optimization_rejects_rehashed_foreign_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool, change: str
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    if not complete:
        (root / "summary.json").unlink()
    inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    trial = read_json(trial_path(root))
    export = root / cast(str, trial["artifact_location"])
    rewrite_benchmark(export, change)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ManifestError):
        inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("holdout", [False, True], ids=["fold", "holdout"])
@pytest.mark.parametrize("change", ["initial_capital", "benchmark_id"])
def test_validation_rejects_rehashed_foreign_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, holdout: bool, change: str
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
            run_id="benchmark-holdout",
        )
        assert consumed.result_reference is not None
        path = ledger.root / cast(str, consumed.result_reference.to_primitive()["path"])
        result = read_record(path)
        artifact = mapping(result["artifact"])
        export = path.parent / "evaluation" / cast(str, artifact["export_location"])
    else:
        fold_root = completed.study.study_path / "folds" / source.folds[0].fold_id
        path = fold_root / "oos.json"
        artifact = read_record(path)
        export = fold_root / "test" / cast(str, artifact["export_location"])
    block_research(monkeypatch)
    inspect_validation(
        source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
    )
    mapping(artifact["result"])["manifest"] = rewrite_benchmark(export, change)
    artifact["export_fingerprint"] = configuration_identity(
        {"integrity": (export / "integrity.json").read_text()}
    )
    if holdout:
        result = read_record(path)
        result["artifact"] = artifact
        result["artifact_sha256"] = configuration_identity(artifact)
        write_record(path, result)
    else:
        write_record(path, artifact)
        state_path = path.parent / "state.json"
        state = read_record(state_path)
        state["artifact_id"] = configuration_identity(artifact)
        write_record(state_path, state)
        source = load_oos_source(source.plan, completed.study.study_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ManifestError, match="backtest benchmark"):
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )
    assert {path: path.read_bytes() for path in before} == before
    if ledger is not None:
        assert ledger.state(source).state.value == "consumed"
