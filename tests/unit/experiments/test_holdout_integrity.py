from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import (
    EvaluationInterval,
    FixedCommission,
    export_backtest_result,
    run_backtest,
    validate_backtest_result_artifact,
)
from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.experiments import ManifestError, inspect_validation
from quantforge.experiments.persistence import read_producer_record
from quantforge.oos import HoldoutEvaluation, HoldoutLedger
from quantforge.walk_forward import BacktestEvaluator
from quantforge.walk_forward.models import BacktestOOSArtifact
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.helpers import make_dataset
from tests.unit.oos.conftest import complete_study


@pytest.mark.parametrize(
    "change", ["metric", "missing_metric", "extra_metric", "empty", "null"]
)
@pytest.mark.parametrize("prediction", [False, True], ids=["backtest", "prediction"])
def test_consumed_summary_must_match_captured_performance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str, prediction: bool
) -> None:
    completed = complete_study(tmp_path, prediction=prediction)
    source = completed.source
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    consumed = ledger.consume(
        HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        ),
        run_id="summary-holdout",
    )
    assert consumed.result_reference is not None
    path = ledger.root / cast(str, consumed.result_reference.to_primitive()["path"])
    result = read_record(path)
    artifact = deepcopy(result["artifact"])
    block_research(monkeypatch)
    inspect_validation(
        source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
    )
    summary = cast(PrimitiveMapping, result["summary"])
    metric = "accuracy" if prediction else "total_return"
    if change == "metric":
        summary[metric] = "999"
    elif change == "missing_metric":
        del summary[metric]
    elif change == "extra_metric":
        summary["foreign_metric"] = "999"
    else:
        result["summary"] = {} if change == "empty" else None
    write_record(path, result)
    assert read_record(path)["artifact"] == artifact
    state = ledger.state(source)
    assert state.state.value == "consumed"
    assert state.result_reference != consumed.result_reference
    with pytest.raises(ManifestError, match=r"holdout (backtest|prediction) summary"):
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )
    assert ledger.state(source) == state
    assert read_record(path) == result


def test_consumed_backtest_binds_dataset_hash_even_when_run_id_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = complete_study(tmp_path, prediction=False)
    source = completed.source
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    consumed = ledger.consume(
        HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        ),
        run_id="dataset-hash-holdout",
    )
    assert consumed.result_reference is not None
    path = ledger.root / cast(str, consumed.result_reference.to_primitive()["path"])
    result = read_record(path)
    artifact = cast(PrimitiveMapping, result["artifact"])
    manifest = cast(
        PrimitiveMapping, cast(PrimitiveMapping, artifact["result"])["manifest"]
    )
    # QF-3 data_sha256 is provenance; QF-5 hashes bars_fingerprint into its run ID.
    cast(PrimitiveMapping, manifest["market_data"])["data_sha256"] = "0" * 64
    export = path.parent / "evaluation" / cast(str, artifact["export_location"])
    write_json(export / "manifest.json", manifest)
    integrity, _ = read_producer_record(export / "integrity.json")
    cast(PrimitiveMapping, integrity["files"])["manifest.json"] = sha256(
        (export / "manifest.json").read_bytes()
    ).hexdigest()
    write_json(export / "integrity.json", integrity)
    artifact["export_fingerprint"] = configuration_identity(
        {"integrity": (export / "integrity.json").read_text()}
    )
    result["artifact_sha256"] = configuration_identity(artifact)
    write_record(path, result)
    assert validate_backtest_result_artifact(export) == export
    assert ledger.state(source).state.value == "consumed"
    block_research(monkeypatch)
    with pytest.raises(
        ManifestError, match="holdout backtest differs from requested dataset"
    ):
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )


@pytest.mark.parametrize(
    "change",
    [
        "strategy",
        "capital",
        "commission",
        "dataset_id",
        "dataset_content",
        "evaluation_start",
        "evaluation_end",
        "unbounded",
    ],
)
def test_consumed_backtest_rejects_self_consistent_run_outside_frozen_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    completed = complete_study(tmp_path, prediction=False)
    source = completed.source
    evaluator = completed.evaluator
    assert isinstance(evaluator, BacktestEvaluator)
    evaluation = HoldoutEvaluation.prepare(
        source, evaluator, selection_fold_id=source.folds[-1].fold_id
    )
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    consumed = ledger.consume(evaluation, run_id="frozen-holdout")
    assert consumed.result_reference is not None
    path = ledger.root / cast(str, consumed.result_reference.to_primitive()["path"])
    result = read_record(path)
    original = cast(PrimitiveMapping, result["artifact"])
    frozen = cast(PrimitiveMapping, evaluation.selection.to_primitive()["candidate"])
    candidate = next(
        item
        for item in evaluator.universe.candidates
        if item.combination_id == frozen["combination_id"]
    )
    if change == "strategy":
        candidate = next(
            item
            for item in evaluator.universe.candidates
            if item.combination_id != candidate.combination_id
        )
    strategy = evaluator.factory.build(candidate.parameters.to_primitive())
    dataset = evaluation.permitted.dataset
    sessions = evaluation.permitted.sessions
    interval = EvaluationInterval(sessions[0], sessions[-1])
    config = replace(evaluator.grid_config.backtest, evaluation_interval=interval)
    if change == "capital":
        config = replace(config, initial_capital=config.initial_capital + Decimal(1))
    elif change == "commission":
        config = replace(config, commission=FixedCommission(Decimal(17)))
    elif change.startswith("dataset"):
        dataset = make_dataset(
            tuple(
                str(
                    bar.close
                    + (Decimal(1) if change == "dataset_content" else Decimal(0))
                )
                for bar in dataset.bars
            ),
            sessions=tuple(bar.session_date for bar in dataset.bars),
            dataset_id="foreign-holdout",
        )
    elif change == "evaluation_start":
        config = replace(
            config, evaluation_interval=replace(interval, start_session=sessions[-1])
        )
    elif change == "evaluation_end":
        config = replace(
            config, evaluation_interval=replace(interval, end_session=sessions[0])
        )
    elif change == "unbounded":
        config = replace(config, evaluation_interval=None)
    replacement = run_backtest(dataset, strategy, config)
    assert replacement.run_id != original["result_id"]
    export = export_backtest_result(replacement, path.parent / "evaluation")
    artifact = BacktestOOSArtifact(
        evaluation.selection.selection_id,
        replacement.run_id,
        PrimitiveMappingSnapshot.capture(replacement.to_primitive()),
        export.name,
        configuration_identity({"integrity": (export / "integrity.json").read_text()}),
    ).to_primitive()
    result["artifact"] = artifact
    result["artifact_sha256"] = configuration_identity(artifact)
    write_record(path, result)
    assert ledger.state(source).state.value == "consumed"
    block_research(monkeypatch)
    with pytest.raises(
        ManifestError,
        match=r"holdout backtest differs from (frozen configuration|requested dataset)",
    ):
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )


@pytest.mark.parametrize("prediction", [False, True], ids=["backtest", "prediction"])
@pytest.mark.parametrize("change", ["selection_id", "result_id", "manifest"])
def test_consumed_result_preserves_frozen_selection_and_result_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction: bool, change: str
) -> None:
    completed = complete_study(tmp_path, prediction=prediction)
    source = completed.source
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    consumed = ledger.consume(
        HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        ),
        run_id="fixture-holdout",
    )
    assert consumed.result_reference is not None
    path = ledger.root / cast(str, consumed.result_reference.to_primitive()["path"])
    result = read_record(path)
    artifact = cast(PrimitiveMapping, result["artifact"])
    if change == "manifest":
        manifest = cast(
            PrimitiveMapping, cast(PrimitiveMapping, artifact["result"])["manifest"]
        )
        if prediction:
            manifest["window_result_id"] = "0" * 64
        else:
            manifest["initiated_at"] = "2000-01-01T00:00:00+00:00"
    else:
        artifact[change] = "0" * 64
    result["artifact_sha256"] = configuration_identity(artifact)
    write_record(path, result)
    assert ledger.state(source).state.value == "consumed"
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=r"(holdout|window)"):
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )


@pytest.mark.parametrize(
    "change",
    ["foreign_selection", "foreign_partition", "window_identity", "window_counts"],
)
def test_consumed_prediction_rejects_rehashed_foreign_or_inconsistent_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    completed = complete_study(tmp_path, prediction=True)
    source = completed.source
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    consumed = ledger.consume(
        HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        ),
        run_id="fixture-holdout",
    )
    assert consumed.result_reference is not None
    path = ledger.root / cast(str, consumed.result_reference.to_primitive()["path"])
    result = read_record(path)
    original = cast(PrimitiveMapping, result["artifact"])
    if change.startswith("foreign"):
        fold_artifact = source.folds[0].artifact
        assert fold_artifact is not None
        artifact = deepcopy(fold_artifact.to_primitive())
        if change == "foreign_partition":
            artifact["selection_id"] = original["selection_id"]
        result["artifact"] = artifact
    else:
        artifact = original
        manifest = cast(
            PrimitiveMapping, cast(PrimitiveMapping, artifact["result"])["manifest"]
        )
        if change == "window_identity":
            manifest["window_result_id"] = "0" * 64
        else:
            cast(PrimitiveMapping, manifest["record_counts"])["scheduled_decisions"] = (
                999
            )
    result["artifact_sha256"] = configuration_identity(artifact)
    write_record(path, result)
    assert ledger.state(source).state.value == "consumed"
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=r"(holdout|window)"):
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )


@pytest.mark.parametrize("change", ["missing", "version", "window", "summary"])
def test_prediction_holdout_requires_bound_captured_summary_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    completed = complete_study(tmp_path, prediction=True)
    source = completed.source
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    consumed = ledger.consume(
        HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        ),
        run_id="captured-summary",
    )
    assert consumed.result_reference is not None
    path = ledger.root / cast(str, consumed.result_reference.to_primitive()["path"])
    result = read_record(path)
    artifact = cast(PrimitiveMapping, result["artifact"])
    captured = cast(PrimitiveMapping, artifact["holdout_summary"])
    assert captured["summary"] == result["summary"]
    assert captured["window_result_id"] == artifact["result_id"]
    if change == "missing":
        del artifact["holdout_summary"]
    elif change == "version":
        captured["schema_version"] = "future"
    elif change == "window":
        captured["window_result_id"] = "0" * 64
    else:
        cast(PrimitiveMapping, captured["summary"])["accuracy"] = "999"
    result["artifact_sha256"] = configuration_identity(artifact)
    write_record(path, result)
    state = ledger.state(source)
    assert state.state.value == "consumed"
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="holdout prediction summary"):
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )
    assert ledger.state(source) == state
    assert read_record(path) == result
