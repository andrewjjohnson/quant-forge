"""Compact bounded ancestry, finalized holdout ingestion, and offline presentation."""

import socket
from dataclasses import replace
from pathlib import Path

import pytest

from quantforge.experiments import (
    create_manifest,
    inspect_validation,
    verify_artifacts,
    write_manifest,
)
from quantforge.oos import (
    HoldoutEvaluation,
    HoldoutLedger,
    OOSIntegrityError,
    aggregate_prediction,
    export_oos_aggregate,
    load_oos_aggregate,
    load_oos_source,
)
from quantforge.prediction.window_compact import CompactPredictionWindowDecision
from quantforge.prediction.window_encoding import mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.reporting import build_research_report, export_research_report
from quantforge.walk_forward import PredictionEvaluator, WalkForwardStudy
from tests.integration.test_bounded_prediction_workflow import workflow_fixture
from tests.integration.test_intraday_prediction_provenance import cached_fixture
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import execution
from tests.unit.helpers import SESSIONS
from tests.unit.walk_forward.test_incremental_prediction import compact_adapter


def test_bounded_finalized_holdout_and_offline_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = cached_fixture(tmp_path / "cache", session_dates=SESSIONS[:7])
    config, original = workflow_fixture(tmp_path, fixture)
    adapter = compact_adapter(original)
    study = WalkForwardStudy(config, adapter, tmp_path / "studies")
    study.run()
    source = load_oos_source(config.plan, study.study_path)
    aggregate = aggregate_prediction(source)
    aggregate_path = export_oos_aggregate(aggregate, tmp_path / "aggregate")
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    reserved = ledger.reserve(source)
    assert reserved.state == "reserved_unconsumed"
    evaluation = HoldoutEvaluation.prepare(
        source, adapter, selection_fold_id=source.folds[-1].fold_id
    )
    # An externally finalized result is accepted only after the same frozen
    # configuration/partition validation used by the normal evaluation path.
    artifact = adapter.evaluate_partition(
        config.plan, evaluation.permitted, evaluation.selection, tmp_path / "finalized"
    )
    final_path = tmp_path / "finalized" / "prediction-window.jsonl"
    evaluation = replace(evaluation, finalized_prediction_window=final_path)
    assert ledger.state(source) == reserved

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail(
            "offline downstream processing attempted research or network access"
        )

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(PredictionEvaluator, "evaluate_partition", forbidden)
    monkeypatch.setattr(CompactPredictionWindowDecision, "from_embedded", forbidden)
    wrong_window = next(r.path for r in source.prediction_windows if r is not None)
    invalid_ledger = HoldoutLedger.create(tmp_path / "invalid-ledger")
    invalid_ledger.reserve(source)
    with pytest.raises(ValueError, match=r"incompatible|scope|identity"):
        invalid_ledger.consume(
            replace(evaluation, finalized_prediction_window=wrong_window),
            run_id="foreign-window",
        )
    assert invalid_ledger.state(source).state == "consumed"
    assert ledger.state(source) == reserved
    consumed = ledger.consume(evaluation, run_id="accept-finalized")
    assert consumed.state == "consumed"
    final_path.unlink()  # The caller-owned import is no longer needed for replay.
    assert (
        mapping(ledger.result(evaluation).to_primitive()["artifact"])["result_id"]
        == artifact.window_result_id
    )
    assert (
        HoldoutLedger(ledger.root).consume(evaluation, run_id="compatible-rerun")
        == consumed
    )
    other_freeze = HoldoutEvaluation.prepare(
        source, adapter, selection_fold_id=source.folds[0].fold_id
    )
    with pytest.raises(OOSIntegrityError, match="reselection"):
        ledger.consume(other_freeze, run_id="incompatible")
    assert ledger.state(source) == consumed
    block_research(monkeypatch)
    assert load_oos_aggregate(aggregate_path).to_primitive() == aggregate.to_primitive()
    reloaded = load_oos_source(config.plan, study.study_path)
    artifacts = inspect_validation(
        reloaded,
        study.study_path,
        artifact_root=tmp_path,
        ledger=ledger,
        aggregate_path=aggregate_path,
    )
    verify_artifacts(artifacts.index, tmp_path).require_valid()
    manifest = create_manifest(artifacts, execution())
    manifest_path = write_manifest(
        manifest, tmp_path / "manifests", artifact_root=tmp_path
    )
    # QF-41 must not walk decisions just to display authoritative aggregates.
    monkeypatch.setattr(PredictionWindowReader, "iterate_decisions", forbidden)
    report = build_research_report(
        manifest_path,
        artifact_root=tmp_path,
        holdout_source=reloaded,
        holdout_ledger=ledger,
    )
    assert report.header.to_primitive()["holdout_state"] == "consumed"
    assert (
        "consumed" in export_research_report(report, tmp_path / "reports").read_text()
    )
    assert ledger.state(reloaded) == consumed
