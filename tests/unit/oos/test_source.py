"""Fail closed on non-OOS membership, provenance drift and damaged artifacts."""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.oos import (
    OOSIntegrityError,
    aggregate_backtest,
    aggregate_prediction,
    load_oos_source,
)
from quantforge.oos._records import mapping, records
from quantforge.walk_forward.models import FoldStatus, WalkForwardPersistenceError
from quantforge.walk_forward.persistence import read_record, write_record

from .conftest import CompletedStudy


def fold_root(study: CompletedStudy, index: int = 0) -> Path:
    return study.study.study_path / "folds" / study.source.folds[index].fold_id


def rewrite_oos(root: Path, payload: PrimitiveMapping) -> None:
    write_record(root / "oos.json", payload)
    state = read_record(root / "state.json")
    state["artifact_id"] = configuration_identity(payload)
    write_record(root / "state.json", state)


def test_source_requires_no_execution_or_selection(
    completed_study: CompletedStudy, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail("aggregation must not execute research")

    evaluator = completed_study.evaluator
    monkeypatch.setattr(type(evaluator), "select", fail)
    monkeypatch.setattr(type(evaluator), "evaluate", fail)
    monkeypatch.setattr(type(evaluator), "configuration", fail)
    source = load_oos_source(
        completed_study.source.plan, completed_study.study.study_path
    )
    assert source == completed_study.source


def test_wrong_validation_plan_rejected(completed_study: CompletedStudy) -> None:
    plan = replace(completed_study.source.plan, name="another plan")
    with pytest.raises(OOSIntegrityError, match="validation plan"):
        load_oos_source(plan, completed_study.study.study_path)


def test_fold_order_and_duplicates_rejected(prediction_study: CompletedStudy) -> None:
    source = prediction_study.source
    for folds in (source.folds[::-1], (source.folds[0], source.folds[0])):
        with pytest.raises(OOSIntegrityError, match="unique"):
            aggregate_prediction(replace(source, folds=folds))


def test_duplicate_persisted_fold_rejected(completed_study: CompletedStudy) -> None:
    root = fold_root(completed_study)
    shutil.copytree(root, root.parent / "duplicate")
    with pytest.raises(OOSIntegrityError, match="duplicate"):
        load_oos_source(completed_study.source.plan, completed_study.study.study_path)


@pytest.mark.parametrize("role", ["development", "selection"])
def test_selection_artifact_cannot_masquerade_as_test(
    completed_study: CompletedStudy, role: str
) -> None:
    root = fold_root(completed_study)
    selection = read_record(root / "selection.json")
    membership = mapping(selection["membership"])
    membership["test"] = membership[role]
    selection["selection_id"] = configuration_identity(
        {k: v for k, v in selection.items() if k != "selection_id"}
    )
    write_record(root / "selection.json", selection)
    with pytest.raises((OOSIntegrityError, TypeError), match=r"window|record"):
        load_oos_source(completed_study.source.plan, completed_study.study.study_path)


def test_failed_and_missing_folds_explicit(completed_study: CompletedStudy) -> None:
    root = fold_root(completed_study)
    state = read_record(root / "state.json")
    state["status"] = "failed"
    state["artifact_id"] = None
    state["failures"] = [{"stage": "test", "error_type": "FixtureFailure"}]
    write_record(root / "state.json", state)
    # An already-present oos.json cannot turn failed state into success.
    shutil.rmtree(fold_root(completed_study, 1))
    source = load_oos_source(
        completed_study.source.plan, completed_study.study.study_path
    )
    assert [f.status for f in source.folds] == [FoldStatus.FAILED, FoldStatus.PENDING]
    assert all(f.artifact is None for f in source.folds)
    aggregate = (
        aggregate_prediction(source)
        if source.plan.environment.study_type.value == "prediction"
        else aggregate_backtest(source)
    )
    completeness = mapping(aggregate.summary.to_primitive()["completeness"])
    assert completeness["complete"] is False
    assert completeness["failed_windows"] == [source.folds[0].fold_id]
    assert completeness["missing_windows"] == [source.folds[1].fold_id]
    assert source.references[0].to_primitive()["state"] == state


def test_orphaned_and_corrupt_sources_rejected(completed_study: CompletedStudy) -> None:
    root = fold_root(completed_study)
    (root / "state.json").unlink()
    with pytest.raises(OOSIntegrityError, match="orphaned"):
        load_oos_source(completed_study.source.plan, completed_study.study.study_path)


def test_corrupt_envelope_rejected(prediction_study: CompletedStudy) -> None:
    path = fold_root(prediction_study) / "oos.json"
    path.write_text(path.read_text().replace('"fingerprint":"', '"fingerprint":"0', 1))
    with pytest.raises(WalkForwardPersistenceError, match="corrupt"):
        load_oos_source(prediction_study.source.plan, prediction_study.study.study_path)


def test_prediction_training_schedule_cannot_enter_oos(
    prediction_study: CompletedStudy,
) -> None:
    root = fold_root(prediction_study)
    artifact = read_record(root / "oos.json")
    payload = mapping(artifact["result"])
    decision = records(payload["decisions"])[0]
    decision["decision_timestamp"] = "2024-07-08T17:30:00+00:00"
    rewrite_oos(root, artifact)
    with pytest.raises(
        ValueError, match=r"identity|timestamp|backend|provenance|context"
    ):
        load_oos_source(prediction_study.source.plan, prediction_study.study.study_path)


@pytest.mark.parametrize("field", ["daily_equity", "benchmark_daily_equity"])
def test_backtest_training_equity_rejected(
    backtest_study: CompletedStudy, field: str
) -> None:
    root = fold_root(backtest_study)
    artifact = read_record(root / "oos.json")
    payload = mapping(artifact["result"])
    records(payload[field])[0]["session"] = "2024-07-08"
    rewrite_oos(root, artifact)
    with pytest.raises(OOSIntegrityError, match="non-test"):
        load_oos_source(backtest_study.source.plan, backtest_study.study.study_path)


def test_backtest_snapshot_must_match_native_export(
    backtest_study: CompletedStudy,
) -> None:
    root = fold_root(backtest_study)
    artifact = read_record(root / "oos.json")
    payload = mapping(artifact["result"])
    records(payload["daily_equity"])[0]["total_equity"] = "999999"
    rewrite_oos(root, artifact)
    with pytest.raises(OOSIntegrityError, match="tabular export"):
        load_oos_source(backtest_study.source.plan, backtest_study.study.study_path)


@pytest.mark.parametrize(
    "key", ["indicator_backend_environment", "dataset_family_fingerprint"]
)
def test_prediction_provenance_cannot_drift(
    prediction_study: CompletedStudy, key: str
) -> None:
    root = fold_root(prediction_study)
    artifact = read_record(root / "oos.json")
    mapping(mapping(artifact["result"])["manifest"])[key] = "different"
    rewrite_oos(root, artifact)
    with pytest.raises(
        ValueError, match=r"identity|timestamp|backend|provenance|context"
    ):
        load_oos_source(prediction_study.source.plan, prediction_study.study.study_path)


def test_incompatible_frozen_selection_rejected(
    completed_study: CompletedStudy,
) -> None:
    root = fold_root(completed_study)
    selection = read_record(root / "selection.json")
    selection["candidate_universe_id"] = "different"
    selection["selection_id"] = configuration_identity(
        {k: v for k, v in selection.items() if k != "selection_id"}
    )
    write_record(root / "selection.json", selection)
    with pytest.raises(OOSIntegrityError, match="lineage"):
        load_oos_source(completed_study.source.plan, completed_study.study.study_path)


def test_snapshot_detachment(prediction_study: CompletedStudy) -> None:
    source = prediction_study.source
    copy = source.definition.to_primitive()
    copy.clear()
    assert source.definition == PrimitiveMappingSnapshot.capture(
        read_record(prediction_study.study.study_path / "manifest.json")
    )


def test_unrelated_future_append_cannot_change_immutable_aggregate(
    completed_study: CompletedStudy,
    tmp_path: Path,
) -> None:
    source = completed_study.source
    aggregate = (
        aggregate_prediction
        if source.plan.environment.study_type.value == "prediction"
        else aggregate_backtest
    )
    original = aggregate(source)
    # New provider extracts are independent from already completed QF-39 files.
    future = tmp_path / "future-provider-extract.json"
    future.write_text('{"session":"2030-01-02","close":"1000000"}')
    future.write_text(future.read_text() + '\n{"session":"2030-01-03","close":"1"}')
    reloaded = load_oos_source(source.plan, completed_study.study.study_path)
    assert aggregate(reloaded).aggregate_id == original.aggregate_id
    assert aggregate(reloaded).provenance == original.provenance


def test_replaced_source_artifact_cannot_bypass_verified_references(
    prediction_study: CompletedStudy,
) -> None:
    source = prediction_study.source
    first, second = source.folds
    changed = replace(first, artifact=second.artifact)
    with pytest.raises(OOSIntegrityError, match="verified"):
        aggregate_prediction(replace(source, folds=(changed, second)))


def test_replaced_lineage_rejected(prediction_study: CompletedStudy) -> None:
    source = replace(
        prediction_study.source,
        lineage=PrimitiveMappingSnapshot.capture({"changed": True}),
    )
    with pytest.raises(OOSIntegrityError, match="lineage identity"):
        aggregate_prediction(source)
