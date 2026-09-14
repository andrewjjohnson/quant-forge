"""One-way consumption, exact retries, failure injection and cross-process locks."""

import multiprocessing
from dataclasses import replace
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

import quantforge.oos.holdout as holdout_module
from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.oos import (
    HoldoutEvaluation,
    HoldoutLedger,
    HoldoutState,
    OOSIntegrityError,
    aggregate_prediction,
    load_oos_source,
)
from quantforge.oos._records import mapping, records
from quantforge.oos.source import validation_lineage
from quantforge.walk_forward import WalkForwardStudy
from quantforge.walk_forward.models import OOSArtifact, WalkForwardPersistenceError
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.helpers import SESSIONS

from .conftest import CompletedStudy


def prepared(study: CompletedStudy) -> HoldoutEvaluation:
    return HoldoutEvaluation.prepare(
        study.source, study.evaluator, selection_fold_id=study.source.folds[-1].fold_id
    )


def test_explicit_first_consumption_and_exact_reproduction(
    completed_study: CompletedStudy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    source = completed_study.source
    evaluation = prepared(completed_study)
    reserved = ledger.reserve(source)
    assert reserved.state is HoldoutState.RESERVED
    assert reserved.to_primitive()["pristine"] is True
    assert ledger.state(source) == reserved
    calls: list[Path] = []
    original = HoldoutEvaluation._evaluate  # pyright: ignore[reportPrivateUsage]

    def evaluate(self: HoldoutEvaluation, output_root: Path) -> OOSArtifact:
        # Read the durable marker directly during evaluation (the store is locked).
        marker = read_record(
            tmp_path / "ledger" / "exposures" / f"{source.lineage_id}.json"
        )
        assert marker["state"] == "consumed"
        assert (
            mapping(marker["request"])["frozen_selection"]
            == evaluation.selection.to_primitive()
        )
        calls.append(output_root)
        return original(self, output_root)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("holdout must never call selection")

    monkeypatch.setattr(type(completed_study.evaluator), "select", forbidden)
    monkeypatch.setattr(HoldoutEvaluation, "_evaluate", evaluate)
    consumed = ledger.consume(evaluation, run_id="first-run")
    assert consumed.state is HoldoutState.CONSUMED
    assert consumed.to_primitive()["pristine"] is False
    assert consumed.consumption is not None
    marker = consumed.consumption.to_primitive()
    assert marker["consumption_run_id"] == "first-run"
    assert marker["consumed_at"]
    assert marker["request"] == evaluation.configuration()
    assert ledger.reserve(source) == consumed
    assert ledger.consume(evaluation, run_id="idempotent-retry") == consumed
    assert len(calls) == 1
    assert ledger.consume(evaluation, run_id="reproduce", reproduce=True) == consumed
    assert len(calls) == 2
    reopened = HoldoutLedger(tmp_path / "ledger")
    assert reopened.state(source) == consumed
    result = reopened.result(evaluation).to_primitive()
    assert result["kind"] == "final_holdout_result"
    artifact = mapping(result["artifact"])
    assert artifact["selection_id"] == evaluation.selection.selection_id
    assert mapping(result["summary"])
    payload = mapping(artifact["result"])
    if artifact["kind"] == "prediction":
        for decision in records(payload["decisions"]):
            for row in records(mapping(decision["prediction_study"])["rows"]):
                assert (
                    mapping(row["prediction"])["signal_session"]
                    == SESSIONS[13].isoformat()
                )
                assert (
                    mapping(row["outcome"])["outcome_session"]
                    == SESSIONS[14].isoformat()
                )
    else:
        equity = records(payload["daily_equity"])
        assert [row["session"] for row in equity] == [
            s.isoformat() for s in SESSIONS[13:15]
        ]
        assert equity[0]["cash"] == "100000"
        assert equity[0]["shares"] == 0


def test_normal_aggregation_does_not_consume_or_claim_pristine(
    prediction_study: CompletedStudy, tmp_path: Path
) -> None:
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    source = prediction_study.source
    original = ledger.reserve(source)
    aggregate = aggregate_prediction(source)
    assert ledger.state(source) == original
    ledger.consume(prepared(prediction_study), run_id="consume")
    assert aggregate_prediction(source) == aggregate
    assert (
        aggregate.provenance.to_primitive()["holdout_state"]
        == "consult_holdout_ledger; aggregation_does_not_consume"
    )
    assert ledger.state(source).state is HoldoutState.CONSUMED


def test_consumed_configuration_cannot_change(
    completed_study: CompletedStudy, tmp_path: Path
) -> None:
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    source = completed_study.source
    ledger.reserve(source)
    original = ledger.consume(prepared(completed_study), run_id="first")
    different = HoldoutEvaluation.prepare(
        source, completed_study.evaluator, selection_fold_id=source.folds[0].fold_id
    )
    with pytest.raises(OOSIntegrityError, match="reselection"):
        ledger.consume(different, run_id="after-view")
    assert ledger.state(source) == original


def test_material_lineage_changes_and_cosmetic_names(
    completed_study: CompletedStudy, tmp_path: Path
) -> None:
    source = completed_study.source
    definition = source.definition.to_primitive()
    mapping(definition["configuration"])["name"] = "renamed"
    mapping(definition["configuration"])["retry_failed"] = True
    assert validation_lineage(definition) == source.lineage
    universe = mapping(mapping(definition["adapter"])["universe"])
    grid = mapping(universe["grid_definition"])
    grid["new_material_policy"] = "changed"
    assert validation_lineage(definition) != source.lineage
    study = WalkForwardStudy(
        replace(completed_study.study.config, minimum_test_observations=2),
        completed_study.evaluator,
        tmp_path / "distinct-lineage",
    )
    study.run()
    changed = load_oos_source(study.config.plan, study.study_path)
    assert changed.lineage_id != source.lineage_id
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    ledger.consume(prepared(completed_study), run_id="first")
    with pytest.raises(OOSIntegrityError, match="another lineage"):
        ledger.reserve(changed)
    with pytest.raises(OOSIntegrityError, match="another lineage"):
        ledger.state(changed)


@pytest.mark.parametrize(
    "failure_stage",
    ["before_evaluation", "after_evaluation", "artifact_write", "directory_sync"],
)
def test_interrupted_consumption_is_permanent_and_retryable(
    completed_study: CompletedStudy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    source = completed_study.source
    evaluation = prepared(completed_study)
    ledger.reserve(source)
    original_evaluate = HoldoutEvaluation._evaluate  # pyright: ignore[reportPrivateUsage]
    original_write = holdout_module._durable_write  # pyright: ignore[reportPrivateUsage]
    original_sync = holdout_module._sync_directory  # pyright: ignore[reportPrivateUsage]

    def evaluate(self: HoldoutEvaluation, output_root: Path) -> OOSArtifact:
        if failure_stage == "before_evaluation":
            raise RuntimeError("interrupted")
        result = original_evaluate(self, output_root)
        if failure_stage == "after_evaluation":
            raise RuntimeError("interrupted after evaluation")
        return result

    def write(path: Path, payload: PrimitiveMapping) -> None:
        if failure_stage == "artifact_write" and path.name == "result.json":
            raise OSError("interrupted artifact write")
        original_write(path, payload)

    def sync(path: Path) -> None:
        if failure_stage == "directory_sync" and path.name == "exposures":
            raise OSError("interrupted directory sync")
        original_sync(path)

    with monkeypatch.context() as patch:
        patch.setattr(HoldoutEvaluation, "_evaluate", evaluate)
        patch.setattr(holdout_module, "_durable_write", write)
        patch.setattr(holdout_module, "_sync_directory", sync)
        with pytest.raises((RuntimeError, OSError), match="interrupted"):
            ledger.consume(evaluation, run_id="interrupted-first")
    consumed = HoldoutLedger(tmp_path / "ledger").state(source)
    assert consumed.state is HoldoutState.CONSUMED
    assert consumed.result_reference is None
    assert consumed.consumption is not None
    marker = consumed.consumption
    retry = ledger.consume(evaluation, run_id="recovery")
    assert retry.state is HoldoutState.CONSUMED
    assert retry.consumption == marker
    assert retry.result_reference is not None
    assert ledger.consume(evaluation, run_id="duplicate") == retry


def test_failure_before_durable_consumption_does_not_evaluate(
    completed_study: CompletedStudy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(completed_study.source)
    evaluation = prepared(completed_study)
    original = holdout_module._durable_write  # pyright: ignore[reportPrivateUsage]

    def write(path: Path, payload: PrimitiveMapping) -> None:
        if path.parent.name == "exposures":
            raise OSError("cannot persist consumption")
        original(path, payload)

    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail("evaluation was attempted before durable consumed state")

    monkeypatch.setattr(holdout_module, "_durable_write", write)
    monkeypatch.setattr(HoldoutEvaluation, "_evaluate", fail)
    with pytest.raises(OSError, match="cannot persist"):
        ledger.consume(evaluation, run_id="failed-write")
    assert ledger.state(completed_study.source).state is HoldoutState.RESERVED


@pytest.mark.parametrize("missing", ["exposure", "reservation", "store"])
def test_orphaned_result_cannot_be_pristine(
    completed_study: CompletedStudy, tmp_path: Path, missing: str
) -> None:
    root = tmp_path / "ledger"
    ledger = HoldoutLedger.create(root)
    source = completed_study.source
    ledger.reserve(source)
    ledger.consume(prepared(completed_study), run_id="first")
    path = {
        "exposure": root / "exposures" / f"{source.lineage_id}.json",
        "reservation": root / "lineages" / source.lineage_id / "reservation.json",
        "store": root / "store.json",
    }[missing]
    path.unlink()
    with pytest.raises(
        (OOSIntegrityError, WalkForwardPersistenceError), match=r"without|corrupt"
    ):
        ledger.reserve(source)
    with pytest.raises(OOSIntegrityError, match="already exists"):
        HoldoutLedger.create(root)


def test_foreign_result_and_missing_result_preserve_consumption(
    completed_study: CompletedStudy, tmp_path: Path
) -> None:
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    source = completed_study.source
    ledger.reserve(source)
    evaluation = prepared(completed_study)
    ledger.consume(evaluation, run_id="first")
    path = tmp_path / "ledger" / "lineages" / source.lineage_id / "result.json"
    original = read_record(path)
    result = dict(original)
    result["request_id"] = "different"
    write_record(path, result)
    with pytest.raises(OOSIntegrityError, match="compatible"):
        ledger.state(source)
    path.unlink()
    assert ledger.state(source).state is HoldoutState.CONSUMED
    assert ledger.consume(evaluation, run_id="recovery").state is HoldoutState.CONSUMED
    assert read_record(path) == original


def _lock_process(
    root: Path,
    ready: Event,
    release: Event,
) -> None:
    ledger = HoldoutLedger(root)
    with ledger._locked():  # pyright: ignore[reportPrivateUsage]
        ready.set()
        release.wait(10)


def test_conflicting_process_attempt_is_rejected(
    prediction_study: CompletedStudy, tmp_path: Path
) -> None:
    root = tmp_path / "ledger"
    ledger = HoldoutLedger.create(root)
    ledger.reserve(prediction_study.source)
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    child = context.Process(target=_lock_process, args=(root, ready, release))
    child.start()
    try:
        assert ready.wait(10)
        with pytest.raises(OOSIntegrityError, match="conflicting"):
            ledger.consume(prepared(prediction_study), run_id="conflict")
    finally:
        release.set()
        child.join(10)
    assert child.exitcode == 0
    assert ledger.state(prediction_study.source).state is HoldoutState.RESERVED


def test_frozen_and_provenance_drift_rejected_before_exposure(
    prediction_study: CompletedStudy, tmp_path: Path
) -> None:
    source = prediction_study.source
    evaluation = prepared(prediction_study)
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    invalid = replace(
        evaluation,
        adapter_snapshot=PrimitiveMappingSnapshot.capture({"backend": "changed"}),
    )
    with pytest.raises(OOSIntegrityError, match="changed"):
        ledger.consume(invalid, run_id="invalid")
    assert ledger.state(source).state is HoldoutState.RESERVED
    with pytest.raises(OOSIntegrityError, match="existing"):
        HoldoutEvaluation.prepare(
            source, prediction_study.evaluator, selection_fold_id="holdout-selected"
        )
