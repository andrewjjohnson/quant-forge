"""QF-64 schema 4: decision coverage receipts separate from rich observations."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

import quantforge.prediction.window_execution as window_execution
from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.indicators import NATIVE_INDICATOR_BACKEND
from quantforge.prediction import (
    PredictionContextFailurePolicy,
    PredictionRuleContext,
    PredictionStrategyOutput,
    PredictionWindowResult,
    run_prediction_window,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.study import prepare_prediction_study_dataset
from quantforge.prediction.window_compact import (
    CompactDecisionReceipt,
    CompactPredictionWindowDecision,
    CompactPredictionWindowResult,
    PredictionWindowEvidence,
)
from quantforge.prediction.window_compact_validation import (
    validate_prediction_window_reader,
)
from quantforge.prediction.window_coverage import PredictionDecisionReceipt
from quantforge.prediction.window_encoding import canonical, mapping
from quantforge.prediction.window_execution import (
    run_incremental_prediction_window_in_session,
)
from quantforge.prediction.window_incremental import IncrementalPredictionWindowWriter
from quantforge.prediction.window_membership import prediction_context
from quantforge.prediction.window_reader import PredictionWindowReader
from tests.unit.prediction.test_compact_prediction_window import (
    records,
    rehash_decision,
    verify,
    write_compact,
    write_records,
)
from tests.unit.prediction.test_incremental_prediction_window import validator
from tests.unit.prediction.test_multi_timeframe_study import (
    _prediction_dataset,  # pyright: ignore[reportPrivateUsage]
    _requirements,  # pyright: ignore[reportPrivateUsage]
    _study,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_window import (
    START,
    WindowProvider,
    WindowRule,
    _provider_with_final_session,  # pyright: ignore[reportPrivateUsage]
    run_window,
    schedule,
)
from tests.unit.prediction.test_technical_confluence import (
    _rule as confluence_rule,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_technical_confluence import (
    _study as confluence_study,  # pyright: ignore[reportPrivateUsage]
)

PHYSICAL = ("decision_id", "shared_evidence_id")
EVENT_MINUTES = frozenset({40, 50})
ENVIRONMENT: PrimitiveMapping = {"provider": "immutable_fixture", "version": "1"}
RECEIPT_KEYS = {
    "context_id",
    "observation_count",
    "prediction_study_id",
    "receipt_id",
    "record_type",
    "sequence",
    "status",
}


class EventRule(WindowRule):
    """Keep the fixture signal at two of the four scheduled bar ends only."""

    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> PredictionStrategyOutput:
        output = super().generate_with_context(context)
        primary = self.context_requirements.primary.timeframe
        if context.latest_bar_for(primary).end_timestamp.minute in EVENT_MINUTES:
            return output
        return replace(output, signals=())


def run_events(
    provider: WindowProvider | None = None,
) -> PredictionWindowResult[Any, Any, Any]:
    provider = provider or WindowProvider()
    return run_prediction_window(
        _prediction_dataset(),
        _study(EventRule(_requirements())),
        schedule=schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment=ENVIRONMENT,
    )


@pytest.fixture(scope="module")
def events() -> PredictionWindowResult[Any, Any, Any]:
    return run_events()


def sparse(
    window: PredictionWindowResult[Any, Any, Any],
) -> CompactPredictionWindowResult:
    return CompactPredictionWindowResult.from_window(window, schema_version="4")


def normalized(
    window: PredictionWindowResult[Any, Any, Any],
) -> CompactPredictionWindowResult:
    return CompactPredictionWindowResult.from_window(window, schema_version="3")


def execute(
    path: Path, provider: WindowProvider, schema_version: str = "4"
) -> PredictionWindowReader:
    return run_incremental_prediction_window_in_session(
        prepare_prediction_study_dataset(_prediction_dataset()),
        _study(EventRule(_requirements())),
        path=path,
        schedule=schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment=ENVIRONMENT,
        schema_version=schema_version,
    )


def logical(record: PrimitiveMapping) -> PrimitiveMapping:
    return {key: value for key, value in record.items() if key not in PHYSICAL}


def coverage(receipt: PredictionDecisionReceipt) -> tuple[object, ...]:
    return (
        receipt.sequence,
        receipt.decision_timestamp,
        receipt.status,
        receipt.context_id,
        receipt.prediction_study_id,
        receipt.observation_start,
        receipt.observation_count,
    )


def observed(reader: PredictionWindowReader) -> list[tuple[object, ...]]:
    return [
        (
            item.observation_index,
            item.sequence,
            item.signal_index,
            item.decision_timestamp,
            item.context_id,
            item.prediction_study_id,
            canonical(item.signal()),
            None
            if item.row() is None
            else canonical(cast(PrimitiveMapping, item.row())),
        )
        for item in reader.iterate_observations()
    ]


def rehash_receipt(record: PrimitiveMapping) -> None:
    record["receipt_id"] = configuration_identity(
        {key: value for key, value in record.items() if key != "receipt_id"}
    )


def nested(record: PrimitiveMapping) -> dict[str, Any]:
    return cast(dict[str, Any], record["decision"])


def rich_and_bare(items: list[PrimitiveMapping]) -> tuple[list[int], list[int]]:
    rich = [index for index, item in enumerate(items[2:], 2) if "decision" in item]
    bare = [index for index, item in enumerate(items[2:], 2) if "decision" not in item]
    return rich, bare


def test_sparse_window_preserves_schema_3_coverage_observations_and_identities(
    events: PredictionWindowResult[Any, Any, Any], tmp_path: Path
) -> None:
    counts = events.counts_primitive()
    assert (counts["no_prediction_decisions"], counts["generated_predictions"]) == (
        2,
        2,
    )
    old = write_compact(tmp_path / "v3.jsonl", normalized(events))
    new = write_compact(tmp_path / "v4.jsonl", sparse(events))
    assert sparse(events).serialize() == (tmp_path / "v4.jsonl").read_bytes()
    assert (new.schema_version, new.decision_count) == ("4", 4)
    assert new.header()["record_counts"] == old.header()["record_counts"] == counts
    verify(new, events)
    for before, after, original in zip(
        old.iterate_decision_receipts(),
        new.iterate_decision_receipts(),
        events.decisions,
        strict=True,
    ):
        assert coverage(before) == coverage(after)
        assert after.prediction_study_id == original.result.study_id
        assert after.context_id == original.context_id
        if after.status == "no_prediction":
            assert after.decision is None
            assert isinstance(after.record, CompactDecisionReceipt)
            assert set(after.record.to_primitive()) == RECEIPT_KEYS
        else:
            assert after.decision is not None
            assert before.decision is not None
            assert canonical(logical(after.decision.expanded_record())) == canonical(
                logical(before.decision.expanded_record())
            )
    assert observed(new) == observed(old)
    assert [item[1] for item in observed(new)] == [1, 3]
    # Scientific scope/schedule are shared; physical identities are versioned.
    assert new.evidence.identity_snapshot == old.evidence.identity_snapshot
    assert new.header()["schedule_id"] == old.header()["schedule_id"]
    assert new.evidence.evidence_id != old.evidence.evidence_id
    assert new.header()["window_id"] != old.header()["window_id"]
    assert new.header()["window_result_id"] != old.header()["window_result_id"]
    lines = (tmp_path / "v4.jsonl").read_bytes().splitlines()[2:]
    bare = [line for line in lines if "decision" not in json.loads(line)]
    assert len(bare) == 2
    assert all(len(line) < 400 for line in bare)
    assert b"prediction_context" not in b"".join(bare)


@pytest.mark.parametrize("version", ["1", "2", "3", "4"])
def test_coverage_and_observation_views_serve_every_version(
    events: PredictionWindowResult[Any, Any, Any], tmp_path: Path, version: str
) -> None:
    if version == "1":
        reader = PredictionWindowReader.from_snapshot(events.to_primitive())
    else:
        reader = write_compact(
            tmp_path / "window.jsonl",
            CompactPredictionWindowResult.from_window(events, schema_version=version),
        )
    receipts = list(reader.iterate_decision_receipts())
    originals = [item.to_primitive() for item in events.decisions]
    assert [item.sequence for item in receipts] == [0, 1, 2, 3]
    assert tuple(item.decision_timestamp for item in receipts) == (
        events.schedule.decision_timestamps
    )
    assert [item.status for item in receipts] == [item["status"] for item in originals]
    assert [item.observation_count for item in receipts] == [0, 1, 0, 1]
    assert [item.observation_start for item in receipts] == [0, 0, 1, 1]
    observations = list(reader.iterate_observations())
    assert [item.signal() for item in observations] == [
        signal
        for original in originals
        for signal in cast(list[PrimitiveMapping], original["generated_signals"])
    ]
    assert [item.row() for item in observations] == [
        row
        for original in originals
        for row in cast(
            list[PrimitiveMapping], mapping(original["prediction_study"])["rows"]
        )
    ]
    if version == "4":
        with pytest.raises(InvalidPredictionOutputError, match="sparse windows"):
            next(reader.iterate_decisions())
    else:
        assert len(list(reader.iterate_decisions())) == 4


@pytest.mark.parametrize(
    "kind", ["skipped", "unlabeled", "confluence-0", "confluence-12.5"]
)
def test_exceptional_and_candidate_dispositions_keep_rich_evidence(
    tmp_path: Path, kind: str
) -> None:
    parameters = None
    if kind == "skipped":
        provider = WindowProvider()
        provider.series = ()
        window = run_window(
            provider,
            requirements=_requirements(
                failure_policy=PredictionContextFailurePolicy.SKIP
            ),
        )
    elif kind == "unlabeled":
        tomorrow = START + timedelta(days=1)
        window = run_window(
            _provider_with_final_session(),
            decision_schedule=schedule(tomorrow, tomorrow),
        )
    else:
        provider = WindowProvider()
        rule = confluence_rule(
            threshold=Decimal(kind.split("-")[1]), backend_id=NATIVE_INDICATOR_BACKEND
        )
        study, _ = confluence_study(rule)
        parameters = rule.parameters.to_primitive()
        window = run_prediction_window(
            _prediction_dataset(),
            study,
            schedule=schedule(),
            context_provider=provider,
            dataset_family_fingerprint=provider.family.family_id,
            context_environment={},
        )
    reader = write_compact(tmp_path / "window.jsonl", sparse(window))
    if parameters is None:
        verify(reader, window)
    else:
        validate_prediction_window_reader(
            reader,
            expected_identity=window.identity_snapshot,
            schedule=window.schedule,
            outcome_sessions=tuple(
                bar.session_date for bar in _prediction_dataset().bars
            ),
            strategy_parameters=parameters,
        )
    assert reader.header()["record_counts"] == window.counts_primitive()
    for receipt, original in zip(
        reader.iterate_decision_receipts(), window.decisions, strict=True
    ):
        record = original.to_primitive()
        assert (receipt.status, receipt.context_id) == (
            record["status"],
            record["context_id"],
        )
        if record["status"] == "no_prediction":
            assert receipt.decision is None
            continue
        # Evaluated and skipped decisions retain their complete evidence.
        assert receipt.decision is not None
        assert canonical(logical(receipt.decision.expanded_record())) == canonical(
            logical(
                CompactPredictionWindowDecision.from_embedded(
                    record,
                    sequence=receipt.sequence,
                    evidence=PredictionWindowEvidence(window.identity_snapshot),
                ).to_primitive()
            )
        )
    statuses = {item.status for item in reader.iterate_decision_receipts()}
    if kind == "skipped":
        assert statuses == {"skipped"}
        assert not list(reader.iterate_observations())
    if kind == "unlabeled":
        (observation,) = reader.iterate_observations()
        assert observation.row() is None
        assert mapping(reader.header()["record_counts"])["unavailable_outcomes"] == 1


def test_accepted_and_rejected_candidates_are_both_observations(tmp_path: Path) -> None:
    dispositions: set[str] = set()
    for threshold in ("0", "12.5"):
        provider = WindowProvider()
        rule = confluence_rule(
            threshold=Decimal(threshold), backend_id=NATIVE_INDICATOR_BACKEND
        )
        study, _ = confluence_study(rule)
        window = run_prediction_window(
            _prediction_dataset(),
            study,
            schedule=schedule(),
            context_provider=provider,
            dataset_family_fingerprint=provider.family.family_id,
            context_environment={},
        )
        reader = write_compact(tmp_path / f"{threshold}.jsonl", sparse(window))
        for observation in reader.iterate_observations():
            values = mapping(mapping(observation.signal()["prediction"])["values"])
            dispositions.add(cast(str, values.get("disposition", "unclassified")))
    assert {"accepted", "rejected"} <= dispositions


def test_receipts_support_zero_one_and_many_observations(
    events: PredictionWindowResult[Any, Any, Any], tmp_path: Path
) -> None:
    compact = sparse(events)
    receipts = list(compact.receipts)
    decision = receipts[1].decision
    assert decision is not None
    record = decision.to_primitive()
    signals = cast(list[dict[str, Any]], record["generated_signals"])
    extra = deepcopy(signals[0])
    extra["features"] = {**extra["features"], "calls": "999"}
    record["generated_signals"] = [signals[0], extra]
    rehash_decision(record)
    receipts[1] = CompactDecisionReceipt.with_decision(
        CompactPredictionWindowDecision(PrimitiveMappingSnapshot.capture(record))
    )
    many = CompactPredictionWindowResult(compact.evidence, receipts=tuple(receipts))
    reader = write_compact(tmp_path / "many.jsonl", many)
    reader.verify_integrity()
    assert [item.observation_count for item in reader.iterate_decision_receipts()] == [
        0,
        2,
        0,
        1,
    ]
    assert [item.observation_start for item in reader.iterate_decision_receipts()] == [
        0,
        0,
        2,
        2,
    ]
    observations = list(reader.iterate_observations())
    assert [
        (item.sequence, item.signal_index, item.observation_index)
        for item in observations
    ] == [(1, 0, 0), (1, 1, 1), (3, 0, 2)]
    assert observations[0].row() is not None
    assert observations[1].row() is None
    assert mapping(reader.header()["record_counts"])["generated_predictions"] == 3
    # The representation is zero-to-many; QF-42 historical validation still
    # admits at most one generated signal per decision session.
    with pytest.raises(InvalidPredictionOutputError, match="per session"):
        verify(reader, events)


def _corrupt(
    items: list[PrimitiveMapping], change: str, evidence_v3: PredictionWindowEvidence
) -> list[PrimitiveMapping] | bytes:
    rich, bare = rich_and_bare(items)
    first_rich, first_bare = items[rich[0]], items[bare[0]]
    if change == "missing_receipt":
        del items[bare[0]]
    elif change == "duplicate_receipt":
        items.insert(bare[0], deepcopy(first_bare))
    elif change == "reordered_receipts":
        items[2], items[3] = items[3], items[2]
    elif change == "wrong_sequence":
        first_bare["sequence"] = 7
        rehash_receipt(first_bare)
    elif change == "wrong_timestamp":
        nested(first_rich)["decision_timestamp"] = START.isoformat()
        rehash_decision(nested(first_rich))
        rehash_receipt(first_rich)
    elif change == "no_observation_with_evidence":
        first_bare["decision"] = deepcopy(nested(first_rich))
        rehash_receipt(first_bare)
    elif change == "observation_without_evidence":
        del first_rich["decision"]
        rehash_receipt(first_rich)
    elif change == "count_mismatch":
        first_rich["observation_count"] = 2
        rehash_receipt(first_rich)
    elif change == "orphan_observation":
        items.insert(bare[0], deepcopy(nested(first_rich)))
    elif change == "wrong_decision":
        second = items[rich[1]]
        first_rich["decision"], second["decision"] = (
            second["decision"],
            first_rich["decision"],
        )
        rehash_receipt(first_rich)
        rehash_receipt(second)
    elif change == "truncated_payload":
        encoded = b"".join(canonical(item) + b"\n" for item in items)
        return encoded[: encoded.rindex(b'"generated_signals"')]
    elif change == "corrupt_receipt_digest":
        first_bare["receipt_id"] = "0" * 64
    elif change == "corrupt_observation_digest":
        nested(first_rich)["decision_id"] = "0" * 64
        rehash_receipt(first_rich)
    elif change == "skipped_without_evidence":
        first_rich.update(status="skipped", observation_count=0)
        del first_rich["decision"]
        rehash_receipt(first_rich)
    elif change == "unknown_status":
        first_bare["status"] = "failed"
        rehash_receipt(first_bare)
    elif change == "malformed_identity":
        first_bare["context_id"] = None
        rehash_receipt(first_bare)
    elif change == "foreign_catalogue_range":
        context = prediction_context(nested(first_rich))
        source = mapping(context["source_context"])
        timeframe = cast(list[dict[str, Any]], source["timeframes"])[0]
        timeframe["visible_bar_range"]["stop_index"] += 100
        rehash_decision(nested(first_rich))
        rehash_receipt(first_rich)
    elif change == "foreign_evidence":
        nested(first_rich)["shared_evidence_id"] = evidence_v3.evidence_id
        rehash_decision(nested(first_rich))
        rehash_receipt(first_rich)
    elif change == "rehashed_bare_identity":
        first_bare["prediction_study_id"] = "a" * 64
        rehash_receipt(first_bare)
    elif change == "header_counts":
        counts = cast(dict[str, int], items[0]["record_counts"])
        counts["no_prediction_decisions"] += 1
    elif change == "decision_record_in_v4":
        items[bare[0]] = deepcopy(nested(first_rich))
    elif change == "unsupported_version":
        items[0]["schema_version"] = "5"
    return items


@pytest.mark.parametrize(
    "change",
    [
        "missing_receipt",
        "duplicate_receipt",
        "reordered_receipts",
        "wrong_sequence",
        "wrong_timestamp",
        "no_observation_with_evidence",
        "observation_without_evidence",
        "count_mismatch",
        "orphan_observation",
        "wrong_decision",
        "truncated_payload",
        "corrupt_receipt_digest",
        "corrupt_observation_digest",
        "skipped_without_evidence",
        "unknown_status",
        "malformed_identity",
        "foreign_catalogue_range",
        "foreign_evidence",
        "rehashed_bare_identity",
        "header_counts",
        "decision_record_in_v4",
        "unsupported_version",
    ],
)
def test_sparse_corruption_fails_closed(
    events: PredictionWindowResult[Any, Any, Any], tmp_path: Path, change: str
) -> None:
    corrupted = _corrupt(
        records(sparse(events)),
        change,
        PredictionWindowEvidence(events.identity_snapshot, "3"),
    )
    path = tmp_path / "corrupt.jsonl"
    if isinstance(corrupted, bytes):
        path.write_bytes(corrupted)
    else:
        write_records(path, corrupted)
    with pytest.raises(InvalidPredictionOutputError):
        list(PredictionWindowReader.open(path).iterate_observations())
    with pytest.raises(InvalidPredictionOutputError):
        verify(PredictionWindowReader.open(path), events)


@pytest.mark.parametrize("change", ["study_identity", "duplicate_signal"])
def test_rehashed_rich_evidence_fails_scientific_validation(
    events: PredictionWindowResult[Any, Any, Any], tmp_path: Path, change: str
) -> None:
    compact = sparse(events)
    receipts = list(compact.receipts)
    decision = receipts[3].decision
    assert decision is not None
    record = decision.to_primitive()
    if change == "study_identity":
        record["prediction_study_id"] = "b" * 64
        mapping(mapping(record["prediction_study"])["manifest"])["study_id"] = "b" * 64
    else:
        signals = cast(list[PrimitiveMapping], record["generated_signals"])
        record["generated_signals"] = [signals[0], deepcopy(signals[0])]
    rehash_decision(record)
    receipts[3] = CompactDecisionReceipt.with_decision(
        CompactPredictionWindowDecision(PrimitiveMappingSnapshot.capture(record))
    )
    reader = write_compact(
        tmp_path / "rehashed.jsonl",
        CompactPredictionWindowResult(compact.evidence, receipts=tuple(receipts)),
    )
    reader.verify_integrity()  # Self-consistent physical encoding.
    with pytest.raises(InvalidPredictionOutputError):
        verify(reader, events)


def test_versions_and_record_forms_do_not_mix(
    events: PredictionWindowResult[Any, Any, Any], tmp_path: Path
) -> None:
    v3 = records(normalized(events))
    v4 = records(sparse(events))
    receipt = deepcopy(v4[2])
    for items, index, replacement in ((v3, 2, receipt), (v4, 2, deepcopy(v3[2]))):
        items[index] = replacement
        path = tmp_path / "mixed.jsonl"
        write_records(path, items)
        with pytest.raises(InvalidPredictionOutputError):
            PredictionWindowReader.open(path).verify_integrity()
    with pytest.raises(InvalidPredictionOutputError, match="schema 4"):
        CompactPredictionWindowResult(
            normalized(events).evidence, receipts=sparse(events).receipts
        )
    with pytest.raises(InvalidPredictionOutputError, match="receipts"):
        CompactPredictionWindowResult(
            sparse(events).evidence, normalized(events).decisions
        )
    decision = normalized(events).decisions[0]
    assert decision.to_primitive()["status"] == "no_prediction"
    with pytest.raises(InvalidPredictionOutputError, match="without rich evidence"):
        CompactDecisionReceipt.with_decision(decision)


@pytest.mark.parametrize("prefix", [0, 1, 2, 3, 4])
def test_incremental_receipts_resume_to_identical_bytes(
    tmp_path: Path, events: PredictionWindowResult[Any, Any, Any], prefix: int
) -> None:
    compact = sparse(events)
    path = tmp_path / "window.jsonl"
    writer = IncrementalPredictionWindowWriter.open(
        path, validator=validator(events), schema_version="4"
    )
    assert writer.checkpoint()["schema_version"] == "4"
    for receipt in compact.receipts[:prefix]:
        writer.append(receipt)
    # A committed receipt proves execution, including an ordinary no-prediction
    # decision; a decision without a committed receipt was never executed.
    committed = [
        json.loads(line) for line in writer.journal_path.read_bytes().splitlines()
    ]
    assert [item["sequence"] for item in committed] == list(range(prefix))
    assert [item["status"] for item in committed] == [
        "no_prediction",
        "evaluated",
        "no_prediction",
        "evaluated",
    ][:prefix]
    resumed = IncrementalPredictionWindowWriter.open(
        path, validator=validator(events), schema_version="4"
    )
    assert resumed.checkpoint() == writer.checkpoint()
    assert resumed.completed_count == prefix
    for receipt in compact.receipts[prefix:]:
        resumed.append(receipt)
    reader = resumed.finalize()
    assert path.read_bytes() == compact.serialize()
    verify(reader, events)
    assert IncrementalPredictionWindowWriter.open(
        path, validator=validator(events), schema_version="4"
    ).finalized


@pytest.mark.parametrize("prefix", [0, 1, 2, 3])
def test_interrupted_sparse_execution_resumes_with_fresh_runtime_and_no_recomputation(
    tmp_path: Path,
    events: PredictionWindowResult[Any, Any, Any],
    prefix: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = WindowProvider()
    original = provider.get_context_at
    calls: list[object] = []

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        if len(calls) == prefix:
            raise KeyboardInterrupt
        calls.append(kwargs["as_of"])
        return original(*args, **kwargs)

    monkeypatch.setattr(provider, "get_context_at", interrupted)
    path = tmp_path / "window.jsonl"
    with pytest.raises(KeyboardInterrupt):
        execute(path, provider)
    assert not path.exists()
    # Restart with a fresh provider, dataset session and study: nothing from
    # the interrupted process's in-memory preparation is reused.
    fresh = WindowProvider()
    fresh_original = fresh.get_context_at
    resumed_calls: list[object] = []

    def resumed(*args: Any, **kwargs: Any) -> Any:
        resumed_calls.append(kwargs["as_of"])
        return fresh_original(*args, **kwargs)

    monkeypatch.setattr(fresh, "get_context_at", resumed)
    reader = execute(path, fresh)
    assert resumed_calls == list(events.schedule.decision_timestamps[prefix:])
    uninterrupted = execute(tmp_path / "uninterrupted.jsonl", WindowProvider())
    assert path.read_bytes() == uninterrupted.path.read_bytes()
    assert path.read_bytes() == sparse(events).serialize()
    verify(reader, events)
    resumed_calls.clear()
    execute(path, fresh)
    assert not resumed_calls


def test_no_prediction_decisions_build_no_rich_decision(
    tmp_path: Path,
    events: PredictionWindowResult[Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []
    original = window_execution.PredictionWindowDecision

    def counted(*args: Any, **kwargs: Any) -> Any:
        decision = original(*args, **kwargs)
        built.append(cast(str, decision.to_primitive()["status"]))
        return decision

    monkeypatch.setattr(window_execution, "PredictionWindowDecision", counted)
    execute(tmp_path / "window.jsonl", WindowProvider())
    assert built == ["evaluated", "evaluated"]
    built.clear()
    execute(tmp_path / "normalized.jsonl", WindowProvider(), schema_version="3")
    assert built == []  # Schema 3 uses the unchanged decision iterator.


def test_committed_prefix_tail_and_checkpoint_corruption(
    tmp_path: Path, events: PredictionWindowResult[Any, Any, Any]
) -> None:
    compact = sparse(events)
    path = tmp_path / "window.jsonl"
    writer = IncrementalPredictionWindowWriter.open(
        path, validator=validator(events), schema_version="4"
    )
    for receipt in compact.receipts[:3]:
        writer.append(receipt)
    committed = writer.journal_path.read_bytes()
    with writer.journal_path.open("ab") as stream:
        stream.write(b'{"record_type":"decision_receipt","sequence":3')
    recovered = IncrementalPredictionWindowWriter.open(
        path, validator=validator(events), schema_version="4"
    )
    assert recovered.completed_count == 3
    assert writer.journal_path.read_bytes() == committed  # Only the tail went.
    lines = committed.splitlines(keepends=True)
    changed = mapping(json.loads(lines[2]))
    changed["prediction_study_id"] = "c" * 64
    rehash_receipt(changed)
    corrupt = b"".join([*lines[:2], canonical(changed) + b"\n"])
    writer.journal_path.write_bytes(corrupt)
    for schema_version in ("4", "3"):
        with pytest.raises(InvalidPredictionOutputError):
            IncrementalPredictionWindowWriter.open(
                path, validator=validator(events), schema_version=schema_version
            )
        assert writer.journal_path.read_bytes() == corrupt  # Never truncated/reset.
    writer.journal_path.write_bytes(committed)
    checkpoint = writer.staging_path / "checkpoint.json"
    envelope = mapping(json.loads(checkpoint.read_bytes()))
    mapping(envelope["checkpoint"])["completed_count"] = 2
    envelope["fingerprint"] = configuration_identity(mapping(envelope["checkpoint"]))
    checkpoint.write_bytes(canonical(envelope) + b"\n")
    with pytest.raises(InvalidPredictionOutputError, match="checkpoint"):
        IncrementalPredictionWindowWriter.open(
            path, validator=validator(events), schema_version="4"
        )
    assert writer.journal_path.read_bytes() == committed


def test_writer_rejects_foreign_forms_and_rolls_back_rich_catalogue_growth(
    tmp_path: Path, events: PredictionWindowResult[Any, Any, Any]
) -> None:
    compact = sparse(events)
    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "window.jsonl", validator=validator(events), schema_version="4"
    )
    with pytest.raises(InvalidPredictionOutputError, match="receipt"):
        writer.append(normalized(events).decisions[0])
    writer.append(compact.receipts[0])
    with pytest.raises(InvalidPredictionOutputError, match="coverage/order"):
        writer.append(compact.receipts[2])
    assert writer.membership is not None
    before = writer.membership.lengths()
    decision = compact.receipts[1].decision
    assert decision is not None
    corrupt = decision.to_primitive()
    corrupt["prediction_study_id"] = "0" * 64
    mapping(mapping(corrupt["prediction_study"])["manifest"])["study_id"] = "0" * 64
    rehash_decision(corrupt)
    with pytest.raises(InvalidPredictionOutputError):
        writer.append(
            CompactDecisionReceipt.with_decision(
                CompactPredictionWindowDecision(
                    PrimitiveMappingSnapshot.capture(corrupt)
                )
            )
        )
    assert writer.membership.lengths() == before
    assert writer.completed_count == 1
    for receipt in compact.receipts[1:]:
        writer.append(receipt)
    assert writer.finalize().path.read_bytes() == compact.serialize()
    legacy = IncrementalPredictionWindowWriter.open(
        tmp_path / "legacy.jsonl", validator=validator(events), schema_version="3"
    )
    with pytest.raises(InvalidPredictionOutputError, match="schema 4"):
        legacy.append(compact.receipts[0])


@pytest.mark.parametrize(
    "change",
    [
        "sequence",
        "session",
        "as_of",
        "family",
        "requirements",
        "study_identity",
        "context_identity",
        "counts",
        "skipped",
        "feed",
    ],
)
def test_no_prediction_receipts_are_bound_to_their_executed_context(
    events: PredictionWindowResult[Any, Any, Any], change: str
) -> None:
    trusted = validator(events)
    result = events.decisions[0].result
    snapshot = result.prediction_context_snapshot
    assert snapshot is not None
    context = snapshot.to_primitive()
    counts: PrimitiveMapping = {
        "generated_predictions": 0,
        "labeled_rows": 0,
        "unavailable_outcomes": 0,
    }
    assert (
        trusted.validate_no_prediction(
            context, 0, prediction_study_id=result.study_id, record_counts=counts
        )
        == events.decisions[0].context_id
    )
    sequence, study_id = 0, result.study_id
    source = cast(dict[str, Any], context["source_context"])
    if change == "sequence":
        sequence = 1
    elif change == "session":
        context["decision_session"] = "2024-07-12"
    elif change == "as_of":
        source["as_of"] = (START + timedelta(minutes=5)).isoformat()
    elif change == "family":
        source["source_consistency"]["family_id"] = "f" * 64
    elif change == "requirements":
        cast(dict[str, Any], context["requirements"])["failure_policy"] = "skip"
    elif change == "study_identity":
        study_id = "d" * 64
    elif change == "context_identity":
        source["context_id"] = "e" * 64
    elif change == "counts":
        counts["generated_predictions"] = 1
    elif change == "skipped":
        context["status"] = "skipped"
    else:
        cast(dict[str, Any], context["requirements"])["primary"]["feed_scope"] = {}
    with pytest.raises(InvalidPredictionOutputError):
        trusted.validate_no_prediction(
            context, sequence, prediction_study_id=study_id, record_counts=counts
        )
