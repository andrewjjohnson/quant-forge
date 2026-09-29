"""QF-64 storage shape: 4,095 QF-45-like decisions with 0/25/100/4,095 observations.

This is a storage-shape fixture, not a claimed market dataset. An existing
fixture decision supplies the rich observation structure and QF-62's synthetic
QF-45 membership pattern supplies its visible bars. Every other decision is an
ordinary no-prediction receipt. Real QF-11 execution and scientific validation
live in the unit/integration tests and the documented QF-45 measurement.
"""

import hashlib
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.prediction.window_compact import (
    CompactDecisionReceipt,
    CompactPredictionWindowDecision,
    CompactPredictionWindowResult,
    PredictionWindowEvidence,
)
from quantforge.prediction.window_encoding import canonical
from quantforge.prediction.window_membership import (
    CATALOGUE_SEGMENTS_FIELD,
    MembershipCatalogues,
    prediction_context,
    with_prediction_context,
)
from quantforge.prediction.window_reader import PredictionWindowReader
from tests.performance.test_normalized_window_membership_scale import (
    DAILY,
    DECISIONS,
    PRIMARY,
    membership,
    with_membership,
)
from tests.unit.prediction.test_compact_prediction_window import rehash_decision
from tests.unit.prediction.test_prediction_window import START, run_window, schedule

OBSERVATION_COUNTS = (0, 25, 100, DECISIONS)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def observation_sequences(count: int) -> frozenset[int]:
    """Evenly spread deterministic event sequences, including late decisions."""
    if count == 0:
        return frozenset()
    step = DECISIONS // count
    return frozenset(DECISIONS - 1 - index * step for index in range(count))


def build(
    evidence: PredictionWindowEvidence,
    template: dict[str, Any],
    timestamps: tuple[Any, ...],
    events: frozenset[int],
) -> tuple[CompactPredictionWindowResult, int]:
    """Receipts for every decision; rich QF-62-normalized evidence for events."""
    context_template = prediction_context(template)
    state = MembershipCatalogues()
    receipts: list[CompactDecisionReceipt] = []
    segment_bytes = 0
    for sequence, timestamp in enumerate(timestamps):
        if sequence not in events:
            receipts.append(
                CompactDecisionReceipt.no_prediction(
                    sequence=sequence,
                    context_id=digest(f"context-{sequence}"),
                    prediction_study_id=digest(f"study-{sequence}"),
                )
            )
            continue
        logical = with_membership(context_template, *membership(sequence))
        record = with_prediction_context(template, logical)
        record.update(
            sequence=sequence,
            decision_timestamp=timestamp.isoformat(),
            shared_evidence_id=evidence.evidence_id,
        )
        context, segments = state.normalize(logical)
        record = with_prediction_context(record, context)
        record[CATALOGUE_SEGMENTS_FIELD] = segments
        segment_bytes += len(canonical({"s": segments})) - len(b'{"s":}')
        rehash_decision(record)
        state.accept(record)
        receipts.append(
            CompactDecisionReceipt.with_decision(
                CompactPredictionWindowDecision(
                    PrimitiveMappingSnapshot.capture(record)
                )
            )
        )
    return CompactPredictionWindowResult(
        evidence, receipts=tuple(receipts)
    ), segment_bytes


def test_sparse_storage_scales_with_observations_not_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = run_window()
    broad = schedule(START, START + timedelta(days=100))
    exact = schedule(START, broad.decision_timestamps[DECISIONS - 1])
    assert len(exact.decision_timestamps) == DECISIONS
    identity = original.identity_snapshot.to_primitive()
    identity["schedule"] = exact.to_primitive()
    evidence = PredictionWindowEvidence(PrimitiveMappingSnapshot.capture(identity), "4")
    template = cast(
        dict[str, Any],
        CompactPredictionWindowResult.from_window(original).decisions[0].to_primitive(),
    )
    assert template["status"] == "evaluated"
    template.pop("decision_id")
    header_size = shared_size = 0
    report: dict[str, Any] = {"decisions": DECISIONS}
    totals: dict[int, int] = {}
    bare_sizes: set[int] = set()
    for count in OBSERVATION_COUNTS:
        began = perf_counter()
        events = observation_sequences(count)
        assert len(events) == count
        compact, segment_bytes = build(
            evidence, template, exact.decision_timestamps, events
        )
        path = tmp_path / f"window-{count}.jsonl"
        with path.open("wb") as stream:
            stream.writelines(compact.iter_serialized_records())
        built = perf_counter()
        header_size = len(canonical(compact.header_snapshot.to_primitive())) + 1
        shared_size = len(canonical(evidence.to_primitive())) + 1
        sizes = {
            receipt.sequence: len(receipt.snapshot.canonical_json.encode()) + 1
            for receipt in compact.receipts
        }
        bare = [size for sequence, size in sizes.items() if sequence not in events]
        rich = [size for sequence, size in sizes.items() if sequence in events]
        nested = [
            len(receipt.decision.snapshot.canonical_json.encode()) + 1
            for receipt in compact.receipts
            if receipt.decision is not None
        ]
        total = path.stat().st_size
        assert total == header_size + shared_size + sum(sizes.values())
        totals[count] = total
        bare_sizes.update(bare)

        constructed: list[int] = []
        validate = CompactPredictionWindowDecision.__post_init__

        def counted(self: CompactPredictionWindowDecision) -> None:
            constructed.append(1)
            validate(self)

        monkeypatch.setattr(CompactPredictionWindowDecision, "__post_init__", counted)
        reader = PredictionWindowReader.open(path)
        started = perf_counter()
        observations = sum(1 for _ in reader.iterate_observations())
        streamed = perf_counter()
        monkeypatch.undo()
        assert observations == count
        # Rich objects are built only for observations (checked, then given their
        # catalogue view); never for a no-prediction receipt.
        assert len(constructed) == 2 * count
        report[str(count)] = {
            "total_bytes": total,
            "receipt_bytes_no_observation": sum(bare),
            "receipt_bytes_with_observation": sum(rich),
            "nested_rich_decision_bytes": sum(nested),
            "catalogue_segment_bytes": segment_bytes,
            "no_observation_receipt_bytes_min_max": [min(bare), max(bare)]
            if bare
            else None,
            "observation_receipt_bytes_mean": sum(rich) / len(rich) if rich else None,
            "build_seconds": round(built - began, 2),
            "observation_stream_seconds": round(streamed - started, 2),
        }
        if count == DECISIONS:
            # All decisions observed: the receipt envelope around each version 3
            # decision is the only overhead over one rich record per decision.
            envelope = sum(rich) - sum(nested)
            report["dense_envelope_bytes_per_decision"] = envelope / DECISIONS
            assert envelope / DECISIONS < 400
    # An ordinary no-prediction receipt has a constant tiny size.
    assert max(bare_sizes) - min(bare_sizes) <= 4
    assert max(bare_sizes) < 400
    # O(decisions x receipt) + O(observations x rich payload) + shared evidence.
    base = header_size + shared_size + DECISIONS * max(bare_sizes)
    assert totals[0] <= base
    per_observation = (totals[100] - totals[25]) / 75
    assert 1_000 < per_observation < 200_000
    assert totals[25] < totals[0] + 26 * per_observation + len(PRIMARY + DAILY) * 70
    assert totals[DECISIONS] > 20 * totals[25]
    report.update(
        header_bytes=header_size,
        shared_record_bytes=shared_size,
        marginal_bytes_per_observation=round(per_observation, 1),
    )
    print(canonical(cast(Any, report)).decode())
