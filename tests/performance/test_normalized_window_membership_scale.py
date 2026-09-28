"""QF-45-shaped membership growth: 4,095 decisions stored as catalogues and ranges.

This is a storage-shape fixture, not a claimed market dataset: an existing
fixture decision supplies the context structure and synthetic bar IDs reproduce
QF-45's measured pattern (2m 62 -> 4,156 prefix growth; daily 51 bars with one
rolling start shift at the first session close, then growth to 71). Real QF-11
execution and scientific validation live in the unit/integration tests.
"""

import hashlib
import re
from collections import Counter
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.prediction.window_compact import (
    CompactPredictionWindowDecision,
    CompactPredictionWindowResult,
    PredictionWindowEvidence,
)
from quantforge.prediction.window_encoding import canonical, mapping
from quantforge.prediction.window_membership import (
    CATALOGUE_SEGMENTS_FIELD,
    MembershipCatalogues,
    MembershipView,
    prediction_context,
    with_prediction_context,
)
from quantforge.prediction.window_reader import PredictionWindowReader
from tests.unit.prediction.test_compact_prediction_window import rehash_decision
from tests.unit.prediction.test_prediction_window import START, run_window, schedule

DECISIONS = 4_095
SESSION_DECISIONS = 195


def ids(prefix: str, count: int) -> list[str]:
    return [
        hashlib.sha256(f"{prefix}-{index}".encode()).hexdigest()
        for index in range(count)
    ]


PRIMARY = ids("2m", 62 + DECISIONS - 1)
DAILY = ids("daily", 72)


def membership(sequence: int) -> tuple[list[str], list[str]]:
    """Exact QF-45 counts: prefix-growing 2m; one rolling daily start shift."""
    closed = (sequence + 1) // SESSION_DECISIONS
    daily = DAILY[:51] if closed == 0 else DAILY[1 : 51 + closed]
    return PRIMARY[: 62 + sequence], daily


def daily_index(context: PrimitiveMapping) -> int:
    source = cast(dict[str, Any], context["source_context"])
    return next(
        index
        for index, entry in enumerate(cast(list[dict[str, Any]], source["timeframes"]))
        if entry["requirement"]["timeframe"]["configuration"]["interval"]
        == {"kind": "exchange_sessions", "session_count": 1}
    )


def with_membership(
    template: PrimitiveMapping, primary: list[str], daily: list[str]
) -> PrimitiveMapping:
    """Replace the primary and daily lists in every source/rule/indicator copy."""
    context = deepcopy(template)
    source = cast(dict[str, Any], context["source_context"])
    lists = {0: primary, daily_index(context): daily}
    for index, members in lists.items():
        source["timeframes"][index]["visible_bar_ids"] = list(members)
        rule = cast(list[dict[str, Any]], context["timeframes"])[index]
        rule["visible_bar_ids"] = list(members)
        for indicator in rule["indicators"]:
            indicator["visible_bar_ids"] = list(members)
    return context


def test_qf45_membership_pattern_is_stored_once_with_bounded_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    began = perf_counter()
    original = run_window()
    broad = schedule(START, START + timedelta(days=100))
    exact = schedule(START, broad.decision_timestamps[DECISIONS - 1])
    assert len(exact.decision_timestamps) == DECISIONS
    identity = original.identity_snapshot.to_primitive()
    identity["schedule"] = exact.to_primitive()
    evidence = PredictionWindowEvidence(PrimitiveMappingSnapshot.capture(identity), "3")
    template = (
        CompactPredictionWindowResult.from_window(original).decisions[0].to_primitive()
    )
    context_template = prediction_context(template)
    state = MembershipCatalogues()
    decisions: list[CompactPredictionWindowDecision] = []
    expanded_bytes = 0
    for sequence, timestamp in enumerate(exact.decision_timestamps):
        logical = with_membership(context_template, *membership(sequence))
        record = with_prediction_context(template, logical)
        record.update(
            sequence=sequence,
            decision_timestamp=timestamp.isoformat(),
            shared_evidence_id=evidence.evidence_id,
        )
        record.pop("decision_id")
        # Exact schema 2 equivalent of this decision (same ID field widths).
        rehash_decision(record)
        expanded_bytes += len(canonical(record)) + 1
        record.pop("decision_id")
        context, segments = state.normalize(logical)
        record = with_prediction_context(record, context)
        record[CATALOGUE_SEGMENTS_FIELD] = segments
        rehash_decision(record)
        state.accept(record)
        decisions.append(
            CompactPredictionWindowDecision(PrimitiveMappingSnapshot.capture(record))
        )
    converted = perf_counter()
    compact = CompactPredictionWindowResult(evidence, tuple(decisions))
    path = tmp_path / "window.jsonl"
    with path.open("wb") as stream:
        stream.writelines(compact.iter_serialized_records())
    header = compact.header_snapshot.to_primitive()
    header_size = len(canonical(header)) + 1
    shared_size = len(canonical(evidence.to_primitive())) + 1
    sizes = [len(item.snapshot.canonical_json.encode()) + 1 for item in decisions]
    shared_decision_bytes = header_size + shared_size
    expanded_total = shared_decision_bytes + expanded_bytes
    assert path.stat().st_size == shared_decision_bytes + sum(sizes)
    catalogues = {
        item["bar_count"]
        for item in cast(list[dict[str, Any]], header["membership_catalogues"])
    }
    assert {len(PRIMARY), len(DAILY)} <= catalogues  # 4,156 and 72, like QF-45.
    # The first decision introduces the initial catalogues; afterwards storage per
    # decision no longer grows with visible history (at most two new IDs each).
    assert max(sizes[1:]) - min(sizes[1:]) < 1_000
    assert sizes[0] - sizes[1] < (62 + 51) * 70 + 4_000
    # Remaining bytes are fixture context/outcome evidence, not repeated membership.
    assert path.stat().st_size * 10 < expanded_total
    occurrences = Counter(re.findall(rb'"([0-9a-f]{64})"', path.read_bytes()))
    assert all(occurrences[bar.encode()] == 1 for bar in (*PRIMARY, *DAILY))

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("streaming verification must not re-expand membership")

    monkeypatch.setattr(MembershipView, "expand", forbidden)
    reader = PredictionWindowReader.open(path)
    started = perf_counter()
    reader.verify_integrity()
    verified = perf_counter()
    starts: list[tuple[int, int]] = []
    daily = daily_index(context_template)
    for item in reader.iterate_decisions():
        source = mapping(prediction_context(item.to_primitive())["source_context"])
        entries = cast(list[dict[str, Any]], source["timeframes"])
        starts.append(
            (
                entries[0]["visible_bar_range"]["start_index"],
                entries[daily]["visible_bar_range"]["start_index"],
            )
        )
    # 2m always starts at the catalogue origin; daily rolls once at the first close.
    assert starts == [(0, 0)] * (SESSION_DECISIONS - 1) + [(0, 1)] * (
        DECISIONS - SESSION_DECISIONS + 1
    )
    report = {
        "decisions": DECISIONS,
        "header_bytes": header_size,
        "shared_record_bytes": shared_size,
        "decision_bytes_min": min(sizes),
        "decision_bytes_max": max(sizes),
        "normalized_total_bytes": path.stat().st_size,
        "expanded_equivalent_total_bytes": expanded_total,
        "fixture_build_seconds": round(converted - began, 2),
        "verify_integrity_seconds": round(verified - started, 2),
    }
    print(canonical(cast(Any, report)).decode())
