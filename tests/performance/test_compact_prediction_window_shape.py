"""QF-45-sized structural storage proof, without an 18.8 GB legacy allocation.

The large payload is synthetic immutable market evidence, not a claimed market
dataset. Real QF-52 ancestry and outcome validation live in the integration tests.
Only one copy of the heavy shared bytes and lightweight decision records exist.
"""

from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.prediction import PredictionContextFailurePolicy
from quantforge.prediction.window_compact import (
    CompactPredictionWindowDecision,
    CompactPredictionWindowResult,
    PredictionWindowEvidence,
)
from quantforge.prediction.window_encoding import canonical, mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from tests.unit.prediction.test_compact_prediction_window import rehash_decision
from tests.unit.prediction.test_multi_timeframe_study import (
    _requirements,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_window import (
    START,
    WindowProvider,
    run_window,
    schedule,
)

DECISIONS = 5_760
SHARED_PAYLOAD_BYTES = 3_263_326
MARKER = "qf55-shared-scale-evidence:"


def scale_fixture() -> CompactPredictionWindowResult:
    provider = WindowProvider()
    provider.series = ()
    original = run_window(
        provider,
        requirements=_requirements(failure_policy=PredictionContextFailurePolicy.SKIP),
        decision_schedule=schedule(START, START),
    )
    broad = schedule(START, START + timedelta(days=130))
    exact = schedule(START, broad.decision_timestamps[DECISIONS - 1])
    assert len(exact.decision_timestamps) == DECISIONS
    identity = original.identity_snapshot.to_primitive()
    identity["schedule"] = exact.to_primitive()
    mapping(identity["market_data"])["synthetic_scale_evidence"] = MARKER + "x" * (
        SHARED_PAYLOAD_BYTES - len(MARKER)
    )
    evidence = PredictionWindowEvidence(PrimitiveMappingSnapshot.capture(identity))
    template = (
        CompactPredictionWindowResult.from_window(original).decisions[0].to_primitive()
    )
    decisions: list[CompactPredictionWindowDecision] = []
    for index, timestamp in enumerate(exact.decision_timestamps):
        record = {
            **template,
            "sequence": index,
            "decision_timestamp": timestamp.isoformat(),
            "shared_evidence_id": evidence.evidence_id,
        }
        rehash_decision(record)
        decisions.append(
            CompactPredictionWindowDecision(PrimitiveMappingSnapshot.capture(record))
        )
    return CompactPredictionWindowResult(evidence, tuple(decisions))


def test_qf45_scale_is_shared_once_plus_small_decision_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compact = scale_fixture()
    path = tmp_path / "window.jsonl"
    # Finalized serialization iterator, not incremental execution/checkpointing.
    with path.open("wb") as stream:
        stream.writelines(compact.iter_serialized_records())
    shared_size = len(canonical(compact.evidence.to_primitive())) + 1
    header_size = len(canonical(compact.header_snapshot.to_primitive())) + 1
    decision_sizes = [
        len(item.snapshot.canonical_json.encode()) + 1 for item in compact.decisions
    ]
    assert path.stat().st_size == shared_size + header_size + sum(decision_sizes)
    assert max(decision_sizes) < SHARED_PAYLOAD_BYTES // 100
    assert shared_size > SHARED_PAYLOAD_BYTES
    assert SHARED_PAYLOAD_BYTES * DECISIONS == 18_796_757_760
    assert path.stat().st_size < SHARED_PAYLOAD_BYTES * DECISIONS // 100
    with path.open("rb") as stream:
        assert sum(line.count(MARKER.encode()) for line in stream) == 1
    reader = PredictionWindowReader.open(path)
    shared_decodes = 0
    decode = PrimitiveMappingSnapshot.to_primitive

    def counted(snapshot: PrimitiveMappingSnapshot) -> dict[str, Any]:
        nonlocal shared_decodes
        if snapshot is reader.evidence.identity_snapshot:
            shared_decodes += 1
        return decode(snapshot)

    monkeypatch.setattr(PrimitiveMappingSnapshot, "to_primitive", counted)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("compact reading must not construct embedded decisions")

    monkeypatch.setattr(CompactPredictionWindowDecision, "from_embedded", forbidden)
    assert reader.decision_count == DECISIONS
    assert shared_decodes == 0
    reader.verify_integrity()
    assert shared_decodes == 1  # One shared preamble check, independent of N.
    report = {
        "decisions": DECISIONS,
        "shared_payload_bytes": SHARED_PAYLOAD_BYTES,
        "shared_record_bytes": shared_size,
        "header_bytes": header_size,
        "decision_bytes_min": min(decision_sizes),
        "decision_bytes_max": max(decision_sizes),
        "decision_bytes_mean": sum(decision_sizes) / DECISIONS,
        "compact_bytes": path.stat().st_size,
        "theoretical_legacy_repeated_payload_bytes": SHARED_PAYLOAD_BYTES * DECISIONS,
    }
    print(canonical(cast(Any, report)).decode())
