"""5,760 durable decisions with large shared evidence and constant writer state."""

import json
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.prediction import PredictionContextFailurePolicy
from quantforge.prediction.window_compact import (
    CompactPredictionWindowDecision,
    CompactPredictionWindowResult,
)
from quantforge.prediction.window_compact_validation import (
    PredictionWindowDecisionValidator,
    validate_prediction_window_reader,
)
from quantforge.prediction.window_encoding import (
    StudyIdentity,
    WindowResultIdentity,
    canonical,
    mapping,
)
from quantforge.prediction.window_incremental import IncrementalPredictionWindowWriter
from tests.performance.test_compact_prediction_window_shape import (
    DECISIONS,
    MARKER,
    SHARED_PAYLOAD_BYTES,
)
from tests.unit.prediction.test_compact_prediction_window import rehash_decision
from tests.unit.prediction.test_multi_timeframe_study import (
    FixtureParameters,
    _requirements,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_window import (
    START,
    WindowProvider,
    run_window,
    schedule,
)


def test_5760_decisions_interrupt_resume_and_validate(tmp_path: Path) -> None:
    began = perf_counter()
    provider = WindowProvider()
    provider.series = ()
    original = run_window(
        provider,
        requirements=_requirements(failure_policy=PredictionContextFailurePolicy.SKIP),
        decision_schedule=schedule(START, START),
    )
    broad = schedule(START, START + timedelta(days=130))
    exact = schedule(START, broad.decision_timestamps[DECISIONS - 1])
    identity = original.identity_snapshot.to_primitive()
    identity["schedule"] = exact.to_primitive()
    mapping(identity["market_data"])["synthetic_scale_evidence"] = MARKER + "x" * (
        SHARED_PAYLOAD_BYTES - len(MARKER)
    )
    expected = PrimitiveMappingSnapshot.capture(identity)
    validator = PredictionWindowDecisionValidator(
        expected_identity=expected,
        schedule=exact,
        outcome_sessions=(),
        strategy_parameters=FixtureParameters().to_primitive(),
    )
    template = (
        CompactPredictionWindowResult.from_window(original).decisions[0].to_primitive()
    )
    study_id = StudyIdentity(identity).for_context(
        mapping(
            mapping(mapping(template["prediction_study"])["manifest"])[
                "prediction_context"
            ]
        )
    )
    template["prediction_study_id"] = study_id
    mapping(mapping(template["prediction_study"])["manifest"])["study_id"] = study_id
    path = tmp_path / "window.jsonl"
    writer = IncrementalPredictionWindowWriter.open(path, validator=validator)
    uninterrupted = WindowResultIdentity(writer.evidence.window_id)
    sizes: list[int] = []  # diagnostic integers only, not completed records
    executions: list[int] = []
    interruption = 3000
    initial_fields = set(vars(writer))
    initial_shared = (writer.staging_path / "shared.json").read_bytes()

    def record(sequence: int) -> CompactPredictionWindowDecision:
        primitive: PrimitiveMapping = {
            **template,
            "sequence": sequence,
            "decision_timestamp": exact.decision_timestamps[sequence].isoformat(),
            "shared_evidence_id": writer.evidence.evidence_id,
        }
        rehash_decision(primitive)
        return CompactPredictionWindowDecision(
            PrimitiveMappingSnapshot.capture(primitive)
        )

    def execute_until(stop: int) -> None:
        for index in range(writer.completed_count, stop):
            decision = record(index)
            executions.append(index)
            writer.append(decision)
            uninterrupted.update(decision.to_primitive())
            sizes.append(len(decision.snapshot.canonical_json.encode()) + 1)
            assert set(vars(writer)) == initial_fields
            assert not any(
                isinstance(item, (list, tuple, CompactPredictionWindowDecision))
                for item in vars(writer).values()
            )
            del decision

    execute_until(interruption)
    assert len(list(writer.staging_path.iterdir())) == 3
    checkpoint = writer.checkpoint()
    # Simulate a crash while the next line is being written, before checkpoint.
    with writer.journal_path.open("ab") as stream:
        stream.write(b'{"decision_id":"partial')
    del writer
    writer = IncrementalPredictionWindowWriter.open(path, validator=validator)
    assert writer.completed_count == interruption
    assert writer.checkpoint() == checkpoint
    assert writer.journal_path.stat().st_size == checkpoint["byte_offset"]
    assert (writer.staging_path / "shared.json").read_bytes() == initial_shared
    execute_until(DECISIONS)
    assert executions == list(range(DECISIONS))
    reader = writer.finalize()
    assert reader.header()["window_result_id"] == uninterrupted.hexdigest()
    validate_prediction_window_reader(
        reader,
        expected_identity=expected,
        schedule=exact,
        outcome_sessions=(),
        strategy_parameters=FixtureParameters().to_primitive(),
    )
    with path.open("rb") as stream:
        assert sum(line.count(MARKER.encode()) for line in stream) == 1
    shared_bytes = len(initial_shared)
    header_bytes = len(canonical(reader.header())) + 1
    assert path.stat().st_size == header_bytes + shared_bytes + sum(sizes)
    assert reader.decision_count == DECISIONS
    report: dict[str, Any] = {
        "completed_count": DECISIONS,
        "artifact_bytes": path.stat().st_size,
        "shared_evidence_bytes": shared_bytes,
        "record_bytes_min": min(sizes),
        "record_bytes_max": max(sizes),
        "record_bytes_mean": sum(sizes) / DECISIONS,
        "checkpoints": DECISIONS,
        "interruption_after": interruption,
        "resume_sequence": interruption,
        "resume_decision_ordinal": interruption + 1,
        "completed_decisions_recomputed": 0,
        "retained_completed_records": 0,
        "peak_current_record_buffer": 1,
        "staging_files": 3,
        "elapsed_seconds": perf_counter() - began,
    }
    (tmp_path / "scale-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(canonical(cast(PrimitiveMapping, report)).decode())
