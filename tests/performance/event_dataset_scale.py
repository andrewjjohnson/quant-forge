"""QF-67 SYNTHETIC scale fixture: thousands of schema-4 event observations.

This is a benchmark fixture, not market or research evidence. One real QF-49
rich observation from a QF-39 trial window of the synthetic QF-45 composition
is the template. Its causal features, timestamps and outcome values are
rewritten per synthetic event (positive, zero, negative and unavailable labels
in rotation) and every other scheduled decision is an ordinary no-trigger
receipt. Windows are streamed one receipt at a time through the unchanged
QF-64 ``checked_receipts``/``window_header`` contracts, so the structural
reader accepts them. They cannot pass QF-39 scientific validation, so their
``EventSourceWindow`` is created with the loaders' private token: only the
assembly, artifact and reading costs are representative, while plan-bound
verification is measured on the real fixtures and real QF-45 evidence.
"""

import copy
import hashlib
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.examples.spy_ema import TWO_MINUTES
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.prediction import PredictionDecisionSchedule
from quantforge.prediction.window_compact import (
    CompactDecisionReceipt,
    CompactPredictionWindowDecision,
    PredictionWindowEvidence,
    WindowRecordCounts,
    checked_receipts,
    window_header,
)
from quantforge.prediction.window_encoding import WindowResultIdentity, canonical
from quantforge.prediction.window_membership import (
    CATALOGUE_SEGMENTS_FIELD,
    MembershipCatalogues,
    prediction_context,
    with_prediction_context,
)
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.validation import (
    PartitionRole,
    TimestampBoundary,
    ValidationPlan,
    ValidationWindow,
)
from quantforge.walk_forward import PredictionEvaluator, WalkForwardConfig
from tests.integration import rapid_scan_fixtures
from tests.integration.rapid_scan_fixtures import windowed_plan

NEW_YORK = ZoneInfo("America/New_York")
# Outcome rotation: positive, exactly zero (negative), negative, unavailable.
_RETURNS = ("0.001", "0", "-0.001", None)


def at(day: str, clock: time) -> datetime:
    return datetime.combine(date.fromisoformat(day), clock, NEW_YORK).astimezone(UTC)


def long_plan(
    inputs: SmokeInputs,
    root: Path,
    windows: dict[PartitionRole, tuple[datetime, datetime]],
) -> tuple[WalkForwardConfig, PredictionEvaluator]:
    """The unchanged QF-45 plan wiring over long synthetic windows (never run)."""
    with pytest.MonkeyPatch.context() as patch:
        for role, interval in windows.items():
            patch.setitem(rapid_scan_fixtures.WINDOWS, role, interval)
        return windowed_plan(inputs, root, fixed=True)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def template_record(reader: PredictionWindowReader) -> PrimitiveMapping:
    """The first rich decision of a real window, as its logical record."""
    for receipt in reader.iterate_decision_receipts():
        if receipt.decision is not None and receipt.observation_count == 1:
            record = receipt.decision.expanded_record()
            record.pop("decision_id")
            return record
    raise AssertionError("template window has no observation")


def _event(
    template: PrimitiveMapping, timestamp: datetime, session: date, index: int
) -> PrimitiveMapping:
    record = copy.deepcopy(template)
    signal = cast(dict[str, Any], cast(list[Any], record["generated_signals"])[0])
    prediction = cast(dict[str, Any], signal["prediction"])
    prediction["signal_session"] = session.isoformat()
    prediction["values"]["decision_timestamp"] = timestamp.isoformat()
    signal["features"] = {
        name: decimal_to_primitive(Decimal(value) + Decimal(index) / 10_000)
        for name, value in cast(dict[str, str], signal["features"]).items()
    }
    row = cast(
        dict[str, Any], cast(dict[str, Any], record["prediction_study"])["rows"][0]
    )
    row["prediction"] = copy.deepcopy(prediction)
    row["features"] = copy.deepcopy(signal["features"])
    outcome, evaluation = row["outcome"], row["evaluation"]
    values = cast(dict[str, Any], outcome["values"])
    raw = _RETURNS[index % len(_RETURNS)]
    values.update(
        decision_timestamp=timestamp.isoformat(),
        signal_session=session.isoformat(),
        raw_return=raw,
        status="available" if raw is not None else "session_overflow",
        available=raw is not None,
    )
    if raw is None:
        values.update(
            outcome_price=None,
            expected_observation_timestamp=None,
            resolved_observation_timestamp=None,
            observation_id=None,
        )
    outcome_id = digest(f"outcome-{index}")
    outcome.update(
        outcome_id=outcome_id,
        signal_session=session.isoformat(),
        outcome_session=session.isoformat(),
    )
    evaluation.update(outcome_id=outcome_id, values=copy.deepcopy(values))
    record["decision_timestamp"] = timestamp.isoformat()
    return record


@dataclass(frozen=True)
class SyntheticWindow:
    path: Path
    decisions: int
    events: int
    bytes: int


def write_synthetic_window(
    path: Path,
    reader: PredictionWindowReader,
    plan: ValidationPlan,
    window: ValidationWindow,
    *,
    every: int,
) -> SyntheticWindow:
    """Stream one schema-4 window over ``window``'s scheduled decisions."""
    membership = plan.prediction_membership
    assert membership is not None
    start = cast(TimestampBoundary, window.interval.start).timestamp
    end = cast(TimestampBoundary, window.interval.end).timestamp
    schedule = PredictionDecisionSchedule(TWO_MINUTES, start, end)
    identity = reader.evidence.identity_snapshot.to_primitive()
    identity["schedule"] = schedule.to_primitive()
    evidence = PredictionWindowEvidence(PrimitiveMappingSnapshot.capture(identity), "4")
    template = template_record(reader)
    logical = prediction_context(template)
    generation = MembershipCatalogues()
    events = 0

    def receipts() -> Iterator[CompactDecisionReceipt]:
        nonlocal events
        for sequence, timestamp in enumerate(schedule.decision_timestamps):
            if sequence % every:
                yield CompactDecisionReceipt.no_prediction(
                    sequence=sequence,
                    context_id=digest(f"context-{sequence}"),
                    prediction_study_id=digest(f"study-{sequence}"),
                )
                continue
            record = dict(
                _event(template, timestamp, membership.session_for(timestamp), events)
            )
            record.update(
                sequence=sequence,
                shared_evidence_id=evidence.evidence_id,
                record_type="decision",
            )
            context, segments = generation.normalize(logical)
            record = with_prediction_context(record, context)
            record[CATALOGUE_SEGMENTS_FIELD] = segments
            record["decision_id"] = configuration_identity(
                {k: v for k, v in record.items() if k != "decision_id"}
            )
            generation.accept(record)
            events += 1
            yield CompactDecisionReceipt.with_decision(
                CompactPredictionWindowDecision(
                    PrimitiveMappingSnapshot.capture(record)
                )
            )

    catalogues = MembershipCatalogues()
    counts = WindowRecordCounts()
    result = WindowResultIdentity(evidence.window_id)
    body = path.with_name(path.name + ".body")
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions = 0
    with body.open("wb") as stream:
        for receipt in checked_receipts(receipts(), evidence):
            if receipt.decision is not None:
                catalogues.accept(receipt.decision.to_primitive())
            counts.update_receipt(receipt)
            result.update(receipt.to_primitive())
            stream.write(receipt.snapshot.canonical_json.encode() + b"\n")
            decisions += 1
    header = window_header(
        evidence,
        window_result_id=result.hexdigest(),
        decision_count=decisions,
        record_counts=counts.totals,
        membership=catalogues,
    )
    with path.open("wb") as stream, body.open("rb") as source:
        stream.write(canonical(header) + b"\n")
        stream.write(canonical(evidence.to_primitive()) + b"\n")
        shutil.copyfileobj(source, stream, length=1 << 20)
    body.unlink()
    return SyntheticWindow(path, decisions, events, path.stat().st_size)
