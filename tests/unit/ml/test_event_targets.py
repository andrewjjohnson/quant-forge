"""QF-67 binary target: strict comparison, explicit unavailability, binding."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.examples.spy_ema import TWO_MINUTES
from quantforge.ml import (
    BoundEventTarget,
    EventDatasetIntegrityError,
    EventTargetError,
    ForwardReturnBinaryTarget,
    TargetLabel,
)
from quantforge.ml.targets import require_label
from quantforge.prediction import (
    IntradayForwardReturnEvaluator,
    IntradayForwardReturnOutcomeLabeler,
)
from quantforge.prediction.outcome_resolution import OutcomeResolutionStatus
from quantforge.prediction.outcome_temporal import OutcomeTemporalConfiguration
from quantforge.validation import ValidationPlan

DECISION = datetime(2024, 12, 23, 16, 0, tzinfo=UTC)
BOUND = BoundEventTarget(
    ForwardReturnBinaryTarget(), "outcome-config", "evaluator-config", timedelta(0)
)


def row(**values: Any) -> PrimitiveMapping:
    payload: PrimitiveMapping = {
        "decision_timestamp": DECISION.isoformat(),
        "elapsed_duration_microseconds": 1_800_000_000,
        "status": "available",
        "available": True,
        "raw_return": "0.0016",
        "reference_price": "130",
    }
    payload.update(values)
    return {
        "outcome": {
            "outcome_configuration_id": "outcome-config",
            "outcome_id": "outcome-1",
            "values": payload,
        },
        "evaluation": {
            "evaluator_configuration_id": "evaluator-config",
            "outcome_id": "outcome-1",
            "values": dict(payload),
        },
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0.0016", True), ("0.0000001", True), ("0", False), ("-0.0016", False)],
)
def test_available_returns_use_a_strict_greater_than_zero_comparison(
    raw: str, expected: bool
) -> None:
    label = BOUND.label(row(raw_return=raw), DECISION)
    assert label.value is expected
    assert (label.status, label.source_value, label.outcome_id) == (
        "available",
        raw,
        "outcome-1",
    )


UNAVAILABLE = [
    status.value
    for status in OutcomeResolutionStatus
    if status is not OutcomeResolutionStatus.AVAILABLE
]


@pytest.mark.parametrize("status", UNAVAILABLE)
def test_unavailable_outcomes_stay_null_with_status_never_false_or_zero(
    status: str,
) -> None:
    label = BOUND.label(row(status=status, available=False, raw_return=None), DECISION)
    assert label.value is None
    assert label.value is not False
    assert label.source_value is None
    assert label.status == status


def test_threshold_is_explicit_and_part_of_identity() -> None:
    strict = BoundEventTarget(
        ForwardReturnBinaryTarget(threshold=Decimal("0.001")),
        "outcome-config",
        "evaluator-config",
        timedelta(0),
    )
    assert strict.label(row(raw_return="0.001"), DECISION).value is False
    assert strict.label(row(raw_return="0.0011"), DECISION).value is True
    assert strict.target.target_id != BOUND.target.target_id
    assert strict.configuration_id != BOUND.configuration_id


@pytest.mark.parametrize(
    "corruption",
    [
        {"available": True, "raw_return": None},
        {"status": "session_overflow", "available": False, "raw_return": "0"},
        {"status": "available", "available": False, "raw_return": None},
        {"status": "session_overflow", "available": True},
        {"status": "unknown_status", "available": False, "raw_return": None},
        {"available": "true"},
        {"raw_return": "not-a-number"},
        {"raw_return": "NaN"},
        {"raw_return": "Infinity"},
        {"raw_return": "0.00160"},
        {"raw_return": "1.6E-3"},
        {"raw_return": 0.0016},
        {"decision_timestamp": "2024-12-23T16:02:00+00:00"},
        {"elapsed_duration_microseconds": 600_000_000},
    ],
)
def test_contradictory_or_foreign_outcomes_fail_closed(
    corruption: dict[str, Any],
) -> None:
    with pytest.raises(EventDatasetIntegrityError):
        BOUND.label(row(**corruption), DECISION)


def test_rows_must_belong_to_the_bound_outcome_and_evaluator() -> None:
    with pytest.raises(EventDatasetIntegrityError, match="explicit availability"):
        BOUND.label(None, DECISION)
    for envelope, key in (
        ("outcome", "outcome_configuration_id"),
        ("evaluation", "evaluator_configuration_id"),
        ("evaluation", "outcome_id"),
    ):
        foreign = row()
        cast(dict[str, Any], foreign[envelope])[key] = "other"
        with pytest.raises(EventDatasetIntegrityError, match="bound target"):
            BOUND.label(foreign, DECISION)
    divergent = row()
    cast(dict[str, Any], cast(dict[str, Any], divergent["evaluation"])["values"])[
        "raw_return"
    ] = "-1"
    with pytest.raises(EventDatasetIntegrityError, match="bound target"):
        BOUND.label(divergent, DECISION)
    with pytest.raises(EventDatasetIntegrityError, match="incomplete"):
        BOUND.label({"outcome": {}}, DECISION)


def test_stored_labels_are_rechecked_against_their_outcome_values() -> None:
    threshold = Decimal(0)
    for valid in (
        TargetLabel(True, "available", "0.0016", "o"),
        TargetLabel(False, "available", "0", "o"),
        TargetLabel(None, "session_overflow", None, "o"),
    ):
        require_label(valid, threshold)
    for invalid, message in (
        (TargetLabel(None, "unknown", None, "o"), "unknown target status"),
        (TargetLabel(False, "session_overflow", None, "o"), "inconsistent"),
        (TargetLabel(None, "dataset_end", "0", "o"), "inconsistent"),
        (TargetLabel(True, "available", "Infinity", "o"), "canonical"),
        (TargetLabel(True, "available", "0.00160", "o"), "canonical"),
        (TargetLabel(True, "available", "0", "o"), "inconsistent"),
        (TargetLabel(None, "available", "0.1", "o"), "inconsistent"),
        (TargetLabel(True, "available", "0.1", ""), "outcome identity"),
    ):
        with pytest.raises(EventDatasetIntegrityError, match=message):
            require_label(invalid, threshold)


def test_target_configuration_round_trips_and_rejects_tampering() -> None:
    target = ForwardReturnBinaryTarget()
    primitive = target.to_primitive()
    assert ForwardReturnBinaryTarget.from_primitive(primitive) == target
    assert primitive["comparison"] == "strictly_greater_than"
    assert primitive["direction_adjustment"] == "none"
    assert primitive["missing_label"] == "unavailable_outcome_is_null_with_status"
    tampered = dict(primitive)
    tampered["comparison"] = "greater_than_or_equal"
    with pytest.raises(EventTargetError):
        ForwardReturnBinaryTarget.from_primitive(tampered)
    invalid: tuple[dict[str, Any], ...] = (
        {"horizon": timedelta(0)},
        {"horizon": timedelta(microseconds=1) / 2},
        {"threshold": Decimal("NaN")},
        {"threshold": 0.0},
        {"name": ""},
    )
    for values in invalid:
        with pytest.raises(EventTargetError):
            ForwardReturnBinaryTarget(**values)


def definition(horizon: timedelta = timedelta(minutes=30)) -> PrimitiveMapping:
    labeler = IntradayForwardReturnOutcomeLabeler(
        OutcomeTemporalConfiguration.elapsed_duration(horizon, TWO_MINUTES)
    )
    evaluator = IntradayForwardReturnEvaluator()
    return {
        "outcome_labeler": {
            "name": labeler.name,
            "implementation_version": labeler.implementation_version,
            "configuration": labeler.configuration(),
            "configuration_id": labeler.configuration_id,
        },
        "evaluator": {
            "name": evaluator.name,
            "implementation_version": evaluator.implementation_version,
            "configuration": evaluator.configuration(),
            "configuration_id": evaluator.configuration_id,
        },
    }


def test_binding_requires_the_exact_qf49_outcome_and_horizon() -> None:
    # These checks precede any plan access; the plan is never consulted here.
    plan = cast(ValidationPlan, object())
    target = ForwardReturnBinaryTarget()
    with pytest.raises(EventTargetError, match="horizon"):
        target.bind(definition(timedelta(minutes=10)), plan)
    edited = definition()
    labeler = cast(dict[str, Any], edited["outcome_labeler"])
    configuration = cast(dict[str, Any], labeler["configuration"])
    configuration["parameters"] = {
        **configuration["parameters"],
        "return_formula": "log(outcome_close / reference_close)",
    }
    with pytest.raises(EventTargetError, match="differ"):
        target.bind(edited, plan)
    foreign = definition()
    cast(dict[str, Any], foreign["evaluator"])["name"] = "direction_evaluator"
    with pytest.raises(EventTargetError, match="differ"):
        target.bind(foreign, plan)
    with pytest.raises(EventTargetError, match="QF-49"):
        target.bind({"outcome_labeler": {"configuration": {}}}, plan)
