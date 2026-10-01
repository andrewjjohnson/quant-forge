"""QF-72: one QF-45 decision kernel and candidate builder serve both paths."""

from datetime import UTC, date, datetime, time
from decimal import Decimal

import pytest

from quantforge.examples.spy_ema import (
    DAILY,
    DECISION_END,
    DECISION_START,
    DECISION_TIMEZONE,
    TWO_MINUTES,
    EmaParameters,
    EmaSmokeRule,
    midday_bullish_cross,
)
from quantforge.prediction import PredictionDirection
from quantforge.rapid import RapidBarInput, RapidIndicatorInput
from tests.unit.test_spy_ema_smoke import rule_context


@pytest.mark.parametrize(
    ("clock", "values", "expected"),
    [
        (time(11), ("99", "100", "101", "100", "110", "100"), True),
        (time(14), ("100", "100", "101", "100", "110", "100"), True),
        (time(10, 58), ("99", "100", "101", "100", "110", "100"), False),
        (time(14, 2), ("99", "100", "101", "100", "110", "100"), False),
        (time(12), ("101", "100", "102", "100", "110", "100"), False),
        (time(12), ("99", "100", "100", "100", "110", "100"), False),
        (time(12), ("99", "100", "101", "100", "100", "100"), False),
        (time(12), ("99", "100", "101", "100", "110", None), False),
        (time(12), (None, "100", "101", "100", "110", "100"), False),
    ],
)
def test_kernel_truth_table(
    clock: time, values: tuple[str | None, ...], expected: bool
) -> None:
    decimals = tuple(None if value is None else Decimal(value) for value in values)
    assert midday_bullish_cross(clock, *decimals) is expected
    decide = EmaSmokeRule().rapid_specification().decide
    assert decide(clock, decimals) is (PredictionDirection.UP if expected else None)


def test_rapid_specification_declares_the_rule_inputs_and_window() -> None:
    rule = EmaSmokeRule(EmaParameters(12, 60))
    specification = rule.rapid_specification()
    assert specification.clock_timezone == DECISION_TIMEZONE
    window = specification.decision_window
    assert window is not None
    configured = rule.configuration()["decision_interval"]
    assert isinstance(configured, dict)
    assert (window.start, window.end) == (DECISION_START, DECISION_END)
    assert (window.start.isoformat(), window.end.isoformat()) == (
        configured["start"],
        configured["end"],
    )
    assert configured["timezone"] == DECISION_TIMEZONE
    assert specification.inputs == (
        RapidIndicatorInput(
            "previous_fast", TWO_MINUTES, "fast", "exponential_moving_average", 1
        ),
        RapidIndicatorInput(
            "previous_slow", TWO_MINUTES, "slow", "exponential_moving_average", 1
        ),
        RapidIndicatorInput(
            "current_fast", TWO_MINUTES, "fast", "exponential_moving_average"
        ),
        RapidIndicatorInput(
            "current_slow", TWO_MINUTES, "slow", "exponential_moving_average"
        ),
        RapidBarInput("daily_close", DAILY, "close"),
        RapidIndicatorInput(
            "daily_ema50", DAILY, "trend", "exponential_moving_average"
        ),
    )
    assert sorted(item.name for item in specification.inputs) == [
        field.name for field in rule.strategy_feature_definitions
    ]


def test_candidate_builder_is_the_authoritative_candidate() -> None:
    rule = EmaSmokeRule()
    context = rule_context(clock="15:00")
    (authoritative,) = rule.generate_with_context(context).signals
    values = (
        Decimal(99),
        Decimal(100),
        Decimal(101),
        Decimal(100),
        Decimal(110),
        Decimal(100),
    )
    rapid = rule.rapid_specification().candidate(
        "SPY", date(2025, 6, 2), datetime(2025, 6, 2, 15, tzinfo=UTC), values
    )
    assert rapid == authoritative


def test_promoted_parameters_rebuild_exactly() -> None:
    parameters = EmaParameters(12, 60)
    assert EmaParameters.from_primitive(parameters.to_primitive()) == parameters
    with pytest.raises(ValueError, match="daily=50"):
        EmaParameters.from_primitive({"fast": 12, "slow": 60, "daily": 20})
