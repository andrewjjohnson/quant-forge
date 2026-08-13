from decimal import ROUND_DOWN, ROUND_UP, Decimal, localcontext
from typing import cast

import pytest

from quantforge.indicators import (
    EXPONENTIAL_MOVING_AVERAGE_OUTPUT,
    ExponentialMovingAverage,
    ExponentialMovingAverageParameters,
    InvalidIndicatorParametersError,
    MarketField,
)

from ..helpers import make_dataset


def _values(closes: tuple[str, ...], *, period: int = 3) -> tuple[Decimal | None, ...]:
    output = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(period)
    ).calculate(make_dataset(closes))
    return output.values_for(EXPONENTIAL_MOVING_AVERAGE_OUTPUT)


def test_hand_calculated_sma_seed_and_recursive_values_preserve_alignment() -> None:
    dataset = make_dataset(("10", "11", "12", "13", "14"))

    output = ExponentialMovingAverage(ExponentialMovingAverageParameters(3)).calculate(
        dataset
    )

    assert output.session_dates == tuple(bar.session_date for bar in dataset.bars)
    assert output.values_for(EXPONENTIAL_MOVING_AVERAGE_OUTPUT) == (
        None,
        None,
        Decimal(11),
        Decimal(12),
        Decimal(13),
    )


def test_period_one_is_available_from_first_observation() -> None:
    assert _values(("1", "2", "3"), period=1) == (
        Decimal(1),
        Decimal(2),
        Decimal(3),
    )


@pytest.mark.parametrize("period", [0, -1])
def test_rejects_nonpositive_period(period: int) -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="at least 1"):
        ExponentialMovingAverageParameters(period)


def test_rejects_noninteger_period_and_unsupported_source_field() -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="integer"):
        ExponentialMovingAverageParameters(cast(int, 2.5))
    with pytest.raises(InvalidIndicatorParametersError, match="source_field"):
        ExponentialMovingAverageParameters(2, cast(MarketField, "adjusted_close"))


def test_selected_source_field_controls_values() -> None:
    dataset = make_dataset(("100", "100", "100"), opens=("1", "3", "5"))
    output = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(2, source_field=MarketField.OPEN)
    ).calculate(dataset)

    assert output.values_for(EXPONENTIAL_MOVING_AVERAGE_OUTPUT) == (
        None,
        Decimal(2),
        Decimal("4"),
    )


def test_missing_source_emits_none_and_requires_a_fresh_full_seed() -> None:
    assert _values(("1", "2", "3", "NaN", "10", "20", "30")) == (
        None,
        None,
        Decimal(2),
        None,
        None,
        None,
        Decimal(20),
    )


def test_calculation_is_deterministic_and_does_not_mutate_input() -> None:
    dataset = make_dataset(("1", "2", "8", "4"))
    original = dataset
    indicator = ExponentialMovingAverage(ExponentialMovingAverageParameters(3))

    with localcontext() as low_precision:
        low_precision.prec = 8
        low_precision.rounding = ROUND_DOWN
        low_result = indicator.calculate(dataset)
    with localcontext() as high_precision:
        high_precision.prec = 50
        high_precision.rounding = ROUND_UP
        high_result = indicator.calculate(dataset)

    assert low_result == high_result
    assert dataset == original


def test_configuration_serialization_and_identity_are_stable() -> None:
    first = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3, MarketField.CLOSE)
    )
    same = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3, MarketField.CLOSE)
    )
    other_period = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(4, MarketField.CLOSE)
    )
    other_field = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3, MarketField.OPEN)
    )

    assert first.configuration_id == same.configuration_id
    assert first.configuration_id != other_period.configuration_id
    assert first.configuration_id != other_field.configuration_id
    assert first.parameters.to_primitive() == {
        "period": 3,
        "source_field": "close",
    }
    assert first.configuration()["formula"] == {
        "smoothing_factor": "2 / (period + 1)",
        "initialization": "simple_average_of_first_period_consecutive_observations",
        "recurrence": ("((period - 1) * previous + 2 * current) / (period + 1)"),
        "missing_value_policy": "emit_none_and_restart_initialization",
    }


def test_appended_future_bars_do_not_change_historical_values() -> None:
    through_cutoff = _values(("3", "2", "1", "2", "3"))
    with_future = _values(("3", "2", "1", "2", "3", "1000", "1.5"))

    assert with_future[:5] == through_cutoff
