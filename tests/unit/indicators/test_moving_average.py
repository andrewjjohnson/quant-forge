from decimal import ROUND_DOWN, ROUND_UP, Decimal, localcontext
from typing import cast

import pytest

from quantforge.indicators import (
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    Indicator,
    InvalidIndicatorParametersError,
    MarketField,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
)

from ..helpers import make_dataset


def calculate_through_contract(
    indicator: Indicator, closes: tuple[str, ...]
) -> tuple[Decimal | None, ...]:
    return indicator.calculate(make_dataset(closes)).values_for(
        SIMPLE_MOVING_AVERAGE_OUTPUT
    )


def test_window_one_and_exact_full_window_values_preserve_alignment() -> None:
    dataset = make_dataset(("1", "2", "3", "6"))
    one = calculate_through_contract(
        SimpleMovingAverage(SimpleMovingAverageParameters(1)),
        ("1", "2", "3"),
    )
    three = SimpleMovingAverage(SimpleMovingAverageParameters(3)).calculate(dataset)

    assert one == (Decimal(1), Decimal(2), Decimal(3))
    assert three.session_dates == tuple(bar.session_date for bar in dataset.bars)
    assert three.values_for(SIMPLE_MOVING_AVERAGE_OUTPUT) == (
        None,
        None,
        Decimal(2),
        Decimal("3.666666666666666666666666666666667"),
    )


@pytest.mark.parametrize("window", [0, -1])
def test_rejects_nonpositive_window(window: int) -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="at least 1"):
        SimpleMovingAverageParameters(window)


def test_rejects_noninteger_and_unknown_source_fields() -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="integer"):
        SimpleMovingAverageParameters(cast(int, 2.5))
    with pytest.raises(InvalidIndicatorParametersError, match="source_field"):
        SimpleMovingAverageParameters(2, cast(MarketField, "adjusted_close"))


def test_window_larger_than_dataset_preserves_unavailable_rows() -> None:
    values = calculate_through_contract(
        SimpleMovingAverage(SimpleMovingAverageParameters(4)), ("1", "2", "3")
    )
    assert values == (None, None, None)


def test_missing_source_value_invalidates_each_containing_full_window() -> None:
    values = calculate_through_contract(
        SimpleMovingAverage(SimpleMovingAverageParameters(2)),
        ("1", "NaN", "3", "5"),
    )
    assert values == (None, None, None, Decimal(4))


def test_calculation_does_not_mutate_input() -> None:
    dataset = make_dataset(("1", "2", "3"))
    original = dataset

    SimpleMovingAverage(SimpleMovingAverageParameters(2)).calculate(dataset)

    assert dataset == original


def test_indicator_configuration_is_stable_and_inspectable() -> None:
    first = SimpleMovingAverage(SimpleMovingAverageParameters(3, MarketField.CLOSE))
    second = SimpleMovingAverage(SimpleMovingAverageParameters(3, MarketField.CLOSE))
    different = SimpleMovingAverage(SimpleMovingAverageParameters(4, MarketField.CLOSE))

    assert first.configuration_id == second.configuration_id
    assert first.configuration_id != different.configuration_id
    assert first.configuration()["required_fields"] == ["close"]
    assert first.configuration()["warm_up_observations"] == 3
    assert first.configuration()["arithmetic"] == {
        "decimal_precision": 34,
        "rounding": "ROUND_HALF_EVEN",
    }
    assert first.missing_value is None


def test_moving_average_ignores_ambient_decimal_context() -> None:
    indicator = SimpleMovingAverage(SimpleMovingAverageParameters(3))
    dataset = make_dataset(("1", "2", "8"))

    with localcontext() as low_precision:
        low_precision.prec = 8
        low_precision.rounding = ROUND_DOWN
        low_result = indicator.calculate(dataset)
    with localcontext() as high_precision:
        high_precision.prec = 50
        high_precision.rounding = ROUND_UP
        high_result = indicator.calculate(dataset)

    expected = Decimal("3.666666666666666666666666666666667")
    assert low_result.values_for(SIMPLE_MOVING_AVERAGE_OUTPUT)[-1] == expected
    assert high_result.values_for(SIMPLE_MOVING_AVERAGE_OUTPUT)[-1] == expected


def test_appended_future_bars_do_not_change_historical_values() -> None:
    indicator = SimpleMovingAverage(SimpleMovingAverageParameters(3))
    through_cutoff = indicator.calculate(make_dataset(("3", "2", "1", "2", "3")))
    with_future = indicator.calculate(
        make_dataset(("3", "2", "1", "2", "3", "1000", "1.5"))
    )

    assert with_future.session_dates[:5] == through_cutoff.session_dates
    assert with_future.values_for(SIMPLE_MOVING_AVERAGE_OUTPUT)[:5] == (
        through_cutoff.values_for(SIMPLE_MOVING_AVERAGE_OUTPUT)
    )


def test_output_rows_use_null_for_unavailable_values() -> None:
    output = SimpleMovingAverage(SimpleMovingAverageParameters(2)).calculate(
        make_dataset(("1", "3"))
    )
    assert output.to_rows() == [
        {"session_date": "2024-07-01", "simple_moving_average": None},
        {"session_date": "2024-07-02", "simple_moving_average": "2"},
    ]
