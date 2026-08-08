from decimal import Decimal

from quantforge.indicators import (
    AVERAGE_DIRECTIONAL_INDEX_OUTPUT,
    NEGATIVE_DIRECTIONAL_INDICATOR_OUTPUT,
    POSITIVE_DIRECTIONAL_INDICATOR_OUTPUT,
    WILDER_AVERAGE_TRUE_RANGE_OUTPUT,
    WILDER_RSI_OUTPUT,
    WilderAverageTrueRange,
    WilderAverageTrueRangeParameters,
    WilderDirectionalMovement,
    WilderDirectionalMovementParameters,
    WilderRelativeStrengthIndex,
    WilderRelativeStrengthIndexParameters,
)

from ..helpers import make_dataset


def test_wilder_rsi_uses_initial_period_then_recursive_smoothing() -> None:
    indicator = WilderRelativeStrengthIndex(WilderRelativeStrengthIndexParameters(2))

    output = indicator.calculate(make_dataset(("1", "2", "3", "2", "2")))

    assert output.values_for(WILDER_RSI_OUTPUT) == (
        None,
        None,
        Decimal(100),
        Decimal(50),
        Decimal(50),
    )
    assert indicator.warm_up_observations == 3


def test_wilder_rsi_flat_window_is_neutral() -> None:
    output = WilderRelativeStrengthIndex(
        WilderRelativeStrengthIndexParameters(2)
    ).calculate(make_dataset(("5", "5", "5")))

    assert output.values_for(WILDER_RSI_OUTPUT)[-1] == Decimal(50)


def test_directional_movement_aligns_di_and_delayed_adx_warmup() -> None:
    indicator = WilderDirectionalMovement(WilderDirectionalMovementParameters(2))

    output = indicator.calculate(make_dataset(("1", "2", "3", "2", "1")))

    positive_di = output.values_for(POSITIVE_DIRECTIONAL_INDICATOR_OUTPUT)
    negative_di = output.values_for(NEGATIVE_DIRECTIONAL_INDICATOR_OUTPUT)
    adx = output.values_for(AVERAGE_DIRECTIONAL_INDEX_OUTPUT)
    assert positive_di[:2] == negative_di[:2] == (None, None)
    assert positive_di[2:] == (
        Decimal(50),
        Decimal("33.33333333333333333333333333333333"),
        Decimal(20),
    )
    assert negative_di[2:] == (
        Decimal(0),
        Decimal("33.33333333333333333333333333333333"),
        Decimal(60),
    )
    assert adx == (None, None, None, Decimal(50), Decimal(50))
    assert indicator.warm_up_observations == 4


def test_appended_future_bars_do_not_change_wilder_history() -> None:
    rsi = WilderRelativeStrengthIndex(WilderRelativeStrengthIndexParameters(2))
    dmi = WilderDirectionalMovement(WilderDirectionalMovementParameters(2))
    through_cutoff = make_dataset(("1", "2", "3", "2", "1"))
    with_future = make_dataset(("1", "2", "3", "2", "1", "1000", "0.5"))

    rsi_cutoff = rsi.calculate(through_cutoff)
    rsi_future = rsi.calculate(with_future)
    dmi_cutoff = dmi.calculate(through_cutoff)
    dmi_future = dmi.calculate(with_future)

    assert rsi_future.values_for(WILDER_RSI_OUTPUT)[:5] == rsi_cutoff.values_for(
        WILDER_RSI_OUTPUT
    )
    for field_name in dmi.output_fields:
        assert dmi_future.values_for(field_name)[:5] == dmi_cutoff.values_for(
            field_name
        )


def test_wilder_atr_is_aligned_causal_and_hand_auditable() -> None:
    dataset = make_dataset(
        ("10", "11", "10", "12"),
        highs=("11", "12", "12", "13"),
        lows=("9", "10", "9", "10"),
    )
    indicator = WilderAverageTrueRange(WilderAverageTrueRangeParameters(2))

    values = indicator.calculate(dataset).values_for(WILDER_AVERAGE_TRUE_RANGE_OUTPUT)

    assert values == (None, None, Decimal("2.5"), Decimal("2.75"))
    assert indicator.warm_up_observations == 3


def test_wilder_atr_does_not_change_when_future_bars_are_appended() -> None:
    original = make_dataset(
        ("10", "11", "10"),
        highs=("11", "12", "12"),
        lows=("9", "10", "9"),
    )
    extended = make_dataset(
        ("10", "11", "10", "999"),
        highs=("11", "12", "12", "1000"),
        lows=("9", "10", "9", "1"),
    )
    indicator = WilderAverageTrueRange(WilderAverageTrueRangeParameters(2))

    original_values = indicator.calculate(original).values_for(
        WILDER_AVERAGE_TRUE_RANGE_OUTPUT
    )
    extended_values = indicator.calculate(extended).values_for(
        WILDER_AVERAGE_TRUE_RANGE_OUTPUT
    )

    assert original_values == extended_values[: len(original_values)]
