from decimal import ROUND_DOWN, ROUND_UP, Decimal, localcontext
from typing import cast

import pytest

from quantforge.indicators import (
    BOLLINGER_BANDWIDTH_OUTPUT,
    BOLLINGER_LOWER_BAND_OUTPUT,
    BOLLINGER_MIDDLE_BAND_OUTPUT,
    BOLLINGER_UPPER_BAND_OUTPUT,
    BollingerBands,
    BollingerBandsParameters,
    InvalidIndicatorParametersError,
    MarketField,
)

from ..helpers import make_dataset


def test_population_standard_deviation_and_full_window_values_are_exact() -> None:
    dataset = make_dataset(("1", "3", "5"))

    output = BollingerBands(BollingerBandsParameters(2, Decimal(2))).calculate(dataset)

    assert output.session_dates == tuple(bar.session_date for bar in dataset.bars)
    assert output.values_for(BOLLINGER_MIDDLE_BAND_OUTPUT) == (
        None,
        Decimal(2),
        Decimal(4),
    )
    assert output.values_for(BOLLINGER_UPPER_BAND_OUTPUT) == (
        None,
        Decimal(4),
        Decimal(6),
    )
    assert output.values_for(BOLLINGER_LOWER_BAND_OUTPUT) == (
        None,
        Decimal(0),
        Decimal(2),
    )
    assert output.values_for(BOLLINGER_BANDWIDTH_OUTPUT) == (
        None,
        Decimal(2),
        Decimal(1),
    )


@pytest.mark.parametrize("period", [0, -1])
def test_rejects_nonpositive_period(period: int) -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="at least 1"):
        BollingerBandsParameters(period)


def test_rejects_noninteger_period_and_unsupported_source_field() -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="integer"):
        BollingerBandsParameters(cast(int, 2.5))
    with pytest.raises(InvalidIndicatorParametersError, match="source_field"):
        BollingerBandsParameters(2, source_field=cast(MarketField, "price"))


@pytest.mark.parametrize(
    "multiplier",
    [Decimal(0), Decimal(-1), Decimal("NaN"), Decimal("Infinity")],
)
def test_rejects_nonpositive_or_nonfinite_multiplier(multiplier: Decimal) -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="greater than zero"):
        BollingerBandsParameters(2, multiplier)


def test_rejects_non_decimal_multiplier() -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="must be a Decimal"):
        BollingerBandsParameters(2, cast(Decimal, 2))


def test_selected_source_field_controls_every_band() -> None:
    dataset = make_dataset(("100", "100"), opens=("1", "3"))
    output = BollingerBands(
        BollingerBandsParameters(2, Decimal(1), MarketField.OPEN)
    ).calculate(dataset)

    assert output.values_for(BOLLINGER_MIDDLE_BAND_OUTPUT) == (None, Decimal(2))
    assert output.values_for(BOLLINGER_UPPER_BAND_OUTPUT) == (None, Decimal(3))
    assert output.values_for(BOLLINGER_LOWER_BAND_OUTPUT) == (None, Decimal(1))
    assert output.values_for(BOLLINGER_BANDWIDTH_OUTPUT) == (None, Decimal(1))


def test_missing_source_invalidates_only_each_containing_full_window() -> None:
    output = BollingerBands(BollingerBandsParameters(3)).calculate(
        make_dataset(("1", "2", "3", "NaN", "4", "5", "6"))
    )

    for field_name in (
        BOLLINGER_MIDDLE_BAND_OUTPUT,
        BOLLINGER_UPPER_BAND_OUTPUT,
        BOLLINGER_LOWER_BAND_OUTPUT,
        BOLLINGER_BANDWIDTH_OUTPUT,
    ):
        values = output.values_for(field_name)
        assert values[:2] == (None, None)
        assert values[3:6] == (None, None, None)
        assert values[2] is not None
        assert values[6] is not None


def test_constant_price_has_zero_width_and_zero_bandwidth() -> None:
    output = BollingerBands(BollingerBandsParameters(3)).calculate(
        make_dataset(("5", "5", "5"))
    )

    assert output.values_for(BOLLINGER_MIDDLE_BAND_OUTPUT)[-1] == Decimal(5)
    assert output.values_for(BOLLINGER_UPPER_BAND_OUTPUT)[-1] == Decimal(5)
    assert output.values_for(BOLLINGER_LOWER_BAND_OUTPUT)[-1] == Decimal(5)
    assert output.values_for(BOLLINGER_BANDWIDTH_OUTPUT)[-1] == Decimal(0)


def test_zero_middle_with_nonzero_width_has_unavailable_bandwidth() -> None:
    output = BollingerBands(BollingerBandsParameters(2)).calculate(
        make_dataset(("-1", "1"))
    )

    assert output.values_for(BOLLINGER_MIDDLE_BAND_OUTPUT)[-1] == Decimal(0)
    assert output.values_for(BOLLINGER_UPPER_BAND_OUTPUT)[-1] == Decimal(2)
    assert output.values_for(BOLLINGER_LOWER_BAND_OUTPUT)[-1] == Decimal(-2)
    assert output.values_for(BOLLINGER_BANDWIDTH_OUTPUT)[-1] is None


def test_period_one_zero_price_is_deterministic() -> None:
    output = BollingerBands(BollingerBandsParameters(1)).calculate(make_dataset(("0",)))

    for field_name in (
        BOLLINGER_MIDDLE_BAND_OUTPUT,
        BOLLINGER_UPPER_BAND_OUTPUT,
        BOLLINGER_LOWER_BAND_OUTPUT,
        BOLLINGER_BANDWIDTH_OUTPUT,
    ):
        assert output.values_for(field_name) == (Decimal(0),)


def test_calculation_is_context_independent_and_does_not_mutate_input() -> None:
    dataset = make_dataset(("1", "2", "8", "4"))
    original = dataset
    indicator = BollingerBands(BollingerBandsParameters(3, Decimal("1.5")))

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


def test_configuration_serialization_and_identity_include_formula_parameters() -> None:
    first = BollingerBands(BollingerBandsParameters(3, Decimal("2.0")))
    same = BollingerBands(BollingerBandsParameters(3, Decimal(2)))
    other_period = BollingerBands(BollingerBandsParameters(4, Decimal(2)))
    other_multiplier = BollingerBands(BollingerBandsParameters(3, Decimal(3)))
    other_field = BollingerBands(
        BollingerBandsParameters(3, Decimal(2), MarketField.OPEN)
    )

    assert first.configuration_id == same.configuration_id
    assert (
        len(
            {
                first.configuration_id,
                other_period.configuration_id,
                other_multiplier.configuration_id,
                other_field.configuration_id,
            }
        )
        == 4
    )
    assert first.parameters.to_primitive() == {
        "period": 3,
        "source_field": "close",
        "standard_deviation_multiplier": "2",
    }
    formula = cast(dict[str, object], first.configuration()["formula"])
    assert formula["standard_deviation_degrees_of_freedom"] == 0


def test_output_rows_serialize_all_fields_stably() -> None:
    output = BollingerBands(BollingerBandsParameters(2)).calculate(
        make_dataset(("1", "3"))
    )

    assert output.to_rows() == [
        {
            "session_date": "2024-07-01",
            "bollinger_middle_band": None,
            "bollinger_upper_band": None,
            "bollinger_lower_band": None,
            "bollinger_bandwidth": None,
        },
        {
            "session_date": "2024-07-02",
            "bollinger_middle_band": "2",
            "bollinger_upper_band": "4",
            "bollinger_lower_band": "0",
            "bollinger_bandwidth": "2",
        },
    ]


def test_appended_future_bars_do_not_change_historical_values() -> None:
    indicator = BollingerBands(BollingerBandsParameters(3))
    cutoff = indicator.calculate(make_dataset(("3", "2", "1", "2", "3")))
    extended = indicator.calculate(
        make_dataset(("3", "2", "1", "2", "3", "1000", "1.5"))
    )

    assert extended.session_dates[:5] == cutoff.session_dates
    for field_name in indicator.output_fields:
        assert extended.values_for(field_name)[:5] == cutoff.values_for(field_name)
