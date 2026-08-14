from decimal import ROUND_DOWN, ROUND_UP, Decimal, localcontext
from fractions import Fraction
from typing import cast

import pytest

import quantforge.indicators.backends.native as native_backend_module
from quantforge.indicators import (
    BOLLINGER_BANDWIDTH_OUTPUT,
    BOLLINGER_LOWER_BAND_OUTPUT,
    BOLLINGER_MIDDLE_BAND_OUTPUT,
    BOLLINGER_UPPER_BAND_OUTPUT,
    BollingerBands,
    BollingerBandsParameters,
    IndicatorCalculationError,
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


@pytest.mark.parametrize(
    "multiplier",
    [
        Decimal("1E+1000000000"),
        Decimal("1E-1000000000"),
        Decimal("1" * 69),
    ],
)
def test_rejects_resource_unbounded_multiplier_before_serialization(
    multiplier: Decimal,
) -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="resource bounds"):
        BollingerBandsParameters(2, multiplier)


def test_accepts_multiplier_at_fixed_point_serialization_boundary() -> None:
    parameters = BollingerBandsParameters(2, Decimal("1E+255"))
    serialized_multiplier = cast(
        str, parameters.to_primitive()["standard_deviation_multiplier"]
    )

    assert len(serialized_multiplier) == 256


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


def test_mature_windows_update_statistics_without_rescanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rebuild = (
        native_backend_module._rebuild_window_moments  # pyright: ignore[reportPrivateUsage]
    )
    rebuild_count = 0

    def counting_rebuild(
        observations: tuple[Decimal, ...],
    ) -> tuple[Fraction, Fraction]:
        nonlocal rebuild_count
        rebuild_count += 1
        return rebuild(observations)

    monkeypatch.setattr(
        native_backend_module,
        "_rebuild_window_moments",
        counting_rebuild,
    )

    BollingerBands(BollingerBandsParameters(7)).calculate(
        make_dataset(tuple(str(value) for value in range(1, 16)))
    )

    assert rebuild_count == 1


def test_constant_window_rebases_positive_residue_after_scale_change() -> None:
    output = BollingerBands(BollingerBandsParameters(2)).calculate(
        make_dataset(("0", "1", "1E+34", "1E+34"))
    )

    assert output.values_for(BOLLINGER_MIDDLE_BAND_OUTPUT)[-1] == Decimal("1E+34")
    assert output.values_for(BOLLINGER_UPPER_BAND_OUTPUT)[-1] == Decimal("1E+34")
    assert output.values_for(BOLLINGER_LOWER_BAND_OUTPUT)[-1] == Decimal("1E+34")
    assert output.values_for(BOLLINGER_BANDWIDTH_OUTPUT)[-1] == Decimal(0)


def test_near_constant_window_rebuilds_impossible_residue_after_scale_change() -> None:
    output = BollingerBands(BollingerBandsParameters(2)).calculate(
        make_dataset(("0", "1", "1E+34", "10000000000000000000000000000000001"))
    )

    assert output.values_for(BOLLINGER_MIDDLE_BAND_OUTPUT)[-1] == Decimal("1E+34")
    assert output.values_for(BOLLINGER_UPPER_BAND_OUTPUT)[-1] == Decimal("1E+34")
    assert output.values_for(BOLLINGER_LOWER_BAND_OUTPUT)[-1] == Decimal(
        "9999999999999999999999999999999999"
    )
    assert output.values_for(BOLLINGER_BANDWIDTH_OUTPUT)[-1] == Decimal("1E-34")


@pytest.mark.parametrize(
    "closes",
    [
        ("0", "1E+34", "9999999999999999999999999999999999"),
        ("0", "1", "1E+34", "10000000000000000000000000000000001"),
        ("0", "1", "1E+34", "10000000000000000100000000000000000"),
        ("0", "-1E+34", "-9999999999999999999999999999999999"),
        ("0", "1E+20", "99999999999999999999"),
    ],
)
def test_scale_transition_result_depends_only_on_active_window(
    closes: tuple[str, ...],
) -> None:
    indicator = BollingerBands(BollingerBandsParameters(2))
    transitioned = indicator.calculate(make_dataset(closes))
    direct = indicator.calculate(make_dataset(closes[-2:]))

    for field_name in indicator.output_fields:
        assert (
            transitioned.values_for(field_name)[-1] == direct.values_for(field_name)[-1]
        )


@pytest.mark.parametrize(
    "source_value",
    [
        Decimal("1E-1000000000"),
        Decimal("1E+1000000000"),
        Decimal("1E-999999"),
        Decimal("1" * 69),
    ],
)
def test_rejects_resource_unbounded_source_before_exact_moment_conversion(
    source_value: Decimal,
) -> None:
    indicator = BollingerBands(BollingerBandsParameters(1))
    source_text = str(source_value)

    with pytest.raises(IndicatorCalculationError, match="resource bounds"):
        indicator.calculate(
            make_dataset(
                (source_text,),
                opens=(source_text,),
                highs=(source_text,),
                lows=(source_text,),
            )
        )


def test_accepts_source_at_practical_exact_moment_boundary() -> None:
    source_text = "1E-2047"
    output = BollingerBands(BollingerBandsParameters(1)).calculate(
        make_dataset(
            (source_text,),
            opens=(source_text,),
            highs=(source_text,),
            lows=(source_text,),
        )
    )

    assert output.values_for(BOLLINGER_MIDDLE_BAND_OUTPUT) == (Decimal(source_text),)


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
    assert first.configuration()["parameter_bounds"] == {
        "standard_deviation_multiplier": {
            "maximum_coefficient_digits": 68,
            "maximum_fixed_point_characters": 256,
        },
    }
    formula = cast(dict[str, object], first.configuration()["formula"])
    assert formula["standard_deviation_degrees_of_freedom"] == 0
    assert formula["window_update"] == ("exact_rational_rolling_sum_and_sum_of_squares")
    arithmetic = cast(dict[str, object], first.configuration()["arithmetic"])
    assert arithmetic["rolling_moment_accumulation"] == "exact_rational"
    assert arithmetic["exact_moment_input_bounds"] == {
        "maximum_coefficient_digits": 68,
        "maximum_source_integer_digits": 2048,
        "maximum_squared_integer_digits": 4096,
        "minimum_stored_exponent": -999999,
        "maximum_stored_exponent": 999999,
        "minimum_adjusted_exponent": -999999,
        "maximum_adjusted_exponent": 999999,
    }


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
