from copy import deepcopy
from decimal import Decimal
from typing import cast

import pytest
import talib

from quantforge.configuration import PrimitiveMapping
from quantforge.indicators import (
    NATIVE_INDICATOR_BACKEND,
    STOCHASTIC_D_OUTPUT,
    STOCHASTIC_K_OUTPUT,
    STOCHASTIC_SMOOTHING_METHOD,
    TALIB_INDICATOR_BACKEND,
    IndicatorBackendVersionError,
    IndicatorComputationRequest,
    InvalidIndicatorBackendError,
    InvalidIndicatorParametersError,
    StochasticOscillator,
    StochasticOscillatorParameters,
    TalibIndicatorBackend,
    UnsupportedIndicatorBackendError,
)

from ..helpers import make_dataset

CLOSES = (
    "10",
    "11",
    "12",
    "11",
    "13",
    "13",
    "12",
    "14",
    "15",
    "14",
)


@pytest.mark.parametrize(
    ("k_period", "k_smoothing_period", "d_period"),
    [
        (0, 3, 3),
        (-1, 3, 3),
        (5, 0, 3),
        (5, 3, 0),
    ],
)
def test_rejects_nonpositive_periods(
    k_period: int, k_smoothing_period: int, d_period: int
) -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="greater than zero"):
        StochasticOscillatorParameters(k_period, k_smoothing_period, d_period)


@pytest.mark.parametrize("parameter_index", [0, 1, 2])
def test_rejects_noninteger_periods(parameter_index: int) -> None:
    periods = [5, 3, 3]
    periods[parameter_index] = cast(int, 2.5)

    with pytest.raises(InvalidIndicatorParametersError, match="integer"):
        StochasticOscillatorParameters(*periods)


def test_default_definition_uses_talib_without_exposing_backend_names() -> None:
    indicator = StochasticOscillator(StochasticOscillatorParameters(3, 2, 2))

    assert indicator.backend_identity.backend_id == TALIB_INDICATOR_BACKEND
    assert indicator.backend_identity.function_name == "STOCH"
    assert indicator.standard_definition.to_primitive() == {
        "name": "stochastic_oscillator",
        "parameters": {
            "d_period": 2,
            "k_period": 3,
            "k_smoothing_period": 2,
            "smoothing_method": "simple_moving_average",
        },
        "input_fields": ["high", "low", "close"],
        "output_fields": ["k", "d"],
    }
    serialized_definition = str(indicator.standard_definition.to_primitive())
    assert all(
        backend_name not in serialized_definition
        for backend_name in (
            "fastk_period",
            "slowk_period",
            "slowk_matype",
            "slowd_period",
            "slowd_matype",
            "slowk",
            "slowd",
        )
    )


def test_talib_tuple_is_normalized_to_aligned_named_outputs() -> None:
    dataset = make_dataset(CLOSES)
    indicator = StochasticOscillator(StochasticOscillatorParameters(3, 2, 3))

    result = TalibIndicatorBackend().compute(
        IndicatorComputationRequest(indicator.standard_definition, dataset.bars)
    )

    assert tuple(field.name for field in result.fields) == (
        STOCHASTIC_K_OUTPUT,
        STOCHASTIC_D_OUTPUT,
    )
    assert result.metadata() == {
        "definition_name": "stochastic_oscillator",
        "backend": indicator.backend_identity.to_primitive(),
        "normalized_parameters": {
            "d_period": 3,
            "k_period": 3,
            "k_smoothing_period": 2,
            "smoothing_method": "simple_moving_average",
        },
        "normalized_input_fields": ["high", "low", "close"],
        "normalized_output_fields": ["k", "d"],
        "observation_count": len(CLOSES),
    }
    assert result.fields[0].values == (
        None,
        None,
        None,
        None,
        None,
        Decimal("66.66666666666667"),
        Decimal("33.333333333333336"),
        Decimal("33.333333333333336"),
        Decimal("70.83333333333334"),
        Decimal("37.50000000000001"),
    )
    assert result.fields[1].values == (
        None,
        None,
        None,
        None,
        None,
        Decimal("44.44444444444445"),
        Decimal("44.44444444444445"),
        Decimal("44.44444444444445"),
        Decimal("45.833333333333336"),
        Decimal("47.22222222222222"),
    )


def test_warm_up_is_explicit_aligned_and_serialized() -> None:
    indicator = StochasticOscillator(StochasticOscillatorParameters(3, 2, 2))
    output = indicator.calculate(make_dataset(CLOSES))

    assert indicator.warm_up_observations == 5
    assert indicator.configuration()["warm_up_observations"] == 5
    for field_name in indicator.output_fields:
        values = output.values_for(field_name)
        assert values[:4] == (None,) * 4
        assert values[4] is not None


def test_zero_high_low_range_follows_talib_without_formula_substitution() -> None:
    prices = ("10",) * len(CLOSES)
    output = StochasticOscillator(StochasticOscillatorParameters(3, 2, 2)).calculate(
        make_dataset(prices, highs=prices, lows=prices)
    )

    assert output.values_for(STOCHASTIC_K_OUTPUT) == (None,) * 4 + (Decimal(0),) * 6
    assert output.values_for(STOCHASTIC_D_OUTPUT) == (None,) * 4 + (Decimal(0),) * 6


def test_missing_source_values_remain_explicitly_unavailable_without_fill() -> None:
    output = StochasticOscillator(StochasticOscillatorParameters(3, 2, 2)).calculate(
        make_dataset(("10", "11", "12", "11", "13", "NaN", "12", "14", "15", "14"))
    )

    for field_name in (STOCHASTIC_K_OUTPUT, STOCHASTIC_D_OUTPUT):
        values = output.values_for(field_name)
        assert values[:4] == (None,) * 4
        assert values[4] is not None
        assert values[5:] == (None,) * 5


def test_talib_period_limits_fail_through_backend_domain_error() -> None:
    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="stochastic k_smoothing_period must be from 1 through 100000",
    ):
        StochasticOscillator(StochasticOscillatorParameters(5, 100_001, 3))


def test_native_backend_combination_is_explicitly_unsupported() -> None:
    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="native_v1 does not support indicator",
    ):
        StochasticOscillator(
            StochasticOscillatorParameters(3, 2, 2),
            backend_id=NATIVE_INDICATOR_BACKEND,
        )


def test_configuration_round_trip_binds_periods_smoothing_and_backend_version() -> None:
    indicator = StochasticOscillator(StochasticOscillatorParameters(3, 2, 2))
    configuration = indicator.configuration()
    restored = StochasticOscillator.from_configuration(configuration)

    assert configuration["contract_version"] == "2"
    assert configuration["parameters"] == {
        "d_period": 2,
        "k_period": 3,
        "k_smoothing_period": 2,
        "smoothing_method": STOCHASTIC_SMOOTHING_METHOD,
    }
    assert restored.configuration() == configuration
    assert restored.configuration_id == indicator.configuration_id
    assert configuration["backend"] == indicator.backend_identity.to_primitive()
    assert indicator.backend_identity.library_version == talib.__version__
    assert indicator.backend_identity.runtime_library_version == "0.7.1"


def test_configuration_restore_rejects_backend_version_and_smoothing_drift() -> None:
    indicator = StochasticOscillator(StochasticOscillatorParameters(3, 2, 2))
    configuration = indicator.configuration()
    changed_backend: PrimitiveMapping = {
        **indicator.backend_identity.to_primitive(),
        "library_version": "0.7.2",
    }
    changed_backend_configuration: PrimitiveMapping = {
        **configuration,
        "backend": changed_backend,
    }
    changed_parameters: PrimitiveMapping = {
        "d_period": 2,
        "k_period": 3,
        "k_smoothing_period": 2,
        "smoothing_method": "exponential_moving_average",
    }
    changed_smoothing_configuration: PrimitiveMapping = {
        **configuration,
        "parameters": changed_parameters,
    }

    with pytest.raises(IndicatorBackendVersionError, match="installed backend"):
        StochasticOscillator.from_configuration(changed_backend_configuration)
    with pytest.raises(InvalidIndicatorBackendError, match="smoothing method"):
        StochasticOscillator.from_configuration(changed_smoothing_configuration)


def test_configuration_identity_changes_with_each_normalized_period() -> None:
    indicators = (
        StochasticOscillator(StochasticOscillatorParameters(2, 2, 2)),
        StochasticOscillator(StochasticOscillatorParameters(3, 2, 2)),
        StochasticOscillator(StochasticOscillatorParameters(2, 3, 2)),
        StochasticOscillator(StochasticOscillatorParameters(2, 2, 3)),
    )

    assert len({indicator.configuration_id for indicator in indicators}) == len(
        indicators
    )


def test_calculation_does_not_mutate_canonical_input() -> None:
    dataset = make_dataset(CLOSES)
    original = deepcopy(dataset)

    StochasticOscillator(StochasticOscillatorParameters(3, 2, 2)).calculate(dataset)

    assert dataset == original


def test_appending_future_bars_does_not_change_historical_values() -> None:
    indicator = StochasticOscillator(StochasticOscillatorParameters(3, 2, 2))
    cutoff = indicator.calculate(make_dataset(CLOSES[:8]))
    extended = indicator.calculate(make_dataset((*CLOSES[:8], "1000", "-1000")))

    assert extended.session_dates[:8] == cutoff.session_dates
    for field_name in indicator.output_fields:
        assert extended.values_for(field_name)[:8] == cutoff.values_for(field_name)
