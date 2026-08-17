from collections.abc import Callable
from copy import deepcopy
from decimal import ROUND_DOWN, Decimal, Overflow, localcontext
from typing import cast

import pytest
import talib

from quantforge.configuration import PrimitiveMapping
from quantforge.indicators import (
    MACD_HISTOGRAM_OUTPUT,
    MACD_OUTPUT,
    MACD_SIGNAL_OUTPUT,
    NATIVE_INDICATOR_BACKEND,
    TALIB_INDICATOR_BACKEND,
    IndicatorBackendVersionError,
    IndicatorComputationRequest,
    InvalidIndicatorBackendError,
    InvalidIndicatorParametersError,
    MarketField,
    MovingAverageConvergenceDivergence,
    MovingAverageConvergenceDivergenceParameters,
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
    ("fast_period", "slow_period", "signal_period"),
    [
        (0, 26, 9),
        (-1, 26, 9),
        (12, 0, 9),
        (12, 26, 0),
    ],
)
def test_rejects_nonpositive_periods(
    fast_period: int, slow_period: int, signal_period: int
) -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="greater than zero"):
        MovingAverageConvergenceDivergenceParameters(
            fast_period,
            slow_period,
            signal_period,
        )


@pytest.mark.parametrize("fast_period", [3, 4])
def test_fast_period_must_be_less_than_slow_period(fast_period: int) -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="less than"):
        MovingAverageConvergenceDivergenceParameters(fast_period, 3, 2)


def test_rejects_noninteger_period_and_invalid_source_field() -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="integer"):
        MovingAverageConvergenceDivergenceParameters(cast(int, 2.5), 5, 2)
    with pytest.raises(InvalidIndicatorParametersError, match="source_field"):
        MovingAverageConvergenceDivergenceParameters(
            2,
            5,
            2,
            cast(MarketField, "price"),
        )


def test_default_definition_uses_talib_without_exposing_backend_names() -> None:
    indicator = MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(3, 5, 2)
    )

    assert indicator.backend_identity.backend_id == TALIB_INDICATOR_BACKEND
    assert indicator.backend_identity.function_name == "MACD"
    assert indicator.standard_definition.to_primitive() == {
        "name": "moving_average_convergence_divergence",
        "parameters": {
            "fast_period": 3,
            "signal_period": 2,
            "slow_period": 5,
            "source_field": "close",
        },
        "input_fields": ["close"],
        "output_fields": ["macd", "signal", "histogram"],
    }
    serialized_definition = str(indicator.standard_definition.to_primitive())
    assert all(
        backend_name not in serialized_definition
        for backend_name in (
            "fastperiod",
            "slowperiod",
            "signalperiod",
            "macdsignal",
            "macdhist",
        )
    )


def test_talib_tuple_is_normalized_to_aligned_named_outputs() -> None:
    dataset = make_dataset(CLOSES)
    indicator = MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(3, 5, 2)
    )

    result = TalibIndicatorBackend().compute(
        IndicatorComputationRequest(indicator.standard_definition, dataset.bars)
    )

    assert tuple(field.name for field in result.fields) == (
        MACD_OUTPUT,
        MACD_SIGNAL_OUTPUT,
        MACD_HISTOGRAM_OUTPUT,
    )
    assert result.metadata() == {
        "definition_name": "moving_average_convergence_divergence",
        "backend": indicator.backend_identity.to_primitive(),
        "normalized_parameters": {
            "fast_period": 3,
            "signal_period": 2,
            "slow_period": 5,
            "source_field": "close",
        },
        "normalized_input_fields": ["close"],
        "normalized_output_fields": ["macd", "signal", "histogram"],
        "observation_count": len(CLOSES),
    }
    assert result.fields[0].values == (
        None,
        None,
        None,
        None,
        None,
        Decimal("0.5666666666666664"),
        Decimal("0.2944444444444443"),
        Decimal("0.48796296296296227"),
        Decimal("0.6378086419753082"),
        Decimal("0.41478909465020486"),
    )


def test_histogram_is_exact_normalized_macd_minus_signal() -> None:
    output = MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(3, 5, 2)
    ).calculate(make_dataset(CLOSES))

    macd = output.values_for(MACD_OUTPUT)
    signal = output.values_for(MACD_SIGNAL_OUTPUT)
    histogram = output.values_for(MACD_HISTOGRAM_OUTPUT)
    for macd_value, signal_value, histogram_value in zip(
        macd, signal, histogram, strict=True
    ):
        if macd_value is None or signal_value is None:
            assert histogram_value is None
        else:
            assert histogram_value == macd_value - signal_value


def test_normalized_histogram_uses_a_fully_specified_decimal_context() -> None:
    indicator = MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(3, 5, 2)
    )
    scaled_closes = tuple(str(int(close) * 100_000_000) for close in CLOSES)
    dataset = make_dataset(scaled_closes)
    expected = indicator.calculate(dataset)

    with localcontext() as ambient:
        ambient.prec = 6
        ambient.rounding = ROUND_DOWN
        ambient.Emin = -5
        ambient.Emax = 5
        ambient.capitals = 0
        ambient.clamp = 1
        for signal in ambient.traps:
            ambient.traps[signal] = False
        ambient.traps[Overflow] = True
        constrained = indicator.calculate(dataset)

    assert constrained == expected


def test_warm_up_is_explicit_aligned_and_serialized() -> None:
    indicator = MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(3, 5, 2)
    )
    output = indicator.calculate(make_dataset(CLOSES))

    assert indicator.warm_up_observations == 6
    assert indicator.configuration()["warm_up_observations"] == 6
    for field_name in indicator.output_fields:
        values = output.values_for(field_name)
        assert values[:5] == (None,) * 5
        assert values[5] is not None


def test_missing_source_values_remain_explicitly_unavailable_without_fill() -> None:
    indicator = MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(2, 3, 2)
    )
    output = indicator.calculate(
        make_dataset(("1", "2", "3", "4", "5", "NaN", "7", "8", "9", "10"))
    )

    for field_name in indicator.output_fields:
        values = output.values_for(field_name)
        assert values[:3] == (None,) * 3
        assert values[3:5] != (None, None)
        assert values[5:] == (None,) * 5


def test_selected_source_field_controls_macd_input() -> None:
    dataset = make_dataset(
        ("100",) * len(CLOSES),
        opens=CLOSES,
        highs=tuple(str(101 + index) for index in range(len(CLOSES))),
        lows=tuple(str(99 - index) for index in range(len(CLOSES))),
    )
    close_output = MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(3, 5, 2)
    ).calculate(dataset)
    open_output = MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(
            3,
            5,
            2,
            MarketField.OPEN,
        )
    ).calculate(dataset)

    assert close_output.values_for(MACD_OUTPUT)[-1] == Decimal(0)
    assert open_output.values_for(MACD_OUTPUT)[-1] == Decimal("0.41478909465020486")


def test_talib_period_limits_fail_through_backend_domain_error() -> None:
    parameters = MovingAverageConvergenceDivergenceParameters(1, 3, 1)

    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="MACD fast_period must be from 2 through 100000",
    ):
        MovingAverageConvergenceDivergence(parameters)


def test_native_backend_combination_is_explicitly_unsupported() -> None:
    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="native_v1 does not support indicator",
    ):
        MovingAverageConvergenceDivergence(
            MovingAverageConvergenceDivergenceParameters(3, 5, 2),
            backend_id=NATIVE_INDICATOR_BACKEND,
        )


def test_configuration_round_trip_binds_parameters_and_exact_backend_version() -> None:
    indicator = MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(
            3,
            5,
            2,
            MarketField.OPEN,
        )
    )
    restored = MovingAverageConvergenceDivergence.from_configuration(
        indicator.configuration()
    )

    assert restored.configuration() == indicator.configuration()
    assert restored.configuration_id == indicator.configuration_id
    assert indicator.configuration()["backend"] == (
        indicator.backend_identity.to_primitive()
    )
    assert indicator.backend_identity.library_version == talib.__version__
    assert indicator.backend_identity.runtime_library_version == "0.7.1"


def test_configuration_restore_rejects_backend_version_drift() -> None:
    indicator = MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(3, 5, 2)
    )
    configuration = indicator.configuration()
    changed_backend: PrimitiveMapping = {
        **indicator.backend_identity.to_primitive(),
        "library_version": "0.7.2",
    }
    changed_configuration: PrimitiveMapping = {
        **configuration,
        "backend": changed_backend,
    }

    with pytest.raises(IndicatorBackendVersionError, match="installed backend"):
        MovingAverageConvergenceDivergence.from_configuration(changed_configuration)


def test_configuration_identity_changes_with_each_normalized_parameter() -> None:
    indicators = (
        MovingAverageConvergenceDivergence(
            MovingAverageConvergenceDivergenceParameters(2, 5, 2)
        ),
        MovingAverageConvergenceDivergence(
            MovingAverageConvergenceDivergenceParameters(3, 5, 2)
        ),
        MovingAverageConvergenceDivergence(
            MovingAverageConvergenceDivergenceParameters(3, 6, 2)
        ),
        MovingAverageConvergenceDivergence(
            MovingAverageConvergenceDivergenceParameters(3, 5, 3)
        ),
        MovingAverageConvergenceDivergence(
            MovingAverageConvergenceDivergenceParameters(
                3,
                5,
                2,
                MarketField.OPEN,
            )
        ),
    )

    assert len({indicator.configuration_id for indicator in indicators}) == len(
        indicators
    )


def test_calculation_does_not_mutate_canonical_input() -> None:
    dataset = make_dataset(CLOSES)
    original = deepcopy(dataset)

    MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(3, 5, 2)
    ).calculate(dataset)

    assert dataset == original


def test_appending_future_bars_does_not_change_historical_values() -> None:
    indicator = MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(3, 5, 2)
    )
    cutoff = indicator.calculate(make_dataset(CLOSES[:8]))
    extended = indicator.calculate(make_dataset((*CLOSES[:8], "1000", "-1000")))

    assert extended.session_dates[:8] == cutoff.session_dates
    for field_name in indicator.output_fields:
        assert extended.values_for(field_name)[:8] == cutoff.values_for(field_name)


def test_macd_fails_closed_when_talib_ema_global_state_has_drifted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indicator = MovingAverageConvergenceDivergence(
        MovingAverageConvergenceDivergenceParameters(3, 5, 2)
    )
    get_unstable_period = cast(
        Callable[[str], int],
        talib.get_unstable_period,  # pyright: ignore[reportUnknownMemberType]
    )

    def drifted_unstable_period(function_name: str) -> int:
        return 1 if function_name == "EMA" else get_unstable_period(function_name)

    monkeypatch.setattr(
        talib,
        "get_unstable_period",
        drifted_unstable_period,
    )

    with pytest.raises(
        InvalidIndicatorBackendError,
        match="default compatibility and zero unstable period",
    ):
        indicator.calculate(make_dataset(CLOSES))
