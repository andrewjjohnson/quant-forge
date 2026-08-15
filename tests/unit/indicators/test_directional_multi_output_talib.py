from collections.abc import Callable
from copy import deepcopy
from decimal import Decimal
from typing import cast

import pytest
import talib

from quantforge.configuration import PrimitiveMapping
from quantforge.indicators import (
    AVERAGE_DIRECTIONAL_INDEX_OUTPUT,
    BOLLINGER_BANDWIDTH_OUTPUT,
    BOLLINGER_LOWER_BAND_OUTPUT,
    BOLLINGER_MIDDLE_BAND_OUTPUT,
    BOLLINGER_UPPER_BAND_OUTPUT,
    NATIVE_INDICATOR_BACKEND,
    NEGATIVE_DIRECTIONAL_INDICATOR_OUTPUT,
    POSITIVE_DIRECTIONAL_INDICATOR_OUTPUT,
    TALIB_INDICATOR_BACKEND,
    BollingerBands,
    BollingerBandsParameters,
    IndicatorBackendVersionError,
    IndicatorComputationRequest,
    TalibIndicatorBackend,
    UnsupportedIndicatorBackendError,
    WilderDirectionalMovement,
    WilderDirectionalMovementParameters,
)
from quantforge.prediction.overnight_gap import (
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
)

from ..helpers import make_dataset

LEGACY_DIRECTIONAL_PERIOD_2_ID = (
    "f2a1a615a442f87232efd8992b519cd01575900d5b4a6c4db2e0cf30e3cf6ea6"
)
LEGACY_BOLLINGER_PERIOD_3_ID = (
    "da0a5aa3bb94b2815608afbb531f2c9e51af75ae6c31c20c220140dbe9504564"
)

CLOSES = ("10", "11", "12", "11", "13", "13", "12", "14")
HIGHS = ("11", "12", "13", "12", "14", "14", "13", "15")
LOWS = ("9", "10", "11", "10", "12", "12", "11", "13")


def _dataset():
    return make_dataset(CLOSES, highs=HIGHS, lows=LOWS)


def _assert_talib_values(
    actual: tuple[Decimal | None, ...],
    expected: tuple[Decimal | None, ...],
) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if expected_value is None:
            assert actual_value is None
        else:
            assert actual_value is not None
            assert float(actual_value) == pytest.approx(
                float(expected_value), rel=1e-14, abs=1e-14
            )


def test_directional_and_bollinger_use_backend_neutral_public_definitions() -> None:
    native_directional = WilderDirectionalMovement(
        WilderDirectionalMovementParameters(2),
        backend_id=NATIVE_INDICATOR_BACKEND,
    )
    talib_directional = WilderDirectionalMovement(
        WilderDirectionalMovementParameters(2),
        backend_id=TALIB_INDICATOR_BACKEND,
    )
    native_bollinger = BollingerBands(
        BollingerBandsParameters(3), backend_id=NATIVE_INDICATOR_BACKEND
    )
    talib_bollinger = BollingerBands(
        BollingerBandsParameters(3), backend_id=TALIB_INDICATOR_BACKEND
    )

    for native, talib_indicator in (
        (native_directional, talib_directional),
        (native_bollinger, talib_bollinger),
    ):
        assert type(native) is type(talib_indicator)
        assert native.standard_definition == talib_indicator.standard_definition
        assert native.backend_identity.backend_id == NATIVE_INDICATOR_BACKEND
        assert talib_indicator.backend_identity.backend_id == TALIB_INDICATOR_BACKEND


def test_talib_directional_outputs_are_named_aligned_and_initialized_by_talib() -> None:
    indicator = WilderDirectionalMovement(
        WilderDirectionalMovementParameters(2),
        backend_id=TALIB_INDICATOR_BACKEND,
    )

    output = indicator.calculate(_dataset())

    _assert_talib_values(
        output.values_for(POSITIVE_DIRECTIONAL_INDICATOR_OUTPUT),
        (
            None,
            None,
            Decimal("50.0"),
            Decimal("21.428571428571427"),
            Decimal("50.0"),
            Decimal("27.142857142857142"),
            Decimal("14.17910447761194"),
            Decimal("45.0920245398773"),
        ),
    )
    _assert_talib_values(
        output.values_for(NEGATIVE_DIRECTIONAL_INDICATOR_OUTPUT),
        (
            None,
            None,
            Decimal("0.0"),
            Decimal("28.57142857142857"),
            Decimal("10.526315789473683"),
            Decimal("5.714285714285714"),
            Decimal("26.865671641791046"),
            Decimal("11.042944785276074"),
        ),
    )
    _assert_talib_values(
        output.values_for(AVERAGE_DIRECTIONAL_INDEX_OUTPUT),
        (
            None,
            None,
            None,
            Decimal("57.142857142857146"),
            Decimal("61.18012422360249"),
            Decimal("63.19875776397516"),
            Decimal("47.053924336533036"),
            Decimal("53.85483102072553"),
        ),
    )
    assert indicator.backend_identity.function_name == "PLUS_DI+MINUS_DI+ADX"


def test_one_directional_request_calls_each_talib_function_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"PLUS_DI": 0, "MINUS_DI": 0, "ADX": 0}

    for function_name in calls:
        original = cast(Callable[..., object], getattr(talib, function_name))

        def counted(
            *args: object,
            _name: str = function_name,
            _original: Callable[..., object] = original,
            **kwargs: object,
        ) -> object:
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(talib, function_name, counted)

    indicator = WilderDirectionalMovement(
        WilderDirectionalMovementParameters(2),
        backend_id=TALIB_INDICATOR_BACKEND,
    )
    indicator.calculate(_dataset())

    assert calls == {"PLUS_DI": 1, "MINUS_DI": 1, "ADX": 1}


def test_talib_bollinger_tuple_is_normalized_and_bandwidth_is_derived_by_name() -> None:
    indicator = BollingerBands(
        BollingerBandsParameters(3, Decimal(2)),
        backend_id=TALIB_INDICATOR_BACKEND,
    )

    result = TalibIndicatorBackend().compute(
        IndicatorComputationRequest(indicator.standard_definition, _dataset().bars)
    )

    assert tuple(field.name for field in result.fields) == indicator.output_fields
    assert result.metadata()["normalized_output_fields"] == [
        BOLLINGER_MIDDLE_BAND_OUTPUT,
        BOLLINGER_UPPER_BAND_OUTPUT,
        BOLLINGER_LOWER_BAND_OUTPUT,
        BOLLINGER_BANDWIDTH_OUTPUT,
    ]
    middle = result.fields[0].values[-1]
    upper = result.fields[1].values[-1]
    lower = result.fields[2].values[-1]
    bandwidth = result.fields[3].values[-1]
    assert middle == Decimal("13.0")
    assert upper == Decimal("14.63299316185544")
    assert lower == Decimal("11.36700683814456")
    assert bandwidth == Decimal("0.25122971720852927")


def test_native_fixtures_and_legacy_configuration_ids_are_unchanged() -> None:
    directional = WilderDirectionalMovement(WilderDirectionalMovementParameters(2))
    explicit_directional = WilderDirectionalMovement(
        WilderDirectionalMovementParameters(2),
        backend_id=NATIVE_INDICATOR_BACKEND,
    )
    bollinger = BollingerBands(BollingerBandsParameters(3))
    explicit_bollinger = BollingerBands(
        BollingerBandsParameters(3), backend_id=NATIVE_INDICATOR_BACKEND
    )

    assert directional.configuration_id == LEGACY_DIRECTIONAL_PERIOD_2_ID
    assert bollinger.configuration_id == LEGACY_BOLLINGER_PERIOD_3_ID
    assert (
        explicit_directional.calculate(_dataset()).fields
        == directional.calculate(_dataset()).fields
    )
    assert (
        explicit_bollinger.calculate(_dataset()).fields
        == bollinger.calculate(_dataset()).fields
    )
    assert "backend" not in directional.configuration()
    assert "backend" not in bollinger.configuration()


def test_overnight_gap_keeps_implicit_native_directional_semantics() -> None:
    strategy = OvernightGapPredictionStrategy(OvernightGapPredictionParameters())
    directional = next(
        indicator
        for indicator in strategy.required_indicators
        if isinstance(indicator, WilderDirectionalMovement)
    )

    assert directional.uses_legacy_native_configuration
    assert directional.backend_identity.backend_id == NATIVE_INDICATOR_BACKEND
    assert "backend" not in directional.configuration()


def test_explicit_backend_configurations_round_trip_and_have_distinct_ids() -> None:
    talib_directional = WilderDirectionalMovement(
        WilderDirectionalMovementParameters(2),
        backend_id=TALIB_INDICATOR_BACKEND,
    )
    talib_bollinger = BollingerBands(
        BollingerBandsParameters(3),
        backend_id=TALIB_INDICATOR_BACKEND,
    )
    pairs = (
        (
            talib_directional,
            WilderDirectionalMovement.from_configuration(
                talib_directional.configuration()
            ),
            WilderDirectionalMovement(
                WilderDirectionalMovementParameters(2),
                backend_id=NATIVE_INDICATOR_BACKEND,
            ),
        ),
        (
            talib_bollinger,
            BollingerBands.from_configuration(talib_bollinger.configuration()),
            BollingerBands(
                BollingerBandsParameters(3),
                backend_id=NATIVE_INDICATOR_BACKEND,
            ),
        ),
    )

    for indicator, restored, explicit_native in pairs:
        assert restored.configuration_id == indicator.configuration_id
        assert explicit_native.configuration_id != indicator.configuration_id
        assert indicator.configuration()["backend"] == (
            indicator.backend_identity.to_primitive()
        )


def test_explicit_configurations_reject_backend_version_drift() -> None:
    indicator = BollingerBands(
        BollingerBandsParameters(3), backend_id=TALIB_INDICATOR_BACKEND
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
        BollingerBands.from_configuration(changed_configuration)


def test_backend_specific_initialization_and_period_limits_are_explicit() -> None:
    native = WilderDirectionalMovement(
        WilderDirectionalMovementParameters(2),
        backend_id=NATIVE_INDICATOR_BACKEND,
    ).calculate(_dataset())
    talib_output = WilderDirectionalMovement(
        WilderDirectionalMovementParameters(2),
        backend_id=TALIB_INDICATOR_BACKEND,
    ).calculate(_dataset())

    assert native.values_for(AVERAGE_DIRECTIONAL_INDEX_OUTPUT)[:3] == (None,) * 3
    assert talib_output.values_for(AVERAGE_DIRECTIONAL_INDEX_OUTPUT)[:3] == (None,) * 3
    assert native.values_for(AVERAGE_DIRECTIONAL_INDEX_OUTPUT)[3] == Decimal(50)
    first_talib_adx = talib_output.values_for(AVERAGE_DIRECTIONAL_INDEX_OUTPUT)[3]
    assert first_talib_adx is not None
    assert float(first_talib_adx) == pytest.approx(
        57.142857142857146,
        rel=1e-14,
        abs=1e-14,
    )
    assert BollingerBands(
        BollingerBandsParameters(1), backend_id=NATIVE_INDICATOR_BACKEND
    )
    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="Bollinger Bands period must be from 2 through 100000",
    ):
        BollingerBands(BollingerBandsParameters(1), backend_id=TALIB_INDICATOR_BACKEND)


def test_talib_multi_output_computation_does_not_mutate_canonical_bars() -> None:
    dataset = _dataset()
    original = deepcopy(dataset)

    WilderDirectionalMovement(
        WilderDirectionalMovementParameters(2),
        backend_id=TALIB_INDICATOR_BACKEND,
    ).calculate(dataset)
    BollingerBands(
        BollingerBandsParameters(3),
        backend_id=TALIB_INDICATOR_BACKEND,
    ).calculate(dataset)

    assert dataset == original


def test_appending_future_bars_does_not_change_talib_multi_output_history() -> None:
    cutoff = 6
    cutoff_dataset = make_dataset(
        CLOSES[:cutoff], highs=HIGHS[:cutoff], lows=LOWS[:cutoff]
    )
    indicators = (
        WilderDirectionalMovement(
            WilderDirectionalMovementParameters(2),
            backend_id=TALIB_INDICATOR_BACKEND,
        ),
        BollingerBands(
            BollingerBandsParameters(3),
            backend_id=TALIB_INDICATOR_BACKEND,
        ),
    )

    for indicator in indicators:
        historical = indicator.calculate(cutoff_dataset)
        extended = indicator.calculate(_dataset())
        for field_name in indicator.output_fields:
            assert extended.values_for(field_name)[:cutoff] == historical.values_for(
                field_name
            )
