import ast
from decimal import Decimal
from pathlib import Path

import pytest

import quantforge
from quantforge.configuration import PrimitiveMapping
from quantforge.indicators import (
    EXPONENTIAL_MOVING_AVERAGE_OUTPUT,
    NATIVE_INDICATOR_BACKEND,
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    TALIB_INDICATOR_BACKEND,
    WILDER_AVERAGE_TRUE_RANGE_OUTPUT,
    WILDER_RSI_OUTPUT,
    ExponentialMovingAverage,
    ExponentialMovingAverageParameters,
    IndicatorBackendVersionError,
    IndicatorComputationRequest,
    MarketField,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
    TalibIndicatorBackend,
    UnsupportedIndicatorBackendError,
    WilderAverageTrueRange,
    WilderAverageTrueRangeParameters,
    WilderRelativeStrengthIndex,
    WilderRelativeStrengthIndexParameters,
)

from ..helpers import make_dataset

type CoreIndicator = (
    SimpleMovingAverage
    | ExponentialMovingAverage
    | WilderRelativeStrengthIndex
    | WilderAverageTrueRange
)

LEGACY_SMA_PERIOD_3_ID = (
    "a278b8c1464be1c5d84d9be9ff54a0f3f8e148cc3bcc5682dcb24a37dddfcb46"
)
LEGACY_RSI_PERIOD_3_ID = (
    "db1c9b1ca01385b13ae988701882067c47bea6d1f457e755ca47f8531e115ec2"
)
LEGACY_ATR_PERIOD_3_ID = (
    "da67f055ef917b614d3054d48de5cf6f8844617644f6c4200b938a9e35ed3ede"
)

CLOSES = ("10", "11", "12", "11", "13", "13", "12", "14")
HIGHS = ("11", "12", "13", "12", "14", "14", "13", "15")
LOWS = ("9", "10", "11", "10", "12", "12", "11", "13")


def _dataset():
    return make_dataset(CLOSES, highs=HIGHS, lows=LOWS)


def _backend_pairs() -> tuple[tuple[CoreIndicator, CoreIndicator, str], ...]:
    return (
        (
            SimpleMovingAverage(
                SimpleMovingAverageParameters(3),
                backend_id=NATIVE_INDICATOR_BACKEND,
            ),
            SimpleMovingAverage(
                SimpleMovingAverageParameters(3),
                backend_id=TALIB_INDICATOR_BACKEND,
            ),
            SIMPLE_MOVING_AVERAGE_OUTPUT,
        ),
        (
            ExponentialMovingAverage(
                ExponentialMovingAverageParameters(3),
                backend_id=NATIVE_INDICATOR_BACKEND,
            ),
            ExponentialMovingAverage(
                ExponentialMovingAverageParameters(3),
                backend_id=TALIB_INDICATOR_BACKEND,
            ),
            EXPONENTIAL_MOVING_AVERAGE_OUTPUT,
        ),
        (
            WilderRelativeStrengthIndex(
                WilderRelativeStrengthIndexParameters(3),
                backend_id=NATIVE_INDICATOR_BACKEND,
            ),
            WilderRelativeStrengthIndex(
                WilderRelativeStrengthIndexParameters(3),
                backend_id=TALIB_INDICATOR_BACKEND,
            ),
            WILDER_RSI_OUTPUT,
        ),
        (
            WilderAverageTrueRange(
                WilderAverageTrueRangeParameters(3),
                backend_id=NATIVE_INDICATOR_BACKEND,
            ),
            WilderAverageTrueRange(
                WilderAverageTrueRangeParameters(3),
                backend_id=TALIB_INDICATOR_BACKEND,
            ),
            WILDER_AVERAGE_TRUE_RANGE_OUTPUT,
        ),
    )


def test_core_indicators_use_one_backend_neutral_definition_per_indicator() -> None:
    for native, talib_indicator, _ in _backend_pairs():
        assert type(native) is type(talib_indicator)
        assert native.standard_definition == talib_indicator.standard_definition
        assert native.backend_identity.backend_id == NATIVE_INDICATOR_BACKEND
        assert talib_indicator.backend_identity.backend_id == TALIB_INDICATOR_BACKEND


def test_talib_core_outputs_are_aligned_and_normalized_to_public_names() -> None:
    expected = {
        SIMPLE_MOVING_AVERAGE_OUTPUT: (
            None,
            None,
            Decimal("11.0"),
            Decimal("11.333333333333334"),
            Decimal("12.0"),
            Decimal("12.333333333333334"),
            Decimal("12.666666666666666"),
            Decimal("13.0"),
        ),
        WILDER_RSI_OUTPUT: (
            None,
            None,
            None,
            Decimal("66.66666666666666"),
            Decimal("83.33333333333334"),
            Decimal("83.33333333333333"),
            Decimal("53.333333333333336"),
            Decimal("77.56410256410257"),
        ),
        WILDER_AVERAGE_TRUE_RANGE_OUTPUT: (
            None,
            None,
            None,
            Decimal("2.0"),
            Decimal("2.3333333333333335"),
            Decimal("2.2222222222222223"),
            Decimal("2.1481481481481484"),
            Decimal("2.432098765432099"),
        ),
    }

    talib_indicators: tuple[tuple[CoreIndicator, str], ...] = (
        (_backend_pairs()[0][1], SIMPLE_MOVING_AVERAGE_OUTPUT),
        (_backend_pairs()[2][1], WILDER_RSI_OUTPUT),
        (_backend_pairs()[3][1], WILDER_AVERAGE_TRUE_RANGE_OUTPUT),
    )
    for indicator, output_name in talib_indicators:
        output = indicator.calculate(_dataset())
        assert output.session_dates == tuple(
            bar.session_date for bar in _dataset().bars
        )
        assert output.values_for(output_name) == expected[output_name]


def test_talib_sma_maps_the_configured_market_field() -> None:
    dataset = make_dataset(("100", "100", "100", "100"), opens=("10", "12", "14", "16"))
    indicator = SimpleMovingAverage(
        SimpleMovingAverageParameters(2, source_field=MarketField.OPEN),
        backend_id=TALIB_INDICATOR_BACKEND,
    )

    assert indicator.calculate(dataset).values_for(SIMPLE_MOVING_AVERAGE_OUTPUT) == (
        None,
        Decimal("11.0"),
        Decimal("13.0"),
        Decimal("15.0"),
    )


def test_native_and_talib_match_semantics_without_forcing_decimal_equality() -> None:
    dataset = _dataset()
    for native, talib_indicator, output_name in _backend_pairs():
        native_values = native.calculate(dataset).values_for(output_name)
        talib_values = talib_indicator.calculate(dataset).values_for(output_name)
        for native_value, talib_value in zip(native_values, talib_values, strict=True):
            if native_value is None or talib_value is None:
                assert native_value is talib_value
            else:
                assert float(talib_value) == pytest.approx(
                    float(native_value), rel=1e-14, abs=1e-14
                )


def test_talib_flat_rsi_difference_is_explicit() -> None:
    dataset = make_dataset(("5", "5", "5", "5", "5", "5"))
    native = WilderRelativeStrengthIndex(
        WilderRelativeStrengthIndexParameters(3),
        backend_id=NATIVE_INDICATOR_BACKEND,
    )
    talib_indicator = WilderRelativeStrengthIndex(
        WilderRelativeStrengthIndexParameters(3),
        backend_id=TALIB_INDICATOR_BACKEND,
    )

    assert native.calculate(dataset).values_for(WILDER_RSI_OUTPUT) == (
        None,
        None,
        None,
        Decimal(50),
        Decimal(50),
        Decimal(50),
    )
    assert talib_indicator.calculate(dataset).values_for(WILDER_RSI_OUTPUT) == (
        None,
        None,
        None,
        Decimal(0),
        Decimal(0),
        Decimal(0),
    )


def test_legacy_core_configurations_deserialize_to_stable_native_semantics() -> None:
    sma = SimpleMovingAverage(SimpleMovingAverageParameters(3))
    rsi = WilderRelativeStrengthIndex(WilderRelativeStrengthIndexParameters(3))
    atr = WilderAverageTrueRange(WilderAverageTrueRangeParameters(3))

    restored_sma = SimpleMovingAverage.from_configuration(sma.configuration())
    restored_rsi = WilderRelativeStrengthIndex.from_configuration(rsi.configuration())
    restored_atr = WilderAverageTrueRange.from_configuration(atr.configuration())

    assert (
        sma.configuration_id == restored_sma.configuration_id == LEGACY_SMA_PERIOD_3_ID
    )
    assert (
        rsi.configuration_id == restored_rsi.configuration_id == LEGACY_RSI_PERIOD_3_ID
    )
    assert (
        atr.configuration_id == restored_atr.configuration_id == LEGACY_ATR_PERIOD_3_ID
    )
    assert restored_sma.uses_legacy_native_configuration
    assert restored_rsi.uses_legacy_native_configuration
    assert restored_atr.uses_legacy_native_configuration
    assert restored_sma.backend_identity.backend_id == NATIVE_INDICATOR_BACKEND
    assert restored_rsi.backend_identity.backend_id == NATIVE_INDICATOR_BACKEND
    assert restored_atr.backend_identity.backend_id == NATIVE_INDICATOR_BACKEND


def test_explicit_backend_configs_have_distinct_ids_and_complete_metadata() -> None:
    for native, talib_indicator, _ in _backend_pairs():
        repeated_talib = type(talib_indicator).from_configuration(
            talib_indicator.configuration()
        )
        assert native.configuration_id != talib_indicator.configuration_id
        assert repeated_talib.configuration_id == talib_indicator.configuration_id
        configuration = talib_indicator.configuration()
        assert configuration["parameters"] == talib_indicator.parameters.to_primitive()
        assert configuration["backend"] == {
            "backend_id": "talib_v1",
            "contract_version": "1",
            "library_name": "TA-Lib",
            "library_version": "0.7.1",
            "function_name": talib_indicator.backend_identity.function_name,
            "runtime_library_name": "TA-Lib C",
            "runtime_library_version": "0.7.1",
        }
        metadata = (
            TalibIndicatorBackend()
            .compute(
                IndicatorComputationRequest(
                    talib_indicator.standard_definition, _dataset().bars
                )
            )
            .metadata()
        )
        assert metadata["normalized_parameters"] == (
            talib_indicator.parameters.to_primitive()
        )
        assert metadata["normalized_input_fields"] == [
            field.value for field in talib_indicator.standard_definition.input_fields
        ]
        assert metadata["normalized_output_fields"] == list(
            talib_indicator.output_fields
        )
        assert metadata["backend"] == talib_indicator.backend_identity.to_primitive()


def test_explicit_core_configurations_reject_installed_version_drift() -> None:
    for _, talib_indicator, _ in _backend_pairs():
        configuration = talib_indicator.configuration()
        changed_backend: PrimitiveMapping = {
            **talib_indicator.backend_identity.to_primitive(),
            "library_version": "0.7.2",
        }
        changed_configuration: PrimitiveMapping = {
            **configuration,
            "backend": changed_backend,
        }

        with pytest.raises(IndicatorBackendVersionError, match="installed backend"):
            type(talib_indicator).from_configuration(changed_configuration)


def test_talib_period_limits_fail_during_configuration() -> None:
    native_period_one_rsi = WilderRelativeStrengthIndex(
        WilderRelativeStrengthIndexParameters(1),
        backend_id=NATIVE_INDICATOR_BACKEND,
    )

    assert native_period_one_rsi.backend_identity.backend_id == NATIVE_INDICATOR_BACKEND
    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="SMA window must be from 1 through 100000",
    ):
        SimpleMovingAverage(
            SimpleMovingAverageParameters(100_001),
            backend_id=TALIB_INDICATOR_BACKEND,
        )
    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="RSI period must be from 2 through 100000",
    ):
        WilderRelativeStrengthIndex(
            WilderRelativeStrengthIndexParameters(1),
            backend_id=TALIB_INDICATOR_BACKEND,
        )
    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="ATR period must be from 1 through 100000",
    ):
        WilderAverageTrueRange(
            WilderAverageTrueRangeParameters(100_001),
            backend_id=TALIB_INDICATOR_BACKEND,
        )


def test_appending_future_bars_does_not_change_talib_history() -> None:
    cutoff = 6
    cutoff_dataset = make_dataset(
        CLOSES[:cutoff], highs=HIGHS[:cutoff], lows=LOWS[:cutoff]
    )
    extended_dataset = _dataset()

    for _, talib_indicator, output_name in _backend_pairs():
        historical = talib_indicator.calculate(cutoff_dataset).values_for(output_name)
        extended = talib_indicator.calculate(extended_dataset).values_for(output_name)
        assert extended[:cutoff] == historical


def test_talib_import_is_confined_to_the_backend_adapter() -> None:
    package_file = quantforge.__file__
    assert package_file is not None
    package_root = Path(package_file).parent
    importers: list[Path] = []

    for source_path in package_root.rglob("*.py"):
        syntax_tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imports_talib = any(
            (
                isinstance(node, ast.Import)
                and any(alias.name == "talib" for alias in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and (node.module == "talib" or node.module.startswith("talib."))
            )
            for node in ast.walk(syntax_tree)
        )
        if imports_talib:
            importers.append(source_path.relative_to(package_root))

    assert importers == [Path("indicators/backends/talib.py")]
