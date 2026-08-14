from decimal import Decimal

import pytest
import talib

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.indicators import (
    EXPONENTIAL_MOVING_AVERAGE_OUTPUT,
    NATIVE_INDICATOR_BACKEND,
    TALIB_INDICATOR_BACKEND,
    ExponentialMovingAverage,
    ExponentialMovingAverageParameters,
    IndicatorBackendIdentity,
    IndicatorBackendRegistry,
    IndicatorBackendVersionError,
    IndicatorComputationRequest,
    IndicatorComputationResult,
    IndicatorFieldOutput,
    MarketField,
    StandardIndicatorDefinition,
    TalibIndicatorBackend,
    UnsupportedIndicatorBackendError,
    default_indicator_backend_registry,
)

from ..helpers import make_dataset

LEGACY_EMA_PERIOD_3_ID = (
    "00590cc239ae24cf8878e2d0af5095900a04963fc2cc4cd49c3d48f9a7d05bce"
)


class FutureIndicatorBackend:
    """Small test adapter proving a new backend needs no new indicator class."""

    backend_id = "future_v1"

    def __init__(self, library_version: str) -> None:
        self._library_version = library_version

    def identity_for(
        self, definition: StandardIndicatorDefinition
    ) -> IndicatorBackendIdentity:
        return IndicatorBackendIdentity(
            backend_id=self.backend_id,
            library_name="future-library",
            library_version=self._library_version,
            function_name=f"mapped_{definition.name}",
        )

    def compute(
        self, request: IndicatorComputationRequest
    ) -> IndicatorComputationResult:
        fields = tuple(
            IndicatorFieldOutput(
                output_name,
                tuple(Decimal(42) for _ in request.bars),
            )
            for output_name in request.definition.output_fields
        )
        return IndicatorComputationResult(
            definition_name=request.definition.name,
            backend_identity=self.identity_for(request.definition),
            normalized_parameters=request.definition.parameters,
            normalized_input_fields=request.definition.input_fields,
            fields=fields,
            observation_count=len(request.bars),
        )


def test_default_registry_resolves_stable_native_and_talib_ids() -> None:
    registry = default_indicator_backend_registry()

    assert registry.backend_ids == (NATIVE_INDICATOR_BACKEND, TALIB_INDICATOR_BACKEND)
    assert registry.resolve(NATIVE_INDICATOR_BACKEND).backend_id == "native_v1"
    assert registry.resolve(TALIB_INDICATOR_BACKEND).backend_id == "talib_v1"

    with pytest.raises(
        UnsupportedIndicatorBackendError, match="unsupported indicator backend"
    ):
        registry.resolve("missing_v1")


def test_native_and_talib_use_one_backend_neutral_ema_definition() -> None:
    parameters = ExponentialMovingAverageParameters(3)
    native = ExponentialMovingAverage(parameters, backend_id=NATIVE_INDICATOR_BACKEND)
    talib_indicator = ExponentialMovingAverage(
        parameters, backend_id=TALIB_INDICATOR_BACKEND
    )

    assert type(native) is ExponentialMovingAverage
    assert type(talib_indicator) is ExponentialMovingAverage
    assert native.standard_definition == talib_indicator.standard_definition
    assert native.backend_identity.backend_id == NATIVE_INDICATOR_BACKEND
    assert talib_indicator.backend_identity.backend_id == TALIB_INDICATOR_BACKEND


def test_talib_ema_maps_inputs_parameters_and_aligned_output() -> None:
    dataset = make_dataset(("100", "100", "100", "100"), opens=("10", "11", "12", "13"))
    indicator = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3, source_field=MarketField.OPEN),
        backend_id=TALIB_INDICATOR_BACKEND,
    )

    output = indicator.calculate(dataset)

    assert output.session_dates == tuple(bar.session_date for bar in dataset.bars)
    assert output.values_for(EXPONENTIAL_MOVING_AVERAGE_OUTPUT) == (
        None,
        None,
        Decimal("11.0"),
        Decimal("12.0"),
    )
    result = TalibIndicatorBackend().compute(
        IndicatorComputationRequest(indicator.standard_definition, dataset.bars)
    )
    assert result.metadata() == {
        "definition_name": "exponential_moving_average",
        "backend": {
            "backend_id": "talib_v1",
            "contract_version": "1",
            "library_name": "TA-Lib",
            "library_version": "0.7.1",
            "function_name": "EMA",
        },
        "normalized_parameters": {"period": 3, "source_field": "open"},
        "normalized_input_fields": ["open"],
        "normalized_output_fields": ["exponential_moving_average"],
        "observation_count": 4,
    }


def test_talib_exact_installed_version_participates_in_configuration_identity() -> None:
    indicator = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3),
        backend_id=TALIB_INDICATOR_BACKEND,
    )
    configuration = indicator.configuration()

    assert talib.__version__ == "0.7.1"
    assert configuration["backend"] == indicator.backend_identity.to_primitive()
    assert indicator.backend_identity.library_version == talib.__version__

    version_one = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3),
        backend_id="future_v1",
        backend_registry=IndicatorBackendRegistry((FutureIndicatorBackend("1.0"),)),
    )
    version_two = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3),
        backend_id="future_v1",
        backend_registry=IndicatorBackendRegistry((FutureIndicatorBackend("2.0"),)),
    )
    assert version_one.configuration_id != version_two.configuration_id


def test_explicit_backend_configuration_round_trips_and_rejects_version_drift() -> None:
    indicator = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3),
        backend_id=TALIB_INDICATOR_BACKEND,
    )
    configuration = indicator.configuration()

    restored = ExponentialMovingAverage.from_configuration(configuration)
    changed_backend: PrimitiveMapping = {
        **indicator.backend_identity.to_primitive(),
        "library_version": "0.7.2",
    }
    changed_version: PrimitiveMapping = {
        **configuration,
        "backend": changed_backend,
    }

    assert restored.configuration_id == indicator.configuration_id
    assert restored.backend_identity == indicator.backend_identity
    with pytest.raises(IndicatorBackendVersionError, match="installed backend"):
        ExponentialMovingAverage.from_configuration(changed_version)


def test_unsupported_backend_indicator_combination_is_a_clear_domain_error() -> None:
    unsupported = StandardIndicatorDefinition(
        name="moving_average_convergence_divergence",
        parameters=PrimitiveMappingSnapshot.capture({}),
        input_fields=(MarketField.CLOSE,),
        output_fields=("macd",),
    )

    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="talib_v1 does not support indicator",
    ):
        TalibIndicatorBackend().identity_for(unsupported)


def test_legacy_configuration_resolves_explicit_native_semantics_unchanged() -> None:
    legacy = ExponentialMovingAverage(ExponentialMovingAverageParameters(3))
    legacy_configuration = legacy.configuration()

    restored = ExponentialMovingAverage.from_configuration(legacy_configuration)

    assert "backend" not in legacy_configuration
    assert legacy.configuration_id == LEGACY_EMA_PERIOD_3_ID
    assert restored.configuration() == legacy_configuration
    assert restored.configuration_id == LEGACY_EMA_PERIOD_3_ID
    assert restored.backend_identity.backend_id == NATIVE_INDICATOR_BACKEND
    assert restored.uses_legacy_native_configuration


def test_explicit_native_preserves_historical_ema_values() -> None:
    dataset = make_dataset(("10", "11", "12", "13", "14"))
    legacy = ExponentialMovingAverage(ExponentialMovingAverageParameters(3))
    explicit_native = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3),
        backend_id=NATIVE_INDICATOR_BACKEND,
    )

    assert explicit_native.calculate(dataset).fields == legacy.calculate(dataset).fields
    assert explicit_native.configuration_id != legacy.configuration_id


def test_future_backend_adapter_reuses_ema_without_backend_specific_class() -> None:
    registry = IndicatorBackendRegistry((FutureIndicatorBackend("9.4"),))
    indicator = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(2),
        backend_id="future_v1",
        backend_registry=registry,
    )

    output = indicator.calculate(make_dataset(("1", "2", "3")))

    assert type(indicator) is ExponentialMovingAverage
    assert output.values_for(EXPONENTIAL_MOVING_AVERAGE_OUTPUT) == (
        Decimal(42),
        Decimal(42),
        Decimal(42),
    )
