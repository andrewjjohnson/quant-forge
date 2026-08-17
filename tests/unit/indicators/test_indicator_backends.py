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
    InvalidIndicatorBackendError,
    MarketField,
    NativeIndicatorBackend,
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


class MismatchedIdentityBackend(FutureIndicatorBackend):
    """Malformed adapter whose resolved ID differs from its identity."""

    def identity_for(
        self, definition: StandardIndicatorDefinition
    ) -> IndicatorBackendIdentity:
        return IndicatorBackendIdentity(
            backend_id="different_v1",
            library_name="future-library",
            library_version=self._library_version,
            function_name=f"mapped_{definition.name}",
        )


class TruncatedIndicatorBackend(FutureIndicatorBackend):
    """Malformed adapter returning internally aligned but truncated values."""

    def compute(
        self, request: IndicatorComputationRequest
    ) -> IndicatorComputationResult:
        observation_count = len(request.bars) - 1
        fields = tuple(
            IndicatorFieldOutput(
                output_name,
                tuple(Decimal(42) for _ in range(observation_count)),
            )
            for output_name in request.definition.output_fields
        )
        return IndicatorComputationResult(
            definition_name=request.definition.name,
            backend_identity=self.identity_for(request.definition),
            normalized_parameters=request.definition.parameters,
            normalized_input_fields=request.definition.input_fields,
            fields=fields,
            observation_count=observation_count,
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
            "runtime_library_name": "TA-Lib C",
            "runtime_library_version": "0.7.1",
        },
        "normalized_parameters": {"period": 3, "source_field": "open"},
        "normalized_input_fields": ["open"],
        "normalized_output_fields": ["exponential_moving_average"],
        "observation_count": 4,
    }


def test_talib_state_drift_during_calculation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indicator = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3),
        backend_id=TALIB_INDICATOR_BACKEND,
    )
    compatibility_states = iter((0, 1))
    monkeypatch.setattr(talib, "get_compatibility", lambda: next(compatibility_states))

    with pytest.raises(
        InvalidIndicatorBackendError,
        match="default compatibility and zero unstable period",
    ):
        indicator.calculate(make_dataset(("1", "2", "3", "4")))


def test_talib_exact_installed_versions_participate_in_configuration_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indicator = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3),
        backend_id=TALIB_INDICATOR_BACKEND,
    )
    configuration = indicator.configuration()

    assert talib.__version__ == "0.7.1"
    assert configuration["backend"] == indicator.backend_identity.to_primitive()
    assert indicator.backend_identity.library_version == talib.__version__
    assert indicator.backend_identity.runtime_library_name == "TA-Lib C"
    assert indicator.backend_identity.runtime_library_version == "0.7.1"

    monkeypatch.setattr(talib, "__ta_version__", b"0.7.2 (different runtime)")
    changed_runtime = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3),
        backend_id=TALIB_INDICATOR_BACKEND,
    )
    assert changed_runtime.backend_identity.runtime_library_version == "0.7.2"
    assert changed_runtime.configuration_id != indicator.configuration_id

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


def test_talib_ema_period_range_is_validated_during_configuration() -> None:
    period_one = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(1),
        backend_id=TALIB_INDICATOR_BACKEND,
    )
    native_above_talib_limit = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(100_001),
        backend_id=NATIVE_INDICATOR_BACKEND,
    )
    unsupported_parameters: PrimitiveMapping = {
        "period": 100_001,
        "source_field": "close",
    }
    unsupported_serialized: PrimitiveMapping = {
        **period_one.configuration(),
        "parameters": unsupported_parameters,
    }

    assert period_one.calculate(make_dataset(("1", "2", "3"))).values_for(
        EXPONENTIAL_MOVING_AVERAGE_OUTPUT
    ) == (Decimal(1), Decimal(2), Decimal(3))
    assert native_above_talib_limit.backend_identity.backend_id == "native_v1"
    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="EMA period must be from 1 through 100000",
    ):
        ExponentialMovingAverage(
            ExponentialMovingAverageParameters(100_001),
            backend_id=TALIB_INDICATOR_BACKEND,
        )
    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="EMA period must be from 1 through 100000",
    ):
        ExponentialMovingAverage.from_configuration(unsupported_serialized)


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
        name="unmapped_indicator",
        parameters=PrimitiveMappingSnapshot.capture({}),
        input_fields=(MarketField.CLOSE,),
        output_fields=("macd",),
    )

    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="talib_v1 does not support indicator",
    ):
        TalibIndicatorBackend().identity_for(unsupported)


@pytest.mark.parametrize(
    "input_fields",
    [
        (MarketField.CLOSE, MarketField.OPEN),
        (MarketField.OPEN,),
    ],
)
def test_native_ema_rejects_noncanonical_input_mappings(
    input_fields: tuple[MarketField, ...],
) -> None:
    definition = StandardIndicatorDefinition(
        name="exponential_moving_average",
        parameters=PrimitiveMappingSnapshot.capture(
            {"period": 3, "source_field": "close"}
        ),
        input_fields=input_fields,
        output_fields=(EXPONENTIAL_MOVING_AVERAGE_OUTPUT,),
    )

    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="native_v1 input mapping is unavailable",
    ):
        NativeIndicatorBackend().identity_for(definition)


@pytest.mark.parametrize("period", [-2, 0, True])
def test_native_ema_rejects_invalid_periods_through_direct_backend_contract(
    period: int | bool,
) -> None:
    definition = StandardIndicatorDefinition(
        name="exponential_moving_average",
        parameters=PrimitiveMappingSnapshot.capture(
            {"period": period, "source_field": "close"}
        ),
        input_fields=(MarketField.CLOSE,),
        output_fields=(EXPONENTIAL_MOVING_AVERAGE_OUTPUT,),
    )

    with pytest.raises(
        UnsupportedIndicatorBackendError,
        match="native_v1 EMA period must be a positive integer",
    ):
        NativeIndicatorBackend().identity_for(definition)


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


def test_selected_registry_id_must_match_backend_identity() -> None:
    registry = IndicatorBackendRegistry((MismatchedIdentityBackend("9.4"),))

    with pytest.raises(
        InvalidIndicatorBackendError,
        match="identity does not match the selected registry id: future_v1",
    ):
        ExponentialMovingAverage(
            ExponentialMovingAverageParameters(2),
            backend_id="future_v1",
            backend_registry=registry,
        )


def test_direct_calculation_rejects_truncated_backend_results() -> None:
    registry = IndicatorBackendRegistry((TruncatedIndicatorBackend("9.4"),))
    indicator = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(2),
        backend_id="future_v1",
        backend_registry=registry,
    )
    bars = make_dataset(("1", "2", "3")).bars

    with pytest.raises(
        InvalidIndicatorBackendError,
        match="observation count does not match the canonical input bars",
    ):
        indicator.calculate_bar_fields(bars)
