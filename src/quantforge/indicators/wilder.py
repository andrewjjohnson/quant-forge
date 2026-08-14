"""Causal Wilder RSI and directional-movement indicators."""

from dataclasses import dataclass
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.models import MarketDataset
from quantforge.indicators.backends import (
    NATIVE_INDICATOR_BACKEND,
    IndicatorBackend,
    IndicatorBackendIdentity,
    IndicatorBackendRegistry,
    IndicatorComputationRequest,
    StandardIndicatorDefinition,
    default_indicator_backend_registry,
)
from quantforge.indicators.base import (
    DevelopingBarSupport,
    IndicatorBar,
    IndicatorParameters,
    validate_indicator_alignment,
    validate_indicator_bars,
    validate_market_input,
)
from quantforge.indicators.exceptions import (
    IndicatorBackendVersionError,
    IndicatorCalculationError,
    InvalidIndicatorBackendError,
    InvalidIndicatorParametersError,
)
from quantforge.indicators.models import (
    IndicatorFieldOutput,
    IndicatorOutput,
    IndicatorValue,
    MarketField,
)

WILDER_RSI_OUTPUT = "wilder_rsi"
POSITIVE_DIRECTIONAL_INDICATOR_OUTPUT = "positive_directional_indicator"
NEGATIVE_DIRECTIONAL_INDICATOR_OUTPUT = "negative_directional_indicator"
AVERAGE_DIRECTIONAL_INDEX_OUTPUT = "average_directional_index"
WILDER_AVERAGE_TRUE_RANGE_OUTPUT = "wilder_average_true_range"

_DECIMAL_PRECISION = 34
_DECIMAL_ROUNDING = ROUND_HALF_EVEN
_DECIMAL_EMIN = -999_999
_DECIMAL_EMAX = 999_999
_DECIMAL_CAPITALS = 1
_DECIMAL_CLAMP = 0
_DECIMAL_TRAPS: tuple[type[DecimalException], ...] = (
    DivisionByZero,
    InvalidOperation,
    Overflow,
)
_DECIMAL_TRAP_NAMES = tuple(signal.__name__ for signal in _DECIMAL_TRAPS)
_ONE_HUNDRED = Decimal(100)


@dataclass(frozen=True, slots=True)
class WilderRelativeStrengthIndexParameters:
    """Period for Wilder's recursively smoothed relative-strength index."""

    period: int = 14

    def __post_init__(self) -> None:
        _validate_period(self.period)

    def to_primitive(self) -> PrimitiveMapping:
        return {"period": self.period}


class WilderRelativeStrengthIndex:
    """Wilder RSI using the current and historical bar closes only."""

    name = "wilder_relative_strength_index"
    implementation_version = "1"
    output_fields = (WILDER_RSI_OUTPUT,)
    missing_value = None
    developing_bar_support = DevelopingBarSupport.DEVELOPING_AS_OF

    def __init__(
        self,
        parameters: WilderRelativeStrengthIndexParameters,
        *,
        backend_id: str | None = None,
        backend_registry: IndicatorBackendRegistry | None = None,
    ) -> None:
        self._parameters = parameters
        self._legacy_native_configuration = backend_id is None
        selected_backend_id = _validate_backend_id(
            NATIVE_INDICATOR_BACKEND if backend_id is None else backend_id
        )
        self._definition = StandardIndicatorDefinition(
            name=self.name,
            parameters=PrimitiveMappingSnapshot.capture(parameters.to_primitive()),
            input_fields=(MarketField.CLOSE,),
            output_fields=self.output_fields,
        )
        registry = backend_registry or default_indicator_backend_registry()
        self._backend: IndicatorBackend = registry.resolve(selected_backend_id)
        self._backend_identity = self._backend.identity_for(self._definition)
        if self._backend_identity.backend_id != selected_backend_id:
            raise InvalidIndicatorBackendError(
                "indicator backend identity does not match the selected registry id: "
                f"{selected_backend_id}"
            )

    @property
    def parameters(self) -> IndicatorParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((MarketField.CLOSE,))

    @property
    def warm_up_observations(self) -> int:
        return self._parameters.period + 1

    @property
    def standard_definition(self) -> StandardIndicatorDefinition:
        """Return the canonical Wilder RSI definition shared by every backend."""
        return self._definition

    @property
    def backend_identity(self) -> IndicatorBackendIdentity:
        """Return the resolved backend and exact library identity."""
        return self._backend_identity

    @property
    def uses_legacy_native_configuration(self) -> bool:
        """Whether serialization intentionally preserves the pre-QF-36 identity."""
        return self._legacy_native_configuration

    def configuration(self) -> PrimitiveMapping:
        legacy_configuration = _configuration(
            component_name=self.name,
            implementation_version=self.implementation_version,
            parameters=self._parameters.to_primitive(),
            required_fields=self.required_fields,
            warm_up_observations=self.warm_up_observations,
            output_fields=self.output_fields,
        )
        if self._legacy_native_configuration:
            return legacy_configuration
        return _backend_configuration(
            component_name=self.name,
            definition_version=self.implementation_version,
            parameters=self._parameters.to_primitive(),
            required_fields=self.required_fields,
            warm_up_observations=self.warm_up_observations,
            output_fields=self.output_fields,
            definition=self._definition,
            backend_identity=self._backend_identity,
        )

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    @classmethod
    def from_configuration(
        cls,
        configuration: PrimitiveMapping,
        *,
        backend_registry: IndicatorBackendRegistry | None = None,
    ) -> "WilderRelativeStrengthIndex":
        """Restore legacy/native or explicit backend semantics without drift."""
        period = _serialized_period(configuration, "RSI")
        backend_id = _serialized_backend_id(configuration, "RSI")
        indicator = cls(
            WilderRelativeStrengthIndexParameters(period),
            backend_id=backend_id,
            backend_registry=backend_registry,
        )
        if indicator.configuration() != configuration:
            raise IndicatorBackendVersionError(
                "serialized RSI configuration does not match the installed backend"
            )
        return indicator

    def calculate(self, dataset: MarketDataset) -> IndicatorOutput:
        validate_market_input(dataset, self.required_fields)
        fields = self.calculate_bar_fields(dataset.bars)
        return _aligned_output(self, dataset, fields)

    def calculate_bar_fields(
        self, bars: tuple[IndicatorBar, ...]
    ) -> tuple[IndicatorFieldOutput, ...]:
        """Calculate RSI against any validated canonical timeframe series."""
        validate_indicator_bars(bars, self.required_fields)
        return _calculate_backend_fields(
            backend=self._backend,
            backend_identity=self._backend_identity,
            definition=self._definition,
            bars=bars,
        )


@dataclass(frozen=True, slots=True)
class WilderDirectionalMovementParameters:
    """Period for Wilder +DI, -DI, and ADX smoothing."""

    period: int = 14

    def __post_init__(self) -> None:
        _validate_period(self.period)

    def to_primitive(self) -> PrimitiveMapping:
        return {"period": self.period}


class WilderDirectionalMovement:
    """Aligned Wilder +DI, -DI, and ADX over canonical OHLC bars."""

    name = "wilder_directional_movement"
    implementation_version = "1"
    output_fields = (
        POSITIVE_DIRECTIONAL_INDICATOR_OUTPUT,
        NEGATIVE_DIRECTIONAL_INDICATOR_OUTPUT,
        AVERAGE_DIRECTIONAL_INDEX_OUTPUT,
    )
    missing_value = None
    developing_bar_support = DevelopingBarSupport.DEVELOPING_AS_OF

    def __init__(self, parameters: WilderDirectionalMovementParameters) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> IndicatorParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((MarketField.HIGH, MarketField.LOW, MarketField.CLOSE))

    @property
    def warm_up_observations(self) -> int:
        return self._parameters.period * 2

    def configuration(self) -> PrimitiveMapping:
        return _configuration(
            component_name=self.name,
            implementation_version=self.implementation_version,
            parameters=self._parameters.to_primitive(),
            required_fields=self.required_fields,
            warm_up_observations=self.warm_up_observations,
            output_fields=self.output_fields,
        )

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def calculate(self, dataset: MarketDataset) -> IndicatorOutput:
        validate_market_input(dataset, self.required_fields)
        fields = self.calculate_bar_fields(dataset.bars)
        return _aligned_output(self, dataset, fields)

    def calculate_bar_fields(
        self, bars: tuple[IndicatorBar, ...]
    ) -> tuple[IndicatorFieldOutput, ...]:
        """Calculate DMI/ADX against any validated canonical timeframe series."""
        validate_indicator_bars(bars, self.required_fields)
        count = len(bars)
        positive_di: list[IndicatorValue] = [None] * count
        negative_di: list[IndicatorValue] = [None] * count
        adx: list[IndicatorValue] = [None] * count
        period = self._parameters.period
        if count <= period:
            return tuple(
                IndicatorFieldOutput(name, values)
                for name, values in zip(
                    self.output_fields,
                    (tuple(positive_di), tuple(negative_di), tuple(adx)),
                    strict=True,
                )
            )

        context = _arithmetic_context()
        try:
            with localcontext(context):
                true_ranges: list[Decimal] = []
                positive_movements: list[Decimal] = []
                negative_movements: list[Decimal] = []
                for index in range(1, count):
                    current = bars[index]
                    previous = bars[index - 1]
                    true_ranges.append(
                        max(
                            current.high - current.low,
                            abs(current.high - previous.close),
                            abs(current.low - previous.close),
                        )
                    )
                    upward_move = current.high - previous.high
                    downward_move = previous.low - current.low
                    positive_movements.append(
                        upward_move
                        if upward_move > downward_move and upward_move > 0
                        else Decimal(0)
                    )
                    negative_movements.append(
                        downward_move
                        if downward_move > upward_move and downward_move > 0
                        else Decimal(0)
                    )

                smoothed_true_range = sum(true_ranges[:period], Decimal(0))
                smoothed_positive = sum(positive_movements[:period], Decimal(0))
                smoothed_negative = sum(negative_movements[:period], Decimal(0))
                dx_values: list[Decimal] = []

                for index in range(period, count):
                    if index > period:
                        movement_index = index - 1
                        divisor = Decimal(period)
                        smoothed_true_range = (
                            smoothed_true_range
                            - smoothed_true_range / divisor
                            + true_ranges[movement_index]
                        )
                        smoothed_positive = (
                            smoothed_positive
                            - smoothed_positive / divisor
                            + positive_movements[movement_index]
                        )
                        smoothed_negative = (
                            smoothed_negative
                            - smoothed_negative / divisor
                            + negative_movements[movement_index]
                        )
                    current_positive, current_negative, current_dx = _di_and_dx(
                        smoothed_true_range,
                        smoothed_positive,
                        smoothed_negative,
                    )
                    positive_di[index] = current_positive
                    negative_di[index] = current_negative
                    dx_values.append(current_dx)

                first_adx_index = period * 2 - 1
                if count > first_adx_index:
                    divisor = Decimal(period)
                    current_adx = sum(dx_values[:period], Decimal(0)) / divisor
                    adx[first_adx_index] = current_adx
                    for index in range(first_adx_index + 1, count):
                        current_dx = dx_values[index - period]
                        current_adx = (
                            current_adx * Decimal(period - 1) + current_dx
                        ) / divisor
                        adx[index] = current_adx
        except DecimalException as error:
            raise IndicatorCalculationError(
                "Wilder directional-movement arithmetic failed under its configured "
                "decimal policy"
            ) from error

        return tuple(
            IndicatorFieldOutput(name, values)
            for name, values in zip(
                self.output_fields,
                (tuple(positive_di), tuple(negative_di), tuple(adx)),
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class WilderAverageTrueRangeParameters:
    """Period for Wilder's recursively smoothed average true range."""

    period: int = 14

    def __post_init__(self) -> None:
        _validate_period(self.period)

    def to_primitive(self) -> PrimitiveMapping:
        return {"period": self.period}


class WilderAverageTrueRange:
    """Aligned Wilder ATR using only current and prior canonical OHLC bars."""

    name = "wilder_average_true_range"
    implementation_version = "1"
    output_fields = (WILDER_AVERAGE_TRUE_RANGE_OUTPUT,)
    missing_value = None
    developing_bar_support = DevelopingBarSupport.DEVELOPING_AS_OF

    def __init__(
        self,
        parameters: WilderAverageTrueRangeParameters,
        *,
        backend_id: str | None = None,
        backend_registry: IndicatorBackendRegistry | None = None,
    ) -> None:
        self._parameters = parameters
        self._legacy_native_configuration = backend_id is None
        selected_backend_id = _validate_backend_id(
            NATIVE_INDICATOR_BACKEND if backend_id is None else backend_id
        )
        self._definition = StandardIndicatorDefinition(
            name=self.name,
            parameters=PrimitiveMappingSnapshot.capture(parameters.to_primitive()),
            input_fields=(MarketField.HIGH, MarketField.LOW, MarketField.CLOSE),
            output_fields=self.output_fields,
        )
        registry = backend_registry or default_indicator_backend_registry()
        self._backend: IndicatorBackend = registry.resolve(selected_backend_id)
        self._backend_identity = self._backend.identity_for(self._definition)
        if self._backend_identity.backend_id != selected_backend_id:
            raise InvalidIndicatorBackendError(
                "indicator backend identity does not match the selected registry id: "
                f"{selected_backend_id}"
            )

    @property
    def parameters(self) -> IndicatorParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((MarketField.HIGH, MarketField.LOW, MarketField.CLOSE))

    @property
    def warm_up_observations(self) -> int:
        return self._parameters.period + 1

    @property
    def standard_definition(self) -> StandardIndicatorDefinition:
        """Return the canonical Wilder ATR definition shared by every backend."""
        return self._definition

    @property
    def backend_identity(self) -> IndicatorBackendIdentity:
        """Return the resolved backend and exact library identity."""
        return self._backend_identity

    @property
    def uses_legacy_native_configuration(self) -> bool:
        """Whether serialization intentionally preserves the pre-QF-36 identity."""
        return self._legacy_native_configuration

    def configuration(self) -> PrimitiveMapping:
        legacy_configuration = _configuration(
            component_name=self.name,
            implementation_version=self.implementation_version,
            parameters=self._parameters.to_primitive(),
            required_fields=self.required_fields,
            warm_up_observations=self.warm_up_observations,
            output_fields=self.output_fields,
        )
        if self._legacy_native_configuration:
            return legacy_configuration
        return _backend_configuration(
            component_name=self.name,
            definition_version=self.implementation_version,
            parameters=self._parameters.to_primitive(),
            required_fields=self.required_fields,
            warm_up_observations=self.warm_up_observations,
            output_fields=self.output_fields,
            definition=self._definition,
            backend_identity=self._backend_identity,
        )

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    @classmethod
    def from_configuration(
        cls,
        configuration: PrimitiveMapping,
        *,
        backend_registry: IndicatorBackendRegistry | None = None,
    ) -> "WilderAverageTrueRange":
        """Restore legacy/native or explicit backend semantics without drift."""
        period = _serialized_period(configuration, "ATR")
        backend_id = _serialized_backend_id(configuration, "ATR")
        indicator = cls(
            WilderAverageTrueRangeParameters(period),
            backend_id=backend_id,
            backend_registry=backend_registry,
        )
        if indicator.configuration() != configuration:
            raise IndicatorBackendVersionError(
                "serialized ATR configuration does not match the installed backend"
            )
        return indicator

    def calculate(self, dataset: MarketDataset) -> IndicatorOutput:
        validate_market_input(dataset, self.required_fields)
        fields = self.calculate_bar_fields(dataset.bars)
        return _aligned_output(self, dataset, fields)

    def calculate_bar_fields(
        self, bars: tuple[IndicatorBar, ...]
    ) -> tuple[IndicatorFieldOutput, ...]:
        """Calculate ATR against any validated canonical timeframe series."""
        validate_indicator_bars(bars, self.required_fields)
        return _calculate_backend_fields(
            backend=self._backend,
            backend_identity=self._backend_identity,
            definition=self._definition,
            bars=bars,
        )


def _di_and_dx(
    smoothed_true_range: Decimal,
    smoothed_positive: Decimal,
    smoothed_negative: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    if smoothed_true_range == 0:
        return Decimal(0), Decimal(0), Decimal(0)
    positive = _ONE_HUNDRED * smoothed_positive / smoothed_true_range
    negative = _ONE_HUNDRED * smoothed_negative / smoothed_true_range
    directional_sum = positive + negative
    dx = (
        Decimal(0)
        if directional_sum == 0
        else _ONE_HUNDRED * abs(positive - negative) / directional_sum
    )
    return positive, negative, dx


def _configuration(
    *,
    component_name: str,
    implementation_version: str,
    parameters: PrimitiveMapping,
    required_fields: frozenset[MarketField],
    warm_up_observations: int,
    output_fields: tuple[str, ...],
) -> PrimitiveMapping:
    return {
        "component_type": "indicator",
        "component_name": component_name,
        "contract_version": "1",
        "implementation_version": implementation_version,
        "parameters": parameters,
        "required_fields": [field.value for field in sorted(required_fields)],
        "warm_up_observations": warm_up_observations,
        "output_fields": list(output_fields),
        "missing_value": None,
        "arithmetic": {
            "decimal_precision": _DECIMAL_PRECISION,
            "rounding": "ROUND_HALF_EVEN",
            "decimal_emin": _DECIMAL_EMIN,
            "decimal_emax": _DECIMAL_EMAX,
            "capitals": _DECIMAL_CAPITALS,
            "clamp": _DECIMAL_CLAMP,
            "initial_flags": [],
            "traps": list(_DECIMAL_TRAP_NAMES),
        },
    }


def _backend_configuration(
    *,
    component_name: str,
    definition_version: str,
    parameters: PrimitiveMapping,
    required_fields: frozenset[MarketField],
    warm_up_observations: int,
    output_fields: tuple[str, ...],
    definition: StandardIndicatorDefinition,
    backend_identity: IndicatorBackendIdentity,
) -> PrimitiveMapping:
    normalized_input_fields: list[Primitive] = [
        field.value for field in definition.input_fields
    ]
    normalized_parameter_names: list[Primitive] = []
    normalized_parameter_names.extend(sorted(parameters))
    return {
        "component_type": "indicator",
        "component_name": component_name,
        "contract_version": "2",
        "definition_version": definition_version,
        "parameters": parameters,
        "required_fields": [field.value for field in sorted(required_fields)],
        "warm_up_observations": warm_up_observations,
        "output_fields": list(output_fields),
        "missing_value": None,
        "backend": backend_identity.to_primitive(),
        "normalization": {
            "input_fields": normalized_input_fields,
            "parameter_names": normalized_parameter_names,
            "output_fields": list(output_fields),
            "unavailable_output": None,
        },
    }


def _calculate_backend_fields(
    *,
    backend: IndicatorBackend,
    backend_identity: IndicatorBackendIdentity,
    definition: StandardIndicatorDefinition,
    bars: tuple[IndicatorBar, ...],
) -> tuple[IndicatorFieldOutput, ...]:
    result = backend.compute(IndicatorComputationRequest(definition, bars))
    if result.observation_count != len(bars):
        raise InvalidIndicatorBackendError(
            "indicator backend result observation count does not match the "
            "canonical input bars"
        )
    if (
        result.backend_identity != backend_identity
        or result.definition_name != definition.name
        or result.normalized_parameters != definition.parameters
        or result.normalized_input_fields != definition.input_fields
        or tuple(field.name for field in result.fields) != definition.output_fields
    ):
        raise InvalidIndicatorBackendError(
            "indicator backend result metadata changed during calculation"
        )
    return result.fields


def _serialized_period(configuration: PrimitiveMapping, label: str) -> int:
    parameters_value = configuration.get("parameters")
    if not isinstance(parameters_value, dict):
        raise InvalidIndicatorBackendError(
            f"serialized {label} parameters must be an object"
        )
    period = parameters_value.get("period")
    if isinstance(period, bool) or not isinstance(period, int):
        raise InvalidIndicatorBackendError(
            f"serialized {label} period must be an integer"
        )
    return period


def _serialized_backend_id(configuration: PrimitiveMapping, label: str) -> str | None:
    backend_value = configuration.get("backend")
    if backend_value is None:
        return None
    if not isinstance(backend_value, dict):
        raise InvalidIndicatorBackendError(
            f"serialized {label} backend must be an object"
        )
    backend_id = backend_value.get("backend_id")
    if not isinstance(backend_id, str):
        raise InvalidIndicatorBackendError(f"serialized {label} backend id is invalid")
    return backend_id


def _aligned_output(
    indicator: (
        WilderRelativeStrengthIndex | WilderDirectionalMovement | WilderAverageTrueRange
    ),
    dataset: MarketDataset,
    fields: tuple[IndicatorFieldOutput, ...],
) -> IndicatorOutput:
    output = IndicatorOutput(
        indicator.name,
        indicator.configuration_id,
        tuple(bar.session_date for bar in dataset.bars),
        fields,
    )
    validate_indicator_alignment(dataset, output)
    return output


def _arithmetic_context() -> Context:
    return Context(
        prec=_DECIMAL_PRECISION,
        rounding=_DECIMAL_ROUNDING,
        Emin=_DECIMAL_EMIN,
        Emax=_DECIMAL_EMAX,
        capitals=_DECIMAL_CAPITALS,
        clamp=_DECIMAL_CLAMP,
        flags=[],
        traps=list(_DECIMAL_TRAPS),
    )


def _validate_period(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidIndicatorParametersError("period must be an integer")
    if value < 1:
        raise InvalidIndicatorParametersError("period must be at least 1")


def _validate_backend_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidIndicatorBackendError(
            "indicator backend id must be a non-empty string"
        )
    return value
