"""Adapter for historical QuantForge-native standard indicator mathematics."""

from collections.abc import Callable
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
from itertools import pairwise

from quantforge.configuration import PrimitiveMapping
from quantforge.indicators.backends.base import (
    NATIVE_INDICATOR_BACKEND,
    IndicatorBackendIdentity,
    IndicatorComputationRequest,
    IndicatorComputationResult,
    StandardIndicatorDefinition,
)
from quantforge.indicators.exceptions import (
    IndicatorCalculationError,
    MissingMarketFieldError,
    UnsupportedIndicatorBackendError,
)
from quantforge.indicators.models import IndicatorFieldOutput, IndicatorValue

_SMA_NAME = "simple_moving_average"
_EMA_NAME = "exponential_moving_average"
_RSI_NAME = "wilder_relative_strength_index"
_ATR_NAME = "wilder_average_true_range"
_DECIMAL_PRECISION = 34
_DECIMAL_EMIN = -999_999
_DECIMAL_EMAX = 999_999
_DECIMAL_TRAPS: tuple[type[DecimalException], ...] = (
    DivisionByZero,
    InvalidOperation,
    Overflow,
)
type _NativeFunction = Callable[
    [tuple[tuple[Decimal, ...], ...], PrimitiveMapping],
    tuple[tuple[IndicatorValue, ...], ...],
]


@dataclass(frozen=True, slots=True)
class _NativeMapping:
    function_name: str
    implementation_version: str
    parameter_names: frozenset[str]
    input_fields: tuple[str, ...] | None
    source_field_parameter: str | None
    output_names: tuple[str, ...]
    validate_parameters: Callable[[PrimitiveMapping], None]
    function: _NativeFunction


_MAPPINGS = {
    _SMA_NAME: _NativeMapping(
        function_name="quantforge_native_sma",
        implementation_version="1",
        parameter_names=frozenset(("source_field", "window")),
        input_fields=None,
        source_field_parameter="source_field",
        output_names=("simple_moving_average",),
        validate_parameters=lambda parameters: _validate_positive_period(
            parameters, "window", "SMA"
        ),
        function=lambda inputs, parameters: (_native_sma(inputs[0], parameters),),
    ),
    _EMA_NAME: _NativeMapping(
        function_name="quantforge_native_ema",
        implementation_version="1",
        parameter_names=frozenset(("period", "source_field")),
        input_fields=None,
        source_field_parameter="source_field",
        output_names=("exponential_moving_average",),
        validate_parameters=lambda parameters: _validate_positive_period(
            parameters, "period", "EMA"
        ),
        function=lambda inputs, parameters: (_native_ema(inputs[0], parameters),),
    ),
    _RSI_NAME: _NativeMapping(
        function_name="quantforge_native_wilder_rsi",
        implementation_version="1",
        parameter_names=frozenset(("period",)),
        input_fields=("close",),
        source_field_parameter=None,
        output_names=("wilder_rsi",),
        validate_parameters=lambda parameters: _validate_positive_period(
            parameters, "period", "RSI"
        ),
        function=lambda inputs, parameters: (_native_rsi(inputs[0], parameters),),
    ),
    _ATR_NAME: _NativeMapping(
        function_name="quantforge_native_wilder_atr",
        implementation_version="1",
        parameter_names=frozenset(("period",)),
        input_fields=("high", "low", "close"),
        source_field_parameter=None,
        output_names=("wilder_average_true_range",),
        validate_parameters=lambda parameters: _validate_positive_period(
            parameters, "period", "ATR"
        ),
        function=lambda inputs, parameters: (
            _native_atr(inputs[0], inputs[1], inputs[2], parameters),
        ),
    ),
}


class NativeIndicatorBackend:
    """Execute mapped standard indicators with versioned QuantForge math."""

    backend_id = NATIVE_INDICATOR_BACKEND

    def identity_for(
        self, definition: StandardIndicatorDefinition
    ) -> IndicatorBackendIdentity:
        mapping = _mapping_for(definition)
        return IndicatorBackendIdentity(
            backend_id=self.backend_id,
            library_name="quantforge_native_indicators",
            library_version=mapping.implementation_version,
            function_name=mapping.function_name,
        )

    def compute(
        self, request: IndicatorComputationRequest
    ) -> IndicatorComputationResult:
        """Translate canonical bars and parameters, then normalize native outputs."""
        definition = request.definition
        mapping = _mapping_for(definition)
        parameters = definition.parameters.to_primitive()
        inputs = tuple(
            _decimal_input(request, field.value) for field in definition.input_fields
        )
        raw_outputs = mapping.function(inputs, parameters)
        if len(raw_outputs) != len(definition.output_fields):
            raise IndicatorCalculationError(
                f"{self.backend_id} returned an unexpected output count for "
                f"indicator: {definition.name}"
            )
        fields = tuple(
            IndicatorFieldOutput(name, values)
            for name, values in zip(definition.output_fields, raw_outputs, strict=True)
        )
        return IndicatorComputationResult(
            definition_name=definition.name,
            backend_identity=self.identity_for(definition),
            normalized_parameters=definition.parameters,
            normalized_input_fields=definition.input_fields,
            fields=fields,
            observation_count=len(request.bars),
        )


def _mapping_for(definition: StandardIndicatorDefinition) -> _NativeMapping:
    try:
        mapping = _MAPPINGS[definition.name]
    except KeyError as error:
        raise UnsupportedIndicatorBackendError(
            f"{NATIVE_INDICATOR_BACKEND} does not support indicator: {definition.name}"
        ) from error
    if definition.output_fields != mapping.output_names:
        raise UnsupportedIndicatorBackendError(
            f"{NATIVE_INDICATOR_BACKEND} output mapping is unavailable for "
            f"indicator: {definition.name}"
        )
    parameters = definition.parameters.to_primitive()
    if frozenset(parameters) != mapping.parameter_names:
        raise UnsupportedIndicatorBackendError(
            f"{NATIVE_INDICATOR_BACKEND} parameter mapping is unavailable for "
            f"indicator: {definition.name}"
        )
    mapping.validate_parameters(parameters)
    actual_input_fields = tuple(field.value for field in definition.input_fields)
    expected_input_fields = mapping.input_fields
    if mapping.source_field_parameter is not None:
        source_field = parameters.get(mapping.source_field_parameter)
        expected_input_fields = (source_field,) if isinstance(source_field, str) else ()
    if actual_input_fields != expected_input_fields:
        raise UnsupportedIndicatorBackendError(
            f"{NATIVE_INDICATOR_BACKEND} input mapping is unavailable for "
            f"indicator: {definition.name}"
        )
    return mapping


def _validate_positive_period(
    parameters: PrimitiveMapping, parameter_name: str, indicator_label: str
) -> None:
    period = parameters.get(parameter_name)
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise UnsupportedIndicatorBackendError(
            f"{NATIVE_INDICATOR_BACKEND} {indicator_label} {parameter_name} must be "
            "a positive integer"
        )


def _decimal_input(
    request: IndicatorComputationRequest, field_name: str
) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    for bar in request.bars:
        try:
            raw_value = getattr(bar, field_name)
        except AttributeError as error:
            raise MissingMarketFieldError(
                f"market data is missing required field: {field_name}"
            ) from error
        if not isinstance(raw_value, Decimal):
            raise MissingMarketFieldError(
                f"market field must be a compatible Decimal: {field_name}"
            )
        values.append(raw_value)
    return tuple(values)


def _native_ema(
    source_values: tuple[Decimal, ...], parameters: PrimitiveMapping
) -> tuple[IndicatorValue, ...]:
    period_value = parameters.get("period")
    if isinstance(period_value, bool) or not isinstance(period_value, int):
        raise IndicatorCalculationError("native EMA period must be an integer")
    period = period_value
    values: list[IndicatorValue] = []
    seed: list[Decimal] = []
    previous: Decimal | None = None

    try:
        with localcontext(_arithmetic_context()):
            two = Decimal(2)
            previous_weight = Decimal(period - 1)
            recurrence_divisor = Decimal(period + 1)
            seed_divisor = Decimal(period)
            for source in source_values:
                if not source.is_finite():
                    values.append(None)
                    seed.clear()
                    previous = None
                    continue
                if previous is None:
                    seed.append(source)
                    if len(seed) < period:
                        values.append(None)
                        continue
                    previous = sum(seed, Decimal(0)) / seed_divisor
                else:
                    previous = (
                        previous_weight * previous + two * source
                    ) / recurrence_divisor
                values.append(previous)
    except DecimalException as error:
        raise IndicatorCalculationError(
            "exponential moving average arithmetic failed under its configured "
            "decimal policy"
        ) from error
    return tuple(values)


def _native_sma(
    source_values: tuple[Decimal, ...], parameters: PrimitiveMapping
) -> tuple[IndicatorValue, ...]:
    window_value = parameters.get("window")
    if isinstance(window_value, bool) or not isinstance(window_value, int):
        raise IndicatorCalculationError("native SMA window must be an integer")
    window = window_value
    divisor = Decimal(window)
    source = tuple(value if value.is_finite() else None for value in source_values)
    values: list[IndicatorValue] = []

    try:
        with localcontext(_arithmetic_context()):
            for index in range(len(source)):
                if index + 1 < window:
                    values.append(None)
                    continue
                observations = source[index + 1 - window : index + 1]
                if any(observation is None for observation in observations):
                    values.append(None)
                    continue
                total = sum(
                    (value for value in observations if value is not None),
                    start=Decimal(0),
                )
                values.append(total / divisor)
    except DecimalException as error:
        raise IndicatorCalculationError(
            "simple moving average arithmetic failed under its configured "
            "decimal policy"
        ) from error
    return tuple(values)


def _native_rsi(
    close_values: tuple[Decimal, ...], parameters: PrimitiveMapping
) -> tuple[IndicatorValue, ...]:
    period_value = parameters.get("period")
    if isinstance(period_value, bool) or not isinstance(period_value, int):
        raise IndicatorCalculationError("native RSI period must be an integer")
    period = period_value
    values: list[IndicatorValue] = [None] * len(close_values)
    if len(close_values) <= period:
        return tuple(values)

    try:
        with localcontext(_arithmetic_context()):
            gains: list[Decimal] = []
            losses: list[Decimal] = []
            for previous, current in pairwise(close_values):
                change = current - previous
                gains.append(max(change, Decimal(0)))
                losses.append(max(-change, Decimal(0)))

            divisor = Decimal(period)
            average_gain = sum(gains[:period], Decimal(0)) / divisor
            average_loss = sum(losses[:period], Decimal(0)) / divisor
            values[period] = _rsi(average_gain, average_loss)

            for index in range(period + 1, len(close_values)):
                gain = gains[index - 1]
                loss = losses[index - 1]
                average_gain = (average_gain * Decimal(period - 1) + gain) / divisor
                average_loss = (average_loss * Decimal(period - 1) + loss) / divisor
                values[index] = _rsi(average_gain, average_loss)
    except DecimalException as error:
        raise IndicatorCalculationError(
            "Wilder RSI arithmetic failed under its configured decimal policy"
        ) from error
    return tuple(values)


def _native_atr(
    high_values: tuple[Decimal, ...],
    low_values: tuple[Decimal, ...],
    close_values: tuple[Decimal, ...],
    parameters: PrimitiveMapping,
) -> tuple[IndicatorValue, ...]:
    period_value = parameters.get("period")
    if isinstance(period_value, bool) or not isinstance(period_value, int):
        raise IndicatorCalculationError("native ATR period must be an integer")
    period = period_value
    values: list[IndicatorValue] = [None] * len(close_values)
    if len(close_values) <= period:
        return tuple(values)

    try:
        with localcontext(_arithmetic_context()):
            true_ranges = tuple(
                max(
                    high_values[index] - low_values[index],
                    abs(high_values[index] - close_values[index - 1]),
                    abs(low_values[index] - close_values[index - 1]),
                )
                for index in range(1, len(close_values))
            )
            divisor = Decimal(period)
            current_atr = sum(true_ranges[:period], Decimal(0)) / divisor
            values[period] = current_atr
            for index in range(period + 1, len(close_values)):
                current_atr = (
                    current_atr * Decimal(period - 1) + true_ranges[index - 1]
                ) / divisor
                values[index] = current_atr
    except DecimalException as error:
        raise IndicatorCalculationError(
            "Wilder ATR arithmetic failed under its configured decimal policy"
        ) from error
    return tuple(values)


def _rsi(average_gain: Decimal, average_loss: Decimal) -> Decimal:
    if average_gain == 0 and average_loss == 0:
        return Decimal(50)
    if average_loss == 0:
        return Decimal(100)
    if average_gain == 0:
        return Decimal(0)
    relative_strength = average_gain / average_loss
    return Decimal(100) - Decimal(100) / (Decimal(1) + relative_strength)


def _arithmetic_context() -> Context:
    return Context(
        prec=_DECIMAL_PRECISION,
        rounding=ROUND_HALF_EVEN,
        Emin=_DECIMAL_EMIN,
        Emax=_DECIMAL_EMAX,
        capitals=1,
        clamp=0,
        flags=[],
        traps=list(_DECIMAL_TRAPS),
    )


__all__ = ["NativeIndicatorBackend"]
