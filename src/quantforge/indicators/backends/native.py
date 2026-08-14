"""Adapter for historical QuantForge-native standard indicator mathematics."""

from collections import deque
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
from fractions import Fraction
from itertools import pairwise
from typing import cast

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
_DIRECTIONAL_MOVEMENT_NAME = "wilder_directional_movement"
_BOLLINGER_NAME = "bollinger_bands"
_DECIMAL_PRECISION = 34
_DECIMAL_EMIN = -999_999
_DECIMAL_EMAX = 999_999
_EXACT_MOMENT_MAX_COEFFICIENT_DIGITS = _DECIMAL_PRECISION * 2
_EXACT_MOMENT_MAX_SOURCE_INTEGER_DIGITS = 2_048
_MULTIPLIER_MAX_COEFFICIENT_DIGITS = _DECIMAL_PRECISION * 2
_MULTIPLIER_MAX_FIXED_POINT_CHARACTERS = 256
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
    _DIRECTIONAL_MOVEMENT_NAME: _NativeMapping(
        function_name="quantforge_native_wilder_directional_movement",
        implementation_version="1",
        parameter_names=frozenset(("period",)),
        input_fields=("high", "low", "close"),
        source_field_parameter=None,
        output_names=(
            "positive_directional_indicator",
            "negative_directional_indicator",
            "average_directional_index",
        ),
        validate_parameters=lambda parameters: _validate_positive_period(
            parameters, "period", "directional movement"
        ),
        function=lambda inputs, parameters: _native_directional_movement(
            inputs[0], inputs[1], inputs[2], parameters
        ),
    ),
    _BOLLINGER_NAME: _NativeMapping(
        function_name="quantforge_native_bollinger_bands",
        implementation_version="8",
        parameter_names=frozenset(
            ("period", "source_field", "standard_deviation_multiplier")
        ),
        input_fields=None,
        source_field_parameter="source_field",
        output_names=(
            "bollinger_middle_band",
            "bollinger_upper_band",
            "bollinger_lower_band",
            "bollinger_bandwidth",
        ),
        validate_parameters=lambda parameters: _validate_bollinger_parameters(
            parameters
        ),
        function=lambda inputs, parameters: _native_bollinger_bands(
            inputs[0], parameters
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


def _validate_bollinger_parameters(parameters: PrimitiveMapping) -> None:
    _validate_positive_period(parameters, "period", "Bollinger Bands")
    raw_multiplier = parameters.get("standard_deviation_multiplier")
    if not isinstance(raw_multiplier, str):
        raise UnsupportedIndicatorBackendError(
            f"{NATIVE_INDICATOR_BACKEND} Bollinger Bands multiplier must be a "
            "normalized decimal string"
        )
    try:
        multiplier = Decimal(raw_multiplier)
    except InvalidOperation as error:
        raise UnsupportedIndicatorBackendError(
            f"{NATIVE_INDICATOR_BACKEND} Bollinger Bands multiplier is invalid"
        ) from error
    _, coefficient_digits, stored_exponent = multiplier.as_tuple()
    if (
        not multiplier.is_finite()
        or multiplier <= 0
        or not isinstance(stored_exponent, int)
        or len(coefficient_digits) > _MULTIPLIER_MAX_COEFFICIENT_DIGITS
        or _fixed_point_render_length(coefficient_digits, stored_exponent)
        > _MULTIPLIER_MAX_FIXED_POINT_CHARACTERS
    ):
        raise UnsupportedIndicatorBackendError(
            f"{NATIVE_INDICATOR_BACKEND} Bollinger Bands multiplier is outside "
            "supported bounds"
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


def _native_directional_movement(
    high_values: tuple[Decimal, ...],
    low_values: tuple[Decimal, ...],
    close_values: tuple[Decimal, ...],
    parameters: PrimitiveMapping,
) -> tuple[tuple[IndicatorValue, ...], ...]:
    period_value = parameters.get("period")
    if isinstance(period_value, bool) or not isinstance(period_value, int):
        raise IndicatorCalculationError(
            "native directional-movement period must be an integer"
        )
    period = period_value
    count = len(close_values)
    positive_di: list[IndicatorValue] = [None] * count
    negative_di: list[IndicatorValue] = [None] * count
    adx: list[IndicatorValue] = [None] * count
    if count <= period:
        return tuple(positive_di), tuple(negative_di), tuple(adx)

    try:
        with localcontext(_arithmetic_context()):
            true_ranges: list[Decimal] = []
            positive_movements: list[Decimal] = []
            negative_movements: list[Decimal] = []
            for index in range(1, count):
                true_ranges.append(
                    max(
                        high_values[index] - low_values[index],
                        abs(high_values[index] - close_values[index - 1]),
                        abs(low_values[index] - close_values[index - 1]),
                    )
                )
                upward_move = high_values[index] - high_values[index - 1]
                downward_move = low_values[index - 1] - low_values[index]
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
            divisor = Decimal(period)

            for index in range(period, count):
                if index > period:
                    movement_index = index - 1
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

    return tuple(positive_di), tuple(negative_di), tuple(adx)


def _di_and_dx(
    smoothed_true_range: Decimal,
    smoothed_positive: Decimal,
    smoothed_negative: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    if smoothed_true_range == 0:
        return Decimal(0), Decimal(0), Decimal(0)
    positive = Decimal(100) * smoothed_positive / smoothed_true_range
    negative = Decimal(100) * smoothed_negative / smoothed_true_range
    denominator = positive + negative
    dx = (
        Decimal(0)
        if denominator == 0
        else Decimal(100) * abs(positive - negative) / denominator
    )
    return positive, negative, dx


def _native_bollinger_bands(
    source_values: tuple[Decimal, ...], parameters: PrimitiveMapping
) -> tuple[tuple[IndicatorValue, ...], ...]:
    period_value = parameters.get("period")
    multiplier_value = parameters.get("standard_deviation_multiplier")
    if isinstance(period_value, bool) or not isinstance(period_value, int):
        raise IndicatorCalculationError(
            "native Bollinger Bands period must be an integer"
        )
    if not isinstance(multiplier_value, str):
        raise IndicatorCalculationError(
            "native Bollinger Bands multiplier must be a decimal string"
        )
    period = period_value
    multiplier = Decimal(multiplier_value)
    source = tuple(value if value.is_finite() else None for value in source_values)
    middle_values: list[IndicatorValue] = []
    upper_values: list[IndicatorValue] = []
    lower_values: list[IndicatorValue] = []
    bandwidth_values: list[IndicatorValue] = []
    window: deque[Decimal | None] = deque()
    missing_count = 0
    statistics_are_valid = False
    rolling_sum: Fraction | None = None
    rolling_sum_of_squares: Fraction | None = None

    try:
        with localcontext(_arithmetic_context()):
            for current in source:
                window_was_full = len(window) == period
                outgoing = window.popleft() if window_was_full else None
                if window_was_full and outgoing is None:
                    missing_count -= 1
                window.append(current)
                if current is None:
                    missing_count += 1

                if len(window) < period or missing_count:
                    statistics_are_valid = False
                    _append_unavailable(
                        middle_values,
                        upper_values,
                        lower_values,
                        bandwidth_values,
                    )
                    continue

                current_value = cast(Decimal, current)
                if statistics_are_valid and window_was_full:
                    rolling_sum, rolling_sum_of_squares = _roll_window_moments(
                        previous_sum=cast(Fraction, rolling_sum),
                        previous_sum_of_squares=cast(Fraction, rolling_sum_of_squares),
                        outgoing=cast(Decimal, outgoing),
                        incoming=current_value,
                    )
                else:
                    rolling_sum, rolling_sum_of_squares = _rebuild_window_moments(
                        cast(tuple[Decimal, ...], tuple(window))
                    )
                statistics_are_valid = True
                middle, population_variance = _decimal_window_statistics(
                    total=rolling_sum,
                    sum_of_squares=rolling_sum_of_squares,
                    period=period,
                )
                _append_bands(
                    middle=middle,
                    population_variance=population_variance,
                    multiplier=multiplier,
                    middle_values=middle_values,
                    upper_values=upper_values,
                    lower_values=lower_values,
                    bandwidth_values=bandwidth_values,
                )
    except DecimalException as error:
        raise IndicatorCalculationError(
            "Bollinger Bands arithmetic failed under its configured decimal policy"
        ) from error

    return (
        tuple(middle_values),
        tuple(upper_values),
        tuple(lower_values),
        tuple(bandwidth_values),
    )


def _rebuild_window_moments(
    observations: tuple[Decimal, ...],
) -> tuple[Fraction, Fraction]:
    total = Fraction()
    sum_of_squares = Fraction()
    for observation in observations:
        exact_observation = _bounded_fraction(observation)
        total += exact_observation
        sum_of_squares += exact_observation * exact_observation
    return total, sum_of_squares


def _roll_window_moments(
    *,
    previous_sum: Fraction,
    previous_sum_of_squares: Fraction,
    outgoing: Decimal,
    incoming: Decimal,
) -> tuple[Fraction, Fraction]:
    outgoing_fraction = _bounded_fraction(outgoing)
    incoming_fraction = _bounded_fraction(incoming)
    return (
        previous_sum - outgoing_fraction + incoming_fraction,
        previous_sum_of_squares
        - outgoing_fraction * outgoing_fraction
        + incoming_fraction * incoming_fraction,
    )


def _decimal_window_statistics(
    *, total: Fraction, sum_of_squares: Fraction, period: int
) -> tuple[Decimal, Decimal]:
    divisor = Fraction(period)
    exact_middle = total / divisor
    exact_population_variance = sum_of_squares / divisor - exact_middle * exact_middle
    return (
        Decimal(exact_middle.numerator) / Decimal(exact_middle.denominator),
        Decimal(exact_population_variance.numerator)
        / Decimal(exact_population_variance.denominator),
    )


def _bounded_fraction(value: Decimal) -> Fraction:
    _, coefficient_digits, stored_exponent = value.as_tuple()
    if not isinstance(stored_exponent, int):
        raise IndicatorCalculationError(
            "Bollinger Bands exact moments require a finite Decimal source"
        )
    adjusted_exponent = value.adjusted()
    if (
        len(coefficient_digits) > _EXACT_MOMENT_MAX_COEFFICIENT_DIGITS
        or _source_integer_digit_bound(coefficient_digits, stored_exponent)
        > _EXACT_MOMENT_MAX_SOURCE_INTEGER_DIGITS
        or stored_exponent < _DECIMAL_EMIN
        or stored_exponent > _DECIMAL_EMAX
        or adjusted_exponent < _DECIMAL_EMIN
        or adjusted_exponent > _DECIMAL_EMAX
    ):
        raise IndicatorCalculationError(
            "Bollinger Bands source exceeds exact-moment resource bounds"
        )
    return Fraction(value)


def _source_integer_digit_bound(
    coefficient_digits: tuple[int, ...], stored_exponent: int
) -> int:
    if not any(coefficient_digits):
        return 1
    trailing_zero_count = 0
    for digit in reversed(coefficient_digits):
        if digit:
            break
        trailing_zero_count += 1
    normalized_digit_count = len(coefficient_digits) - trailing_zero_count
    normalized_exponent = stored_exponent + trailing_zero_count
    numerator_digits = normalized_digit_count + max(normalized_exponent, 0)
    denominator_digits = 1 + max(-normalized_exponent, 0)
    return max(numerator_digits, denominator_digits)


def _append_bands(
    *,
    middle: Decimal,
    population_variance: Decimal,
    multiplier: Decimal,
    middle_values: list[IndicatorValue],
    upper_values: list[IndicatorValue],
    lower_values: list[IndicatorValue],
    bandwidth_values: list[IndicatorValue],
) -> None:
    standard_deviation = population_variance.sqrt()
    offset = multiplier * standard_deviation
    upper = middle + offset
    lower = middle - offset
    width = upper - lower
    if width.is_zero():
        bandwidth: IndicatorValue = Decimal(0)
    elif middle.is_zero():
        bandwidth = None
    else:
        bandwidth = width / middle
    middle_values.append(middle)
    upper_values.append(upper)
    lower_values.append(lower)
    bandwidth_values.append(bandwidth)


def _append_unavailable(*outputs: list[IndicatorValue]) -> None:
    for output in outputs:
        output.append(None)


def _fixed_point_render_length(
    coefficient_digits: tuple[int, ...], stored_exponent: int
) -> int:
    coefficient_length = len(coefficient_digits)
    if stored_exponent >= 0:
        return coefficient_length + stored_exponent
    decimal_point_position = coefficient_length + stored_exponent
    if decimal_point_position > 0:
        return coefficient_length + 1
    return 2 - decimal_point_position + coefficient_length


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
