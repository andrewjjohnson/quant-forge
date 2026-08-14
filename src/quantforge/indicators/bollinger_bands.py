"""Full-window, population-standard-deviation Bollinger Bands."""

from collections import deque
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
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.models import MarketDataset
from quantforge.indicators.base import (
    DevelopingBarSupport,
    IndicatorBar,
    IndicatorParameters,
    validate_indicator_alignment,
    validate_indicator_bars,
    validate_market_input,
)
from quantforge.indicators.exceptions import (
    IndicatorCalculationError,
    InvalidIndicatorParametersError,
)
from quantforge.indicators.models import (
    IndicatorFieldOutput,
    IndicatorOutput,
    IndicatorValue,
    MarketField,
)

BOLLINGER_MIDDLE_BAND_OUTPUT = "bollinger_middle_band"
BOLLINGER_UPPER_BAND_OUTPUT = "bollinger_upper_band"
BOLLINGER_LOWER_BAND_OUTPUT = "bollinger_lower_band"
BOLLINGER_BANDWIDTH_OUTPUT = "bollinger_bandwidth"
_DECIMAL_PRECISION = 34
_DECIMAL_ROUNDING = ROUND_HALF_EVEN
_DECIMAL_EMIN = -999_999
_DECIMAL_EMAX = 999_999
_DECIMAL_CAPITALS = 1
_DECIMAL_CLAMP = 0
_EXACT_MOMENT_MAX_COEFFICIENT_DIGITS = _DECIMAL_PRECISION * 2
_EXACT_MOMENT_MAX_SOURCE_INTEGER_DIGITS = 2_048
_DECIMAL_TRAPS: tuple[type[DecimalException], ...] = (
    DivisionByZero,
    InvalidOperation,
    Overflow,
)
_DECIMAL_TRAP_NAMES = tuple(signal.__name__ for signal in _DECIMAL_TRAPS)


@dataclass(frozen=True, slots=True)
class BollingerBandsParameters:
    """Bar period, population-deviation multiplier, and canonical source field."""

    period: int
    standard_deviation_multiplier: Decimal = Decimal(2)
    source_field: MarketField = MarketField.CLOSE

    def __post_init__(self) -> None:
        _validate_period_type(self.period)
        if self.period < 1:
            raise InvalidIndicatorParametersError("period must be at least 1")
        _validate_multiplier(self.standard_deviation_multiplier)
        _validate_source_field(self.source_field)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "period": self.period,
            "source_field": self.source_field.value,
            "standard_deviation_multiplier": decimal_to_primitive(
                self.standard_deviation_multiplier
            ),
        }


class BollingerBands:
    """Trailing mean and population-deviation bands over a complete bar window.

    A full window containing a non-finite source emits ``None`` for every field.
    Bandwidth is ``(upper - lower) / middle``. A zero-width band has bandwidth
    zero, including when the middle is zero; a nonzero width with a zero middle
    has unavailable bandwidth rather than an invented ratio.
    """

    name = "bollinger_bands"
    implementation_version = "7"
    output_fields = (
        BOLLINGER_MIDDLE_BAND_OUTPUT,
        BOLLINGER_UPPER_BAND_OUTPUT,
        BOLLINGER_LOWER_BAND_OUTPUT,
        BOLLINGER_BANDWIDTH_OUTPUT,
    )
    missing_value = None
    developing_bar_support = DevelopingBarSupport.DEVELOPING_AS_OF

    def __init__(self, parameters: BollingerBandsParameters) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> IndicatorParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((self._parameters.source_field,))

    @property
    def warm_up_observations(self) -> int:
        """Consecutive observations required for the first complete window."""
        return self._parameters.period

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_type": "indicator",
            "component_name": self.name,
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": self._parameters.to_primitive(),
            "required_fields": [field.value for field in sorted(self.required_fields)],
            "warm_up_observations": self.warm_up_observations,
            "output_fields": list(self.output_fields),
            "missing_value": None,
            "formula": {
                "middle_band": "sum(window) / period",
                "standard_deviation": (
                    "sqrt(sum((observation - middle_band) ** 2) / period)"
                ),
                "standard_deviation_degrees_of_freedom": 0,
                "window_update": "exact_rational_rolling_sum_and_sum_of_squares",
                "upper_band": (
                    "middle_band + standard_deviation_multiplier * standard_deviation"
                ),
                "lower_band": (
                    "middle_band - standard_deviation_multiplier * standard_deviation"
                ),
                "bandwidth": "(upper_band - lower_band) / middle_band",
                "zero_width_policy": "bandwidth_zero",
                "zero_middle_nonzero_width_policy": "bandwidth_none",
                "missing_value_policy": "full_window_containing_missing_is_none",
            },
            "arithmetic": {
                "decimal_precision": _DECIMAL_PRECISION,
                "rounding": _DECIMAL_ROUNDING,
                "decimal_emin": _DECIMAL_EMIN,
                "decimal_emax": _DECIMAL_EMAX,
                "capitals": _DECIMAL_CAPITALS,
                "clamp": _DECIMAL_CLAMP,
                "initial_flags": [],
                "traps": list(_DECIMAL_TRAP_NAMES),
                "rolling_moment_accumulation": "exact_rational",
                "exact_moment_input_bounds": {
                    "maximum_coefficient_digits": (
                        _EXACT_MOMENT_MAX_COEFFICIENT_DIGITS
                    ),
                    "maximum_source_integer_digits": (
                        _EXACT_MOMENT_MAX_SOURCE_INTEGER_DIGITS
                    ),
                    "maximum_squared_integer_digits": (
                        _EXACT_MOMENT_MAX_SOURCE_INTEGER_DIGITS * 2
                    ),
                    "minimum_stored_exponent": _DECIMAL_EMIN,
                    "maximum_stored_exponent": _DECIMAL_EMAX,
                    "minimum_adjusted_exponent": _DECIMAL_EMIN,
                    "maximum_adjusted_exponent": _DECIMAL_EMAX,
                },
            },
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def calculate(self, dataset: MarketDataset) -> IndicatorOutput:
        """Return causally aligned bands without partial or filled windows."""
        validate_market_input(dataset, self.required_fields)
        fields = self.calculate_bar_fields(cast(tuple[IndicatorBar, ...], dataset.bars))
        output = IndicatorOutput(
            self.name,
            self.configuration_id,
            tuple(bar.session_date for bar in dataset.bars),
            fields,
        )
        validate_indicator_alignment(dataset, output)
        return output

    def calculate_bar_fields(
        self, bars: tuple[IndicatorBar, ...]
    ) -> tuple[IndicatorFieldOutput, ...]:
        """Calculate against canonical bars whose timeframe is validated upstream."""
        validate_indicator_bars(bars, self.required_fields, require_finite=False)
        source = tuple(
            value if value.is_finite() else None
            for bar in bars
            for value in (
                cast(Decimal, getattr(bar, self._parameters.source_field.value)),
            )
        )
        middle_values: list[IndicatorValue] = []
        upper_values: list[IndicatorValue] = []
        lower_values: list[IndicatorValue] = []
        bandwidth_values: list[IndicatorValue] = []
        period = self._parameters.period
        multiplier = self._parameters.standard_deviation_multiplier
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
                            previous_sum_of_squares=cast(
                                Fraction, rolling_sum_of_squares
                            ),
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
            IndicatorFieldOutput(BOLLINGER_MIDDLE_BAND_OUTPUT, tuple(middle_values)),
            IndicatorFieldOutput(BOLLINGER_UPPER_BAND_OUTPUT, tuple(upper_values)),
            IndicatorFieldOutput(BOLLINGER_LOWER_BAND_OUTPUT, tuple(lower_values)),
            IndicatorFieldOutput(BOLLINGER_BANDWIDTH_OUTPUT, tuple(bandwidth_values)),
        )


def _rebuild_window_moments(
    observations: tuple[Decimal, ...],
) -> tuple[Fraction, Fraction]:
    """Establish exact moments after warm-up or a missing-value gap."""
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
    """Replace one observation in the exact moments without rescanning the window."""
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
    """Round exact window moments once at the declared Decimal boundary."""
    divisor = Fraction(period)
    exact_middle = total / divisor
    exact_population_variance = sum_of_squares / divisor - exact_middle * exact_middle
    return (
        _fraction_to_decimal(exact_middle),
        _fraction_to_decimal(exact_population_variance),
    )


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _bounded_fraction(value: Decimal) -> Fraction:
    """Reject resource-unbounded Decimal encodings before exact conversion."""
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
    """Bound either integer component of the exact source fraction."""
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


def _arithmetic_context() -> Context:
    """Build the complete deterministic Decimal policy for Bollinger arithmetic."""
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


def _validate_period_type(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidIndicatorParametersError("period must be an integer")


def _validate_multiplier(value: object) -> None:
    if not isinstance(value, Decimal):
        raise InvalidIndicatorParametersError(
            "standard_deviation_multiplier must be a Decimal"
        )
    if not value.is_finite() or value <= Decimal(0):
        raise InvalidIndicatorParametersError(
            "standard_deviation_multiplier must be finite and greater than zero"
        )


def _validate_source_field(value: object) -> None:
    if not isinstance(value, MarketField):
        raise InvalidIndicatorParametersError(
            "source_field must be a normalized market field"
        )


__all__ = [
    "BOLLINGER_BANDWIDTH_OUTPUT",
    "BOLLINGER_LOWER_BAND_OUTPUT",
    "BOLLINGER_MIDDLE_BAND_OUTPUT",
    "BOLLINGER_UPPER_BAND_OUTPUT",
    "BollingerBands",
    "BollingerBandsParameters",
]
