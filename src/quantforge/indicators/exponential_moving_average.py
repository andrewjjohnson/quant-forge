"""SMA-seeded, causal exponential moving average."""

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
from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
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

EXPONENTIAL_MOVING_AVERAGE_OUTPUT = "exponential_moving_average"
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


@dataclass(frozen=True, slots=True)
class ExponentialMovingAverageParameters:
    """Bar period and canonical OHLCV source for an EMA."""

    period: int
    source_field: MarketField = MarketField.CLOSE

    def __post_init__(self) -> None:
        _validate_period_type(self.period)
        if self.period < 1:
            raise InvalidIndicatorParametersError("period must be at least 1")
        _validate_source_field(self.source_field)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "period": self.period,
            "source_field": self.source_field.value,
        }


class ExponentialMovingAverage:
    """Recursive EMA initialized by a full-period simple average.

    The smoothing factor is ``2 / (period + 1)``. A missing source value emits
    ``None`` and restarts initialization, so the calculation never carries a
    stale value across a data gap.
    """

    name = "exponential_moving_average"
    implementation_version = "1"
    output_fields = (EXPONENTIAL_MOVING_AVERAGE_OUTPUT,)
    missing_value = None
    developing_bar_support = DevelopingBarSupport.DEVELOPING_AS_OF

    def __init__(self, parameters: ExponentialMovingAverageParameters) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> IndicatorParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((self._parameters.source_field,))

    @property
    def warm_up_observations(self) -> int:
        """Consecutive observations required to seed or reseed the EMA."""
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
                "smoothing_factor": "2 / (period + 1)",
                "initialization": (
                    "simple_average_of_first_period_consecutive_observations"
                ),
                "recurrence": (
                    "((period - 1) * previous + 2 * current) / (period + 1)"
                ),
                "missing_value_policy": "emit_none_and_restart_initialization",
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
            },
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def calculate(self, dataset: MarketDataset) -> IndicatorOutput:
        """Return one causally aligned EMA value or ``None`` per daily bar."""
        validate_market_input(dataset, self.required_fields)
        fields = self.calculate_bar_fields(dataset.bars)
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
        """Calculate against any upstream-validated canonical timeframe bars."""
        validate_indicator_bars(bars, self.required_fields, require_finite=False)
        period = self._parameters.period
        values: list[IndicatorValue] = []
        seed: list[Decimal] = []
        previous: Decimal | None = None
        context = _arithmetic_context()

        try:
            with localcontext(context):
                two = Decimal(2)
                recurrence_previous_weight = Decimal(period - 1)
                recurrence_divisor = Decimal(period + 1)
                seed_divisor = Decimal(period)
                for bar in bars:
                    source = cast(
                        Decimal, getattr(bar, self._parameters.source_field.value)
                    )
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
                            recurrence_previous_weight * previous + two * source
                        ) / recurrence_divisor
                    values.append(previous)
        except DecimalException as error:
            raise IndicatorCalculationError(
                "exponential moving average arithmetic failed under its "
                "configured decimal policy"
            ) from error

        return (IndicatorFieldOutput(EXPONENTIAL_MOVING_AVERAGE_OUTPUT, tuple(values)),)


def _arithmetic_context() -> Context:
    """Build the complete deterministic Decimal policy for EMA arithmetic."""
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


def _validate_source_field(value: object) -> None:
    if not isinstance(value, MarketField):
        raise InvalidIndicatorParametersError(
            "source_field must be a normalized market field"
        )


__all__ = [
    "EXPONENTIAL_MOVING_AVERAGE_OUTPUT",
    "ExponentialMovingAverage",
    "ExponentialMovingAverageParameters",
]
