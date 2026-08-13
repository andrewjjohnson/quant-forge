"""Full-window, population-standard-deviation Bollinger Bands."""

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
    implementation_version = "1"
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
        divisor = Decimal(period)
        multiplier = self._parameters.standard_deviation_multiplier

        try:
            with localcontext(_arithmetic_context()):
                for index in range(len(source)):
                    if index + 1 < period:
                        _append_unavailable(
                            middle_values,
                            upper_values,
                            lower_values,
                            bandwidth_values,
                        )
                        continue
                    window = source[index + 1 - period : index + 1]
                    if any(observation is None for observation in window):
                        _append_unavailable(
                            middle_values,
                            upper_values,
                            lower_values,
                            bandwidth_values,
                        )
                        continue
                    observations = cast(tuple[Decimal, ...], window)
                    middle = sum(observations, Decimal(0)) / divisor
                    variance = (
                        sum(
                            (
                                (observation - middle) * (observation - middle)
                                for observation in observations
                            ),
                            Decimal(0),
                        )
                        / divisor
                    )
                    standard_deviation = variance.sqrt()
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
