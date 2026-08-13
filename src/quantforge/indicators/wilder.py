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
from itertools import pairwise

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

    def __init__(self, parameters: WilderRelativeStrengthIndexParameters) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> IndicatorParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((MarketField.CLOSE,))

    @property
    def warm_up_observations(self) -> int:
        return self._parameters.period + 1

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
        """Calculate RSI against any validated canonical timeframe series."""
        validate_indicator_bars(bars, self.required_fields)
        values: list[IndicatorValue] = [None] * len(bars)
        period = self._parameters.period
        if len(bars) <= period:
            return (IndicatorFieldOutput(WILDER_RSI_OUTPUT, tuple(values)),)

        context = _arithmetic_context()
        try:
            with localcontext(context):
                gains: list[Decimal] = []
                losses: list[Decimal] = []
                for index in range(1, len(bars)):
                    change = bars[index].close - bars[index - 1].close
                    gains.append(max(change, Decimal(0)))
                    losses.append(max(-change, Decimal(0)))

                divisor = Decimal(period)
                average_gain = sum(gains[:period], Decimal(0)) / divisor
                average_loss = sum(losses[:period], Decimal(0)) / divisor
                values[period] = _rsi(average_gain, average_loss)

                for index in range(period + 1, len(bars)):
                    gain = gains[index - 1]
                    loss = losses[index - 1]
                    average_gain = (average_gain * Decimal(period - 1) + gain) / divisor
                    average_loss = (average_loss * Decimal(period - 1) + loss) / divisor
                    values[index] = _rsi(average_gain, average_loss)
        except DecimalException as error:
            raise IndicatorCalculationError(
                "Wilder RSI arithmetic failed under its configured decimal policy"
            ) from error

        return (IndicatorFieldOutput(WILDER_RSI_OUTPUT, tuple(values)),)


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

    def __init__(self, parameters: WilderAverageTrueRangeParameters) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> IndicatorParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((MarketField.HIGH, MarketField.LOW, MarketField.CLOSE))

    @property
    def warm_up_observations(self) -> int:
        return self._parameters.period + 1

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
        """Calculate ATR against any validated canonical timeframe series."""
        validate_indicator_bars(bars, self.required_fields)
        values: list[IndicatorValue] = [None] * len(bars)
        period = self._parameters.period
        if len(bars) <= period:
            return (
                IndicatorFieldOutput(WILDER_AVERAGE_TRUE_RANGE_OUTPUT, tuple(values)),
            )

        context = _arithmetic_context()
        try:
            with localcontext(context):
                true_ranges = tuple(
                    max(
                        current.high - current.low,
                        abs(current.high - previous.close),
                        abs(current.low - previous.close),
                    )
                    for previous, current in pairwise(bars)
                )
                divisor = Decimal(period)
                current_atr = sum(true_ranges[:period], Decimal(0)) / divisor
                values[period] = current_atr
                for index in range(period + 1, len(bars)):
                    current_atr = (
                        current_atr * Decimal(period - 1) + true_ranges[index - 1]
                    ) / divisor
                    values[index] = current_atr
        except DecimalException as error:
            raise IndicatorCalculationError(
                "Wilder ATR arithmetic failed under its configured decimal policy"
            ) from error
        return (IndicatorFieldOutput(WILDER_AVERAGE_TRUE_RANGE_OUTPUT, tuple(values)),)


def _rsi(average_gain: Decimal, average_loss: Decimal) -> Decimal:
    if average_gain == 0 and average_loss == 0:
        return Decimal(50)
    if average_loss == 0:
        return _ONE_HUNDRED
    if average_gain == 0:
        return Decimal(0)
    relative_strength = average_gain / average_loss
    return _ONE_HUNDRED - _ONE_HUNDRED / (Decimal(1) + relative_strength)


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
