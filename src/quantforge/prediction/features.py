"""Causal typed feature derivation for prediction-study summaries."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, DecimalException

from quantforge.configuration import PrimitiveMapping, decimal_to_primitive
from quantforge.data.models import MarketDataset
from quantforge.indicators import (
    AVERAGE_DIRECTIONAL_INDEX_OUTPUT,
    NEGATIVE_DIRECTIONAL_INDICATOR_OUTPUT,
    POSITIVE_DIRECTIONAL_INDICATOR_OUTPUT,
    WILDER_RSI_OUTPUT,
    WilderDirectionalMovement,
    WilderDirectionalMovementParameters,
    WilderRelativeStrengthIndex,
    WilderRelativeStrengthIndexParameters,
)
from quantforge.prediction._arithmetic import arithmetic
from quantforge.prediction.errors import InvalidPredictionConfigurationError


@dataclass(frozen=True, slots=True)
class DerivedFeatureParameters:
    """Causal periods used only for post-signal explanatory features."""

    rsi_period: int = 2
    adx_period: int = 5
    atr_period: int = 14
    average_volume_period: int = 20

    def __post_init__(self) -> None:
        for name in (
            "rsi_period",
            "adx_period",
            "atr_period",
            "average_volume_period",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise InvalidPredictionConfigurationError(
                    f"{name} must be a positive integer"
                )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "adx_period": self.adx_period,
            "atr_period": self.atr_period,
            "average_volume_period": self.average_volume_period,
            "rsi_period": self.rsi_period,
        }


@dataclass(frozen=True, slots=True)
class DerivedFeatureRow:
    """Typed values available after one completed signal session."""

    signal_session: date
    signal_weekday: int
    open: Decimal
    close: Decimal
    volume: Decimal
    rsi: Decimal | None
    previous_rsi: Decimal | None
    rsi_change: Decimal | None
    adx: Decimal | None
    previous_adx: Decimal | None
    adx_change: Decimal | None
    positive_di: Decimal | None
    negative_di: Decimal | None
    di_spread: Decimal | None
    previous_di_spread: Decimal | None
    atr: Decimal | None
    atr_percentage_of_close: Decimal | None
    candle_return: Decimal
    average_volume: Decimal | None
    volume_ratio: Decimal | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "adx": _optional(self.adx),
            "adx_change": _optional(self.adx_change),
            "atr": _optional(self.atr),
            "atr_percentage_of_close": _optional(self.atr_percentage_of_close),
            "average_volume": _optional(self.average_volume),
            "candle_return": decimal_to_primitive(self.candle_return),
            "close": decimal_to_primitive(self.close),
            "di_spread": _optional(self.di_spread),
            "negative_di": _optional(self.negative_di),
            "open": decimal_to_primitive(self.open),
            "positive_di": _optional(self.positive_di),
            "previous_adx": _optional(self.previous_adx),
            "previous_di_spread": _optional(self.previous_di_spread),
            "previous_rsi": _optional(self.previous_rsi),
            "rsi": _optional(self.rsi),
            "rsi_change": _optional(self.rsi_change),
            "signal_session": self.signal_session.isoformat(),
            "signal_weekday": self.signal_weekday,
            "volume": decimal_to_primitive(self.volume),
            "volume_ratio": _optional(self.volume_ratio),
        }


def derive_completed_session_features(
    dataset: MarketDataset,
    parameters: DerivedFeatureParameters = DerivedFeatureParameters(),
) -> tuple[DerivedFeatureRow, ...]:
    """Derive aligned feature rows without mutating or reading future bars."""
    rsi = WilderRelativeStrengthIndex(
        WilderRelativeStrengthIndexParameters(parameters.rsi_period)
    ).calculate(dataset)
    dmi = WilderDirectionalMovement(
        WilderDirectionalMovementParameters(parameters.adx_period)
    ).calculate(dataset)
    rsi_values = rsi.values_for(WILDER_RSI_OUTPUT)
    positive_di_values = dmi.values_for(POSITIVE_DIRECTIONAL_INDICATOR_OUTPUT)
    negative_di_values = dmi.values_for(NEGATIVE_DIRECTIONAL_INDICATOR_OUTPUT)
    adx_values = dmi.values_for(AVERAGE_DIRECTIONAL_INDEX_OUTPUT)
    atr_values = _wilder_atr(dataset, parameters.atr_period)
    average_volumes = _trailing_average_volume(
        dataset, parameters.average_volume_period
    )
    rows: list[DerivedFeatureRow] = []
    try:
        with arithmetic():
            for index, bar in enumerate(dataset.bars):
                current_rsi = rsi_values[index]
                previous_rsi = rsi_values[index - 1] if index else None
                current_adx = adx_values[index]
                previous_adx = adx_values[index - 1] if index else None
                positive_di = positive_di_values[index]
                negative_di = negative_di_values[index]
                previous_positive_di = positive_di_values[index - 1] if index else None
                previous_negative_di = negative_di_values[index - 1] if index else None
                di_spread = _difference(positive_di, negative_di)
                previous_di_spread = _difference(
                    previous_positive_di, previous_negative_di
                )
                atr = atr_values[index]
                average_volume = average_volumes[index]
                rows.append(
                    DerivedFeatureRow(
                        signal_session=bar.session_date,
                        signal_weekday=bar.session_date.weekday(),
                        open=bar.open,
                        close=bar.close,
                        volume=bar.volume,
                        rsi=current_rsi,
                        previous_rsi=previous_rsi,
                        rsi_change=_difference(current_rsi, previous_rsi),
                        adx=current_adx,
                        previous_adx=previous_adx,
                        adx_change=_difference(current_adx, previous_adx),
                        positive_di=positive_di,
                        negative_di=negative_di,
                        di_spread=di_spread,
                        previous_di_spread=previous_di_spread,
                        atr=atr,
                        atr_percentage_of_close=(
                            None if atr is None else atr / bar.close
                        ),
                        candle_return=bar.close / bar.open - Decimal(1),
                        average_volume=average_volume,
                        volume_ratio=(
                            None
                            if average_volume in (None, Decimal(0))
                            else bar.volume / average_volume
                        ),
                    )
                )
    except DecimalException as error:
        raise InvalidPredictionConfigurationError(
            "derived feature arithmetic failed"
        ) from error
    return tuple(rows)


def _wilder_atr(dataset: MarketDataset, period: int) -> tuple[Decimal | None, ...]:
    values: list[Decimal | None] = [None] * len(dataset.bars)
    if len(dataset.bars) <= period:
        return tuple(values)
    with arithmetic():
        true_ranges = tuple(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
            for previous, current in zip(
                dataset.bars[:-1], dataset.bars[1:], strict=True
            )
        )
        divisor = Decimal(period)
        current_atr = sum(true_ranges[:period], Decimal(0)) / divisor
        values[period] = current_atr
        for index in range(period + 1, len(dataset.bars)):
            current_atr = (
                current_atr * Decimal(period - 1) + true_ranges[index - 1]
            ) / divisor
            values[index] = current_atr
    return tuple(values)


def _trailing_average_volume(
    dataset: MarketDataset, period: int
) -> tuple[Decimal | None, ...]:
    values: list[Decimal | None] = [None] * len(dataset.bars)
    if len(dataset.bars) < period:
        return tuple(values)
    with arithmetic():
        rolling = sum((bar.volume for bar in dataset.bars[:period]), Decimal(0))
        divisor = Decimal(period)
        values[period - 1] = rolling / divisor
        for index in range(period, len(dataset.bars)):
            rolling += dataset.bars[index].volume
            rolling -= dataset.bars[index - period].volume
            values[index] = rolling / divisor
    return tuple(values)


def _difference(first: Decimal | None, second: Decimal | None) -> Decimal | None:
    return None if first is None or second is None else first - second


def _optional(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_primitive(value)
