"""Configurable causal contextual features for QF-7 snapshots."""

from dataclasses import dataclass
from decimal import Decimal, DecimalException
from typing import Protocol

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data.lineage import FeedScope
from quantforge.data.models import MarketDataset
from quantforge.indicators import (
    RELATIVE_VOLUME_OUTPUT,
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    WILDER_AVERAGE_TRUE_RANGE_OUTPUT,
    MarketField,
    RelativeVolume,
    RelativeVolumeDenominatorPolicy,
    RelativeVolumeParameters,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
    WilderAverageTrueRange,
    WilderAverageTrueRangeParameters,
)
from quantforge.prediction._arithmetic import arithmetic
from quantforge.prediction.errors import InvalidPredictionConfigurationError
from quantforge.prediction.signal_feature_models import SchemaField, SchemaFieldCategory


class ContextualFeature(Protocol):
    """One independently configured causal feature with a prefix fallback."""

    @property
    def name(self) -> str: ...

    @property
    def definition(self) -> SchemaField: ...

    @property
    def configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...

    def value_from_history(self, history: MarketDataset) -> Decimal | None: ...


@dataclass(frozen=True, slots=True)
class AtrPercentageContext:
    """Wilder ATR divided by the completed signal-session close."""

    period: int = 14

    def __post_init__(self) -> None:
        _validate_period("period", self.period)

    @property
    def name(self) -> str:
        return "atr_percentage_of_close"

    @property
    def definition(self) -> SchemaField:
        return SchemaField(
            self.name,
            SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
            "decimal",
            "ratio",
            True,
            f"QF-4 Wilder ATR({self.period}) / completed-session close",
            "available after the signal-session close; trailing history only",
        )

    def configuration(self) -> PrimitiveMapping:
        indicator = WilderAverageTrueRange(
            WilderAverageTrueRangeParameters(self.period)
        )
        return _configuration(self.name, indicator.configuration())

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def value_from_history(self, history: MarketDataset) -> Decimal | None:
        return self.values_for_dataset(history)[-1]

    def values_for_dataset(self, dataset: MarketDataset) -> tuple[Decimal | None, ...]:
        indicator = WilderAverageTrueRange(
            WilderAverageTrueRangeParameters(self.period)
        )
        atr_values = indicator.calculate(dataset).values_for(
            WILDER_AVERAGE_TRUE_RANGE_OUTPUT
        )
        values: list[Decimal | None] = []
        for bar, atr in zip(dataset.bars, atr_values, strict=True):
            if atr is None:
                values.append(None)
                continue
            try:
                with arithmetic():
                    values.append(atr / bar.close)
            except DecimalException as error:
                raise InvalidPredictionConfigurationError(
                    "ATR percentage feature arithmetic failed"
                ) from error
        return tuple(values)


@dataclass(frozen=True, slots=True)
class VolumeRatioContext:
    """Current completed-session volume divided by its trailing mean."""

    period: int = 20

    def __post_init__(self) -> None:
        _validate_period("period", self.period)

    @property
    def name(self) -> str:
        return "volume_ratio"

    @property
    def definition(self) -> SchemaField:
        return SchemaField(
            self.name,
            SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
            "decimal",
            "ratio",
            True,
            "completed-session volume / QF-4 trailing inclusive "
            f"mean({self.period}) volume",
            "available after the signal-session close; trailing history only",
        )

    def configuration(self) -> PrimitiveMapping:
        indicator = RelativeVolume(
            RelativeVolumeParameters(
                self.period,
                FeedScope.unknown(),
                RelativeVolumeDenominatorPolicy.INCLUDE_CURRENT_BAR,
            )
        )
        return _configuration(self.name, indicator.configuration(), version="3")

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def value_from_history(self, history: MarketDataset) -> Decimal | None:
        return self.values_for_dataset(history)[-1]

    def values_for_dataset(self, dataset: MarketDataset) -> tuple[Decimal | None, ...]:
        indicator = RelativeVolume(
            RelativeVolumeParameters(
                self.period,
                FeedScope.unknown(),
                RelativeVolumeDenominatorPolicy.INCLUDE_CURRENT_BAR,
            )
        )
        return indicator.calculate(dataset).values_for(RELATIVE_VOLUME_OUTPUT)


@dataclass(frozen=True, slots=True)
class TrendDistanceContext:
    """Completed close's percentage distance from a trailing close SMA."""

    period: int = 20

    def __post_init__(self) -> None:
        _validate_period("period", self.period)

    @property
    def name(self) -> str:
        return "trend_distance_percentage"

    @property
    def definition(self) -> SchemaField:
        return SchemaField(
            self.name,
            SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
            "decimal",
            "ratio",
            True,
            f"completed close / QF-4 trailing SMA({self.period}) close - 1",
            "available after the signal-session close; trailing history only",
        )

    def configuration(self) -> PrimitiveMapping:
        indicator = SimpleMovingAverage(
            SimpleMovingAverageParameters(self.period, MarketField.CLOSE)
        )
        return _configuration(self.name, indicator.configuration())

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def value_from_history(self, history: MarketDataset) -> Decimal | None:
        return self.values_for_dataset(history)[-1]

    def values_for_dataset(self, dataset: MarketDataset) -> tuple[Decimal | None, ...]:
        indicator = SimpleMovingAverage(
            SimpleMovingAverageParameters(self.period, MarketField.CLOSE)
        )
        averages = indicator.calculate(dataset).values_for(SIMPLE_MOVING_AVERAGE_OUTPUT)
        values: list[Decimal | None] = []
        for bar, average in zip(dataset.bars, averages, strict=True):
            if average in (None, Decimal(0)):
                values.append(None)
                continue
            try:
                with arithmetic():
                    values.append(bar.close / average - Decimal(1))
            except DecimalException as error:
                raise InvalidPredictionConfigurationError(
                    "trend-distance feature arithmetic failed"
                ) from error
        return tuple(values)


def default_overnight_gap_contextual_features() -> tuple[ContextualFeature, ...]:
    """Return the three documented unused baseline research features."""
    return (
        AtrPercentageContext(),
        TrendDistanceContext(),
        VolumeRatioContext(),
    )


def _configuration(
    name: str, indicator: PrimitiveMapping, *, version: str = "2"
) -> PrimitiveMapping:
    return {
        "component_name": name,
        "component_type": "signal_contextual_feature",
        "contract_version": "1",
        "implementation_version": version,
        "indicator": indicator,
        "timing": "completed_signal_session_and_trailing_history_only",
    }


def _validate_period(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidPredictionConfigurationError(
            f"contextual feature {name} must be a positive integer"
        )
