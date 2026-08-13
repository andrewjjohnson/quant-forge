"""Reusable indicator boundary and canonical input validation."""

from decimal import Decimal
from enum import StrEnum
from itertools import pairwise
from typing import Protocol

from quantforge.configuration import PrimitiveMapping
from quantforge.data.models import MarketDataset
from quantforge.indicators.exceptions import (
    MisalignedIndicatorOutputError,
    MissingMarketFieldError,
    UnorderedMarketDataError,
)
from quantforge.indicators.models import (
    IndicatorFieldOutput,
    IndicatorOutput,
    MarketField,
)


class IndicatorBar(Protocol):
    """OHLCV observation accepted by timeframe-neutral indicator formulas."""

    @property
    def open(self) -> Decimal: ...

    @property
    def high(self) -> Decimal: ...

    @property
    def low(self) -> Decimal: ...

    @property
    def close(self) -> Decimal: ...

    @property
    def volume(self) -> Decimal: ...


class DevelopingBarSupport(StrEnum):
    """Whether an indicator formula may consume a causal developing bar."""

    COMPLETED_ONLY = "completed_bars_only"
    DEVELOPING_AS_OF = "developing_bar_as_of_supported"


class IndicatorParameters(Protocol):
    """Immutable indicator parameters with stable primitive serialization."""

    def to_primitive(self) -> PrimitiveMapping: ...


class Indicator(Protocol):
    """Provider-independent calculation contract for aligned market data."""

    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> IndicatorParameters: ...

    @property
    def required_fields(self) -> frozenset[MarketField]: ...

    @property
    def warm_up_observations(self) -> int: ...

    @property
    def output_fields(self) -> tuple[str, ...]: ...

    @property
    def missing_value(self) -> None: ...

    @property
    def configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...

    def calculate(self, dataset: MarketDataset) -> IndicatorOutput: ...


class TimeframeNeutralIndicator(Indicator, Protocol):
    """Indicator whose formula accepts any canonical OHLCV bar interval."""

    @property
    def developing_bar_support(self) -> DevelopingBarSupport: ...

    def calculate_bar_fields(
        self, bars: tuple[IndicatorBar, ...]
    ) -> tuple[IndicatorFieldOutput, ...]: ...


def validate_market_input(
    dataset: MarketDataset, required_fields: frozenset[MarketField]
) -> None:
    """Reject unavailable fields and non-chronological canonical sessions."""
    sessions = tuple(bar.session_date for bar in dataset.bars)
    if any(current >= following for current, following in pairwise(sessions)):
        raise UnorderedMarketDataError(
            "market sessions must be unique and strictly chronological"
        )
    for field in sorted(required_fields, key=str):
        if any(not hasattr(bar, field.value) for bar in dataset.bars):
            raise MissingMarketFieldError(
                f"market data is missing required field: {field.value}"
            )


def validate_indicator_alignment(
    dataset: MarketDataset, output: IndicatorOutput
) -> None:
    """Require exact session and length preservation from an indicator."""
    input_sessions = tuple(bar.session_date for bar in dataset.bars)
    if output.session_dates != input_sessions:
        raise MisalignedIndicatorOutputError(
            "indicator output sessions do not match input sessions"
        )


def validate_indicator_bars(
    bars: tuple[IndicatorBar, ...],
    required_fields: frozenset[MarketField],
    *,
    require_finite: bool = True,
) -> None:
    """Reject generic bars that do not expose compatible Decimal source fields."""
    for field in sorted(required_fields, key=str):
        for bar in bars:
            try:
                value = getattr(bar, field.value)
            except AttributeError as error:
                raise MissingMarketFieldError(
                    f"market data is missing required field: {field.value}"
                ) from error
            if not isinstance(value, Decimal) or (
                require_finite and not value.is_finite()
            ):
                raise MissingMarketFieldError(
                    f"market field must be a compatible Decimal: {field.value}"
                )
