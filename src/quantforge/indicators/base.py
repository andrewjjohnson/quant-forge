"""Reusable indicator boundary and canonical input validation."""

from itertools import pairwise
from typing import Protocol

from quantforge.configuration import PrimitiveMapping
from quantforge.data.models import MarketDataset
from quantforge.indicators.exceptions import (
    MisalignedIndicatorOutputError,
    MissingMarketFieldError,
    UnorderedMarketDataError,
)
from quantforge.indicators.models import IndicatorOutput, MarketField


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
