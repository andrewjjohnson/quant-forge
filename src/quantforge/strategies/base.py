"""Reusable strategy boundary and shared daily-bar timing semantics."""

from datetime import date, timedelta
from typing import Protocol

from quantforge.configuration import PrimitiveMapping
from quantforge.data.calendar import expected_sessions
from quantforge.data.models import MarketDataset
from quantforge.indicators.base import Indicator
from quantforge.indicators.models import MarketField
from quantforge.strategies.exceptions import UnsupportedTimingConventionError
from quantforge.strategies.models import ExecutionTiming, StrategyOutput
from quantforge.strategies.parameters import StrategyParameters
from quantforge.strategies.sizing import PositionSizingPolicy


class Strategy(Protocol):
    """Engine-neutral strategy contract over one canonical market dataset."""

    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> StrategyParameters: ...

    @property
    def required_fields(self) -> frozenset[MarketField]: ...

    @property
    def required_indicators(self) -> tuple[Indicator, ...]: ...

    @property
    def warm_up_observations(self) -> int: ...

    @property
    def timing(self) -> ExecutionTiming: ...

    @property
    def sizing_policy(self) -> PositionSizingPolicy: ...

    @property
    def asset_assumptions(self) -> tuple[str, ...]: ...

    @property
    def configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...

    def generate(self, dataset: MarketDataset) -> StrategyOutput: ...


def next_exchange_session(signal_session: date, calendar: str) -> date:
    """Resolve the first calendar session after a completed signal session."""
    start = signal_session + timedelta(days=1)
    end = signal_session + timedelta(days=31)
    try:
        sessions = expected_sessions(start, end, calendar)
    except Exception as error:
        raise UnsupportedTimingConventionError(
            f"cannot resolve next session for calendar: {calendar}"
        ) from error
    if not sessions:
        raise UnsupportedTimingConventionError(
            f"calendar has no session within 31 days after {signal_session}"
        )
    return sessions[0]
