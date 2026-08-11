"""Provider boundaries for canonical daily and intraday bars."""

from datetime import date
from typing import Protocol

from quantforge.data.intraday import (
    IntradayBar,
    IntradayBarRequest,
    IntradayProviderCapabilities,
)
from quantforge.data.models import AdjustmentMode, ProviderResponse


class DailyBarProvider(Protocol):
    """Provider adapters implement this interface; SDK types stay behind it."""

    name: str

    def fetch_daily_bars(
        self, symbol: str, start: date, end: date, adjustment: AdjustmentMode
    ) -> ProviderResponse: ...


class IntradayBarProvider(Protocol):
    """Adapters expose capabilities and return only canonical intraday bars."""

    name: str
    intraday_capabilities: IntradayProviderCapabilities

    def fetch_intraday_bars(
        self, request: IntradayBarRequest
    ) -> tuple[IntradayBar, ...]: ...
