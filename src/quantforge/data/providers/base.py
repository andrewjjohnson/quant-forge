"""Provider boundary for daily bars."""

from datetime import date
from typing import Protocol

from quantforge.data.models import AdjustmentMode, ProviderResponse


class DailyBarProvider(Protocol):
    """Provider adapters implement this interface; SDK types stay behind it."""

    name: str

    def fetch_daily_bars(
        self, symbol: str, start: date, end: date, adjustment: AdjustmentMode
    ) -> ProviderResponse: ...
