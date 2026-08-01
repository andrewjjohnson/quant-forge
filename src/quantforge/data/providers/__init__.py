"""Daily market-data provider adapters."""

from quantforge.data.providers.alpha_vantage import AlphaVantageProvider
from quantforge.data.providers.base import DailyBarProvider

__all__ = ["AlphaVantageProvider", "DailyBarProvider"]
