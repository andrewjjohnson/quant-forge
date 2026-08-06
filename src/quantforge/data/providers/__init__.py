"""Daily market-data provider adapters."""

from quantforge.data.providers.alpha_vantage import AlphaVantageProvider
from quantforge.data.providers.base import DailyBarProvider
from quantforge.data.providers.tiingo import TiingoProvider

__all__ = ["AlphaVantageProvider", "DailyBarProvider", "TiingoProvider"]
