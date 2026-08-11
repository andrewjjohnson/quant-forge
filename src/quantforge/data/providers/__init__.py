"""Daily adapters and provider-neutral intraday adapter contracts."""

from quantforge.data.intraday_ingestion import IntradayIngestionProvider
from quantforge.data.providers.alpha_vantage import AlphaVantageProvider
from quantforge.data.providers.base import DailyBarProvider, IntradayBarProvider
from quantforge.data.providers.tiingo import TiingoProvider

__all__ = [
    "AlphaVantageProvider",
    "DailyBarProvider",
    "IntradayBarProvider",
    "IntradayIngestionProvider",
    "TiingoProvider",
]
