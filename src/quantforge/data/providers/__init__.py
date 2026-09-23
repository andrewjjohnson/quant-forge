"""Daily adapters and provider-neutral intraday adapter contracts."""

from quantforge.data.intraday_ingestion import IntradayIngestionProvider
from quantforge.data.intraday_validation import IntradayCoverageReport
from quantforge.data.providers.alpha_vantage import AlphaVantageProvider
from quantforge.data.providers.base import DailyBarProvider, IntradayBarProvider
from quantforge.data.providers.tiingo import TiingoProvider


def can_reuse_intraday_cache(
    provider_name: str, quality_report: IntradayCoverageReport
) -> bool:
    """Resolve credential-free provider policy for an integrity-checked cache.

    Providers without a stricter reuse policy retain diagnostic cache behavior.
    This lookup also serves offline callers that have only a provider name.
    """
    if provider_name == TiingoProvider.name:
        return TiingoProvider.can_reuse_intraday_cache(quality_report)
    return True


__all__ = [
    "AlphaVantageProvider",
    "DailyBarProvider",
    "IntradayBarProvider",
    "IntradayIngestionProvider",
    "TiingoProvider",
    "can_reuse_intraday_cache",
]
