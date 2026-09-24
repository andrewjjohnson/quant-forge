"""Daily adapters and provider-neutral intraday adapter contracts."""

import os

from quantforge.data.exceptions import RequestError
from quantforge.data.intraday_ingestion import IntradayIngestionProvider
from quantforge.data.intraday_validation import IntradayCoverageReport
from quantforge.data.providers.alpha_vantage import AlphaVantageProvider
from quantforge.data.providers.base import DailyBarProvider, IntradayBarProvider
from quantforge.data.providers.massive import MassiveProvider
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
    if provider_name == MassiveProvider.name:
        return MassiveProvider.can_reuse_intraday_cache(quality_report)
    return True


def create_intraday_provider(
    provider_name: str, *, api_key: str | None = None
) -> IntradayIngestionProvider:
    """Select an installed historical adapter at application setup.

    Explicit secrets or the provider's standard environment variable are used
    only here; offline services still need only ``provider_name``.
    """
    if provider_name == MassiveProvider.name:
        return MassiveProvider(
            os.environ.get("MASSIVE_API_KEY", "") if api_key is None else api_key
        )
    if provider_name == TiingoProvider.name:
        return TiingoProvider(
            os.environ.get("TIINGO_API_KEY", "") if api_key is None else api_key
        )
    raise RequestError("unsupported intraday provider name")


__all__ = [
    "AlphaVantageProvider",
    "DailyBarProvider",
    "IntradayBarProvider",
    "IntradayIngestionProvider",
    "MassiveProvider",
    "TiingoProvider",
    "can_reuse_intraday_cache",
    "create_intraday_provider",
]
