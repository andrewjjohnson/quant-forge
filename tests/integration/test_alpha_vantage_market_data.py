import os
from datetime import date
from pathlib import Path

import pytest

from quantforge.data import AdjustmentMode, MarketDataCache, MarketDataService
from quantforge.data.providers import AlphaVantageProvider


@pytest.mark.integration
def test_live_spy_ingest_cache_and_reload(tmp_path: Path) -> None:
    if os.environ.get("QUANTFORGE_RUN_LIVE_MARKET_DATA") != "1":
        pytest.skip("set QUANTFORGE_RUN_LIVE_MARKET_DATA=1 to opt in")
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        pytest.skip("ALPHA_VANTAGE_API_KEY is not configured")
    provider = AlphaVantageProvider(api_key)
    cache = MarketDataCache(tmp_path)
    service = MarketDataService(provider, cache)
    dataset = service.get_daily_bars(
        "SPY", date(2024, 7, 1), date(2024, 7, 3), AdjustmentMode.SPLIT_ADJUSTED
    )
    assert cache.load(dataset.metadata.dataset_id) == dataset
