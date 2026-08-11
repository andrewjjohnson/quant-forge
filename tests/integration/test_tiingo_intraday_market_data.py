import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    FeedScope,
    IntradayBarRequest,
    IntradayMarketDataCache,
    IntradayMarketDataService,
)
from quantforge.data.providers import TiingoProvider
from quantforge.timeframes import IntradayInterval, Timeframe


@pytest.mark.integration
def test_live_tiingo_spy_intraday_cache_and_replay(tmp_path: Path) -> None:
    if os.environ.get("QUANTFORGE_RUN_LIVE_TIINGO_INTRADAY") != "1":
        pytest.skip("set QUANTFORGE_RUN_LIVE_TIINGO_INTRADAY=1 to opt in")
    api_key = os.environ.get("TIINGO_API_KEY")
    if not api_key:
        pytest.skip("TIINGO_API_KEY is not configured")
    requested_feed = os.environ.get("QUANTFORGE_TIINGO_INTRADAY_FEED", "consolidated")
    if requested_feed == "consolidated":
        feed_scope = FeedScope.consolidated()
    elif requested_feed == "iex":
        feed_scope = FeedScope.iex_only()
    else:
        pytest.fail("QUANTFORGE_TIINGO_INTRADAY_FEED must be consolidated or iex")
    request = IntradayBarRequest(
        symbol="SPY",
        start_timestamp=datetime(2024, 7, 1, 13, 30, tzinfo=UTC),
        end_timestamp=datetime(2024, 7, 1, 14, 0, tzinfo=UTC),
        timeframe=Timeframe.us_equity(IntradayInterval(timedelta(minutes=5))),
        feed_scope=feed_scope,
        adjustment_basis=AdjustmentBasis(
            adjustment_mode=AdjustmentMode.UNADJUSTED,
            ohlc_basis="raw_provider",
            volume_basis="raw_provider",
            corporate_action_policy="not_provided_for_intraday_bars",
            adjusted_fields_used=False,
        ),
    )
    cache = IntradayMarketDataCache(tmp_path / "cache")
    dataset = IntradayMarketDataService(
        cache, provider=TiingoProvider(api_key)
    ).get_intraday_bars(request)
    replayed = IntradayMarketDataService(
        cache, provider_name="tiingo"
    ).get_intraday_bars(request)

    assert dataset == replayed
    assert dataset.bars
    assert tuple(bar.start_timestamp for bar in dataset.bars) == tuple(
        sorted(bar.start_timestamp for bar in dataset.bars)
    )
    assert len({bar.start_timestamp for bar in dataset.bars}) == len(dataset.bars)
    assert all(bar.provenance.feed_scope == feed_scope for bar in dataset.bars)
