"""A session-acquired source remains an ordinary QF-51/QF-52 input."""

from datetime import date
from pathlib import Path

import pytest

from quantforge.data import (
    IntradayMarketDataCache,
    IntradayMarketDataService,
    MarketDataCache,
    aggregate_intraday_dataset,
    aggregate_session_dataset,
    validate_market_dataset,
)
from quantforge.data.models import (
    BoundedPredictionProvenance,
    IntradayPredictionProvenance,
)
from quantforge.data.prediction_inputs import prediction_dataset_from_intraday
from quantforge.data.prediction_views import (
    bounded_prediction_view,
    validate_bounded_prediction_ancestry,
)
from quantforge.data.providers import TiingoProvider
from quantforge.timeframes import IntradayInterval, SessionInterval, Timeframe
from tests.unit.data.test_tiingo_calendar_history import (
    TOKEN,
    history_request,
    install_history,
)


def test_session_acquisition_projects_and_bounds_through_unchanged_contracts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from datetime import timedelta

    calls = install_history(monkeypatch)
    request = history_request("2024-07-02T00:00:00Z", "2024-07-06T00:00:00Z")
    source_cache = IntradayMarketDataCache(tmp_path)
    source = IntradayMarketDataService(
        source_cache, provider=TiingoProvider(TOKEN)
    ).get_intraday_bars(request)
    assert calls == [date(2024, 7, 2), date(2024, 7, 3), date(2024, 7, 5)]
    two_minutes = aggregate_intraday_dataset(
        source, Timeframe.us_equity(IntradayInterval(timedelta(minutes=2)))
    )
    assert len(two_minutes.bars) == 495
    daily = aggregate_session_dataset(source, Timeframe.us_equity(SessionInterval(1)))
    cache = MarketDataCache(tmp_path)
    canonical = prediction_dataset_from_intraday(
        source, daily, cache=cache, intraday_cache=source_cache
    )
    assert validate_market_dataset(canonical) == ()
    assert isinstance(
        canonical.metadata.intraday_provenance, IntradayPredictionProvenance
    )
    assert (
        canonical.metadata.intraday_provenance.source_dataset_id
        == source.metadata.dataset_id
    )
    assert canonical.metadata.corporate_actions_complete is False
    view = bounded_prediction_view(canonical, daily.bars[1].end_timestamp)
    validate_bounded_prediction_ancestry(view, canonical)
    assert validate_market_dataset(view) == ()
    assert isinstance(view.metadata.intraday_provenance, BoundedPredictionProvenance)
    assert view.metadata.actual_last_session == date(2024, 7, 3)
    assert len(view.bars) == 2
    assert (
        view.metadata.intraday_provenance.source_dataset_id
        == source.metadata.dataset_id
    )
    assert cache.load(canonical.metadata.dataset_id) == canonical
