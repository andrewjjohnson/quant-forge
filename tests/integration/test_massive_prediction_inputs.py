"""Massive composes with unchanged aggregation and QF-51/QF-52 validation."""

from datetime import UTC, date, datetime, timedelta
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
    CorporateActionAvailability,
    IntradayPredictionProvenance,
    JsonValue,
)
from quantforge.data.prediction_inputs import prediction_dataset_from_intraday
from quantforge.data.prediction_views import (
    bounded_prediction_view,
    validate_bounded_prediction_ancestry,
)
from quantforge.data.providers import create_intraday_provider
from quantforge.timeframes import IntradayInterval, SessionInterval, Timeframe
from quantforge.walk_forward.partitions import prediction_metadata_prefix
from tests.fixtures.massive.helpers import (
    NEXT,
    TOKEN,
    install_responses,
    page,
    request,
    row,
    rows,
)


@pytest.mark.parametrize("adjusted", [False, True])
def test_massive_sessions_aggregation_provenance_and_bounded_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, adjusted: bool
) -> None:
    # Normal day, early close, holiday, normal day: one range, two HTTP pages.
    first = datetime(2024, 7, 2, 13, 30, tzinfo=UTC)
    early = datetime(2024, 7, 3, 13, 30, tzinfo=UTC)
    last = datetime(2024, 7, 5, 13, 30, tzinfo=UTC)
    excluded: list[JsonValue] = [
        row(first - timedelta(minutes=1)),
        row(first + timedelta(minutes=390)),
        row(early + timedelta(minutes=210)),
        row(datetime(2024, 7, 4, 14, tzinfo=UTC)),
    ]
    calls = install_responses(
        monkeypatch,
        [
            page([*rows(first, 390), *rows(early, 210)], NEXT, adjusted=adjusted),
            page([*rows(last, 390), *excluded], adjusted=adjusted),
        ],
    )
    source_cache = IntradayMarketDataCache(tmp_path)
    source = IntradayMarketDataService(
        source_cache, provider=create_intraday_provider("massive", api_key=TOKEN)
    ).get_intraday_bars(
        request(
            datetime(2024, 7, 2, tzinfo=UTC),
            datetime(2024, 7, 6, tzinfo=UTC),
            adjusted=adjusted,
        )
    )
    assert len(calls) == 2
    assert len(source.bars) == 990
    assert source.quality_report.is_complete
    assert (
        len([bar for bar in source.bars if bar.session_date == date(2024, 7, 3)]) == 210
    )
    assert source.metadata.provider_name == "massive"
    two_minutes = aggregate_intraday_dataset(
        source, Timeframe.us_equity(IntradayInterval(timedelta(minutes=2)))
    )
    assert len(two_minutes.bars) == 495
    assert two_minutes.bars[0].volume == source.bars[0].volume + source.bars[1].volume
    daily = aggregate_session_dataset(source, Timeframe.us_equity(SessionInterval(1)))
    assert len(daily.bars) == 3
    assert daily.bars[1].end_timestamp == datetime(2024, 7, 3, 17, tzinfo=UTC)
    market_cache = MarketDataCache(tmp_path)
    canonical = prediction_dataset_from_intraday(
        source, daily, cache=market_cache, intraday_cache=source_cache
    )
    assert validate_market_dataset(canonical) == ()
    assert isinstance(
        canonical.metadata.intraday_provenance, IntradayPredictionProvenance
    )
    assert (
        canonical.metadata.corporate_action_availability
        is CorporateActionAvailability.UNAVAILABLE
    )
    assert canonical.metadata.corporate_actions_complete is False
    assert canonical.metadata.provider_name == "massive"
    assert (
        canonical.metadata.adjustment_mode
        == source.request.adjustment_basis.adjustment_mode
    )
    cutoff = datetime(2024, 7, 3, 16, tzinfo=UTC)
    view = prediction_metadata_prefix(canonical, cutoff)
    validate_bounded_prediction_ancestry(view, canonical)
    assert validate_market_dataset(view) == ()
    assert isinstance(view.metadata.intraday_provenance, BoundedPredictionProvenance)
    assert view.bars == canonical.bars[:1]
    assert (
        view.metadata.intraday_provenance.source_dataset_id
        == source.metadata.dataset_id
    )
    assert view == bounded_prediction_view(canonical, cutoff)
    assert market_cache.load(canonical.metadata.dataset_id) == canonical
    assert source_cache.load(source.metadata.dataset_id, source.request) == source
