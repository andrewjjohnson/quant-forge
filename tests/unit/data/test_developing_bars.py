from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest

from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    AggregationPolicy,
    ContextAvailability,
    ContextCompletionPolicy,
    ContextTimeframeRequirement,
    DatasetFamily,
    DatasetLineage,
    DevelopingBar,
    DevelopingBarValidationError,
    FeedScope,
    IntradayBar,
    IntradayBarBatch,
    IntradayBarProvenance,
    IntradayBarRequest,
    IntradayDataset,
    IntradayDatasetMetadata,
    IntradayFetchResult,
    IntradayMarketDataCache,
    IntradayRawSnapshot,
    IntradayValidationMode,
    TimeframeBarSeries,
    aggregate_intraday_dataset,
    aggregate_session_dataset,
    build_multi_timeframe_context,
    validate_intraday_coverage,
)
from quantforge.data.developing_bars import reconstruct_developing_bar_as_of
from quantforge.data.identity import sha256_hex
from quantforge.timeframes import (
    BarCompletion,
    IntradayInterval,
    SessionInterval,
    Timeframe,
    TradingWeekInterval,
    resolve_exchange_session,
)

NEW_YORK = ZoneInfo("America/New_York")
RETRIEVED_AT = datetime(2025, 1, 1, tzinfo=UTC)


def _adjustment_basis() -> AdjustmentBasis:
    return AdjustmentBasis(
        adjustment_mode=AdjustmentMode.UNADJUSTED,
        ohlc_basis="raw_provider",
        volume_basis="raw_provider",
        corporate_action_policy="not_provided_for_intraday_bars",
        adjusted_fields_used=False,
    )


def _source_timeframe() -> Timeframe:
    return Timeframe.us_equity(IntradayInterval(timedelta(minutes=5)))


def _request(session_dates: tuple[date, ...]) -> IntradayBarRequest:
    first = resolve_exchange_session(session_dates[0])
    last = resolve_exchange_session(session_dates[-1])
    return IntradayBarRequest(
        symbol="SPY",
        start_timestamp=first.open_timestamp,
        end_timestamp=last.close_timestamp,
        timeframe=_source_timeframe(),
        feed_scope=FeedScope.consolidated(),
        adjustment_basis=_adjustment_basis(),
    )


def _source_bars(
    request: IntradayBarRequest,
    session_dates: tuple[date, ...],
    *,
    source_snapshot_id: str,
) -> tuple[IntradayBar, ...]:
    provenance = IntradayBarProvenance(
        provider_name="fixture-provider",
        provider_symbol="SPY",
        adapter_version="fixture-1",
        retrieved_at=RETRIEVED_AT,
        source_request_id=request.request_id,
        source_snapshot_id=source_snapshot_id,
        feed_scope=request.feed_scope,
        adjustment_basis=request.adjustment_basis,
    )
    bars: list[IntradayBar] = []
    sequence = 0
    for session_date in session_dates:
        session = resolve_exchange_session(session_date)
        start_timestamp = session.open_timestamp
        while start_timestamp < session.close_timestamp:
            end_timestamp = min(
                start_timestamp + timedelta(minutes=5), session.close_timestamp
            )
            base = Decimal(100 + sequence)
            bars.append(
                IntradayBar(
                    symbol=request.symbol,
                    session_date=session_date,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    timeframe=request.timeframe,
                    completion=(
                        BarCompletion.COMPLETED
                        if end_timestamp - start_timestamp == timedelta(minutes=5)
                        else BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL
                    ),
                    open=base,
                    high=base + 2,
                    low=base - 1,
                    close=base + 1,
                    volume=Decimal(10 + sequence),
                    provenance=provenance,
                )
            )
            sequence += 1
            start_timestamp = end_timestamp
    return tuple(bars)


def _persisted_source(
    cache_root: Path,
    session_dates: tuple[date, ...],
) -> tuple[IntradayDataset, IntradayMarketDataCache, DatasetFamily]:
    request = _request(session_dates)
    snapshot = IntradayRawSnapshot(
        provider_name="fixture-provider",
        provider_symbol="SPY",
        adapter_version="fixture-1",
        endpoint="fixture/intraday",
        source_request_id=request.request_id,
        chunk_start_timestamp=request.start_timestamp,
        chunk_end_timestamp=request.end_timestamp,
        retrieved_at=RETRIEVED_AT,
        request_parameters=(("interval", "5min"),),
        records=(),
    )
    bars = _source_bars(
        request,
        session_dates,
        source_snapshot_id=snapshot.snapshot_id,
    )
    cache = IntradayMarketDataCache(cache_root)
    dataset = cache.persist(
        IntradayFetchResult(
            IntradayBarBatch(request, bars),
            (snapshot,),
            "fixture-capabilities",
        )
    )
    source_id = dataset.metadata.dataset_id
    family = DatasetFamily(
        canonical_symbol=request.symbol,
        provider_name=dataset.metadata.provider_name,
        feed_scope=request.feed_scope,
        adjustment_basis=request.adjustment_basis,
        aggregation_policy=AggregationPolicy(
            "quantforge_fixture_source", "1", {"missing": "reject"}
        ),
        canonical_source_snapshot_id=source_id,
        datasets=(
            DatasetLineage(
                source_id,
                request.timeframe,
                source_id,
                None,
            ),
        ),
    )
    return dataset, cache, family


def _context(
    source: IntradayDataset,
    cache: IntradayMarketDataCache,
    family: DatasetFamily,
    *,
    as_of: datetime,
    targets: tuple[Timeframe, ...],
    completion_policy: ContextCompletionPolicy,
):
    source_series = TimeframeBarSeries.from_source_dataset(
        source, family=family, cache=cache
    )
    return build_multi_timeframe_context(
        as_of=as_of,
        primary_timeframe=source.request.timeframe,
        required_timeframes=tuple(
            ContextTimeframeRequirement(target) for target in targets
        ),
        series=(source_series,),
        completion_policy=completion_policy,
    )


def test_developing_mode_is_explicit_and_reconstructs_daily_weekly_and_intraday(
    tmp_path: Path,
) -> None:
    session_dates = (
        date(2024, 7, 8),
        date(2024, 7, 9),
        date(2024, 7, 10),
        date(2024, 7, 11),
        date(2024, 7, 12),
    )
    source, cache, family = _persisted_source(tmp_path, session_dates)
    as_of = datetime(2024, 7, 9, 15, 55, tzinfo=NEW_YORK)
    four_hour = Timeframe.us_equity(IntradayInterval(timedelta(hours=4)))
    daily = Timeframe.us_equity(SessionInterval())
    weekly = Timeframe.us_equity(TradingWeekInterval())

    completed = _context(
        source,
        cache,
        family,
        as_of=as_of,
        targets=(four_hour, daily, weekly),
        completion_policy=ContextCompletionPolicy.COMPLETED_BARS_ONLY,
    )
    assert completed.metadata_for(daily).availability is ContextAvailability.MISSING
    assert completed.metadata_for(weekly).availability is ContextAvailability.MISSING

    developing = _context(
        source,
        cache,
        family,
        as_of=as_of,
        targets=(four_hour, daily, weekly),
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
    )
    daily_bar = developing.latest_bar_for(daily)
    weekly_bar = developing.latest_bar_for(weekly)
    intraday_bar = developing.latest_bar_for(four_hour)

    assert isinstance(daily_bar, DevelopingBar)
    assert isinstance(weekly_bar, DevelopingBar)
    assert isinstance(intraday_bar, DevelopingBar)
    assert not daily_bar.complete
    assert daily_bar.completion is BarCompletion.DEVELOPING
    assert daily_bar.as_of == as_of.astimezone(UTC)
    assert daily_bar.observed_end_timestamp == as_of.astimezone(UTC)
    assert daily_bar.expected_completion_boundary == datetime(
        2024, 7, 9, 20, tzinfo=UTC
    )
    assert daily_bar.source_bar_count == 77
    assert weekly_bar.session_dates == (date(2024, 7, 8), date(2024, 7, 9))
    assert weekly_bar.source_bar_count == 155
    assert weekly_bar.expected_completion_boundary == datetime(
        2024, 7, 12, 20, tzinfo=UTC
    )
    assert intraday_bar.observed_start_timestamp == datetime(
        2024, 7, 9, 17, 30, tzinfo=UTC
    )
    assert intraday_bar.expected_completion_boundary == datetime(
        2024, 7, 9, 20, tzinfo=UTC
    )
    primitive = daily_bar.to_primitive()
    source_reference = cast(dict[str, object], primitive["source_dataset_reference"])
    assert primitive["bar_type"] == "developing_bar_as_of"
    assert primitive["complete"] is False
    assert primitive["source_bar_count"] == 77
    assert "feed_scope" not in source_reference
    assert daily_bar.serialize() == daily_bar.serialize()
    assert developing.to_primitive()["completion_policy"] == "developing_bar_as_of"
    assert (
        developing.serialize()
        == _context(
            source,
            cache,
            family,
            as_of=as_of,
            targets=(weekly, daily, four_hour),
            completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
        ).serialize()
    )


def test_future_source_changes_cannot_change_an_earlier_developing_bar(
    tmp_path: Path,
) -> None:
    session_dates = (date(2024, 7, 8), date(2024, 7, 9))
    source, cache, family = _persisted_source(tmp_path, session_dates)
    series = TimeframeBarSeries.from_source_dataset(source, family=family, cache=cache)
    evidence = series._developing_source_evidence  # pyright: ignore[reportPrivateUsage]
    assert evidence is not None
    as_of = datetime(2024, 7, 9, 10, tzinfo=NEW_YORK)
    daily = Timeframe.us_equity(SessionInterval())
    original = reconstruct_developing_bar_as_of(
        as_of=as_of,
        target_timeframe=daily,
        source_timeframe=series.timeframe,
        source_bars=tuple(bar for bar in series.bars if isinstance(bar, IntradayBar)),
        expected_source_intervals=evidence.expected_intervals,
        source_dataset_reference=series.dataset_reference,
    )
    assert original is not None
    future_index = next(
        index for index, bar in enumerate(series.bars) if bar.end_timestamp > as_of
    )
    changed_future = replace(
        series.bars[future_index],
        high=series.bars[future_index].high + Decimal("1000"),
    )
    changed_bars = (
        *series.bars[:future_index],
        changed_future,
        *series.bars[future_index + 1 :],
    )
    changed = reconstruct_developing_bar_as_of(
        as_of=as_of,
        target_timeframe=daily,
        source_timeframe=series.timeframe,
        source_bars=tuple(bar for bar in changed_bars if isinstance(bar, IntradayBar)),
        expected_source_intervals=evidence.expected_intervals,
        source_dataset_reference=series.dataset_reference,
    )

    assert changed == original


def test_developing_reconstruction_fails_closed_on_missing_causal_source(
    tmp_path: Path,
) -> None:
    source, cache, family = _persisted_source(tmp_path, (date(2024, 7, 9),))
    series = TimeframeBarSeries.from_source_dataset(source, family=family, cache=cache)
    evidence = series._developing_source_evidence  # pyright: ignore[reportPrivateUsage]
    assert evidence is not None
    as_of = datetime(2024, 7, 9, 10, tzinfo=NEW_YORK)
    source_bars = tuple(bar for bar in series.bars if isinstance(bar, IntradayBar))

    with pytest.raises(
        DevelopingBarValidationError, match="missing an available source"
    ):
        reconstruct_developing_bar_as_of(
            as_of=as_of,
            target_timeframe=Timeframe.us_equity(SessionInterval()),
            source_timeframe=series.timeframe,
            source_bars=source_bars[1:],
            expected_source_intervals=evidence.expected_intervals,
            source_dataset_reference=series.dataset_reference,
        )


def test_early_close_uses_actual_completion_boundary(tmp_path: Path) -> None:
    source, cache, family = _persisted_source(tmp_path, (date(2024, 7, 3),))
    as_of = datetime(2024, 7, 3, 12, 55, tzinfo=NEW_YORK)
    daily = Timeframe.us_equity(SessionInterval())

    context = _context(
        source,
        cache,
        family,
        as_of=as_of,
        targets=(daily,),
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
    )
    bar = context.latest_bar_for(daily)

    assert isinstance(bar, DevelopingBar)
    assert bar.observed_end_timestamp == datetime(2024, 7, 3, 16, 55, tzinfo=UTC)
    assert bar.expected_completion_boundary == datetime(2024, 7, 3, 17, tzinfo=UTC)
    assert bar.source_bar_count == 41


@pytest.mark.parametrize(
    ("session_date", "expected_completion"),
    [
        (date(2024, 3, 8), datetime(2024, 3, 8, 18, 30, tzinfo=UTC)),
        (date(2024, 3, 11), datetime(2024, 3, 11, 17, 30, tzinfo=UTC)),
    ],
)
def test_intraday_developing_boundaries_follow_exchange_dst(
    tmp_path: Path,
    session_date: date,
    expected_completion: datetime,
) -> None:
    source, cache, family = _persisted_source(tmp_path, (session_date,))
    four_hour = Timeframe.us_equity(IntradayInterval(timedelta(hours=4)))
    as_of = datetime.combine(
        session_date,
        datetime.min.time().replace(hour=10),
        NEW_YORK,
    )

    context = _context(
        source,
        cache,
        family,
        as_of=as_of,
        targets=(four_hour,),
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
    )
    bar = context.latest_bar_for(four_hour)

    assert isinstance(bar, DevelopingBar)
    assert bar.expected_completion_boundary == expected_completion


def test_completed_qf18_and_qf19_aggregation_remains_terminal() -> None:
    request = _request((date(2024, 7, 8),))
    bars = _source_bars(request, (date(2024, 7, 8),), source_snapshot_id="fixture")
    batch = IntradayBarBatch(request, bars)
    report = validate_intraday_coverage(batch, mode=IntradayValidationMode.DIAGNOSTIC)
    source = IntradayDataset(
        request,
        bars,
        IntradayDatasetMetadata(
            dataset_id="source",
            request_id=request.request_id,
            provider_name="fixture-provider",
            provider_symbol="SPY",
            adapter_version="fixture-1",
            retrieved_at=RETRIEVED_AT,
            capabilities_configuration_id="fixture-capabilities",
            batch_id=batch.batch_id,
            bar_count=len(bars),
            raw_snapshot_ids=("fixture",),
            raw_locations=("intraday/raw/fixture.json",),
            normalized_location="intraday/datasets/source/bars.json",
            data_sha256=sha256_hex(batch.serialize()),
            quality_report=report,
        ),
    )
    intraday = aggregate_intraday_dataset(
        source, Timeframe.us_equity(IntradayInterval(timedelta(hours=4)))
    )
    daily = aggregate_session_dataset(source, Timeframe.us_equity(SessionInterval()))

    assert all(bar.complete for bar in intraday.bars)
    assert all(bar.complete for bar in daily.bars)
    assert all(bar.completion is not BarCompletion.DEVELOPING for bar in intraday.bars)
    assert all(bar.completion is BarCompletion.COMPLETED for bar in daily.bars)
