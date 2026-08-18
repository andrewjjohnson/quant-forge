from collections.abc import Callable
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
    AggregatedSessionBar,
    AggregationPolicy,
    ContextAvailability,
    ContextTimeframeRequirement,
    DatasetFamily,
    DatasetLineage,
    FeedScope,
    IntradayAggregationPolicy,
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
    MultiTimeframeContext,
    MultiTimeframeContextValidationError,
    SessionAggregationPolicy,
    TimeframeBarSeries,
    TimeframeContext,
    UnavailableTimeframeError,
    UndeclaredTimeframeError,
    aggregate_intraday_dataset,
    aggregate_session_dataset,
    build_multi_timeframe_context,
    validate_intraday_coverage,
)
from quantforge.data.identity import sha256_hex
from quantforge.timeframes import (
    BarCompletion,
    ExchangeSessionPolicy,
    IntradayInterval,
    SessionInterval,
    SessionScope,
    Timeframe,
    TradingWeekInterval,
    resolve_exchange_session,
    resolve_trading_week,
)

NEW_YORK = ZoneInfo("America/New_York")
AS_OF = datetime(2024, 7, 9, 10, tzinfo=NEW_YORK)
RETRIEVED_AT = datetime(2025, 1, 1, tzinfo=UTC)
SOURCE_ID = "source-5m"
FOUR_HOUR_ID = "derived-4h"
DAILY_ID = "derived-daily"
WEEKLY_ID = "derived-weekly"


def _timeframes() -> tuple[Timeframe, Timeframe, Timeframe, Timeframe]:
    return (
        Timeframe.us_equity(IntradayInterval(timedelta(minutes=5))),
        Timeframe.us_equity(IntradayInterval(timedelta(hours=4))),
        Timeframe.us_equity(SessionInterval()),
        Timeframe.us_equity(TradingWeekInterval()),
    )


def _adjustment_basis() -> AdjustmentBasis:
    return AdjustmentBasis(
        adjustment_mode=AdjustmentMode.UNADJUSTED,
        ohlc_basis="raw_provider",
        volume_basis="raw_provider",
        corporate_action_policy="separate_provider_actions",
        adjusted_fields_used=False,
    )


def _family(
    *,
    feed_scope: FeedScope | None = None,
    source_dataset_id: str = SOURCE_ID,
) -> DatasetFamily:
    five_minute, four_hour, daily, weekly = _timeframes()
    return DatasetFamily(
        canonical_symbol="SPY",
        provider_name="fixture-provider",
        feed_scope=feed_scope or FeedScope.consolidated(),
        adjustment_basis=_adjustment_basis(),
        aggregation_policy=AggregationPolicy(
            "quantforge_fixture_aggregation", "1", {"missing": "reject"}
        ),
        canonical_source_snapshot_id=source_dataset_id,
        datasets=(
            DatasetLineage(
                source_dataset_id,
                five_minute,
                source_dataset_id,
                None,
                (FOUR_HOUR_ID, DAILY_ID, WEEKLY_ID),
            ),
            DatasetLineage(
                FOUR_HOUR_ID,
                four_hour,
                source_dataset_id,
                source_dataset_id,
            ),
            DatasetLineage(DAILY_ID, daily, source_dataset_id, source_dataset_id),
            DatasetLineage(WEEKLY_ID, weekly, source_dataset_id, source_dataset_id),
        ),
    )


def _provenance(
    *,
    source_request_id: str = "fixture-request",
    source_snapshot_id: str = SOURCE_ID,
) -> IntradayBarProvenance:
    return IntradayBarProvenance(
        provider_name="fixture-provider",
        provider_symbol="SPY",
        adapter_version="fixture-1",
        retrieved_at=RETRIEVED_AT,
        source_request_id=source_request_id,
        source_snapshot_id=source_snapshot_id,
        feed_scope=FeedScope.consolidated(),
        adjustment_basis=_adjustment_basis(),
    )


def _intraday_bar(
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
    completion: BarCompletion = BarCompletion.COMPLETED,
) -> IntradayBar:
    return IntradayBar(
        symbol="SPY",
        session_date=start.astimezone(NEW_YORK).date(),
        start_timestamp=start,
        end_timestamp=end,
        timeframe=timeframe,
        completion=completion,
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal("101"),
        volume=Decimal("1000"),
        provenance=_provenance(),
    )


def _session_bar(timeframe: Timeframe, period_start: date) -> AggregatedSessionBar:
    if isinstance(timeframe.interval, SessionInterval):
        sessions = (resolve_exchange_session(period_start),)
    else:
        sessions = resolve_trading_week(period_start).sessions
    return AggregatedSessionBar(
        symbol="SPY",
        timeframe=timeframe,
        period_start_date=period_start,
        session_dates=tuple(session.session_date for session in sessions),
        start_timestamp=sessions[0].open_timestamp,
        end_timestamp=sessions[-1].close_timestamp,
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal("101"),
        volume=Decimal("10000"),
        source_bar_ids=tuple(
            f"source-{period_start.isoformat()}-{index}"
            for index, _ in enumerate(sessions)
        ),
        source_dataset_id=SOURCE_ID,
    )


def _trusted_series(
    family: DatasetFamily,
    dataset_id: str,
    timeframe: Timeframe,
    bars: tuple[IntradayBar | AggregatedSessionBar, ...],
) -> TimeframeBarSeries:
    """Build isolated alignment fixtures after their own helper validation."""
    return TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        family.reference(dataset_id), timeframe, bars
    )


def _source_request() -> IntradayBarRequest:
    five_minute = _timeframes()[0]
    session = resolve_exchange_session(date(2024, 7, 8))
    return IntradayBarRequest(
        symbol="SPY",
        start_timestamp=session.open_timestamp,
        end_timestamp=session.close_timestamp,
        timeframe=five_minute,
        feed_scope=FeedScope.consolidated(),
        adjustment_basis=_adjustment_basis(),
    )


def _source_bars(
    request: IntradayBarRequest,
    *,
    source_snapshot_id: str,
) -> tuple[IntradayBar, ...]:
    session = resolve_exchange_session(date(2024, 7, 8))
    provenance = _provenance(
        source_request_id=request.request_id,
        source_snapshot_id=source_snapshot_id,
    )
    return tuple(
        IntradayBar(
            symbol="SPY",
            session_date=session.session_date,
            start_timestamp=session.open_timestamp + timedelta(minutes=5 * index),
            end_timestamp=session.open_timestamp + timedelta(minutes=5 * (index + 1)),
            timeframe=request.timeframe,
            completion=BarCompletion.COMPLETED,
            open=Decimal(100 + index),
            high=Decimal(102 + index),
            low=Decimal(99 + index),
            close=Decimal(101 + index),
            volume=Decimal("1000"),
            provenance=provenance,
        )
        for index in range(78)
    )


def _complete_source_dataset() -> IntradayDataset:
    request = _source_request()
    bars = _source_bars(request, source_snapshot_id=SOURCE_ID)
    batch = IntradayBarBatch(request, bars)
    report = validate_intraday_coverage(batch, mode=IntradayValidationMode.DIAGNOSTIC)
    return IntradayDataset(
        request,
        bars,
        IntradayDatasetMetadata(
            dataset_id=SOURCE_ID,
            request_id=request.request_id,
            provider_name="fixture-provider",
            provider_symbol="SPY",
            adapter_version="fixture-1",
            retrieved_at=RETRIEVED_AT,
            capabilities_configuration_id="fixture-capabilities",
            batch_id=batch.batch_id,
            bar_count=len(bars),
            raw_snapshot_ids=(SOURCE_ID,),
            raw_locations=(f"intraday/raw/{SOURCE_ID}.json",),
            normalized_location=f"intraday/datasets/{SOURCE_ID}/bars.json",
            data_sha256=sha256_hex(batch.serialize()),
            quality_report=report,
        ),
    )


def _persisted_source_dataset(
    cache_root: Path,
) -> tuple[IntradayDataset, IntradayMarketDataCache]:
    request = _source_request()
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
    return dataset, cache


def _context_family_for_derived(
    artifact_family: DatasetFamily,
    dataset_id: str,
    timeframe: Timeframe,
) -> DatasetFamily:
    source_id = artifact_family.canonical_source_snapshot_id
    return DatasetFamily(
        canonical_symbol=artifact_family.canonical_symbol,
        provider_name=artifact_family.provider_name,
        feed_scope=artifact_family.feed_scope,
        adjustment_basis=artifact_family.adjustment_basis,
        aggregation_policy=AggregationPolicy(
            "quantforge_context_artifact_set",
            "1",
            {"artifact_family_manifest_ids": [artifact_family.manifest_id]},
        ),
        canonical_source_snapshot_id=source_id,
        datasets=(
            DatasetLineage(
                source_id,
                artifact_family.source_timeframe,
                source_id,
                None,
                (dataset_id,),
            ),
            DatasetLineage(dataset_id, timeframe, source_id, source_id),
        ),
    )


def _fixture_series(
    family: DatasetFamily,
) -> tuple[TimeframeBarSeries, ...]:
    five_minute, four_hour, daily, weekly = _timeframes()
    monday = resolve_exchange_session(date(2024, 7, 8))
    tuesday = resolve_exchange_session(date(2024, 7, 9))
    five_minute_bars = tuple(
        _intraday_bar(
            five_minute,
            tuesday.open_timestamp + timedelta(minutes=5 * index),
            tuesday.open_timestamp + timedelta(minutes=5 * (index + 1)),
        )
        for index in range(7)
    )
    four_hour_bars = (
        _intraday_bar(
            four_hour,
            monday.open_timestamp,
            monday.open_timestamp + timedelta(hours=4),
        ),
        _intraday_bar(
            four_hour,
            monday.open_timestamp + timedelta(hours=4),
            monday.close_timestamp,
            BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL,
        ),
        _intraday_bar(
            four_hour,
            tuesday.open_timestamp,
            tuesday.open_timestamp + timedelta(hours=4),
        ),
    )
    daily_bars = (
        _session_bar(daily, date(2024, 7, 3)),
        _session_bar(daily, date(2024, 7, 8)),
        _session_bar(daily, date(2024, 7, 9)),
    )
    weekly_bars = (
        _session_bar(weekly, date(2024, 7, 1)),
        _session_bar(weekly, date(2024, 7, 8)),
    )
    return (
        _trusted_series(family, SOURCE_ID, five_minute, five_minute_bars),
        _trusted_series(family, FOUR_HOUR_ID, four_hour, four_hour_bars),
        _trusted_series(family, DAILY_ID, daily, daily_bars),
        _trusted_series(family, WEEKLY_ID, weekly, weekly_bars),
    )


def _context(
    family: DatasetFamily | None = None,
    series: tuple[TimeframeBarSeries, ...] | None = None,
):
    active_family = family or _family()
    five_minute, four_hour, daily, weekly = _timeframes()
    return build_multi_timeframe_context(
        as_of=AS_OF,
        primary_timeframe=five_minute,
        required_timeframes=(
            ContextTimeframeRequirement(weekly),
            ContextTimeframeRequirement(daily),
            ContextTimeframeRequirement(four_hour),
        ),
        series=_fixture_series(active_family) if series is None else series,
    )


def test_tuesday_context_exposes_only_completed_weekly_daily_4h_and_5m_bars() -> None:
    context = _context()
    five_minute, four_hour, daily, weekly = _timeframes()

    assert context.latest_bar_for(five_minute).end_timestamp == AS_OF.astimezone(UTC)
    assert context.latest_bar_for(four_hour).end_timestamp == datetime(
        2024, 7, 8, 20, tzinfo=UTC
    )
    latest_daily = context.latest_bar_for(daily)
    latest_weekly = context.latest_bar_for(weekly)
    assert isinstance(latest_daily, AggregatedSessionBar)
    assert isinstance(latest_weekly, AggregatedSessionBar)
    assert latest_daily.period_start_date == date(2024, 7, 8)
    assert latest_weekly.period_start_date == date(2024, 7, 1)

    assert all(
        bar.end_timestamp <= context.as_of
        for aligned in context.timeframes
        for bar in aligned.bars
    )
    daily_bars = context.bars_for(daily)
    weekly_bars = context.bars_for(weekly)
    assert all(isinstance(bar, AggregatedSessionBar) for bar in daily_bars)
    assert all(isinstance(bar, AggregatedSessionBar) for bar in weekly_bars)
    typed_daily_bars = cast(tuple[AggregatedSessionBar, ...], daily_bars)
    typed_weekly_bars = cast(tuple[AggregatedSessionBar, ...], weekly_bars)
    assert date(2024, 7, 9) not in {bar.period_start_date for bar in typed_daily_bars}
    assert date(2024, 7, 8) not in {bar.period_start_date for bar in typed_weekly_bars}

    early_close = typed_daily_bars[0]
    holiday_week = latest_weekly
    assert early_close.end_timestamp == datetime(2024, 7, 3, 17, tzinfo=UTC)
    assert holiday_week.session_dates == (
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
    )


def test_appending_future_bars_cannot_change_historical_context() -> None:
    family = _family()
    original_series = _fixture_series(family)
    original = _context(family, original_series)
    five_minute = _timeframes()[0]
    next_session = resolve_exchange_session(date(2024, 7, 10))
    future_bar = _intraday_bar(
        five_minute,
        next_session.open_timestamp,
        next_session.open_timestamp + timedelta(minutes=5),
    )
    appended_series = tuple(
        _trusted_series(
            family,
            item.dataset_reference.dataset_id,
            item.timeframe,
            (*item.bars, future_bar),
        )
        if item.timeframe == five_minute
        else item
        for item in original_series
    )
    appended = _context(family, appended_series)

    assert appended.context_id == original.context_id
    assert appended.serialize() == original.serialize()


def test_context_models_cannot_be_constructed_with_unbound_bars() -> None:
    context = _context()
    aligned = context.timeframes[0]
    timeframe_context_constructor = cast(Callable[..., object], TimeframeContext)
    multi_context_constructor = cast(Callable[..., object], MultiTimeframeContext)

    with pytest.raises(TypeError):
        timeframe_context_constructor(
            aligned.requirement,
            aligned.dataset_reference,
            aligned.availability,
            aligned.bars,
            aligned.latest_completed_bar_timestamp,
            aligned.age,
        )
    with pytest.raises(TypeError):
        multi_context_constructor(
            context.as_of,
            context.primary_timeframe,
            context.required_timeframes,
            context.completion_policy,
            context.source_consistency,
            context.timeframes,
        )


def test_construction_is_deterministic_and_orders_requirements_and_bars() -> None:
    family = _family()
    first_series = _fixture_series(family)
    reverse_series = tuple(
        _trusted_series(
            family,
            item.dataset_reference.dataset_id,
            item.timeframe,
            tuple(reversed(item.bars)),
        )
        for item in reversed(first_series)
    )
    five_minute, four_hour, daily, weekly = _timeframes()
    second = build_multi_timeframe_context(
        as_of=AS_OF,
        primary_timeframe=five_minute,
        required_timeframes=(
            ContextTimeframeRequirement(four_hour),
            ContextTimeframeRequirement(weekly),
            ContextTimeframeRequirement(daily),
        ),
        series=reverse_series,
    )

    assert second.context_id == _context(family, first_series).context_id
    assert tuple(
        requirement.timeframe.configuration_id
        for requirement in second.required_timeframes
    ) == tuple(
        sorted(timeframe.configuration_id for timeframe in (four_hour, daily, weekly))
    )


def test_missing_and_stale_timeframes_are_explicit_and_access_is_guarded() -> None:
    family = _family()
    five_minute, four_hour, daily, weekly = _timeframes()
    fixture_series = _fixture_series(family)
    empty_daily = _trusted_series(family, DAILY_ID, daily, ())
    selected_series = tuple(
        empty_daily if item.timeframe == daily else item
        for item in fixture_series
        if item.timeframe != four_hour
    )
    context = build_multi_timeframe_context(
        as_of=AS_OF,
        primary_timeframe=five_minute,
        required_timeframes=(
            ContextTimeframeRequirement(weekly, maximum_age=timedelta(hours=12)),
            ContextTimeframeRequirement(daily),
            ContextTimeframeRequirement(four_hour),
        ),
        series=selected_series,
    )

    assert context.metadata_for(weekly).availability is ContextAvailability.STALE
    assert context.metadata_for(daily).availability is ContextAvailability.MISSING
    assert context.metadata_for(daily).dataset_id == DAILY_ID
    assert context.metadata_for(four_hour).availability is ContextAvailability.MISSING
    assert context.metadata_for(four_hour).dataset_id is None
    with pytest.raises(UnavailableTimeframeError, match="no completed bar"):
        context.bars_for(daily)
    with pytest.raises(UndeclaredTimeframeError, match="not declared"):
        context.bars_for(Timeframe.us_equity(IntradayInterval(timedelta(minutes=15))))


def test_context_rejects_mixed_dataset_families() -> None:
    family = _family()
    other_family = _family(feed_scope=FeedScope.single_venue("IEX"))
    series = _fixture_series(family)
    mixed = tuple(
        _trusted_series(other_family, DAILY_ID, item.timeframe, item.bars)
        if item.timeframe == _timeframes()[2]
        else item
        for item in series
    )

    with pytest.raises(MultiTimeframeContextValidationError, match="mixed dataset"):
        _context(family, mixed)


def test_series_cannot_be_constructed_from_an_unbound_reference_and_bars() -> None:
    family = _family()
    five_minute = _timeframes()[0]

    with pytest.raises(TypeError):
        TimeframeBarSeries(family.reference(SOURCE_ID), five_minute, ())  # pyright: ignore[reportCallIssue]


def test_source_series_validates_cached_artifact_and_family_identity(
    tmp_path: Path,
) -> None:
    source, cache = _persisted_source_dataset(tmp_path)
    family = _family(source_dataset_id=source.metadata.dataset_id)

    series = TimeframeBarSeries.from_source_dataset(
        source,
        family=family,
        cache=cache,
    )

    assert series.dataset_reference.dataset_id == source.metadata.dataset_id
    assert series.bars == source.bars

    changed_bar = replace(source.bars[0], close=Decimal("100.5"))
    changed_bars = (changed_bar, *source.bars[1:])
    changed_batch = IntradayBarBatch(source.request, changed_bars)
    tampered = replace(
        source,
        bars=changed_bars,
        metadata=replace(
            source.metadata,
            batch_id=changed_batch.batch_id,
            data_sha256=sha256_hex(changed_batch.serialize()),
            quality_report=validate_intraday_coverage(
                changed_batch,
                mode=IntradayValidationMode.DIAGNOSTIC,
            ),
        ),
    )
    with pytest.raises(
        MultiTimeframeContextValidationError, match="immutable cache artifact"
    ):
        TimeframeBarSeries.from_source_dataset(
            tampered,
            family=family,
            cache=cache,
        )

    forged_id = "f" * 64
    forged = replace(
        source,
        metadata=replace(
            source.metadata,
            dataset_id=forged_id,
            normalized_location=f"intraday/datasets/{forged_id}/bars.json",
        ),
    )
    with pytest.raises(
        MultiTimeframeContextValidationError,
        match="immutable cache validation failed",
    ):
        TimeframeBarSeries.from_source_dataset(
            forged,
            family=_family(source_dataset_id=forged_id),
            cache=cache,
        )

    with pytest.raises(MultiTimeframeContextValidationError, match="family identity"):
        TimeframeBarSeries.from_source_dataset(
            source,
            family=replace(family, canonical_symbol="QQQ"),
            cache=cache,
        )


def test_session_series_rejects_bars_unbound_from_derived_dataset() -> None:
    source = _complete_source_dataset()
    daily = _timeframes()[2]
    derived = aggregate_session_dataset(
        source, daily, policy=SessionAggregationPolicy()
    )

    context_family = _context_family_for_derived(
        derived.dataset_family, derived.metadata.dataset_id, daily
    )
    series = TimeframeBarSeries.from_aggregated_session_dataset(
        derived, family=context_family
    )

    assert series.dataset_reference.dataset_id == derived.metadata.dataset_id
    assert series.dataset_reference.family_id == context_family.family_id
    assert series.bars == derived.bars

    fabricated_family = replace(
        context_family,
        aggregation_policy=AggregationPolicy(
            "fabricated_context_policy",
            "1",
            {"artifact_family_manifest_ids": [derived.dataset_family.manifest_id]},
        ),
    )
    with pytest.raises(
        MultiTimeframeContextValidationError, match="artifact family manifest"
    ):
        TimeframeBarSeries.from_aggregated_session_dataset(
            derived,
            family=fabricated_family,
        )

    foreign_bar = replace(derived.bars[0], source_dataset_id="foreign-source")
    tampered = replace(derived, bars=(foreign_bar,))
    with pytest.raises(
        MultiTimeframeContextValidationError, match="provenance mismatch"
    ):
        TimeframeBarSeries.from_aggregated_session_dataset(tampered)


def test_intraday_series_rejects_bars_unbound_from_derived_dataset() -> None:
    source = _complete_source_dataset()
    four_hour = _timeframes()[1]
    derived = aggregate_intraday_dataset(
        source, four_hour, policy=IntradayAggregationPolicy()
    )

    context_family = _context_family_for_derived(
        derived.dataset_family, derived.metadata.dataset_id, four_hour
    )
    series = TimeframeBarSeries.from_aggregated_intraday_dataset(
        derived, family=context_family
    )

    assert series.dataset_reference.dataset_id == derived.metadata.dataset_id
    assert series.dataset_reference.family_id == context_family.family_id
    assert series.bars == derived.bars

    provenance = replace(
        derived.bars[0].provenance, source_snapshot_id="foreign-source"
    )
    foreign_bar = replace(derived.bars[0], provenance=provenance)
    tampered = replace(derived, bars=(foreign_bar, *derived.bars[1:]))
    with pytest.raises(MultiTimeframeContextValidationError, match="validation failed"):
        TimeframeBarSeries.from_aggregated_intraday_dataset(tampered)


def test_context_rejects_incompatible_session_policies() -> None:
    family = _family()
    five_minute, _, _, _ = _timeframes()
    extended_daily = Timeframe(
        SessionInterval(),
        ExchangeSessionPolicy(
            scope=SessionScope.EXTENDED_HOURS,
            extended_hours_start=datetime.min.time().replace(hour=4),
            extended_hours_end=datetime.min.time().replace(hour=20),
        ),
    )

    with pytest.raises(
        MultiTimeframeContextValidationError, match="incompatible exchange session"
    ):
        build_multi_timeframe_context(
            as_of=AS_OF,
            primary_timeframe=five_minute,
            required_timeframes=(ContextTimeframeRequirement(extended_daily),),
            series=(_fixture_series(family)[0],),
        )


def test_context_identity_records_causal_configuration_and_family() -> None:
    context = _context()
    primitive = context.to_primitive()
    aligned_timeframes = cast(list[dict[str, object]], primitive["timeframes"])

    assert primitive["as_of"] == AS_OF.astimezone(UTC).isoformat()
    assert primitive["completion_policy"] == "completed_bars_only"
    assert primitive["source_consistency"] == {
        "mode": "common_dataset_family",
        "family_id": _family().family_id,
        "external_validation_policy_id": None,
    }
    assert len(cast(list[object], primitive["required_timeframes"])) == 3
    assert all(
        "feed_scope" not in cast(dict[str, object], aligned["dataset_reference"])
        for aligned in aligned_timeframes
    )
    assert context.serialize() == _context().serialize()
