from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
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
    IntradayBar,
    IntradayBarProvenance,
    MultiTimeframeContextValidationError,
    TimeframeBarSeries,
    UnavailableTimeframeError,
    UndeclaredTimeframeError,
    build_multi_timeframe_context,
)
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


def _family(*, feed_scope: FeedScope | None = None) -> DatasetFamily:
    five_minute, four_hour, daily, weekly = _timeframes()
    return DatasetFamily(
        canonical_symbol="SPY",
        provider_name="fixture-provider",
        feed_scope=feed_scope or FeedScope.consolidated(),
        adjustment_basis=_adjustment_basis(),
        aggregation_policy=AggregationPolicy(
            "quantforge_fixture_aggregation", "1", {"missing": "reject"}
        ),
        canonical_source_snapshot_id=SOURCE_ID,
        datasets=(
            DatasetLineage(
                SOURCE_ID,
                five_minute,
                SOURCE_ID,
                None,
                (FOUR_HOUR_ID, DAILY_ID, WEEKLY_ID),
            ),
            DatasetLineage(FOUR_HOUR_ID, four_hour, SOURCE_ID, SOURCE_ID),
            DatasetLineage(DAILY_ID, daily, SOURCE_ID, SOURCE_ID),
            DatasetLineage(WEEKLY_ID, weekly, SOURCE_ID, SOURCE_ID),
        ),
    )


def _provenance() -> IntradayBarProvenance:
    return IntradayBarProvenance(
        provider_name="fixture-provider",
        provider_symbol="SPY",
        adapter_version="fixture-1",
        retrieved_at=RETRIEVED_AT,
        source_request_id="fixture-request",
        source_snapshot_id=SOURCE_ID,
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
        TimeframeBarSeries(family.reference(SOURCE_ID), five_minute, five_minute_bars),
        TimeframeBarSeries(family.reference(FOUR_HOUR_ID), four_hour, four_hour_bars),
        TimeframeBarSeries(family.reference(DAILY_ID), daily, daily_bars),
        TimeframeBarSeries(family.reference(WEEKLY_ID), weekly, weekly_bars),
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
        replace(item, bars=(*item.bars, future_bar))
        if item.timeframe == five_minute
        else item
        for item in original_series
    )
    appended = _context(family, appended_series)

    assert appended.context_id == original.context_id
    assert appended.serialize() == original.serialize()


def test_context_model_rejects_moving_as_of_before_an_exposed_bar() -> None:
    context = _context()

    with pytest.raises(
        MultiTimeframeContextValidationError, match="unavailable at context as-of"
    ):
        replace(context, as_of=AS_OF - timedelta(minutes=1))


def test_construction_is_deterministic_and_orders_requirements_and_bars() -> None:
    family = _family()
    first_series = _fixture_series(family)
    reverse_series = tuple(
        replace(item, bars=tuple(reversed(item.bars)))
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
    empty_daily = TimeframeBarSeries(family.reference(DAILY_ID), daily, ())
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
        replace(item, dataset_reference=other_family.reference(DAILY_ID))
        if item.timeframe == _timeframes()[2]
        else item
        for item in series
    )

    with pytest.raises(MultiTimeframeContextValidationError, match="mixed dataset"):
        _context(family, mixed)


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

    assert primitive["as_of"] == AS_OF.astimezone(UTC).isoformat()
    assert primitive["completion_policy"] == "completed_bars_only"
    assert primitive["source_consistency"] == {
        "mode": "common_dataset_family",
        "family_id": _family().family_id,
        "external_validation_policy_id": None,
    }
    assert len(cast(list[object], primitive["required_timeframes"])) == 3
    assert context.serialize() == _context().serialize()
