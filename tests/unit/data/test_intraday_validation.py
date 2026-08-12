import json
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import configuration_identity
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    FeedScope,
    IntradayBar,
    IntradayBarBatch,
    IntradayBarProvenance,
    IntradayBarRequest,
    IntradayContractValidationError,
    IntradayCoverageStatus,
    IntradayCoverageValidationError,
    IntradayValidationMode,
    validate_intraday_coverage,
)
from quantforge.timeframes import (
    BarCompletion,
    ExchangeSessionPolicy,
    IntradayInterval,
    SessionScope,
    Timeframe,
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


def _timeframe(
    duration: timedelta = timedelta(minutes=5),
    *,
    session_scope: SessionScope = SessionScope.REGULAR_HOURS,
) -> Timeframe:
    policy = (
        ExchangeSessionPolicy()
        if session_scope is SessionScope.REGULAR_HOURS
        else ExchangeSessionPolicy(
            scope=SessionScope.EXTENDED_HOURS,
            extended_hours_start=time(4),
            extended_hours_end=time(20),
        )
    )
    return (
        Timeframe.us_equity(
            IntradayInterval(duration),
        )
        if session_scope is SessionScope.REGULAR_HOURS
        else Timeframe(IntradayInterval(duration), session_policy=policy)
    )


def _request(
    start_timestamp: datetime,
    end_timestamp: datetime,
    *,
    timeframe: Timeframe | None = None,
) -> IntradayBarRequest:
    return IntradayBarRequest(
        symbol="SPY",
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        timeframe=timeframe or _timeframe(),
        feed_scope=FeedScope.consolidated(),
        adjustment_basis=_adjustment_basis(),
    )


def _provenance(request: IntradayBarRequest) -> IntradayBarProvenance:
    return IntradayBarProvenance(
        provider_name="fixture",
        provider_symbol="SPY",
        adapter_version="test-1",
        retrieved_at=RETRIEVED_AT,
        source_request_id=request.request_id,
        source_snapshot_id="fixture-snapshot",
        feed_scope=request.feed_scope,
        adjustment_basis=request.adjustment_basis,
    )


def _bar(
    request: IntradayBarRequest,
    session_date: date,
    start_timestamp: datetime,
    end_timestamp: datetime,
    completion: BarCompletion,
    *,
    volume: Decimal = Decimal("100"),
) -> IntradayBar:
    return IntradayBar(
        symbol=request.symbol,
        session_date=session_date,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        timeframe=request.timeframe,
        completion=completion,
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal("101"),
        volume=volume,
        provenance=_provenance(request),
    )


def _session_bars(
    request: IntradayBarRequest,
    session_date: date,
    *,
    missing_start: datetime | None = None,
    zero_volume_start: datetime | None = None,
) -> tuple[IntradayBar, ...]:
    session = resolve_exchange_session(session_date, request.timeframe.session_policy)
    duration = request.source_interval.nominal_duration
    bars: list[IntradayBar] = []
    start_timestamp = session.open_timestamp
    while start_timestamp < session.close_timestamp:
        end_timestamp = min(start_timestamp + duration, session.close_timestamp)
        completion = (
            BarCompletion.COMPLETED
            if end_timestamp - start_timestamp == duration
            else BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL
        )
        if start_timestamp != missing_start:
            bars.append(
                _bar(
                    request,
                    session_date,
                    start_timestamp,
                    end_timestamp,
                    completion,
                    volume=(
                        Decimal(0)
                        if start_timestamp == zero_volume_start
                        else Decimal("100")
                    ),
                )
            )
        start_timestamp = end_timestamp
    return tuple(bars)


def test_complete_normal_session_report_is_serializable_and_nonmutating() -> None:
    request = _request(
        datetime(2024, 7, 1, 9, 30, tzinfo=NEW_YORK),
        datetime(2024, 7, 1, 16, tzinfo=NEW_YORK),
    )
    batch = IntradayBarBatch(request, _session_bars(request, date(2024, 7, 1)))
    before = batch.serialize()

    report = validate_intraday_coverage(batch, mode=IntradayValidationMode.DIAGNOSTIC)

    assert report.status is IntradayCoverageStatus.COMPLETE
    assert report.expected_completed_interval_count == 78
    assert report.observed_bar_count == 78
    assert report.incomplete_sessions == ()
    assert report.source_interval == request.source_interval
    assert report.feed_scope == request.feed_scope
    assert report.session_scope == SessionScope.REGULAR_HOURS.value
    assert json.loads(report.serialize()) == report.to_primitive()
    assert report.report_id == configuration_identity(report.to_primitive())
    assert batch.serialize() == before


def test_missing_five_minute_bar_is_reported_and_strict_mode_raises() -> None:
    request = _request(
        datetime(2024, 7, 1, 9, 30, tzinfo=NEW_YORK),
        datetime(2024, 7, 1, 16, tzinfo=NEW_YORK),
    )
    missing_start = datetime(2024, 7, 1, 10, tzinfo=NEW_YORK).astimezone(UTC)
    batch = IntradayBarBatch(
        request,
        _session_bars(request, date(2024, 7, 1), missing_start=missing_start),
    )

    diagnostic = validate_intraday_coverage(
        batch, mode=IntradayValidationMode.DIAGNOSTIC
    )

    assert diagnostic.status is IntradayCoverageStatus.INCOMPLETE
    assert diagnostic.incomplete_sessions == (date(2024, 7, 1),)
    assert len(diagnostic.missing_intervals) == 1
    assert diagnostic.missing_intervals[0].start_timestamp == missing_start
    assert diagnostic.missing_intervals[0].end_timestamp == missing_start + timedelta(
        minutes=5
    )
    with pytest.raises(IntradayCoverageValidationError) as raised:
        validate_intraday_coverage(batch)
    assert raised.value.report.missing_intervals[0].start_timestamp == missing_start


def test_weekends_holidays_and_early_close_are_not_false_gaps() -> None:
    request = _request(
        datetime(2024, 11, 29, 9, 30, tzinfo=NEW_YORK),
        datetime(2024, 12, 2, 16, tzinfo=NEW_YORK),
    )
    bars = (
        *_session_bars(request, date(2024, 11, 29)),
        *_session_bars(request, date(2024, 12, 2)),
    )

    report = validate_intraday_coverage(
        IntradayBarBatch(request, bars), mode=IntradayValidationMode.DIAGNOSTIC
    )

    assert report.is_complete
    assert tuple(session.session_date for session in report.sessions) == (
        date(2024, 11, 29),
        date(2024, 12, 2),
    )
    assert report.sessions[0].session_close_timestamp == datetime(
        2024, 11, 29, 18, tzinfo=UTC
    )
    assert report.sessions[0].expected_interval_count == 42
    assert report.sessions[1].expected_interval_count == 78

    holiday_request = _request(
        datetime(2024, 7, 4, tzinfo=NEW_YORK),
        datetime(2024, 7, 5, tzinfo=NEW_YORK),
    )
    holiday_report = validate_intraday_coverage(
        IntradayBarBatch(holiday_request, ()),
        mode=IntradayValidationMode.DIAGNOSTIC,
    )
    assert holiday_report.is_complete
    assert holiday_report.expected_completed_interval_count == 0
    assert holiday_report.sessions == ()


def test_early_close_terminal_partial_bar_is_complete() -> None:
    request = _request(
        datetime(2024, 11, 29, 9, 30, tzinfo=NEW_YORK),
        datetime(2024, 11, 29, 13, tzinfo=NEW_YORK),
        timeframe=_timeframe(timedelta(hours=4)),
    )
    bars = _session_bars(request, date(2024, 11, 29))

    report = validate_intraday_coverage(IntradayBarBatch(request, bars))

    assert report.is_complete
    assert report.expected_completed_interval_count == 1
    assert bars[0].completion is BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL
    assert bars[0].actual_duration == timedelta(hours=3, minutes=30)


def test_dst_boundary_uses_actual_exchange_open_offsets() -> None:
    request = _request(
        datetime(2024, 3, 8, 9, 30, tzinfo=NEW_YORK),
        datetime(2024, 3, 11, 16, tzinfo=NEW_YORK),
        timeframe=_timeframe(timedelta(hours=1)),
    )

    report = validate_intraday_coverage(
        IntradayBarBatch(request, ()), mode=IntradayValidationMode.DIAGNOSTIC
    )

    assert tuple(session.session_open_timestamp for session in report.sessions) == (
        datetime(2024, 3, 8, 14, 30, tzinfo=UTC),
        datetime(2024, 3, 11, 13, 30, tzinfo=UTC),
    )
    assert tuple(session.session_date for session in report.sessions) == (
        date(2024, 3, 8),
        date(2024, 3, 11),
    )


def test_extended_hours_scope_is_explicit_and_regular_scope_rejects_bar() -> None:
    extended_timeframe = _timeframe(
        timedelta(hours=1), session_scope=SessionScope.EXTENDED_HOURS
    )
    extended_request = _request(
        datetime(2024, 7, 1, 4, tzinfo=NEW_YORK),
        datetime(2024, 7, 1, 20, tzinfo=NEW_YORK),
        timeframe=extended_timeframe,
    )
    extended_batch = IntradayBarBatch(
        extended_request,
        _session_bars(extended_request, date(2024, 7, 1)),
    )

    report = validate_intraday_coverage(
        extended_batch, mode=IntradayValidationMode.DIAGNOSTIC
    )

    assert report.is_complete
    assert report.session_scope == SessionScope.EXTENDED_HOURS.value
    assert report.expected_completed_interval_count == 16

    regular_request = _request(
        datetime(2024, 7, 1, 4, tzinfo=NEW_YORK),
        datetime(2024, 7, 1, 20, tzinfo=NEW_YORK),
        timeframe=_timeframe(timedelta(hours=1)),
    )
    with pytest.raises(IntradayContractValidationError, match="belong"):
        _bar(
            regular_request,
            date(2024, 7, 1),
            datetime(2024, 7, 1, 4, tzinfo=NEW_YORK),
            datetime(2024, 7, 1, 5, tzinfo=NEW_YORK),
            BarCompletion.COMPLETED,
        )


def test_zero_volume_is_reported_without_inventing_or_rejecting_valid_bar() -> None:
    request = _request(
        datetime(2024, 7, 1, 9, 30, tzinfo=NEW_YORK),
        datetime(2024, 7, 1, 9, 35, tzinfo=NEW_YORK),
    )
    zero_start = request.start_timestamp
    batch = IntradayBarBatch(
        request,
        _session_bars(request, date(2024, 7, 1), zero_volume_start=zero_start)[:1],
    )

    report = validate_intraday_coverage(batch)

    assert report.is_complete
    assert report.has_warnings
    assert len(report.zero_volume_intervals) == 1
    assert report.observed_bar_count == len(batch.bars) == 1


def test_batch_rejects_overlapping_timestamps_defensively() -> None:
    request = _request(
        datetime(2024, 7, 1, 9, 30, tzinfo=NEW_YORK),
        datetime(2024, 7, 1, 9, 40, tzinfo=NEW_YORK),
    )
    first, second = _session_bars(request, date(2024, 7, 1))[:2]
    overlapping_first = replace(first)
    object.__setattr__(
        overlapping_first,
        "end_timestamp",
        second.start_timestamp + timedelta(minutes=1),
    )

    with pytest.raises(IntradayContractValidationError, match="overlapping"):
        IntradayBarBatch(request, (overlapping_first, second))


def test_batch_rejects_bar_outside_requested_range() -> None:
    request = _request(
        datetime(2024, 7, 1, 9, 35, tzinfo=NEW_YORK),
        datetime(2024, 7, 1, 9, 40, tzinfo=NEW_YORK),
    )
    out_of_range = _bar(
        request,
        date(2024, 7, 1),
        datetime(2024, 7, 1, 9, 30, tzinfo=NEW_YORK),
        datetime(2024, 7, 1, 9, 35, tzinfo=NEW_YORK),
        BarCompletion.COMPLETED,
    )

    with pytest.raises(IntradayContractValidationError, match="outside"):
        IntradayBarBatch(request, (out_of_range,))
