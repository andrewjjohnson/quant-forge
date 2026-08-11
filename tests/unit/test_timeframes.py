from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import configuration_identity
from quantforge.data.models import SCHEMA_VERSION, DailyBar
from quantforge.timeframes import (
    DEFAULT_US_EQUITY_TIMEFRAME,
    BarCompletion,
    BarLabel,
    CrossSessionPolicy,
    DevelopingBarExposure,
    ExchangeSessionPolicy,
    IntradayAnchor,
    IntradayBarWindow,
    IntradayInterval,
    SessionInterval,
    SessionScope,
    Timeframe,
    TimeframeValidationError,
    TradingWeekInterval,
    resolve_exchange_session,
    resolve_trading_week,
)

NEW_YORK = ZoneInfo("America/New_York")


def _local_timestamp(
    year: int, month: int, day: int, hour: int, minute: int = 0
) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NEW_YORK)


def _four_hour_timeframe(
    *,
    exposure: DevelopingBarExposure = DevelopingBarExposure.EXCLUDE,
    cross_session_policy: CrossSessionPolicy = CrossSessionPolicy.PROHIBITED,
    label: BarLabel = BarLabel.START,
) -> Timeframe:
    return Timeframe(
        IntradayInterval(timedelta(hours=4), cross_session_policy=cross_session_policy),
        developing_bar_exposure=exposure,
        bar_label=label,
    )


def test_default_us_equity_configuration_has_stable_complete_identity() -> None:
    configuration = DEFAULT_US_EQUITY_TIMEFRAME.to_primitive()

    assert configuration == {
        "schema_version": "1",
        "interval": {"kind": "exchange_sessions", "session_count": 1},
        "session_policy": {
            "calendar": "XNYS",
            "timezone": "America/New_York",
            "scope": "regular_hours",
            "extended_hours_start": None,
            "extended_hours_end": None,
        },
        "bar_label": "bar_start",
        "developing_bar_exposure": "completed_only",
    }
    assert DEFAULT_US_EQUITY_TIMEFRAME.configuration_id == configuration_identity(
        configuration
    )
    assert Timeframe.us_equity().configuration_id == (
        DEFAULT_US_EQUITY_TIMEFRAME.configuration_id
    )


def test_every_material_timeframe_policy_changes_configuration_identity() -> None:
    baseline = _four_hour_timeframe()
    extended_policy = ExchangeSessionPolicy(
        scope=SessionScope.EXTENDED_HOURS,
        extended_hours_start=time(4),
        extended_hours_end=time(20),
    )
    variants = (
        Timeframe.us_equity(IntradayInterval(timedelta(hours=2))),
        Timeframe.us_equity(
            IntradayInterval(
                timedelta(hours=4),
                anchor=IntradayAnchor.CLOCK,
                clock_anchor=time(9, 30),
            )
        ),
        _four_hour_timeframe(cross_session_policy=CrossSessionPolicy.PERMITTED),
        _four_hour_timeframe(label=BarLabel.END),
        _four_hour_timeframe(exposure=DevelopingBarExposure.INCLUDE),
        Timeframe(baseline.interval, session_policy=extended_policy),
        Timeframe.us_equity(SessionInterval(2)),
        Timeframe.us_equity(TradingWeekInterval()),
    )

    identities = {
        baseline.configuration_id,
        *(item.configuration_id for item in variants),
    }
    assert len(identities) == len(variants) + 1


@pytest.mark.parametrize(
    "duration",
    [
        timedelta(minutes=1),
        timedelta(minutes=5),
        timedelta(minutes=15),
        timedelta(minutes=30),
        timedelta(hours=1),
        timedelta(hours=2),
        timedelta(hours=4),
        timedelta(hours=6, minutes=30),
    ],
)
def test_intraday_duration_property_accepts_arbitrary_positive_subday_values(
    duration: timedelta,
) -> None:
    interval = IntradayInterval(duration)
    assert interval.nominal_duration == duration
    assert interval.to_primitive()["nominal_duration_microseconds"] == (
        duration // timedelta(microseconds=1)
    )


@pytest.mark.parametrize(
    "duration", [timedelta(0), timedelta(microseconds=-1), timedelta(days=1)]
)
def test_intraday_interval_cannot_alias_daily_or_weekly_durations(
    duration: timedelta,
) -> None:
    with pytest.raises(TimeframeValidationError, match="shorter than one day"):
        IntradayInterval(duration)


@pytest.mark.parametrize(
    "interval",
    [
        SessionInterval(),
        SessionInterval(3),
        TradingWeekInterval(),
        TradingWeekInterval(2),
    ],
)
def test_daily_and_weekly_intervals_are_exchange_counts_not_timedeltas(
    interval: SessionInterval | TradingWeekInterval,
) -> None:
    primitive = interval.to_primitive()
    assert "nominal_duration_microseconds" not in primitive
    assert primitive["kind"] in {
        "exchange_sessions",
        "exchange_trading_weeks",
    }


@pytest.mark.parametrize(
    "factory", [lambda: SessionInterval(0), lambda: TradingWeekInterval(False)]
)
def test_exchange_counts_require_positive_integers(
    factory: Callable[[], SessionInterval | TradingWeekInterval],
) -> None:
    with pytest.raises(TimeframeValidationError, match="positive integer"):
        factory()


def test_legacy_daily_bar_contract_and_schema_remain_unchanged() -> None:
    legacy_bar = DailyBar(
        "SPY",
        date(2024, 7, 1),
        Decimal("100"),
        Decimal("102"),
        Decimal("99"),
        Decimal("101"),
        Decimal("1000"),
    )

    assert SCHEMA_VERSION == "4"
    assert legacy_bar.session_date == date(2024, 7, 1)
    assert DEFAULT_US_EQUITY_TIMEFRAME.interval == SessionInterval()


def test_four_hour_bars_represent_full_and_completed_partial_duration() -> None:
    timeframe = _four_hour_timeframe()
    full_bar = IntradayBarWindow(
        timeframe,
        date(2024, 7, 1),
        _local_timestamp(2024, 7, 1, 9, 30),
        _local_timestamp(2024, 7, 1, 13, 30),
        BarCompletion.COMPLETED,
    )
    terminal_bar = IntradayBarWindow(
        timeframe,
        date(2024, 7, 1),
        _local_timestamp(2024, 7, 1, 13, 30),
        _local_timestamp(2024, 7, 1, 16),
        BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL,
    )

    assert full_bar.nominal_duration == full_bar.actual_duration == timedelta(hours=4)
    assert not full_bar.is_partial_duration
    assert terminal_bar.nominal_duration == timedelta(hours=4)
    assert terminal_bar.actual_duration == timedelta(hours=2, minutes=30)
    assert terminal_bar.is_partial_duration
    assert terminal_bar.label_timestamp == datetime(2024, 7, 1, 17, 30, tzinfo=UTC)
    assert terminal_bar.to_primitive()["completion"] == (
        "completed_partial_duration_terminal"
    )


def test_completed_partial_terminal_is_distinct_from_developing() -> None:
    session_date = date(2024, 7, 1)
    start = _local_timestamp(2024, 7, 1, 13, 30)

    with pytest.raises(TimeframeValidationError, match="nominal duration"):
        IntradayBarWindow(
            _four_hour_timeframe(),
            session_date,
            start,
            _local_timestamp(2024, 7, 1, 16),
            BarCompletion.COMPLETED,
        )
    with pytest.raises(TimeframeValidationError, match="explicit include-developing"):
        IntradayBarWindow(
            _four_hour_timeframe(),
            session_date,
            start,
            _local_timestamp(2024, 7, 1, 15),
            BarCompletion.DEVELOPING,
        )

    developing_timeframe = _four_hour_timeframe(exposure=DevelopingBarExposure.INCLUDE)
    developing = IntradayBarWindow(
        developing_timeframe,
        session_date,
        start,
        _local_timestamp(2024, 7, 1, 15),
        BarCompletion.DEVELOPING,
    )
    assert developing.completion is BarCompletion.DEVELOPING
    assert not developing.is_partial_duration
    with pytest.raises(TimeframeValidationError, match="not developing"):
        IntradayBarWindow(
            developing_timeframe,
            session_date,
            start,
            _local_timestamp(2024, 7, 1, 16),
            BarCompletion.DEVELOPING,
        )


def test_intraday_bars_cannot_silently_cross_regular_session_close() -> None:
    start = _local_timestamp(2024, 7, 1, 13, 30)
    end = _local_timestamp(2024, 7, 1, 17, 30)

    with pytest.raises(TimeframeValidationError, match="cannot cross"):
        IntradayBarWindow(
            _four_hour_timeframe(),
            date(2024, 7, 1),
            start,
            end,
            BarCompletion.COMPLETED,
        )

    explicitly_permitted = IntradayBarWindow(
        _four_hour_timeframe(cross_session_policy=CrossSessionPolicy.PERMITTED),
        date(2024, 7, 1),
        start,
        end,
        BarCompletion.COMPLETED,
    )
    assert explicitly_permitted.actual_duration == timedelta(hours=4)


def test_session_and_clock_anchors_are_distinct_explicit_policies() -> None:
    with pytest.raises(TimeframeValidationError, match="not aligned"):
        IntradayBarWindow(
            Timeframe.us_equity(IntradayInterval(timedelta(hours=1))),
            date(2024, 7, 1),
            _local_timestamp(2024, 7, 1, 10),
            _local_timestamp(2024, 7, 1, 11),
            BarCompletion.COMPLETED,
        )

    clock_aligned = IntradayBarWindow(
        Timeframe.us_equity(
            IntradayInterval(
                timedelta(hours=1),
                anchor=IntradayAnchor.CLOCK,
                clock_anchor=time(9),
            )
        ),
        date(2024, 7, 1),
        _local_timestamp(2024, 7, 1, 10),
        _local_timestamp(2024, 7, 1, 11),
        BarCompletion.COMPLETED,
    )
    assert clock_aligned.actual_duration == timedelta(hours=1)


def test_early_close_produces_completed_partial_duration_terminal_bar() -> None:
    session = resolve_exchange_session(date(2024, 11, 29))
    assert session.open_timestamp.astimezone(NEW_YORK).time() == time(9, 30)
    assert session.close_timestamp.astimezone(NEW_YORK).time() == time(13)

    terminal = IntradayBarWindow(
        _four_hour_timeframe(),
        session.session_date,
        session.open_timestamp,
        session.close_timestamp,
        BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL,
    )
    assert terminal.actual_duration == timedelta(hours=3, minutes=30)


@pytest.mark.parametrize("non_session", [date(2024, 7, 4), date(2024, 7, 6)])
def test_exchange_holidays_and_weekends_are_not_sessions(non_session: date) -> None:
    with pytest.raises(TimeframeValidationError, match="not an exchange session"):
        resolve_exchange_session(non_session)


def test_exchange_trading_week_excludes_holiday_but_not_neighboring_sessions() -> None:
    week = resolve_trading_week(date(2024, 7, 4))
    assert week.week_start == date(2024, 7, 1)
    assert tuple(session.session_date for session in week.sessions) == (
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
    )


def test_dst_preserves_local_session_times_and_changes_utc() -> None:
    before_dst = resolve_exchange_session(date(2024, 3, 8))
    after_dst = resolve_exchange_session(date(2024, 3, 11))

    assert before_dst.open_timestamp == datetime(2024, 3, 8, 14, 30, tzinfo=UTC)
    assert after_dst.open_timestamp == datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
    assert before_dst.open_timestamp.astimezone(NEW_YORK).time() == time(9, 30)
    assert after_dst.open_timestamp.astimezone(NEW_YORK).time() == time(9, 30)
    assert (
        before_dst.actual_duration
        == after_dst.actual_duration
        == timedelta(hours=6, minutes=30)
    )


def test_end_label_changes_label_only_not_completion_boundary() -> None:
    start_labeled = IntradayBarWindow(
        _four_hour_timeframe(),
        date(2024, 7, 1),
        _local_timestamp(2024, 7, 1, 9, 30),
        _local_timestamp(2024, 7, 1, 13, 30),
        BarCompletion.COMPLETED,
    )
    end_labeled = replace(
        start_labeled,
        timeframe=_four_hour_timeframe(label=BarLabel.END),
    )

    assert start_labeled.label_timestamp == start_labeled.start_timestamp
    assert end_labeled.label_timestamp == end_labeled.end_timestamp
    assert start_labeled.end_timestamp == end_labeled.end_timestamp


def test_extended_hours_require_explicit_valid_local_boundaries() -> None:
    with pytest.raises(TimeframeValidationError, match="must be earlier"):
        ExchangeSessionPolicy(
            scope=SessionScope.EXTENDED_HOURS,
            extended_hours_start=time(20),
            extended_hours_end=time(4),
        )
    with pytest.raises(TimeframeValidationError, match="must match"):
        ExchangeSessionPolicy(timezone_name="UTC")

    policy = ExchangeSessionPolicy(
        scope=SessionScope.EXTENDED_HOURS,
        extended_hours_start=time(4),
        extended_hours_end=time(20),
    )
    session = resolve_exchange_session(date(2024, 7, 1), policy)
    assert session.open_timestamp.astimezone(NEW_YORK).time() == time(4)
    assert session.close_timestamp.astimezone(NEW_YORK).time() == time(20)
