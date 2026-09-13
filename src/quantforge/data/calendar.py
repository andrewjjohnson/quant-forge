"""Exchange-session calculations."""

from collections.abc import Iterable
from datetime import date, datetime, timedelta
from importlib import import_module
from typing import Protocol, cast

NYSE_CALENDAR = "XNYS"


class _ExchangeCalendar(Protocol):
    @property
    def open_offset(self) -> int: ...

    @property
    def close_offset(self) -> int: ...

    @property
    def first_session(self) -> datetime: ...

    @property
    def last_session(self) -> datetime: ...

    def sessions_in_range(self, start: date, end: date) -> Iterable[datetime]: ...

    def date_to_session(self, session_date: date) -> datetime: ...

    def next_session(self, session: datetime) -> datetime: ...


class _ExchangeCalendars(Protocol):
    def get_calendar(self, calendar: str) -> _ExchangeCalendar: ...


def expected_sessions(
    start: date,
    end: date,
    calendar: str = NYSE_CALENDAR,
    *,
    include_overnight: bool = False,
) -> tuple[date, ...]:
    """Return real exchange sessions in the inclusive requested interval.

    With ``include_overnight``, interpret boundaries as exchange-local calendar
    dates and include trade-date labels offset from them by overnight sessions.
    Callers must filter the resulting sessions' actual UTC bar boundaries.
    """
    exchange_calendars = cast(_ExchangeCalendars, import_module("exchange_calendars"))
    exchange = exchange_calendars.get_calendar(calendar)
    if include_overnight:
        earliest_offset = timedelta(
            days=min(0, exchange.open_offset, exchange.close_offset)
        )
        latest_offset = timedelta(
            days=max(0, exchange.open_offset, exchange.close_offset)
        )
        first, last = exchange.first_session.date(), exchange.last_session.date()
        if start < first + earliest_offset or end > last + latest_offset:
            raise ValueError(
                "requested local dates exceed the exchange calendar bounds"
            )
        # Clip only the extra candidate labels, not the requested local dates.
        # This preserves valid intervals at the calendar's first/last session.
        start = max(first, start - latest_offset)
        end = min(last, end - earliest_offset)
    return tuple(
        timestamp.date() for timestamp in exchange.sessions_in_range(start, end)
    )


def next_session_after(session_date: date, calendar: str = NYSE_CALENDAR) -> date:
    """Return the direct calendar successor of a valid exchange session."""
    exchange_calendars = cast(_ExchangeCalendars, import_module("exchange_calendars"))
    exchange = exchange_calendars.get_calendar(calendar)
    session = exchange.date_to_session(session_date)
    return exchange.next_session(session).date()
