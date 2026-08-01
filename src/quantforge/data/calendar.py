"""Exchange-session calculations."""

from collections.abc import Iterable
from datetime import date, datetime
from importlib import import_module
from typing import Protocol, cast

NYSE_CALENDAR = "XNYS"


class _ExchangeCalendar(Protocol):
    def sessions_in_range(self, start: date, end: date) -> Iterable[datetime]: ...


class _ExchangeCalendars(Protocol):
    def get_calendar(self, calendar: str) -> _ExchangeCalendar: ...


def expected_sessions(
    start: date, end: date, calendar: str = NYSE_CALENDAR
) -> tuple[date, ...]:
    """Return real exchange sessions in the inclusive requested interval."""
    exchange_calendars = cast(_ExchangeCalendars, import_module("exchange_calendars"))
    exchange = exchange_calendars.get_calendar(calendar)
    return tuple(
        timestamp.date() for timestamp in exchange.sessions_in_range(start, end)
    )
