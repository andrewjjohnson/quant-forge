"""Exchange-session calculations."""

from datetime import date

import exchange_calendars

NYSE_CALENDAR = "XNYS"


def expected_sessions(
    start: date, end: date, calendar: str = NYSE_CALENDAR
) -> tuple[date, ...]:
    """Return real exchange sessions in the inclusive requested interval."""
    exchange = exchange_calendars.get_calendar(calendar)
    return tuple(
        timestamp.date() for timestamp in exchange.sessions_in_range(start, end)
    )
