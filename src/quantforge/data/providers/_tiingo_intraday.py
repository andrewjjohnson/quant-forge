"""Tiingo-only session acquisition planning; generic chunk contracts stay intact."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from quantforge.data.calendar import expected_sessions
from quantforge.data.intraday import IntradayBarRequest
from quantforge.timeframes import resolve_exchange_session


@dataclass(frozen=True, slots=True)
class TiingoIntradayChunk:
    """One wire response, its retention window, and logical coverage envelope.

    Coverage envelopes partition the logical request, including closed-market
    gaps. Only the intersected session window is requested and retained. A gap
    contributes no expected observations and never requires a separate response.
    """

    session_date: date
    retention_start: datetime
    retention_end: datetime
    coverage_start: datetime
    coverage_end: datetime


def plan_tiingo_intraday_chunks(
    request: IntradayBarRequest, maximum_duration: timedelta
) -> tuple[TiingoIntradayChunk, ...]:
    """Plan ordered XNYS/RTH subrequests without requesting non-session dates."""
    policy = request.timeframe.session_policy
    timezone = ZoneInfo(policy.timezone_name)
    sessions = expected_sessions(
        request.start_timestamp.astimezone(timezone).date(),
        (request.end_timestamp - timedelta(microseconds=1)).astimezone(timezone).date(),
        policy.calendar_name,
    )
    windows: list[tuple[date, datetime, datetime]] = []
    for session_date in sessions:
        session = resolve_exchange_session(session_date, policy)
        cursor = max(request.start_timestamp, session.open_timestamp)
        session_end = min(request.end_timestamp, session.close_timestamp)
        while cursor < session_end:
            chunk_end = min(cursor + maximum_duration, session_end)
            windows.append((session_date, cursor, chunk_end))
            cursor = chunk_end
    return tuple(
        TiingoIntradayChunk(
            session_date=session_date,
            retention_start=start,
            retention_end=end,
            coverage_start=request.start_timestamp if index == 0 else start,
            coverage_end=(
                windows[index + 1][1]
                if index + 1 < len(windows)
                else request.end_timestamp
            ),
        )
        for index, (session_date, start, end) in enumerate(windows)
    )
