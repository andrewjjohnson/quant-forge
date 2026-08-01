"""Deterministic canonical-bar validation."""

from datetime import date
from decimal import Decimal

from quantforge.data.calendar import expected_sessions
from quantforge.data.exceptions import ValidationError
from quantforge.data.models import DailyBar


def validate_bars(
    bars: tuple[DailyBar, ...],
    symbol: str,
    start: date,
    end: date,
    calendar: str,
    *,
    strict: bool = True,
) -> tuple[DailyBar, ...]:
    """Stable-sort bars, reject invalid values, and optionally require every session."""
    if start > end:
        raise ValidationError("start must be on or before end")
    if not bars:
        raise ValidationError("provider returned no bars")
    ordered = tuple(sorted(bars, key=lambda bar: bar.session_date))
    sessions = [bar.session_date for bar in ordered]
    if len(sessions) != len(set(sessions)):
        raise ValidationError("duplicate trading sessions")
    for bar in ordered:
        if bar.symbol != symbol:
            raise ValidationError("inconsistent symbols")
        if not start <= bar.session_date <= end:
            raise ValidationError("session outside requested range")
        values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        if any(not value.is_finite() for value in values):
            raise ValidationError("OHLCV values must be finite")
        if any(value <= Decimal(0) for value in values):
            raise ValidationError("OHLCV values must be positive")
        if bar.high < max(bar.open, bar.low, bar.close):
            raise ValidationError("high is below another OHLC price")
        if bar.low > min(bar.open, bar.high, bar.close):
            raise ValidationError("low is above another OHLC price")
    missing = tuple(
        sorted(set(expected_sessions(start, end, calendar)) - set(sessions))
    )
    if strict and missing:
        rendered = tuple(item.isoformat() for item in missing)
        raise ValidationError(
            f"missing expected sessions: {', '.join(rendered)}",
            missing_sessions=rendered,
        )
    return ordered
