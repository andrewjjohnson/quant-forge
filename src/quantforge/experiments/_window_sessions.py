"""Bind recorded decision instants to exchange-session labels without market bars."""

from bisect import bisect_right
from datetime import datetime, time
from zoneinfo import ZoneInfo

from quantforge.configuration import PrimitiveMapping
from quantforge.data.calendar import expected_sessions
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.timeframes import (
    ExchangeSessionPolicy,
    SessionScope,
    resolve_exchange_session,
)


def scheduled_sessions(schedule: PrimitiveMapping) -> dict[str, str]:
    """Use actual exchange opens/closes, including overnight session labels."""
    session = mapping(mapping(schedule.get("primary_timeframe")).get("session_policy"))
    recorded = schedule.get("decision_timestamps")
    if not isinstance(recorded, list):
        raise ManifestError("window schedule timestamps must be an array")
    try:
        policy = ExchangeSessionPolicy(
            calendar_name=text(session.get("calendar")),
            timezone_name=text(session.get("timezone")),
            scope=SessionScope(text(session.get("scope"))),
            extended_hours_start=None
            if session.get("extended_hours_start") is None
            else time.fromisoformat(text(session["extended_hours_start"])),
            extended_hours_end=None
            if session.get("extended_hours_end") is None
            else time.fromisoformat(text(session["extended_hours_end"])),
        )
        timestamps = [datetime.fromisoformat(text(item)) for item in recorded]
        if any(item.utcoffset() is None for item in timestamps) or timestamps != sorted(
            set(timestamps)
        ):
            raise ManifestError(
                "window schedule timestamps must be ordered unique instants"
            )
        if not timestamps:
            return {}
        timezone = ZoneInfo(policy.timezone_name)
        dates = expected_sessions(
            timestamps[0].astimezone(timezone).date(),
            timestamps[-1].astimezone(timezone).date(),
            policy.calendar_name,
            include_overnight=policy.scope is SessionScope.REGULAR_HOURS,
        )
        result: dict[str, str] = {}
        for label in dates:
            window = resolve_exchange_session(label, policy)
            for index in range(
                bisect_right(timestamps, window.open_timestamp),
                bisect_right(timestamps, window.close_timestamp),
            ):
                key = text(recorded[index])
                if key in result:
                    raise ManifestError(
                        "window decision has ambiguous exchange session"
                    )
                result[key] = label.isoformat()
        if len(result) != len(recorded):
            raise ManifestError(
                "window decision is outside scheduled exchange sessions"
            )
        return result
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError("window schedule exchange sessions are invalid") from error
