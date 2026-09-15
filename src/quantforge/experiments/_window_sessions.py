"""Validate complete recorded QF-42 calendars without loading market observations."""

from datetime import datetime, time, timedelta
from typing import cast

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.prediction.window import PredictionDecisionSchedule
from quantforge.timeframes import (
    BarLabel,
    CrossSessionPolicy,
    DevelopingBarExposure,
    ExchangeSessionPolicy,
    IntradayAnchor,
    IntradayInterval,
    SessionScope,
    Timeframe,
)


def _local_time(value: Primitive) -> time | None:
    return None if value is None else time.fromisoformat(text(value))


def scheduled_sessions(schedule: PrimitiveMapping) -> dict[str, str]:
    """Require every calendar-derived primary bar end in the recorded interval.

    QF-42's schedule is a pure calendar/timeframe contract. This constructs no
    market bars, research contexts, decisions, predictions or outcomes.
    """
    if schedule.get("schema_version") != "1":
        raise ManifestError("unsupported prediction window schedule schema version")
    try:
        configuration = mapping(schedule.get("primary_timeframe"))
        interval = mapping(configuration.get("interval"))
        session = mapping(configuration.get("session_policy"))
        timeframe = Timeframe(
            IntradayInterval(
                timedelta(
                    microseconds=cast(int, interval["nominal_duration_microseconds"])
                ),
                anchor=IntradayAnchor(text(interval.get("anchor"))),
                clock_anchor=_local_time(interval.get("clock_anchor")),
                cross_session_policy=CrossSessionPolicy(
                    text(interval.get("cross_session_policy"))
                ),
            ),
            session_policy=ExchangeSessionPolicy(
                calendar_name=text(session.get("calendar")),
                timezone_name=text(session.get("timezone")),
                scope=SessionScope(text(session.get("scope"))),
                extended_hours_start=_local_time(session.get("extended_hours_start")),
                extended_hours_end=_local_time(session.get("extended_hours_end")),
            ),
            bar_label=BarLabel(text(configuration.get("bar_label"))),
            developing_bar_exposure=DevelopingBarExposure(
                text(configuration.get("developing_bar_exposure"))
            ),
            schema_version=text(configuration.get("schema_version")),
        )
        canonical = PredictionDecisionSchedule(
            timeframe,
            datetime.fromisoformat(text(schedule.get("start_timestamp"))),
            datetime.fromisoformat(text(schedule.get("end_timestamp"))),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ManifestError("window schedule calendar contract is invalid") from error
    if configuration_identity(schedule) != configuration_identity(
        canonical.to_primitive()
    ):
        raise ManifestError(
            "window schedule differs from canonical completed bar boundaries"
        )
    return {
        timestamp.isoformat(): label.isoformat()
        for timestamp, label in zip(
            canonical.decision_timestamps, canonical.decision_sessions, strict=True
        )
    }
