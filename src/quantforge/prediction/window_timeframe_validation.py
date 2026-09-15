"""Validate retained source timeframe definitions using the canonical domain types."""

from datetime import time, timedelta
from typing import cast

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.timeframes import (
    BarInterval,
    BarLabel,
    CrossSessionPolicy,
    DevelopingBarExposure,
    ExchangeSessionPolicy,
    IntervalKind,
    IntradayAnchor,
    IntradayInterval,
    SessionInterval,
    SessionScope,
    Timeframe,
    TradingWeekInterval,
)


def _local_time(value: Primitive) -> time | None:
    return None if value is None else time.fromisoformat(cast(str, value))


def validate_source_timeframe_definition(snapshot: PrimitiveMapping) -> Timeframe:
    """Reject malformed definitions even when the provider's context was skipped."""
    try:
        configuration = cast(PrimitiveMapping, snapshot["configuration"])
        interval = cast(PrimitiveMapping, configuration["interval"])
        session = cast(PrimitiveMapping, configuration["session_policy"])
        kind = IntervalKind(cast(str, interval["kind"]))
        canonical_interval: BarInterval
        if kind is IntervalKind.INTRADAY:
            canonical_interval = IntradayInterval(
                timedelta(
                    microseconds=cast(int, interval["nominal_duration_microseconds"])
                ),
                anchor=IntradayAnchor(cast(str, interval["anchor"])),
                clock_anchor=_local_time(interval["clock_anchor"]),
                cross_session_policy=CrossSessionPolicy(
                    cast(str, interval["cross_session_policy"])
                ),
            )
        elif kind is IntervalKind.EXCHANGE_SESSIONS:
            canonical_interval = SessionInterval(cast(int, interval["session_count"]))
        else:
            canonical_interval = TradingWeekInterval(cast(int, interval["week_count"]))
        canonical = Timeframe(
            canonical_interval,
            session_policy=ExchangeSessionPolicy(
                calendar_name=cast(str, session["calendar"]),
                timezone_name=cast(str, session["timezone"]),
                scope=SessionScope(cast(str, session["scope"])),
                extended_hours_start=_local_time(session["extended_hours_start"]),
                extended_hours_end=_local_time(session["extended_hours_end"]),
            ),
            bar_label=BarLabel(cast(str, configuration["bar_label"])),
            developing_bar_exposure=DevelopingBarExposure(
                cast(str, configuration["developing_bar_exposure"])
            ),
            schema_version=cast(str, configuration["schema_version"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise InvalidPredictionOutputError(
            "source timeframe definition is invalid"
        ) from error
    expected: PrimitiveMapping = {
        "configuration_id": canonical.configuration_id,
        "configuration": canonical.to_primitive(),
    }
    if configuration_identity(snapshot) != configuration_identity(expected):
        raise InvalidPredictionOutputError(
            "source timeframe definition is noncanonical"
        )
    return canonical
