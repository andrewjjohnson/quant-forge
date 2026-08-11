"""Canonical provider-neutral timeframe and exchange-session semantics."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from importlib import import_module
from typing import Protocol, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from quantforge.configuration import PrimitiveMapping, configuration_identity

TIMEFRAME_SCHEMA_VERSION = "1"
XNYS_CALENDAR = "XNYS"
NEW_YORK_TIMEZONE = "America/New_York"
_CLOCK_ANCHOR_EPOCH_DATE = date(1970, 1, 1)
_ONE_DAY = timedelta(days=1)


class TimeframeValidationError(ValueError):
    """A timeframe, session policy, or bar boundary is internally inconsistent."""


class IntervalKind(StrEnum):
    """Provider-neutral interval categories with different temporal semantics."""

    INTRADAY = "intraday"
    EXCHANGE_SESSIONS = "exchange_sessions"
    EXCHANGE_TRADING_WEEKS = "exchange_trading_weeks"


class SessionScope(StrEnum):
    """Which part of a valid exchange session a bar may observe."""

    REGULAR_HOURS = "regular_hours"
    EXTENDED_HOURS = "extended_hours"


class IntradayAnchor(StrEnum):
    """Origin from which consecutive intraday durations are measured."""

    SESSION_OPEN = "exchange_session_open"
    CLOCK = "exchange_local_clock"


class CrossSessionPolicy(StrEnum):
    """Whether an intraday bar may extend beyond its anchor session."""

    PROHIBITED = "prohibited"
    PERMITTED = "permitted"


class BarLabel(StrEnum):
    """Which explicit boundary supplies an intraday bar's timestamp label."""

    START = "bar_start"
    END = "bar_end"


class DevelopingBarExposure(StrEnum):
    """Whether consumers may receive a bar that has not reached a terminal state."""

    EXCLUDE = "completed_only"
    INCLUDE = "include_developing"


class BarCompletion(StrEnum):
    """Mutually exclusive developing, full, and partial completed states."""

    COMPLETED = "completed"
    DEVELOPING = "developing"
    COMPLETED_PARTIAL_DURATION_LEADING = "completed_partial_duration_leading"
    COMPLETED_PARTIAL_DURATION_TERMINAL = "completed_partial_duration_terminal"


class _ExchangeCalendar(Protocol):
    tz: object

    def is_session(self, session_date: date) -> bool: ...

    def session_open(self, session_date: date) -> datetime: ...

    def session_close(self, session_date: date) -> datetime: ...

    def sessions_in_range(
        self, start_date: date, end_date: date
    ) -> Iterable[datetime]: ...


class _ExchangeCalendars(Protocol):
    def get_calendar(self, calendar_name: str) -> _ExchangeCalendar: ...


def _load_calendar(calendar_name: str) -> _ExchangeCalendar:
    try:
        exchange_calendars = cast(
            _ExchangeCalendars, import_module("exchange_calendars")
        )
        return exchange_calendars.get_calendar(calendar_name)
    except Exception as error:
        raise TimeframeValidationError(
            f"cannot resolve exchange calendar: {calendar_name}"
        ) from error


def _calendar_timezone_name(calendar: _ExchangeCalendar) -> str:
    timezone = calendar.tz
    key = getattr(timezone, "key", None)
    return key if isinstance(key, str) else str(timezone)


def _validate_plain_time(value: object, field_name: str) -> time:
    if not isinstance(value, time) or value.tzinfo is not None:
        raise TimeframeValidationError(
            f"{field_name} must be a timezone-naive local time"
        )
    return value


def _time_primitive(value: time | None) -> str | None:
    return None if value is None else value.isoformat(timespec="microseconds")


def _duration_microseconds(duration: timedelta) -> int:
    return (
        duration.days * 86_400 + duration.seconds
    ) * 1_000_000 + duration.microseconds


def _utc_timestamp(timestamp: datetime, field_name: str) -> datetime:
    timestamp_value = cast(object, timestamp)
    if not isinstance(timestamp_value, datetime) or timestamp_value.utcoffset() is None:
        raise TimeframeValidationError(
            f"{field_name} must be a timezone-aware datetime"
        )
    return timestamp_value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ExchangeSessionPolicy:
    """Exchange calendar, timezone, and observable-hours configuration."""

    calendar_name: str = XNYS_CALENDAR
    timezone_name: str = NEW_YORK_TIMEZONE
    scope: SessionScope = SessionScope.REGULAR_HOURS
    extended_hours_start: time | None = None
    extended_hours_end: time | None = None

    def __post_init__(self) -> None:
        calendar_name = cast(object, self.calendar_name)
        timezone_name = cast(object, self.timezone_name)
        scope = cast(object, self.scope)
        if not isinstance(calendar_name, str) or not calendar_name.strip():
            raise TimeframeValidationError("calendar name must be a nonempty string")
        if not isinstance(timezone_name, str) or not timezone_name.strip():
            raise TimeframeValidationError("timezone name must be a nonempty string")
        if not isinstance(scope, SessionScope):
            raise TimeframeValidationError("session scope is invalid")
        try:
            ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as error:
            raise TimeframeValidationError(
                f"cannot resolve exchange timezone: {self.timezone_name}"
            ) from error
        calendar = _load_calendar(self.calendar_name)
        calendar_timezone = _calendar_timezone_name(calendar)
        if calendar_timezone != self.timezone_name:
            raise TimeframeValidationError(
                "exchange timezone must match the calendar timezone: "
                f"{calendar_timezone}"
            )
        if self.scope is SessionScope.REGULAR_HOURS:
            if (
                self.extended_hours_start is not None
                or self.extended_hours_end is not None
            ):
                raise TimeframeValidationError(
                    "regular-hours scope cannot define extended-hours boundaries"
                )
            return
        start = _validate_plain_time(self.extended_hours_start, "extended-hours start")
        end = _validate_plain_time(self.extended_hours_end, "extended-hours end")
        if start >= end:
            raise TimeframeValidationError(
                "extended-hours start must be earlier than its same-day end"
            )

    def to_primitive(self) -> PrimitiveMapping:
        """Return the complete stable session-policy representation."""
        return {
            "calendar": self.calendar_name,
            "timezone": self.timezone_name,
            "scope": self.scope.value,
            "extended_hours_start": _time_primitive(self.extended_hours_start),
            "extended_hours_end": _time_primitive(self.extended_hours_end),
        }


@dataclass(frozen=True, slots=True)
class IntradayInterval:
    """A positive sub-day elapsed duration with explicit anchoring policy."""

    nominal_duration: timedelta
    anchor: IntradayAnchor = IntradayAnchor.SESSION_OPEN
    clock_anchor: time | None = None
    cross_session_policy: CrossSessionPolicy = CrossSessionPolicy.PROHIBITED

    def __post_init__(self) -> None:
        nominal_duration = cast(object, self.nominal_duration)
        anchor = cast(object, self.anchor)
        cross_session_policy = cast(object, self.cross_session_policy)
        if not isinstance(nominal_duration, timedelta) or not (
            timedelta(0) < nominal_duration < _ONE_DAY
        ):
            raise TimeframeValidationError(
                "intraday duration must be positive and shorter than one day"
            )
        if not isinstance(anchor, IntradayAnchor):
            raise TimeframeValidationError("intraday anchor is invalid")
        if not isinstance(cross_session_policy, CrossSessionPolicy):
            raise TimeframeValidationError("cross-session policy is invalid")
        if self.anchor is IntradayAnchor.SESSION_OPEN:
            if self.clock_anchor is not None:
                raise TimeframeValidationError(
                    "session-open anchoring cannot also define a clock anchor"
                )
            return
        _validate_plain_time(self.clock_anchor, "clock anchor")

    @property
    def kind(self) -> IntervalKind:
        return IntervalKind.INTRADAY

    def to_primitive(self) -> PrimitiveMapping:
        """Return an exact integer-duration representation without float rounding."""
        return {
            "kind": self.kind.value,
            "nominal_duration_microseconds": _duration_microseconds(
                self.nominal_duration
            ),
            "anchor": self.anchor.value,
            "clock_anchor": _time_primitive(self.clock_anchor),
            "clock_anchor_epoch_date": (
                _CLOCK_ANCHOR_EPOCH_DATE.isoformat()
                if self.anchor is IntradayAnchor.CLOCK
                else None
            ),
            "cross_session_policy": self.cross_session_policy.value,
        }


@dataclass(frozen=True, slots=True)
class SessionInterval:
    """One or more complete exchange sessions, never a 24-hour duration."""

    session_count: int = 1

    def __post_init__(self) -> None:
        session_count = cast(object, self.session_count)
        if (
            isinstance(session_count, bool)
            or not isinstance(session_count, int)
            or session_count <= 0
        ):
            raise TimeframeValidationError("session count must be a positive integer")

    @property
    def kind(self) -> IntervalKind:
        return IntervalKind.EXCHANGE_SESSIONS

    def to_primitive(self) -> PrimitiveMapping:
        return {"kind": self.kind.value, "session_count": self.session_count}


@dataclass(frozen=True, slots=True)
class TradingWeekInterval:
    """One or more Monday-Sunday exchange trading weeks, not seven-day spans."""

    week_count: int = 1

    def __post_init__(self) -> None:
        week_count = cast(object, self.week_count)
        if (
            isinstance(week_count, bool)
            or not isinstance(week_count, int)
            or week_count <= 0
        ):
            raise TimeframeValidationError("week count must be a positive integer")

    @property
    def kind(self) -> IntervalKind:
        return IntervalKind.EXCHANGE_TRADING_WEEKS

    def to_primitive(self) -> PrimitiveMapping:
        return {"kind": self.kind.value, "week_count": self.week_count}


type BarInterval = IntradayInterval | SessionInterval | TradingWeekInterval


@dataclass(frozen=True, slots=True)
class Timeframe:
    """Complete canonical interval and exchange-session configuration."""

    interval: BarInterval = field(default_factory=SessionInterval)
    session_policy: ExchangeSessionPolicy = field(default_factory=ExchangeSessionPolicy)
    bar_label: BarLabel = BarLabel.START
    developing_bar_exposure: DevelopingBarExposure = DevelopingBarExposure.EXCLUDE
    schema_version: str = TIMEFRAME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        interval = cast(object, self.interval)
        session_policy = cast(object, self.session_policy)
        bar_label = cast(object, self.bar_label)
        developing_bar_exposure = cast(object, self.developing_bar_exposure)
        if not isinstance(
            interval, (IntradayInterval, SessionInterval, TradingWeekInterval)
        ):
            raise TimeframeValidationError("bar interval is invalid")
        if not isinstance(session_policy, ExchangeSessionPolicy):
            raise TimeframeValidationError("exchange session policy is invalid")
        if not isinstance(bar_label, BarLabel):
            raise TimeframeValidationError("bar-label policy is invalid")
        if not isinstance(developing_bar_exposure, DevelopingBarExposure):
            raise TimeframeValidationError("developing-bar exposure policy is invalid")
        if self.schema_version != TIMEFRAME_SCHEMA_VERSION:
            raise TimeframeValidationError(
                f"timeframe schema {TIMEFRAME_SCHEMA_VERSION} is required"
            )

    @classmethod
    def us_equity(cls, interval: BarInterval | None = None) -> "Timeframe":
        """Build the canonical XNYS regular-hours configuration."""
        return cls(SessionInterval() if interval is None else interval)

    def to_primitive(self) -> PrimitiveMapping:
        """Return every material semantic policy in a stable primitive schema."""
        return {
            "schema_version": self.schema_version,
            "interval": self.interval.to_primitive(),
            "session_policy": self.session_policy.to_primitive(),
            "bar_label": self.bar_label.value,
            "developing_bar_exposure": self.developing_bar_exposure.value,
        }

    @property
    def configuration_id(self) -> str:
        """Return the deterministic identity of the complete semantic policy."""
        return configuration_identity(self.to_primitive())


DEFAULT_US_EQUITY_SESSION_POLICY = ExchangeSessionPolicy()
DEFAULT_US_EQUITY_TIMEFRAME = Timeframe()


@dataclass(frozen=True, slots=True)
class ExchangeSession:
    """One valid exchange session with canonical UTC boundary timestamps."""

    session_date: date
    open_timestamp: datetime
    close_timestamp: datetime

    def __post_init__(self) -> None:
        if type(self.session_date) is not date:
            raise TimeframeValidationError("session date must be a date")
        open_timestamp = _utc_timestamp(self.open_timestamp, "session open")
        close_timestamp = _utc_timestamp(self.close_timestamp, "session close")
        if open_timestamp >= close_timestamp:
            raise TimeframeValidationError("session open must be earlier than close")
        object.__setattr__(self, "open_timestamp", open_timestamp)
        object.__setattr__(self, "close_timestamp", close_timestamp)

    @property
    def actual_duration(self) -> timedelta:
        return self.close_timestamp - self.open_timestamp

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "session_date": self.session_date.isoformat(),
            "open_timestamp": self.open_timestamp.isoformat(),
            "close_timestamp": self.close_timestamp.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ExchangeTradingWeek:
    """The actual exchange sessions in one Monday-Sunday local calendar week."""

    week_start: date
    sessions: tuple[ExchangeSession, ...]

    def __post_init__(self) -> None:
        if type(self.week_start) is not date or self.week_start.weekday() != 0:
            raise TimeframeValidationError("trading week must start on Monday")
        if not self.sessions:
            raise TimeframeValidationError("trading week must contain a session")
        session_dates = tuple(session.session_date for session in self.sessions)
        if session_dates != tuple(sorted(set(session_dates))):
            raise TimeframeValidationError(
                "trading-week sessions must be sorted and unique"
            )
        week_end = self.week_start + timedelta(days=6)
        if any(not self.week_start <= value <= week_end for value in session_dates):
            raise TimeframeValidationError(
                "trading-week sessions must fall within its Monday-Sunday range"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "week_start": self.week_start.isoformat(),
            "sessions": [session.to_primitive() for session in self.sessions],
        }


def resolve_exchange_session(
    session_date: date,
    policy: ExchangeSessionPolicy = DEFAULT_US_EQUITY_SESSION_POLICY,
) -> ExchangeSession:
    """Resolve one valid exchange session under regular or explicit extended hours."""
    if type(session_date) is not date:
        raise TimeframeValidationError("session date must be a date")
    policy_value = cast(object, policy)
    if not isinstance(policy_value, ExchangeSessionPolicy):
        raise TimeframeValidationError("exchange session policy is invalid")
    calendar = _load_calendar(policy.calendar_name)
    if not calendar.is_session(session_date):
        raise TimeframeValidationError(
            f"date is not an exchange session for {policy.calendar_name}: "
            f"{session_date.isoformat()}"
        )
    if policy.scope is SessionScope.REGULAR_HOURS:
        return ExchangeSession(
            session_date,
            calendar.session_open(session_date),
            calendar.session_close(session_date),
        )
    timezone = ZoneInfo(policy.timezone_name)
    start = cast(time, policy.extended_hours_start)
    end = cast(time, policy.extended_hours_end)
    return ExchangeSession(
        session_date,
        datetime.combine(session_date, start, timezone),
        datetime.combine(session_date, end, timezone),
    )


def resolve_trading_week(
    reference_date: date,
    policy: ExchangeSessionPolicy = DEFAULT_US_EQUITY_SESSION_POLICY,
) -> ExchangeTradingWeek:
    """Resolve actual sessions in the exchange week containing ``reference_date``."""
    if type(reference_date) is not date:
        raise TimeframeValidationError("trading-week reference must be a date")
    policy_value = cast(object, policy)
    if not isinstance(policy_value, ExchangeSessionPolicy):
        raise TimeframeValidationError("exchange session policy is invalid")
    week_start = reference_date - timedelta(days=reference_date.weekday())
    week_end = week_start + timedelta(days=6)
    calendar = _load_calendar(policy.calendar_name)
    session_dates = tuple(
        timestamp.date()
        for timestamp in calendar.sessions_in_range(week_start, week_end)
    )
    return ExchangeTradingWeek(
        week_start,
        tuple(resolve_exchange_session(value, policy) for value in session_dates),
    )


@dataclass(frozen=True, slots=True)
class IntradayBarWindow:
    """A validated intraday boundary and completion-state record without OHLCV."""

    timeframe: Timeframe
    session_date: date
    start_timestamp: datetime
    end_timestamp: datetime
    completion: BarCompletion

    def __post_init__(self) -> None:
        timeframe = cast(object, self.timeframe)
        completion = cast(object, self.completion)
        if not isinstance(timeframe, Timeframe) or not isinstance(
            timeframe.interval, IntradayInterval
        ):
            raise TimeframeValidationError(
                "intraday bar window requires an intraday timeframe"
            )
        if type(self.session_date) is not date:
            raise TimeframeValidationError("bar session date must be a date")
        if not isinstance(completion, BarCompletion):
            raise TimeframeValidationError("bar completion state is invalid")
        start_timestamp = _utc_timestamp(self.start_timestamp, "bar start")
        end_timestamp = _utc_timestamp(self.end_timestamp, "bar end")
        if start_timestamp >= end_timestamp:
            raise TimeframeValidationError("bar start must be earlier than bar end")
        object.__setattr__(self, "start_timestamp", start_timestamp)
        object.__setattr__(self, "end_timestamp", end_timestamp)
        if (
            self.completion is BarCompletion.DEVELOPING
            and self.timeframe.developing_bar_exposure is DevelopingBarExposure.EXCLUDE
        ):
            raise TimeframeValidationError(
                "developing bars require explicit include-developing exposure"
            )
        session = resolve_exchange_session(
            self.session_date, self.timeframe.session_policy
        )
        interval = cast(IntradayInterval, self.timeframe.interval)
        if not session.open_timestamp <= start_timestamp < session.close_timestamp:
            raise TimeframeValidationError(
                "intraday bar start must belong to its declared exchange session"
            )
        if (
            interval.cross_session_policy is CrossSessionPolicy.PROHIBITED
            and end_timestamp > session.close_timestamp
        ):
            raise TimeframeValidationError(
                "intraday bar cannot cross or extend beyond its exchange session"
            )
        self._validate_anchor(interval, session)
        self._validate_completion(interval, session)

    def _validate_anchor(
        self, interval: IntradayInterval, session: ExchangeSession
    ) -> None:
        if interval.anchor is IntradayAnchor.SESSION_OPEN:
            offset = self.start_timestamp - session.open_timestamp
            if offset >= timedelta(0) and not offset % interval.nominal_duration:
                return
        else:
            bucket_start, bucket_end = self._clock_bucket_boundaries(interval)
            if self.start_timestamp == bucket_start or (
                self.start_timestamp == session.open_timestamp
                and self.start_timestamp < bucket_end
            ):
                return
        raise TimeframeValidationError(
            "intraday bar start is not aligned with its configured anchor"
        )

    def _clock_bucket_boundaries(
        self, interval: IntradayInterval
    ) -> tuple[datetime, datetime]:
        timezone = ZoneInfo(self.timeframe.session_policy.timezone_name)
        anchor = datetime.combine(
            _CLOCK_ANCHOR_EPOCH_DATE,
            cast(time, interval.clock_anchor),
            timezone,
        ).astimezone(UTC)
        elapsed = self.start_timestamp - anchor
        bucket_start = anchor + (elapsed // interval.nominal_duration) * (
            interval.nominal_duration
        )
        bucket_end = bucket_start + interval.nominal_duration
        return bucket_start, bucket_end

    def _anchored_terminal_timestamp(self, interval: IntradayInterval) -> datetime:
        if interval.anchor is IntradayAnchor.CLOCK:
            _, bucket_end = self._clock_bucket_boundaries(interval)
            return bucket_end
        return self.start_timestamp + interval.nominal_duration

    def _terminal_timestamp(
        self, interval: IntradayInterval, session: ExchangeSession
    ) -> datetime:
        terminal_timestamp = self._anchored_terminal_timestamp(interval)
        if interval.cross_session_policy is CrossSessionPolicy.PROHIBITED:
            return min(terminal_timestamp, session.close_timestamp)
        return terminal_timestamp

    def _validate_leading_partial(
        self, interval: IntradayInterval, session: ExchangeSession
    ) -> None:
        anchored_terminal = self._anchored_terminal_timestamp(interval)
        if (
            interval.anchor is not IntradayAnchor.CLOCK
            or self.start_timestamp != session.open_timestamp
            or anchored_terminal >= self.start_timestamp + interval.nominal_duration
            or self.end_timestamp != anchored_terminal
            or self.end_timestamp >= session.close_timestamp
        ):
            raise TimeframeValidationError(
                "completed leading partial-duration bar must start at the session "
                "open and end at the next clock boundary before session close"
            )

    def _validate_completion(
        self, interval: IntradayInterval, session: ExchangeSession
    ) -> None:
        actual_duration = self.actual_duration
        nominal_duration = interval.nominal_duration
        if actual_duration > nominal_duration:
            raise TimeframeValidationError(
                "intraday bar actual duration cannot exceed nominal duration"
            )
        terminal_timestamp = self._terminal_timestamp(interval, session)
        if self.completion is BarCompletion.COMPLETED:
            if (
                actual_duration != nominal_duration
                or self.end_timestamp != terminal_timestamp
            ):
                raise TimeframeValidationError(
                    "completed full bar must have its nominal duration"
                )
            return
        if self.completion is BarCompletion.DEVELOPING:
            if self.end_timestamp >= terminal_timestamp:
                raise TimeframeValidationError(
                    "a bar at its terminal boundary is not developing"
                )
            return
        if self.completion is BarCompletion.COMPLETED_PARTIAL_DURATION_LEADING:
            self._validate_leading_partial(interval, session)
            return
        if interval.cross_session_policy is CrossSessionPolicy.PERMITTED:
            raise TimeframeValidationError(
                "completed terminal partial-duration bar requires prohibited "
                "cross-session continuation"
            )
        if (
            actual_duration >= nominal_duration
            or self.end_timestamp != session.close_timestamp
        ):
            raise TimeframeValidationError(
                "completed partial-duration bar must be shorter than nominal and "
                "end at the actual session close"
            )

    @property
    def nominal_duration(self) -> timedelta:
        return cast(IntradayInterval, self.timeframe.interval).nominal_duration

    @property
    def actual_duration(self) -> timedelta:
        return self.end_timestamp - self.start_timestamp

    @property
    def label_timestamp(self) -> datetime:
        if self.timeframe.bar_label is BarLabel.START:
            return self.start_timestamp
        return self.end_timestamp

    @property
    def is_partial_duration(self) -> bool:
        return self.completion in {
            BarCompletion.COMPLETED_PARTIAL_DURATION_LEADING,
            BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL,
        }

    def to_primitive(self) -> PrimitiveMapping:
        """Return an auditable boundary record with nominal and actual duration."""
        return {
            "timeframe_configuration_id": self.timeframe.configuration_id,
            "session_date": self.session_date.isoformat(),
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "label_timestamp": self.label_timestamp.isoformat(),
            "completion": self.completion.value,
            "nominal_duration_microseconds": _duration_microseconds(
                self.nominal_duration
            ),
            "actual_duration_microseconds": _duration_microseconds(
                self.actual_duration
            ),
        }


__all__ = [
    "DEFAULT_US_EQUITY_SESSION_POLICY",
    "DEFAULT_US_EQUITY_TIMEFRAME",
    "NEW_YORK_TIMEZONE",
    "TIMEFRAME_SCHEMA_VERSION",
    "XNYS_CALENDAR",
    "BarCompletion",
    "BarInterval",
    "BarLabel",
    "CrossSessionPolicy",
    "DevelopingBarExposure",
    "ExchangeSession",
    "ExchangeSessionPolicy",
    "ExchangeTradingWeek",
    "IntervalKind",
    "IntradayAnchor",
    "IntradayBarWindow",
    "IntradayInterval",
    "SessionInterval",
    "SessionScope",
    "Timeframe",
    "TimeframeValidationError",
    "TradingWeekInterval",
    "resolve_exchange_session",
    "resolve_trading_week",
]
