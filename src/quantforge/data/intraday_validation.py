"""Exchange-session coverage validation for canonical intraday bars."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from typing import cast
from zoneinfo import ZoneInfo

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data.calendar import expected_sessions
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import canonical_json_bytes
from quantforge.data.intraday import IntradayBar, IntradayBarBatch
from quantforge.data.lineage import FeedScope
from quantforge.timeframes import (
    BarCompletion,
    CrossSessionPolicy,
    IntradayAnchor,
    IntradayInterval,
    resolve_exchange_session,
)

INTRADAY_QUALITY_REPORT_SCHEMA_VERSION = "1"
_CLOCK_ANCHOR_EPOCH_DATE = date(1970, 1, 1)
_REQUEST_END_EPSILON = timedelta(microseconds=1)


class IntradayValidationMode(StrEnum):
    """Whether material coverage findings are returned or raised."""

    STRICT = "strict"
    DIAGNOSTIC = "diagnostic"


class IntradayCoverageStatus(StrEnum):
    """Material completeness state independent of validation mode."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class IntradayCoverageInterval:
    """One expected, missing, unexpected, developing, or warned interval."""

    session_date: date
    start_timestamp: datetime
    end_timestamp: datetime
    completion: BarCompletion

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "session_date": self.session_date.isoformat(),
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "completion": self.completion.value,
        }


@dataclass(frozen=True, slots=True)
class IntradaySessionCoverage:
    """Coverage facts for the requested portion of one exchange session."""

    session_date: date
    session_open_timestamp: datetime
    session_close_timestamp: datetime
    request_covers_full_session: bool
    expected_interval_count: int
    observed_completed_interval_count: int
    missing_intervals: tuple[IntradayCoverageInterval, ...]
    unexpected_intervals: tuple[IntradayCoverageInterval, ...]
    developing_intervals: tuple[IntradayCoverageInterval, ...]
    zero_volume_intervals: tuple[IntradayCoverageInterval, ...]

    @property
    def is_complete(self) -> bool:
        """Return whether every expected completed interval appears exactly once."""
        return not self.missing_intervals and not self.unexpected_intervals

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "session_date": self.session_date.isoformat(),
            "session_open_timestamp": self.session_open_timestamp.isoformat(),
            "session_close_timestamp": self.session_close_timestamp.isoformat(),
            "request_covers_full_session": self.request_covers_full_session,
            "expected_interval_count": self.expected_interval_count,
            "observed_completed_interval_count": (
                self.observed_completed_interval_count
            ),
            "is_complete": self.is_complete,
            "missing_intervals": [
                interval.to_primitive() for interval in self.missing_intervals
            ],
            "unexpected_intervals": [
                interval.to_primitive() for interval in self.unexpected_intervals
            ],
            "developing_intervals": [
                interval.to_primitive() for interval in self.developing_intervals
            ],
            "zero_volume_intervals": [
                interval.to_primitive() for interval in self.zero_volume_intervals
            ],
        }


@dataclass(frozen=True, slots=True)
class IntradayCoverageReport:
    """Serializable provider-neutral intraday coverage and quality evidence."""

    request_id: str
    batch_id: str
    validation_mode: IntradayValidationMode
    timeframe_configuration_id: str
    source_interval: IntradayInterval
    feed_scope: FeedScope
    session_scope: str
    requested_start_timestamp: datetime
    requested_end_timestamp: datetime
    observed_bar_count: int
    expected_completed_interval_count: int
    missing_intervals: tuple[IntradayCoverageInterval, ...]
    unexpected_intervals: tuple[IntradayCoverageInterval, ...]
    developing_intervals: tuple[IntradayCoverageInterval, ...]
    zero_volume_intervals: tuple[IntradayCoverageInterval, ...]
    incomplete_sessions: tuple[date, ...]
    sessions: tuple[IntradaySessionCoverage, ...]
    schema_version: str = INTRADAY_QUALITY_REPORT_SCHEMA_VERSION

    @property
    def status(self) -> IntradayCoverageStatus:
        """Return the material coverage outcome; warnings do not make it invalid."""
        if self.missing_intervals or self.unexpected_intervals:
            return IntradayCoverageStatus.INCOMPLETE
        return IntradayCoverageStatus.COMPLETE

    @property
    def is_complete(self) -> bool:
        return self.status is IntradayCoverageStatus.COMPLETE

    @property
    def has_warnings(self) -> bool:
        """Return whether valid-but-material observations need consumer attention."""
        return bool(self.developing_intervals or self.zero_volume_intervals)

    def to_primitive(self) -> PrimitiveMapping:
        """Return the complete report as deterministic JSON-compatible values."""
        return {
            "schema_version": self.schema_version,
            "report_type": "intraday_coverage_quality",
            "request_id": self.request_id,
            "batch_id": self.batch_id,
            "validation_mode": self.validation_mode.value,
            "status": self.status.value,
            "is_complete": self.is_complete,
            "has_warnings": self.has_warnings,
            "timeframe_configuration_id": self.timeframe_configuration_id,
            "source_interval": self.source_interval.to_primitive(),
            "feed_scope": self.feed_scope.to_primitive(),
            "session_scope": self.session_scope,
            "requested_start_timestamp": self.requested_start_timestamp.isoformat(),
            "requested_end_timestamp": self.requested_end_timestamp.isoformat(),
            "observed_bar_count": self.observed_bar_count,
            "expected_completed_interval_count": (
                self.expected_completed_interval_count
            ),
            "missing_intervals": [
                interval.to_primitive() for interval in self.missing_intervals
            ],
            "unexpected_intervals": [
                interval.to_primitive() for interval in self.unexpected_intervals
            ],
            "developing_intervals": [
                interval.to_primitive() for interval in self.developing_intervals
            ],
            "zero_volume_intervals": [
                interval.to_primitive() for interval in self.zero_volume_intervals
            ],
            "incomplete_sessions": [
                session_date.isoformat() for session_date in self.incomplete_sessions
            ],
            "sessions": [session.to_primitive() for session in self.sessions],
        }

    @property
    def report_id(self) -> str:
        """Return the deterministic identity of this exact validation outcome."""
        return configuration_identity(self.to_primitive())

    def serialize(self) -> bytes:
        """Serialize the report as canonical sorted JSON bytes."""
        return canonical_json_bytes(self.to_primitive())


class IntradayCoverageValidationError(ValidationError):
    """Strict validation found missing or unexpected completed intervals."""

    def __init__(self, report: IntradayCoverageReport) -> None:
        missing_count = len(report.missing_intervals)
        unexpected_count = len(report.unexpected_intervals)
        super().__init__(
            "intraday coverage validation failed: "
            f"{missing_count} missing and {unexpected_count} unexpected intervals",
            missing_sessions=tuple(
                session_date.isoformat() for session_date in report.incomplete_sessions
            ),
        )
        self.report = report


def _coverage_interval(
    session_date: date,
    start_timestamp: datetime,
    end_timestamp: datetime,
    completion: BarCompletion,
) -> IntradayCoverageInterval:
    return IntradayCoverageInterval(
        session_date=session_date,
        start_timestamp=start_timestamp.astimezone(UTC),
        end_timestamp=end_timestamp.astimezone(UTC),
        completion=completion,
    )


def _bar_interval(bar: IntradayBar) -> IntradayCoverageInterval:
    return _coverage_interval(
        bar.session_date,
        bar.start_timestamp,
        bar.end_timestamp,
        bar.completion,
    )


def _clock_bucket_end(
    session_open_timestamp: datetime,
    interval: IntradayInterval,
    timezone_name: str,
) -> datetime:
    timezone = ZoneInfo(timezone_name)
    anchor = datetime.combine(
        _CLOCK_ANCHOR_EPOCH_DATE,
        cast(time, interval.clock_anchor),
        timezone,
    ).astimezone(UTC)
    elapsed = session_open_timestamp - anchor
    bucket_start = anchor + (elapsed // interval.nominal_duration) * (
        interval.nominal_duration
    )
    return bucket_start + interval.nominal_duration


def _expected_session_intervals(
    batch: IntradayBarBatch, session_date: date
) -> tuple[IntradayCoverageInterval, ...]:
    request = batch.request
    timeframe = request.timeframe
    interval = request.source_interval
    session = resolve_exchange_session(session_date, timeframe.session_policy)
    start_timestamp = session.open_timestamp
    expected: list[IntradayCoverageInterval] = []
    next_clock_end = (
        _clock_bucket_end(
            session.open_timestamp,
            interval,
            timeframe.session_policy.timezone_name,
        )
        if interval.anchor is IntradayAnchor.CLOCK
        else None
    )
    while start_timestamp < session.close_timestamp:
        anchored_end = (
            next_clock_end
            if next_clock_end is not None
            else start_timestamp + interval.nominal_duration
        )
        if interval.cross_session_policy is CrossSessionPolicy.PROHIBITED:
            end_timestamp = min(anchored_end, session.close_timestamp)
        else:
            end_timestamp = anchored_end
        actual_duration = end_timestamp - start_timestamp
        if actual_duration == interval.nominal_duration:
            completion = BarCompletion.COMPLETED
        elif end_timestamp == session.close_timestamp:
            completion = BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL
        else:
            completion = BarCompletion.COMPLETED_PARTIAL_DURATION_LEADING
        expected_interval = _coverage_interval(
            session_date, start_timestamp, end_timestamp, completion
        )
        if (
            request.start_timestamp <= expected_interval.start_timestamp
            and expected_interval.end_timestamp <= request.end_timestamp
        ):
            expected.append(expected_interval)
        start_timestamp = end_timestamp
        if next_clock_end is not None:
            next_clock_end += interval.nominal_duration
    return tuple(expected)


def _interval_key(
    interval: IntradayCoverageInterval,
) -> tuple[datetime, datetime, BarCompletion]:
    return (
        interval.start_timestamp,
        interval.end_timestamp,
        interval.completion,
    )


def validate_intraday_coverage(
    batch: IntradayBarBatch,
    *,
    mode: IntradayValidationMode = IntradayValidationMode.STRICT,
) -> IntradayCoverageReport:
    """Validate completed source-interval coverage without modifying any bar.

    Structural bar, OHLCV, range, session-scope, ordering, duplicate, and overlap
    invariants are enforced by ``IntradayBar`` and ``IntradayBarBatch`` before
    this coverage pass. Diagnostic mode returns the same material findings that
    strict mode attaches to ``IntradayCoverageValidationError``.
    """
    batch_value = cast(object, batch)
    mode_value = cast(object, mode)
    if not isinstance(batch_value, IntradayBarBatch):
        raise TypeError("intraday coverage validation requires an IntradayBarBatch")
    if not isinstance(mode_value, IntradayValidationMode):
        raise TypeError("intraday validation mode is invalid")

    request = batch.request
    timezone = ZoneInfo(request.timeframe.session_policy.timezone_name)
    local_start_date = request.start_timestamp.astimezone(timezone).date()
    local_end_date = (
        (request.end_timestamp - _REQUEST_END_EPSILON).astimezone(timezone).date()
    )
    session_dates = expected_sessions(
        local_start_date,
        local_end_date,
        request.timeframe.session_policy.calendar_name,
    )
    bars_by_session: dict[date, list[IntradayBar]] = {}
    for bar in batch.bars:
        bars_by_session.setdefault(bar.session_date, []).append(bar)

    session_reports: list[IntradaySessionCoverage] = []
    all_missing: list[IntradayCoverageInterval] = []
    all_unexpected: list[IntradayCoverageInterval] = []
    all_developing: list[IntradayCoverageInterval] = []
    all_zero_volume: list[IntradayCoverageInterval] = []
    for session_date in session_dates:
        session = resolve_exchange_session(
            session_date, request.timeframe.session_policy
        )
        expected = _expected_session_intervals(batch, session_date)
        session_bars = bars_by_session.get(session_date, [])
        completed = tuple(
            _bar_interval(bar)
            for bar in session_bars
            if bar.completion is not BarCompletion.DEVELOPING
        )
        developing = tuple(
            _bar_interval(bar)
            for bar in session_bars
            if bar.completion is BarCompletion.DEVELOPING
        )
        zero_volume = tuple(
            _bar_interval(bar) for bar in session_bars if bar.volume == 0
        )
        expected_by_key = {_interval_key(interval): interval for interval in expected}
        completed_by_key = {_interval_key(interval): interval for interval in completed}
        missing = tuple(
            expected_by_key[key]
            for key in sorted(set(expected_by_key) - set(completed_by_key))
        )
        unexpected = tuple(
            completed_by_key[key]
            for key in sorted(set(completed_by_key) - set(expected_by_key))
        )
        if not expected and not session_bars:
            continue
        request_covers_full_session = (
            request.start_timestamp <= session.open_timestamp
            and request.end_timestamp >= session.close_timestamp
        )
        session_report = IntradaySessionCoverage(
            session_date=session_date,
            session_open_timestamp=session.open_timestamp,
            session_close_timestamp=session.close_timestamp,
            request_covers_full_session=request_covers_full_session,
            expected_interval_count=len(expected),
            observed_completed_interval_count=len(completed),
            missing_intervals=missing,
            unexpected_intervals=unexpected,
            developing_intervals=developing,
            zero_volume_intervals=zero_volume,
        )
        session_reports.append(session_report)
        all_missing.extend(missing)
        all_unexpected.extend(unexpected)
        all_developing.extend(developing)
        all_zero_volume.extend(zero_volume)

    report = IntradayCoverageReport(
        request_id=request.request_id,
        batch_id=batch.batch_id,
        validation_mode=mode,
        timeframe_configuration_id=request.timeframe.configuration_id,
        source_interval=request.source_interval,
        feed_scope=request.feed_scope,
        session_scope=request.timeframe.session_policy.scope.value,
        requested_start_timestamp=request.start_timestamp,
        requested_end_timestamp=request.end_timestamp,
        observed_bar_count=len(batch.bars),
        expected_completed_interval_count=sum(
            session.expected_interval_count for session in session_reports
        ),
        missing_intervals=tuple(all_missing),
        unexpected_intervals=tuple(all_unexpected),
        developing_intervals=tuple(all_developing),
        zero_volume_intervals=tuple(all_zero_volume),
        incomplete_sessions=tuple(
            session.session_date
            for session in session_reports
            if not session.is_complete
        ),
        sessions=tuple(session_reports),
    )
    if mode is IntradayValidationMode.STRICT and not report.is_complete:
        raise IntradayCoverageValidationError(report)
    return report


__all__ = [
    "INTRADAY_QUALITY_REPORT_SCHEMA_VERSION",
    "IntradayCoverageInterval",
    "IntradayCoverageReport",
    "IntradayCoverageStatus",
    "IntradayCoverageValidationError",
    "IntradaySessionCoverage",
    "IntradayValidationMode",
    "validate_intraday_coverage",
]
