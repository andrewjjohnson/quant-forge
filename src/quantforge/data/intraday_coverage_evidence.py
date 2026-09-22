"""Validate retained coverage metadata without reading or inventing observations."""

from datetime import UTC, date, datetime
from itertools import pairwise
from typing import cast

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data.intraday import IntradayBarBatch, IntradayBarRequest
from quantforge.data.intraday_validation import (
    IntradayCoverageInterval,
    IntradayCoverageReport,
    IntradaySessionCoverage,
    IntradayValidationMode,
    validate_intraday_coverage,
)
from quantforge.data.lineage import AdjustmentBasis, FeedCoverage, FeedScope
from quantforge.data.models import AdjustmentMode
from quantforge.timeframes import (
    BarCompletion,
    IntradayBarWindow,
    Timeframe,
    resolve_exchange_session,
)


def _record(value: Primitive) -> PrimitiveMapping:
    if not isinstance(value, dict):
        raise ValueError("coverage evidence must contain records")
    return value


def _records(value: Primitive) -> list[PrimitiveMapping]:
    if not isinstance(value, list):
        raise ValueError("coverage evidence must contain arrays")
    return [_record(item) for item in value]


def _text(value: Primitive) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("coverage identifiers must be nonempty strings")
    return value


def _timestamp(value: Primitive) -> datetime:
    timestamp = datetime.fromisoformat(_text(value))
    if timestamp.utcoffset() is None:
        raise ValueError("coverage timestamps must be aware")
    timestamp = timestamp.astimezone(UTC)
    if timestamp.isoformat() != value:
        raise ValueError("coverage timestamps must be canonical UTC")
    return timestamp


def _session_date(value: Primitive) -> date:
    session_date = date.fromisoformat(_text(value))
    if session_date.isoformat() != value:
        raise ValueError("coverage session dates must be canonical")
    return session_date


def intraday_request_from_primitive(
    configuration: PrimitiveMapping,
) -> IntradayBarRequest:
    """Reconstruct a canonical request with its existing domain invariants."""
    feed = _record(configuration["feed_scope"])
    basis = _record(configuration["adjustment_basis"])
    request = IntradayBarRequest(
        symbol=_text(configuration["symbol"]),
        start_timestamp=_timestamp(configuration["start_timestamp"]),
        end_timestamp=_timestamp(configuration["end_timestamp"]),
        timeframe=Timeframe.from_primitive(
            _record(_record(configuration["timeframe"])["configuration"])
        ),
        feed_scope=FeedScope(
            FeedCoverage(_text(feed["coverage"])),
            cast(str | None, feed["market_center"]),
            cast(str | None, feed["provider_scope"]),
        ),
        adjustment_basis=AdjustmentBasis(
            AdjustmentMode(_text(basis["adjustment_mode"])),
            _text(basis["ohlc_basis"]),
            _text(basis["volume_basis"]),
            _text(basis["corporate_action_policy"]),
            cast(bool, basis["adjusted_fields_used"]),
        ),
    )
    if request.request_id != configuration_identity(configuration):
        raise ValueError("retained source request is not canonical")
    return request


def _intervals(
    value: Primitive, request: IntradayBarRequest, session_date: date
) -> tuple[IntradayCoverageInterval, ...]:
    intervals: list[IntradayCoverageInterval] = []
    for record in _records(value):
        window = IntradayBarWindow(
            request.timeframe,
            _session_date(record["session_date"]),
            _timestamp(record["start_timestamp"]),
            _timestamp(record["end_timestamp"]),
            BarCompletion(_text(record["completion"])),
        )
        if (
            window.session_date != session_date
            or window.start_timestamp < request.start_timestamp
            or window.end_timestamp > request.end_timestamp
        ):
            raise ValueError("coverage interval falls outside its session or request")
        interval = IntradayCoverageInterval(
            window.session_date,
            window.start_timestamp,
            window.end_timestamp,
            window.completion,
        )
        if configuration_identity(interval.to_primitive()) != configuration_identity(
            record
        ):
            raise ValueError("coverage interval is not canonical")
        intervals.append(interval)
    keys = [(item.start_timestamp, item.end_timestamp) for item in intervals]
    if keys != sorted(set(keys)):
        raise ValueError("coverage intervals must be unique and ordered")
    return tuple(intervals)


def _sessions(
    records: list[PrimitiveMapping], request: IntradayBarRequest
) -> tuple[IntradaySessionCoverage, ...]:
    # An empty batch supplies only the expected calendar schedule. No prices,
    # volumes, or observed bars are synthesized or loaded for this comparison.
    schedule = validate_intraday_coverage(
        IntradayBarBatch(request, ()), mode=IntradayValidationMode.DIAGNOSTIC
    )
    expected_by_session = {
        session.session_date: session.missing_intervals for session in schedule.sessions
    }
    sessions: list[IntradaySessionCoverage] = []
    for record in records:
        session_date = _session_date(record["session_date"])
        session = resolve_exchange_session(
            session_date, request.timeframe.session_policy
        )
        expected = set(expected_by_session.get(session_date, ()))
        missing = _intervals(record["missing_intervals"], request, session_date)
        unexpected = _intervals(record["unexpected_intervals"], request, session_date)
        developing = _intervals(record["developing_intervals"], request, session_date)
        zero_volume = _intervals(record["zero_volume_intervals"], request, session_date)
        if (
            not set(missing) <= expected
            or set(unexpected) & expected
            or any(item.completion is BarCompletion.DEVELOPING for item in unexpected)
            or any(
                item.completion is not BarCompletion.DEVELOPING for item in developing
            )
        ):
            raise ValueError("coverage findings disagree with the expected intervals")
        completed = (expected - set(missing)) | set(unexpected)
        observed = completed | set(developing)
        if not set(zero_volume) <= observed or (not expected and not observed):
            raise ValueError("coverage session findings lack matching observations")
        ordered = sorted(
            observed, key=lambda item: (item.start_timestamp, item.end_timestamp)
        )
        if any(
            current.start_timestamp < previous.end_timestamp
            for previous, current in pairwise(ordered)
        ):
            raise ValueError("coverage observations overlap")
        sessions.append(
            IntradaySessionCoverage(
                session_date=session_date,
                session_open_timestamp=session.open_timestamp,
                session_close_timestamp=session.close_timestamp,
                request_covers_full_session=(
                    request.start_timestamp <= session.open_timestamp
                    and request.end_timestamp >= session.close_timestamp
                ),
                expected_interval_count=len(expected),
                observed_completed_interval_count=len(completed),
                missing_intervals=missing,
                unexpected_intervals=unexpected,
                developing_intervals=developing,
                zero_volume_intervals=zero_volume,
            )
        )
    session_dates = [session.session_date for session in sessions]
    if session_dates != sorted(
        set(session_dates)
    ) or not expected_by_session.keys() <= set(session_dates):
        raise ValueError(
            "coverage sessions must cover the expected schedule exactly once"
        )
    return tuple(sessions)


def validate_retained_coverage_report(
    manifest: PrimitiveMapping,
) -> IntradayCoverageReport:
    """Reconstruct coverage facts, bind source metadata, and verify the report hash.

    This proves consistency of retained evidence. Actual normalized bars and raw
    extracts remain independently verified by the full intraday cache loader.
    """
    try:
        request_record = _record(manifest["request"])
        request = intraday_request_from_primitive(
            _record(request_record["configuration"])
        )
        quality = _record(manifest["quality_report"])
        recorded = _record(quality["report"])
        sessions = _sessions(_records(recorded["sessions"]), request)
        report = IntradayCoverageReport(
            request_id=request.request_id,
            batch_id=_text(manifest["batch_id"]),
            validation_mode=IntradayValidationMode.DIAGNOSTIC,
            timeframe_configuration_id=request.timeframe.configuration_id,
            source_interval=request.source_interval,
            feed_scope=request.feed_scope,
            session_scope=request.timeframe.session_policy.scope.value,
            requested_start_timestamp=request.start_timestamp,
            requested_end_timestamp=request.end_timestamp,
            observed_bar_count=sum(
                session.observed_completed_interval_count
                + len(session.developing_intervals)
                for session in sessions
            ),
            expected_completed_interval_count=sum(
                session.expected_interval_count for session in sessions
            ),
            missing_intervals=tuple(
                item for session in sessions for item in session.missing_intervals
            ),
            unexpected_intervals=tuple(
                item for session in sessions for item in session.unexpected_intervals
            ),
            developing_intervals=tuple(
                item for session in sessions for item in session.developing_intervals
            ),
            zero_volume_intervals=tuple(
                item for session in sessions for item in session.zero_volume_intervals
            ),
            incomplete_sessions=tuple(
                session.session_date for session in sessions if not session.is_complete
            ),
            sessions=sessions,
        )
        if (
            set(quality) != {"report_id", "report"}
            or report.report_id != quality["report_id"]
            or report.report_id != configuration_identity(recorded)
            or request_record["request_id"] != request.request_id
            or type(manifest["bar_count"]) is not int
            or report.observed_bar_count != manifest["bar_count"]
            or request.feed_scope.to_primitive() != manifest["feed_scope"]
            or request.source_interval.to_primitive() != manifest["source_interval"]
            or report.session_scope != manifest["session_scope"]
        ):
            raise ValueError(
                "coverage fields or report identity differ from the source"
            )
        return report
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"source request coverage report is invalid: {error}"
        ) from error
