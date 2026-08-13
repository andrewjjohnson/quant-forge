"""Causal reconstruction of one developing higher-timeframe OHLCV bar."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import cast
from zoneinfo import ZoneInfo

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.identity import canonical_json_bytes
from quantforge.data.intraday import IntradayBar
from quantforge.data.intraday_validation import IntradayCoverageInterval
from quantforge.data.lineage import DatasetFamilyReference
from quantforge.timeframes import (
    BarCompletion,
    BarInterval,
    CrossSessionPolicy,
    IntradayAnchor,
    IntradayInterval,
    SessionInterval,
    Timeframe,
    TimeframeValidationError,
    resolve_exchange_session,
    resolve_trading_week,
)

DEVELOPING_BAR_SCHEMA_VERSION = "1"
DEVELOPING_BAR_RECONSTRUCTION_POLICY_VERSION = "1"
_RECONSTRUCTION_POLICY_NAME = "quantforge_developing_bar_as_of"
_CLOCK_ANCHOR_EPOCH_DATE = date(1970, 1, 1)


class DevelopingBarValidationError(ValueError):
    """A developing-bar request or result is not causally valid."""


def _utc_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise DevelopingBarValidationError(
            f"{field_name} must be a timezone-aware datetime"
        )
    return value.astimezone(UTC)


def _reconstruction_policy_primitive() -> PrimitiveMapping:
    return {
        "policy_name": _RECONSTRUCTION_POLICY_NAME,
        "policy_version": DEVELOPING_BAR_RECONSTRUCTION_POLICY_VERSION,
        "source_availability": "completed_source_bar_end_at_or_before_as_of",
        "missing_constituent_policy": "reject",
        "ohlcv_policy": "first_open_max_high_min_low_last_close_sum_volume",
    }


@dataclass(frozen=True, slots=True)
class DevelopingBar:
    """One explicitly incomplete bar reconstructed at a historical instant."""

    symbol: str
    timeframe: Timeframe
    period_start_date: date
    session_dates: tuple[date, ...]
    as_of: datetime
    observed_start_timestamp: datetime
    observed_end_timestamp: datetime
    expected_completion_boundary: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    source_bar_ids: tuple[str, ...]
    source_dataset_reference: DatasetFamilyReference
    source_timeframe: Timeframe
    complete: bool = False
    completion: BarCompletion = BarCompletion.DEVELOPING
    schema_version: str = DEVELOPING_BAR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        timeframe = cast(object, self.timeframe)
        source_timeframe = cast(object, self.source_timeframe)
        source_reference = cast(object, self.source_dataset_reference)
        volume_value = cast(object, self.volume)
        if self.schema_version != DEVELOPING_BAR_SCHEMA_VERSION:
            raise DevelopingBarValidationError(
                f"developing-bar schema {DEVELOPING_BAR_SCHEMA_VERSION} is required"
            )
        if self.complete or self.completion is not BarCompletion.DEVELOPING:
            raise DevelopingBarValidationError(
                "developing bars must be structurally incomplete"
            )
        if not isinstance(timeframe, Timeframe) or not isinstance(
            source_timeframe, Timeframe
        ):
            raise DevelopingBarValidationError("developing-bar timeframe is invalid")
        if timeframe.session_policy != source_timeframe.session_policy:
            raise DevelopingBarValidationError(
                "source and target exchange-session policies must match"
            )
        if not isinstance(source_timeframe.interval, IntradayInterval):
            raise DevelopingBarValidationError(
                "developing bars require an intraday canonical source"
            )
        source_interval = source_timeframe.interval
        target_interval = timeframe.interval
        if source_interval.cross_session_policy is not CrossSessionPolicy.PROHIBITED:
            raise DevelopingBarValidationError(
                "developing reconstruction requires prohibited source continuation"
            )
        if isinstance(target_interval, IntradayInterval):
            if (
                target_interval.nominal_duration <= source_interval.nominal_duration
                or target_interval.nominal_duration % source_interval.nominal_duration
                or target_interval.cross_session_policy
                is not CrossSessionPolicy.PROHIBITED
                or target_interval.anchor is not source_interval.anchor
                or target_interval.clock_anchor != source_interval.clock_anchor
            ):
                raise DevelopingBarValidationError(
                    "developing intraday target must be a larger compatible exact "
                    "multiple of its source"
                )
        elif isinstance(target_interval, SessionInterval):
            if target_interval.session_count != 1:
                raise DevelopingBarValidationError(
                    "developing daily bars require exactly one exchange session"
                )
        elif target_interval.week_count != 1:
            raise DevelopingBarValidationError(
                "developing weekly bars require exactly one exchange trading week"
            )
        if type(self.period_start_date) is not date:
            raise DevelopingBarValidationError("period start must be a date")
        if not self.session_dates or self.session_dates != tuple(
            sorted(set(self.session_dates))
        ):
            raise DevelopingBarValidationError(
                "observed exchange sessions must be sorted and unique"
            )
        as_of = _utc_timestamp(self.as_of, "developing-bar as-of")
        observed_start = _utc_timestamp(self.observed_start_timestamp, "observed start")
        observed_end = _utc_timestamp(self.observed_end_timestamp, "observed end")
        expected_completion = _utc_timestamp(
            self.expected_completion_boundary, "expected completion boundary"
        )
        if not observed_start < observed_end <= as_of < expected_completion:
            raise DevelopingBarValidationError(
                "developing-bar boundaries must satisfy observed start < observed "
                "end <= as-of < expected completion"
            )
        target_boundaries = _target_boundaries(as_of, timeframe)
        if target_boundaries is None:
            raise DevelopingBarValidationError(
                "developing bar does not belong to a forming target period"
            )
        (
            expected_period_start,
            expected_session_dates,
            expected_observed_start,
            target_completion,
        ) = target_boundaries
        if (
            self.period_start_date != expected_period_start
            or observed_start != expected_observed_start
            or expected_completion != target_completion
            or any(
                session_date not in expected_session_dates
                for session_date in self.session_dates
            )
        ):
            raise DevelopingBarValidationError(
                "developing-bar period metadata disagrees with target semantics"
            )
        if not self.symbol.strip():
            raise DevelopingBarValidationError("developing-bar symbol is required")
        prices = cast(tuple[object, ...], (self.open, self.high, self.low, self.close))
        if any(
            not isinstance(value, Decimal) or not value.is_finite() or value <= 0
            for value in prices
        ):
            raise DevelopingBarValidationError(
                "developing OHLC must use positive finite Decimal values"
            )
        if (
            not isinstance(volume_value, Decimal)
            or not volume_value.is_finite()
            or volume_value < 0
        ):
            raise DevelopingBarValidationError(
                "developing volume must be a nonnegative finite Decimal"
            )
        if self.high < max(self.open, self.low, self.close) or self.low > min(
            self.open, self.high, self.close
        ):
            raise DevelopingBarValidationError(
                "developing OHLC relationships are invalid"
            )
        if not self.source_bar_ids or len(set(self.source_bar_ids)) != len(
            self.source_bar_ids
        ):
            raise DevelopingBarValidationError(
                "developing source bar IDs must be nonempty and unique"
            )
        if not isinstance(source_reference, DatasetFamilyReference):
            raise DevelopingBarValidationError(
                "developing source dataset reference is invalid"
            )
        if (
            source_reference.timeframe_configuration_id
            != source_timeframe.configuration_id
        ):
            raise DevelopingBarValidationError(
                "developing source timeframe does not match its dataset reference"
            )
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "as_of", as_of)
        object.__setattr__(self, "observed_start_timestamp", observed_start)
        object.__setattr__(self, "observed_end_timestamp", observed_end)
        object.__setattr__(self, "expected_completion_boundary", expected_completion)

    @property
    def nominal_interval(self) -> BarInterval:
        return self.timeframe.interval

    @property
    def start_timestamp(self) -> datetime:
        return self.observed_start_timestamp

    @property
    def end_timestamp(self) -> datetime:
        return self.observed_end_timestamp

    @property
    def source_bar_count(self) -> int:
        return len(self.source_bar_ids)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "bar_type": "developing_bar_as_of",
            "symbol": self.symbol,
            "complete": self.complete,
            "completion": self.completion.value,
            "as_of": self.as_of.isoformat(),
            "nominal_interval": self.nominal_interval.to_primitive(),
            "timeframe": {
                "configuration_id": self.timeframe.configuration_id,
                "configuration": self.timeframe.to_primitive(),
            },
            "period_start_date": self.period_start_date.isoformat(),
            "session_dates": [value.isoformat() for value in self.session_dates],
            "observed_start_timestamp": self.observed_start_timestamp.isoformat(),
            "observed_end_timestamp": self.observed_end_timestamp.isoformat(),
            "expected_completion_boundary": (
                self.expected_completion_boundary.isoformat()
            ),
            "source_bar_count": self.source_bar_count,
            "open": decimal_to_primitive(self.open),
            "high": decimal_to_primitive(self.high),
            "low": decimal_to_primitive(self.low),
            "close": decimal_to_primitive(self.close),
            "volume": decimal_to_primitive(self.volume),
            "source_bar_ids": list(self.source_bar_ids),
            "source_dataset_reference": (self.source_dataset_reference.to_primitive()),
            "source_timeframe": {
                "configuration_id": self.source_timeframe.configuration_id,
                "configuration": self.source_timeframe.to_primitive(),
            },
            "reconstruction_policy": _reconstruction_policy_primitive(),
        }

    @property
    def bar_id(self) -> str:
        return configuration_identity(self.to_primitive())

    def serialize(self) -> bytes:
        return canonical_json_bytes({"bar_id": self.bar_id, **self.to_primitive()})


def _clock_bucket_boundaries(
    as_of: datetime, timeframe: Timeframe
) -> tuple[datetime, datetime]:
    interval = cast(IntradayInterval, timeframe.interval)
    timezone = ZoneInfo(timeframe.session_policy.timezone_name)
    anchor_time = cast(object, interval.clock_anchor)
    if not isinstance(anchor_time, time):
        # ``clock_anchor`` is validated by Timeframe; this keeps strict typing local.
        raise DevelopingBarValidationError("clock-anchored timeframe is invalid")
    anchor = datetime.combine(
        _CLOCK_ANCHOR_EPOCH_DATE, anchor_time, timezone
    ).astimezone(UTC)
    elapsed = as_of - anchor
    bucket_start = (
        anchor + (elapsed // interval.nominal_duration) * interval.nominal_duration
    )
    return bucket_start, bucket_start + interval.nominal_duration


def _target_boundaries(
    as_of: datetime, target_timeframe: Timeframe
) -> tuple[date, tuple[date, ...], datetime, datetime] | None:
    timezone = ZoneInfo(target_timeframe.session_policy.timezone_name)
    local_date = as_of.astimezone(timezone).date()
    interval = target_timeframe.interval
    if isinstance(interval, IntradayInterval):
        try:
            session = resolve_exchange_session(
                local_date, target_timeframe.session_policy
            )
        except TimeframeValidationError:
            return None
        if not session.open_timestamp < as_of < session.close_timestamp:
            return None
        if interval.anchor is IntradayAnchor.SESSION_OPEN:
            elapsed = as_of - session.open_timestamp
            target_start = (
                session.open_timestamp
                + (elapsed // interval.nominal_duration) * interval.nominal_duration
            )
            anchored_completion = target_start + interval.nominal_duration
        else:
            bucket_start, anchored_completion = _clock_bucket_boundaries(
                as_of, target_timeframe
            )
            target_start = max(session.open_timestamp, bucket_start)
        expected_completion = min(anchored_completion, session.close_timestamp)
        if not target_start < as_of < expected_completion:
            return None
        return (
            session.session_date,
            (session.session_date,),
            target_start,
            expected_completion,
        )
    if isinstance(interval, SessionInterval):
        if interval.session_count != 1:
            raise DevelopingBarValidationError(
                "developing daily bars require exactly one exchange session"
            )
        try:
            session = resolve_exchange_session(
                local_date, target_timeframe.session_policy
            )
        except TimeframeValidationError:
            return None
        if not session.open_timestamp < as_of < session.close_timestamp:
            return None
        return (
            session.session_date,
            (session.session_date,),
            session.open_timestamp,
            session.close_timestamp,
        )
    if interval.week_count != 1:
        raise DevelopingBarValidationError(
            "developing weekly bars require exactly one exchange trading week"
        )
    trading_week = resolve_trading_week(local_date, target_timeframe.session_policy)
    first_session = trading_week.sessions[0]
    last_session = trading_week.sessions[-1]
    if not first_session.open_timestamp < as_of < last_session.close_timestamp:
        return None
    return (
        trading_week.week_start,
        tuple(session.session_date for session in trading_week.sessions),
        first_session.open_timestamp,
        last_session.close_timestamp,
    )


def reconstruct_developing_bar_as_of(
    *,
    as_of: datetime,
    target_timeframe: Timeframe,
    source_timeframe: Timeframe,
    source_bars: tuple[IntradayBar, ...],
    expected_source_intervals: tuple[IntradayCoverageInterval, ...],
    source_dataset_reference: DatasetFamilyReference,
) -> DevelopingBar | None:
    """Reconstruct one incomplete target from completed source bars known as-of."""
    decision_timestamp = _utc_timestamp(as_of, "developing-bar as-of")
    if target_timeframe.session_policy != source_timeframe.session_policy:
        raise DevelopingBarValidationError(
            "source and target exchange-session policies must match"
        )
    source_interval = source_timeframe.interval
    if not isinstance(source_interval, IntradayInterval):
        raise DevelopingBarValidationError(
            "developing bars require an intraday canonical source"
        )
    if source_interval.cross_session_policy is not CrossSessionPolicy.PROHIBITED:
        raise DevelopingBarValidationError(
            "developing reconstruction requires prohibited source continuation"
        )
    target_interval = target_timeframe.interval
    if isinstance(target_interval, IntradayInterval):
        if (
            target_interval.nominal_duration <= source_interval.nominal_duration
            or target_interval.nominal_duration % source_interval.nominal_duration
            or target_interval.cross_session_policy is not CrossSessionPolicy.PROHIBITED
            or target_interval.anchor is not source_interval.anchor
            or target_interval.clock_anchor != source_interval.clock_anchor
        ):
            raise DevelopingBarValidationError(
                "developing intraday target must be a larger compatible exact "
                "multiple of its source"
            )
    boundaries = _target_boundaries(decision_timestamp, target_timeframe)
    if boundaries is None:
        return None
    period_start_date, target_session_dates, target_start, expected_completion = (
        boundaries
    )
    expected = tuple(
        interval
        for interval in expected_source_intervals
        if target_start <= interval.start_timestamp
        and interval.end_timestamp <= decision_timestamp
        and interval.end_timestamp <= expected_completion
    )
    if not expected:
        return None
    if expected[0].start_timestamp != target_start:
        raise DevelopingBarValidationError(
            "source range does not cover the developing period from its start"
        )
    observed_by_boundary = {
        (bar.start_timestamp, bar.end_timestamp, bar.completion): bar
        for bar in source_bars
        if bar.completion is not BarCompletion.DEVELOPING
        and bar.end_timestamp <= decision_timestamp
    }
    expected_keys = tuple(
        (interval.start_timestamp, interval.end_timestamp, interval.completion)
        for interval in expected
    )
    missing = tuple(key for key in expected_keys if key not in observed_by_boundary)
    if missing:
        raise DevelopingBarValidationError(
            "developing reconstruction is missing an available source constituent"
        )
    observed = tuple(observed_by_boundary[key] for key in expected_keys)
    observed_session_dates = tuple(dict.fromkeys(bar.session_date for bar in observed))
    if any(value not in target_session_dates for value in observed_session_dates):
        raise DevelopingBarValidationError(
            "developing source bar belongs to another target period"
        )
    return DevelopingBar(
        symbol=observed[0].symbol,
        timeframe=target_timeframe,
        period_start_date=period_start_date,
        session_dates=observed_session_dates,
        as_of=decision_timestamp,
        observed_start_timestamp=observed[0].start_timestamp,
        observed_end_timestamp=observed[-1].end_timestamp,
        expected_completion_boundary=expected_completion,
        open=observed[0].open,
        high=max(bar.high for bar in observed),
        low=min(bar.low for bar in observed),
        close=observed[-1].close,
        volume=sum((bar.volume for bar in observed), start=Decimal(0)),
        source_bar_ids=tuple(bar.bar_id for bar in observed),
        source_dataset_reference=source_dataset_reference,
        source_timeframe=source_timeframe,
    )


__all__ = [
    "DEVELOPING_BAR_RECONSTRUCTION_POLICY_VERSION",
    "DEVELOPING_BAR_SCHEMA_VERSION",
    "DevelopingBar",
    "DevelopingBarValidationError",
]
