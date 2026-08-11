"""Provider-neutral intraday request, bar, and capability contracts."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.exceptions import (
    UnsupportedDateRangeError,
    UnsupportedFeedError,
    UnsupportedIntervalError,
    UnsupportedSessionScopeError,
)
from quantforge.data.identity import canonical_json_bytes
from quantforge.data.lineage import AdjustmentBasis, FeedScope
from quantforge.timeframes import (
    BarCompletion,
    IntradayBarWindow,
    IntradayInterval,
    SessionScope,
    Timeframe,
    TimeframeValidationError,
)

INTRADAY_CONTRACT_SCHEMA_VERSION = "1"


class IntradayContractValidationError(ValueError):
    """An intraday request, bar, provenance, or capability is inconsistent."""


def _validated_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntradayContractValidationError(f"{field_name} must be a nonempty string")
    if value != value.strip():
        raise IntradayContractValidationError(
            f"{field_name} cannot contain leading or trailing whitespace"
        )
    return value


def _canonical_symbol(value: object) -> str:
    if not isinstance(value, str):
        raise IntradayContractValidationError("symbol must be a string")
    symbol = value.strip().upper()
    if not symbol or not all(
        character.isalnum() or character in ".-" for character in symbol
    ):
        raise IntradayContractValidationError(f"unsupported symbol: {value!r}")
    return symbol


def _utc_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise IntradayContractValidationError(
            f"{field_name} must be a timezone-aware datetime"
        )
    return value.astimezone(UTC)


def _duration_microseconds(duration: timedelta) -> int:
    return (
        duration.days * 86_400 + duration.seconds
    ) * 1_000_000 + duration.microseconds


def _validate_decimal(value: object, field_name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise IntradayContractValidationError(f"{field_name} must be a finite Decimal")
    return value


def _primitive_sort_key(value: PrimitiveMapping) -> bytes:
    return canonical_json_bytes(value)


@dataclass(frozen=True, slots=True)
class IntradayBarRequest:
    """One typed half-open intraday coverage request.

    ``start_timestamp`` is inclusive and ``end_timestamp`` is exclusive. Both
    are canonicalized to UTC. The complete QF-13 timeframe retains the exchange
    calendar, timezone, session, anchor, label, and completion-exposure policy.
    """

    symbol: str
    start_timestamp: datetime
    end_timestamp: datetime
    timeframe: Timeframe
    feed_scope: FeedScope
    adjustment_basis: AdjustmentBasis
    schema_version: str = INTRADAY_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        symbol = _canonical_symbol(self.symbol)
        start_timestamp = _utc_timestamp(self.start_timestamp, "request start")
        end_timestamp = _utc_timestamp(self.end_timestamp, "request end")
        timeframe = cast(object, self.timeframe)
        feed_scope = cast(object, self.feed_scope)
        adjustment_basis = cast(object, self.adjustment_basis)
        if start_timestamp >= end_timestamp:
            raise IntradayContractValidationError(
                "request start must be earlier than end"
            )
        if not isinstance(timeframe, Timeframe) or not isinstance(
            timeframe.interval, IntradayInterval
        ):
            raise IntradayContractValidationError(
                "intraday request requires an intraday timeframe"
            )
        if not isinstance(feed_scope, FeedScope):
            raise IntradayContractValidationError("request feed scope is invalid")
        if not isinstance(adjustment_basis, AdjustmentBasis):
            raise IntradayContractValidationError("request adjustment basis is invalid")
        if self.schema_version != INTRADAY_CONTRACT_SCHEMA_VERSION:
            raise IntradayContractValidationError(
                f"intraday contract schema {INTRADAY_CONTRACT_SCHEMA_VERSION} "
                "is required"
            )
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "start_timestamp", start_timestamp)
        object.__setattr__(self, "end_timestamp", end_timestamp)

    @property
    def source_interval(self) -> IntradayInterval:
        """Return the typed source interval requested from the provider."""
        return cast(IntradayInterval, self.timeframe.interval)

    def to_primitive(self) -> PrimitiveMapping:
        """Return every material request policy in stable serializable form."""
        return {
            "schema_version": self.schema_version,
            "contract_type": "intraday_bar_request",
            "symbol": self.symbol,
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "timeframe": {
                "configuration_id": self.timeframe.configuration_id,
                "configuration": self.timeframe.to_primitive(),
            },
            "feed_scope": self.feed_scope.to_primitive(),
            "adjustment_basis": self.adjustment_basis.to_primitive(),
        }

    @property
    def request_id(self) -> str:
        """Return the deterministic identity of this complete request."""
        return configuration_identity(self.to_primitive())

    def serialize(self) -> bytes:
        """Serialize the request as canonical sorted JSON bytes."""
        return canonical_json_bytes(self.to_primitive())


@dataclass(frozen=True, slots=True)
class IntradayBarProvenance:
    """Provider-neutral source facts retained after adapter normalization."""

    provider_name: str
    provider_symbol: str
    adapter_version: str
    retrieved_at: datetime
    source_request_id: str
    source_snapshot_id: str
    feed_scope: FeedScope
    adjustment_basis: AdjustmentBasis

    def __post_init__(self) -> None:
        _validated_text(self.provider_name, "provider name")
        _validated_text(self.provider_symbol, "provider symbol")
        _validated_text(self.adapter_version, "adapter version")
        _validated_text(self.source_request_id, "source request ID")
        _validated_text(self.source_snapshot_id, "source snapshot ID")
        feed_scope = cast(object, self.feed_scope)
        adjustment_basis = cast(object, self.adjustment_basis)
        if not isinstance(feed_scope, FeedScope):
            raise IntradayContractValidationError("provenance feed scope is invalid")
        if not isinstance(adjustment_basis, AdjustmentBasis):
            raise IntradayContractValidationError(
                "provenance adjustment basis is invalid"
            )
        object.__setattr__(
            self,
            "retrieved_at",
            _utc_timestamp(self.retrieved_at, "provenance retrieval timestamp"),
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "provider_name": self.provider_name,
            "provider_symbol": self.provider_symbol,
            "adapter_version": self.adapter_version,
            "retrieved_at": self.retrieved_at.isoformat(),
            "source_request_id": self.source_request_id,
            "source_snapshot_id": self.source_snapshot_id,
            "feed_scope": self.feed_scope.to_primitive(),
            "adjustment_basis": self.adjustment_basis.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class IntradayBar:
    """One canonical provider-neutral intraday OHLCV observation."""

    symbol: str
    session_date: date
    start_timestamp: datetime
    end_timestamp: datetime
    timeframe: Timeframe
    completion: BarCompletion
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    provenance: IntradayBarProvenance
    schema_version: str = INTRADAY_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        symbol = _canonical_symbol(self.symbol)
        if type(self.session_date) is not date:
            raise IntradayContractValidationError("bar session date must be a date")
        provenance = cast(object, self.provenance)
        if not isinstance(provenance, IntradayBarProvenance):
            raise IntradayContractValidationError("bar provenance is invalid")
        if self.schema_version != INTRADAY_CONTRACT_SCHEMA_VERSION:
            raise IntradayContractValidationError(
                f"intraday contract schema {INTRADAY_CONTRACT_SCHEMA_VERSION} "
                "is required"
            )
        try:
            window = IntradayBarWindow(
                self.timeframe,
                self.session_date,
                self.start_timestamp,
                self.end_timestamp,
                self.completion,
            )
        except TimeframeValidationError as error:
            raise IntradayContractValidationError(str(error)) from error
        if self.provenance.retrieved_at < window.end_timestamp:
            raise IntradayContractValidationError(
                "bar retrieval timestamp cannot precede the observed bar end"
            )
        open_price = _validate_decimal(self.open, "bar open")
        high_price = _validate_decimal(self.high, "bar high")
        low_price = _validate_decimal(self.low, "bar low")
        close_price = _validate_decimal(self.close, "bar close")
        volume = _validate_decimal(self.volume, "bar volume")
        if any(
            price <= 0 for price in (open_price, high_price, low_price, close_price)
        ):
            raise IntradayContractValidationError("bar OHLC prices must be positive")
        if volume < 0:
            raise IntradayContractValidationError("bar volume must be nonnegative")
        if high_price < max(open_price, low_price, close_price):
            raise IntradayContractValidationError(
                "bar high cannot be below open, low, or close"
            )
        if low_price > min(open_price, high_price, close_price):
            raise IntradayContractValidationError(
                "bar low cannot be above open, high, or close"
            )
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "start_timestamp", window.start_timestamp)
        object.__setattr__(self, "end_timestamp", window.end_timestamp)

    @property
    def nominal_interval(self) -> IntradayInterval:
        """Return the nominal source interval independently of actual duration."""
        return cast(IntradayInterval, self.timeframe.interval)

    @property
    def nominal_duration(self) -> timedelta:
        return self.nominal_interval.nominal_duration

    @property
    def actual_duration(self) -> timedelta:
        return self.end_timestamp - self.start_timestamp

    @property
    def session_identifier(self) -> str:
        """Return the canonical exchange-session identifier."""
        return self.session_date.isoformat()

    def to_primitive(self) -> PrimitiveMapping:
        """Return the complete normalized bar in stable serializable form."""
        return {
            "schema_version": self.schema_version,
            "contract_type": "intraday_bar",
            "symbol": self.symbol,
            "session_identifier": self.session_identifier,
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "timeframe": {
                "configuration_id": self.timeframe.configuration_id,
                "configuration": self.timeframe.to_primitive(),
            },
            "nominal_duration_microseconds": _duration_microseconds(
                self.nominal_duration
            ),
            "actual_duration_microseconds": _duration_microseconds(
                self.actual_duration
            ),
            "completion": self.completion.value,
            "open": decimal_to_primitive(self.open),
            "high": decimal_to_primitive(self.high),
            "low": decimal_to_primitive(self.low),
            "close": decimal_to_primitive(self.close),
            "volume": decimal_to_primitive(self.volume),
            "provenance": self.provenance.to_primitive(),
        }

    @property
    def bar_id(self) -> str:
        """Return the deterministic identity of this exact normalized bar."""
        return configuration_identity(self.to_primitive())

    def serialize(self) -> bytes:
        """Serialize the bar as canonical sorted JSON bytes."""
        return canonical_json_bytes(self.to_primitive())


@dataclass(frozen=True, slots=True)
class IntradayBarBatch:
    """One request-bound, chronologically ordered canonical bar collection."""

    request: IntradayBarRequest
    bars: tuple[IntradayBar, ...]
    schema_version: str = INTRADAY_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        request = cast(object, self.request)
        bars = cast(object, self.bars)
        if not isinstance(request, IntradayBarRequest):
            raise IntradayContractValidationError(
                "intraday bar batch request is invalid"
            )
        if not isinstance(bars, tuple):
            raise IntradayContractValidationError(
                "intraday bar batch bars must be a tuple"
            )
        untyped_bars = cast(tuple[object, ...], bars)
        if any(not isinstance(bar, IntradayBar) for bar in untyped_bars):
            raise IntradayContractValidationError(
                "intraday bar batch contains an invalid bar"
            )
        typed_bars = cast(tuple[IntradayBar, ...], untyped_bars)
        if self.schema_version != INTRADAY_CONTRACT_SCHEMA_VERSION:
            raise IntradayContractValidationError(
                f"intraday contract schema {INTRADAY_CONTRACT_SCHEMA_VERSION} "
                "is required"
            )
        for bar in typed_bars:
            self._validate_bar_matches_request(bar)
        bar_keys = tuple(
            (bar.symbol, bar.timeframe.configuration_id, bar.start_timestamp)
            for bar in typed_bars
        )
        if len(set(bar_keys)) != len(bar_keys):
            raise IntradayContractValidationError(
                "intraday bar batch contains a duplicate bar key"
            )
        chronological_keys = tuple(
            (bar.start_timestamp, bar.end_timestamp) for bar in typed_bars
        )
        if chronological_keys != tuple(sorted(chronological_keys)):
            raise IntradayContractValidationError(
                "intraday bar batch must be ordered chronologically"
            )
        for previous, current in pairwise(typed_bars):
            if current.start_timestamp < previous.end_timestamp:
                raise IntradayContractValidationError(
                    "intraday bar batch contains overlapping timestamps"
                )

    def _validate_bar_matches_request(self, bar: IntradayBar) -> None:
        if bar.symbol != self.request.symbol:
            raise IntradayContractValidationError(
                "intraday bar symbol does not match its request"
            )
        if bar.timeframe != self.request.timeframe:
            raise IntradayContractValidationError(
                "intraday bar timeframe does not match its request"
            )
        if bar.provenance.source_request_id != self.request.request_id:
            raise IntradayContractValidationError(
                "intraday bar provenance does not match its request identity"
            )
        if bar.provenance.feed_scope != self.request.feed_scope:
            raise IntradayContractValidationError(
                "intraday bar feed scope does not match its request"
            )
        if bar.provenance.adjustment_basis != self.request.adjustment_basis:
            raise IntradayContractValidationError(
                "intraday bar adjustment basis does not match its request"
            )
        if (
            not (
                self.request.start_timestamp
                <= bar.start_timestamp
                < self.request.end_timestamp
            )
            or bar.end_timestamp > self.request.end_timestamp
        ):
            raise IntradayContractValidationError(
                "intraday bar falls outside its request range"
            )

    def to_primitive(self) -> PrimitiveMapping:
        """Return the request binding and ordered canonical bars."""
        return {
            "schema_version": self.schema_version,
            "contract_type": "intraday_bar_batch",
            "request": {
                "request_id": self.request.request_id,
                "configuration": self.request.to_primitive(),
            },
            "bars": [
                {"bar_id": bar.bar_id, "bar": bar.to_primitive()} for bar in self.bars
            ],
        }

    @property
    def batch_id(self) -> str:
        """Return the deterministic identity of this exact ordered collection."""
        return configuration_identity(self.to_primitive())

    def serialize(self) -> bytes:
        """Serialize the batch as canonical sorted JSON bytes."""
        return canonical_json_bytes(self.to_primitive())


@dataclass(frozen=True, slots=True)
class IntradayProviderCapabilities:
    """Serializable provider support for source intervals and coverage policy."""

    provider_name: str
    supported_intervals: tuple[IntradayInterval, ...]
    supported_feed_scopes: tuple[FeedScope, ...]
    supported_session_scopes: tuple[SessionScope, ...]
    available_start_timestamp: datetime | None = None
    available_end_timestamp: datetime | None = None
    schema_version: str = INTRADAY_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validated_text(self.provider_name, "provider name")
        intervals = cast(object, self.supported_intervals)
        feeds = cast(object, self.supported_feed_scopes)
        sessions = cast(object, self.supported_session_scopes)
        if not isinstance(intervals, tuple) or not intervals:
            raise IntradayContractValidationError(
                "provider must declare at least one supported interval"
            )
        if not isinstance(feeds, tuple) or not feeds:
            raise IntradayContractValidationError(
                "provider must declare at least one supported feed scope"
            )
        if not isinstance(sessions, tuple) or not sessions:
            raise IntradayContractValidationError(
                "provider must declare at least one supported session scope"
            )
        untyped_intervals = cast(tuple[object, ...], intervals)
        untyped_feeds = cast(tuple[object, ...], feeds)
        untyped_sessions = cast(tuple[object, ...], sessions)
        if any(not isinstance(item, IntradayInterval) for item in untyped_intervals):
            raise IntradayContractValidationError(
                "provider supported interval is invalid"
            )
        if any(not isinstance(item, FeedScope) for item in untyped_feeds):
            raise IntradayContractValidationError(
                "provider supported feed scope is invalid"
            )
        if any(not isinstance(item, SessionScope) for item in untyped_sessions):
            raise IntradayContractValidationError(
                "provider supported session scope is invalid"
            )
        typed_intervals = cast(tuple[IntradayInterval, ...], untyped_intervals)
        typed_feeds = cast(tuple[FeedScope, ...], untyped_feeds)
        typed_sessions = cast(tuple[SessionScope, ...], untyped_sessions)
        if len(set(typed_intervals)) != len(typed_intervals):
            raise IntradayContractValidationError(
                "provider supported intervals must be unique"
            )
        if len(set(typed_feeds)) != len(typed_feeds):
            raise IntradayContractValidationError(
                "provider supported feed scopes must be unique"
            )
        if len(set(typed_sessions)) != len(typed_sessions):
            raise IntradayContractValidationError(
                "provider supported session scopes must be unique"
            )
        ordered_intervals = tuple(
            sorted(
                typed_intervals,
                key=lambda item: _primitive_sort_key(item.to_primitive()),
            )
        )
        ordered_feeds = tuple(
            sorted(
                typed_feeds,
                key=lambda item: _primitive_sort_key(item.to_primitive()),
            )
        )
        ordered_sessions = tuple(sorted(typed_sessions, key=lambda item: item.value))
        start = (
            None
            if self.available_start_timestamp is None
            else _utc_timestamp(
                self.available_start_timestamp, "capability available start"
            )
        )
        end = (
            None
            if self.available_end_timestamp is None
            else _utc_timestamp(
                self.available_end_timestamp, "capability available end"
            )
        )
        if start is not None and end is not None and start >= end:
            raise IntradayContractValidationError(
                "capability available start must be earlier than end"
            )
        if self.schema_version != INTRADAY_CONTRACT_SCHEMA_VERSION:
            raise IntradayContractValidationError(
                f"intraday contract schema {INTRADAY_CONTRACT_SCHEMA_VERSION} "
                "is required"
            )
        object.__setattr__(self, "supported_intervals", ordered_intervals)
        object.__setattr__(self, "supported_feed_scopes", ordered_feeds)
        object.__setattr__(self, "supported_session_scopes", ordered_sessions)
        object.__setattr__(self, "available_start_timestamp", start)
        object.__setattr__(self, "available_end_timestamp", end)

    def to_primitive(self) -> PrimitiveMapping:
        """Return canonical provider capability metadata."""
        return {
            "schema_version": self.schema_version,
            "contract_type": "intraday_provider_capabilities",
            "provider_name": self.provider_name,
            "supported_intervals": [
                interval.to_primitive() for interval in self.supported_intervals
            ],
            "supported_feed_scopes": [
                feed_scope.to_primitive() for feed_scope in self.supported_feed_scopes
            ],
            "supported_session_scopes": [
                scope.value for scope in self.supported_session_scopes
            ],
            "available_range": {
                "start_timestamp": (
                    None
                    if self.available_start_timestamp is None
                    else self.available_start_timestamp.isoformat()
                ),
                "end_timestamp": (
                    None
                    if self.available_end_timestamp is None
                    else self.available_end_timestamp.isoformat()
                ),
            },
        }

    @property
    def configuration_id(self) -> str:
        """Return the deterministic identity of the declared capabilities."""
        return configuration_identity(self.to_primitive())

    def serialize(self) -> bytes:
        """Serialize capability metadata as canonical sorted JSON bytes."""
        return canonical_json_bytes(self.to_primitive())

    def validate_request(self, request: IntradayBarRequest) -> None:
        """Raise a precise domain exception for an unsupported valid request."""
        request_value = cast(object, request)
        if not isinstance(request_value, IntradayBarRequest):
            raise IntradayContractValidationError(
                "capability validation requires an intraday request"
            )
        if request.source_interval not in self.supported_intervals:
            raise UnsupportedIntervalError(
                f"{self.provider_name} does not support the requested interval"
            )
        if request.feed_scope not in self.supported_feed_scopes:
            raise UnsupportedFeedError(
                f"{self.provider_name} does not support the requested feed scope"
            )
        if request.timeframe.session_policy.scope not in self.supported_session_scopes:
            raise UnsupportedSessionScopeError(
                f"{self.provider_name} does not support the requested session scope"
            )
        if (
            self.available_start_timestamp is not None
            and request.start_timestamp < self.available_start_timestamp
        ) or (
            self.available_end_timestamp is not None
            and request.end_timestamp > self.available_end_timestamp
        ):
            raise UnsupportedDateRangeError(
                f"{self.provider_name} does not support the requested date range"
            )


__all__ = [
    "INTRADAY_CONTRACT_SCHEMA_VERSION",
    "IntradayBar",
    "IntradayBarBatch",
    "IntradayBarProvenance",
    "IntradayBarRequest",
    "IntradayContractValidationError",
    "IntradayProviderCapabilities",
]
