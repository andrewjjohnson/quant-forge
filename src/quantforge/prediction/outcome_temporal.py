"""Time-only outcome contracts; no observations, labels, or membership decisions."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data.calendar import calendar_has_intraday_recesses
from quantforge.prediction.window_timeframe_validation import (
    validate_source_timeframe_definition,
)
from quantforge.timeframes import (
    CrossSessionPolicy,
    IntradayInterval,
    Timeframe,
)

if TYPE_CHECKING:
    from quantforge.validation.models import TemporalOffset


class OutcomeTemporalError(ValueError):
    """An outcome's declared temporal semantics are inconsistent."""


class OutcomeAnchorKind(StrEnum):
    SESSION = "exchange_session"
    TIMESTAMP = "exact_timestamp"


class ObservationAlignment(StrEnum):
    FIRST_COMPLETED_AT_OR_AFTER = "first_expected_completed_at_or_after"


class OutcomeSessionPolicy(StrEnum):
    SAME_SESSION_ONLY = "same_session_only"


def outcome_utc_timestamp(timestamp: object) -> datetime:
    """Reject naive anchors and preserve the exact instant in canonical UTC."""
    if not isinstance(timestamp, datetime) or timestamp.utcoffset() is None:
        raise OutcomeTemporalError("outcome timestamp must be timezone-aware")
    return timestamp.astimezone(UTC)


def _positive_integer(count: object) -> int:
    if type(count) is not int or count < 1:
        raise OutcomeTemporalError("outcome horizon must be a positive integer")
    return count


@dataclass(frozen=True, slots=True)
class ExchangeSessionHorizon:
    count: int

    def __post_init__(self) -> None:
        _positive_integer(self.count)

    def to_primitive(self) -> PrimitiveMapping:
        return {"kind": "exchange_sessions", "count": self.count}


@dataclass(frozen=True, slots=True)
class ElapsedDurationHorizon:
    duration: timedelta

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.duration), timedelta) or self.duration <= (
            timedelta(0)
        ):
            raise OutcomeTemporalError("elapsed outcome horizon must be positive")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "kind": "elapsed_duration",
            "duration_microseconds": self.duration // timedelta(microseconds=1),
        }


type OutcomeHorizon = ExchangeSessionHorizon | ElapsedDurationHorizon


@dataclass(frozen=True, slots=True)
class OutcomeAnchor:
    """The session label is supplied, never inferred from a timestamp's date."""

    kind: OutcomeAnchorKind
    signal_session: date
    decision_timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.kind), OutcomeAnchorKind):
            raise OutcomeTemporalError("outcome anchor kind is invalid")
        if type(self.signal_session) is not date:
            raise OutcomeTemporalError("outcome signal session must be a date")
        if self.decision_timestamp is not None:
            object.__setattr__(
                self,
                "decision_timestamp",
                outcome_utc_timestamp(self.decision_timestamp),
            )
        if self.kind is OutcomeAnchorKind.TIMESTAMP and self.decision_timestamp is None:
            raise OutcomeTemporalError("exact outcome anchor requires a timestamp")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "anchor_kind": self.kind.value,
            "signal_session": self.signal_session.isoformat(),
            "decision_timestamp": (
                None
                if self.decision_timestamp is None
                else self.decision_timestamp.isoformat()
            ),
        }

    @classmethod
    def from_primitive(cls, primitive: PrimitiveMapping) -> "OutcomeAnchor":
        try:
            timestamp = primitive["decision_timestamp"]
            anchor = cls(
                OutcomeAnchorKind(primitive["anchor_kind"]),
                date.fromisoformat(cast(str, primitive["signal_session"])),
                None
                if timestamp is None
                else datetime.fromisoformat(cast(str, timestamp)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise OutcomeTemporalError("invalid serialized outcome anchor") from error
        if configuration_identity(anchor.to_primitive()) != configuration_identity(
            primitive
        ):
            raise OutcomeTemporalError("outcome anchor serialization is noncanonical")
        return anchor


@dataclass(frozen=True, slots=True)
class OutcomeTemporalConfiguration:
    """Explicit horizon/alignment identity, independent of concrete outcome math."""

    anchor_kind: OutcomeAnchorKind
    horizon: OutcomeHorizon
    observation_timeframe: Timeframe | None = None
    alignment: ObservationAlignment | None = None
    session_policy: OutcomeSessionPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.anchor_kind), OutcomeAnchorKind):
            raise OutcomeTemporalError("outcome anchor kind is invalid")
        if isinstance(self.horizon, ExchangeSessionHorizon):
            if self.anchor_kind is not OutcomeAnchorKind.SESSION or any(
                item is not None
                for item in (
                    self.observation_timeframe,
                    self.alignment,
                    self.session_policy,
                )
            ):
                raise OutcomeTemporalError(
                    "session horizons require session anchor semantics"
                )
            return
        timeframe = self.observation_timeframe
        if (
            not isinstance(cast(object, self.horizon), ElapsedDurationHorizon)
            or self.anchor_kind is not OutcomeAnchorKind.TIMESTAMP
            or not isinstance(timeframe, Timeframe)
            or not isinstance(timeframe.interval, IntradayInterval)
            or timeframe.interval.cross_session_policy
            is not CrossSessionPolicy.PROHIBITED
            or self.alignment is not ObservationAlignment.FIRST_COMPLETED_AT_OR_AFTER
            or self.session_policy is not OutcomeSessionPolicy.SAME_SESSION_ONLY
        ):
            raise OutcomeTemporalError(
                "elapsed outcomes require explicit completed same-session semantics"
            )
        if calendar_has_intraday_recesses(timeframe.session_policy.calendar_name):
            raise OutcomeTemporalError(
                "outcome observation windows do not support intraday recesses"
            )
        # Fail at configuration time if the conservative duration cannot be represented.
        try:
            self.required_future_duration
        except OverflowError as error:
            raise OutcomeTemporalError(
                "outcome temporal reach exceeds timedelta range"
            ) from error

    @classmethod
    def exchange_sessions(cls, count: int) -> "OutcomeTemporalConfiguration":
        return cls(OutcomeAnchorKind.SESSION, ExchangeSessionHorizon(count))

    @classmethod
    def elapsed_duration(
        cls, duration: timedelta, observation_timeframe: Timeframe
    ) -> "OutcomeTemporalConfiguration":
        return cls(
            OutcomeAnchorKind.TIMESTAMP,
            ElapsedDurationHorizon(duration),
            observation_timeframe,
            ObservationAlignment.FIRST_COMPLETED_AT_OR_AFTER,
            OutcomeSessionPolicy.SAME_SESSION_ONLY,
        )

    @property
    def required_future_duration(self) -> timedelta:
        """Conservative reach: nominal duration plus one full alignment interval.

        Never clip to session close: overflow is unavailable, not a shorter label.
        This property can be delegated by a QF-8 TimestampOutcomeComponent.
        """
        if not isinstance(self.horizon, ElapsedDurationHorizon):
            raise OutcomeTemporalError("session horizon has no elapsed duration")
        assert self.observation_timeframe is not None
        interval = cast(IntradayInterval, self.observation_timeframe.interval)
        return self.horizon.duration + interval.nominal_duration

    @property
    def future_temporal_reach(self) -> "TemporalOffset":
        """Reuse QF-8 offsets without changing partition membership."""
        from quantforge.validation.models import TemporalOffset

        if isinstance(self.horizon, ExchangeSessionHorizon):
            return TemporalOffset.sessions(self.horizon.count)
        return TemporalOffset.duration(self.required_future_duration)

    def to_primitive(self) -> PrimitiveMapping:
        timeframe = self.observation_timeframe
        return {
            "schema_version": "1",
            "anchor_kind": self.anchor_kind.value,
            "horizon": self.horizon.to_primitive(),
            "observation_timeframe": None
            if timeframe is None
            else {
                "configuration_id": timeframe.configuration_id,
                "configuration": timeframe.to_primitive(),
            },
            "alignment": None if self.alignment is None else self.alignment.value,
            "session_policy": None
            if self.session_policy is None
            else self.session_policy.value,
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.to_primitive())

    @classmethod
    def from_primitive(
        cls, primitive: PrimitiveMapping
    ) -> "OutcomeTemporalConfiguration":
        """Read explicit typed configuration; reject unknown/noncanonical policy."""
        try:
            horizon = primitive["horizon"]
            if not isinstance(horizon, dict):
                raise OutcomeTemporalError("horizon must be an object")
            typed_horizon: OutcomeHorizon
            if horizon.get("kind") == "exchange_sessions":
                typed_horizon = ExchangeSessionHorizon(
                    _positive_integer(horizon.get("count"))
                )
            elif horizon.get("kind") == "elapsed_duration":
                typed_horizon = ElapsedDurationHorizon(
                    timedelta(
                        microseconds=_positive_integer(
                            horizon.get("duration_microseconds")
                        )
                    )
                )
            else:
                raise OutcomeTemporalError("unknown outcome horizon kind")
            timeframe = primitive["observation_timeframe"]
            if timeframe is not None and not isinstance(timeframe, dict):
                raise OutcomeTemporalError("observation timeframe must be an object")
            config = cls(
                OutcomeAnchorKind(primitive["anchor_kind"]),
                typed_horizon,
                None
                if timeframe is None
                else validate_source_timeframe_definition(timeframe),
                None
                if primitive["alignment"] is None
                else ObservationAlignment(primitive["alignment"]),
                None
                if primitive["session_policy"] is None
                else OutcomeSessionPolicy(primitive["session_policy"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise OutcomeTemporalError(
                "invalid serialized outcome temporal configuration"
            ) from error
        if configuration_identity(config.to_primitive()) != configuration_identity(
            primitive
        ):
            raise OutcomeTemporalError("outcome temporal configuration is noncanonical")
        return config


def outcome_temporal_configuration(
    configuration: PrimitiveMapping, *, required_future_sessions: int | None = None
) -> OutcomeTemporalConfiguration:
    """Read typed declarations or historical session counts without rewriting inputs.

    A missing type always means exchange sessions. Legacy configuration hashes
    remain byte-equivalent because this adapter never adds fields to them.
    """
    parameters = configuration.get("parameters", {})
    if not isinstance(parameters, dict):
        raise OutcomeTemporalError("outcome parameters must be an object")
    counts = [
        _positive_integer(container[key])
        for container, key in (
            (parameters, "future_sessions"),
            (parameters, "unavailable_horizon_sessions"),
            (configuration, "required_future_sessions"),
        )
        if key in container
    ]
    if required_future_sessions is not None:
        counts.append(_positive_integer(required_future_sessions))
    if "temporal_configuration" in configuration:
        temporal = configuration["temporal_configuration"]
        if not isinstance(temporal, dict):
            raise OutcomeTemporalError(
                "outcome temporal configuration must be an object"
            )
        result = OutcomeTemporalConfiguration.from_primitive(temporal)
        if counts and (
            not isinstance(result.horizon, ExchangeSessionHorizon)
            or any(count != result.horizon.count for count in counts)
        ):
            raise OutcomeTemporalError("outcome horizon differs from configuration")
        return result
    if not counts or len(set(counts)) != 1:
        raise OutcomeTemporalError(
            "outcome requires one consistent positive session horizon"
        )
    return OutcomeTemporalConfiguration.exchange_sessions(counts[0])
