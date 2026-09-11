"""Immutable, study-neutral validation-plan contracts."""

from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from typing import TYPE_CHECKING, Protocol, cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import (
    AggregationPolicy,
    DatasetFamily,
    DatasetFamilyReference,
    DatasetMetadata,
    MarketDataset,
    validate_market_dataset,
)
from quantforge.data.identity import serialize_metadata_values
from quantforge.indicators import Indicator, IndicatorBackendIdentity
from quantforge.timeframes import (
    DEFAULT_US_EQUITY_SESSION_POLICY,
    ExchangeSessionPolicy,
    IntradayInterval,
    SessionInterval,
    Timeframe,
    TimeframeValidationError,
    resolve_exchange_session,
    resolve_exchange_timezone_name,
)
from quantforge.validation.errors import ValidationPlanError

if TYPE_CHECKING:
    from quantforge.backtesting import BacktestConfig

VALIDATION_PLAN_SCHEMA_VERSION = "1"
VALIDATION_WINDOW_SCHEMA_VERSION = "1"
RESEARCH_ENVIRONMENT_SCHEMA_VERSION = "1"


def _validated_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationPlanError(f"{label} must be a non-empty string")
    return value


def _validated_hash(value: object, label: str) -> str:
    text = _validated_text(value, label)
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ValidationPlanError(f"{label} must be a lowercase SHA-256 value")
    return text


def _duration_microseconds(duration: timedelta) -> int:
    return (
        duration.days * 86_400 + duration.seconds
    ) * 1_000_000 + duration.microseconds


class BoundaryAxis(StrEnum):
    """The ordered domain used to define validation membership."""

    EXCHANGE_SESSION = "exchange_session"
    TIMESTAMP = "timestamp"


@dataclass(frozen=True, slots=True)
class ExchangeSessionBoundary:
    """One actual exchange session used as an inclusive partition boundary."""

    session_date: date
    session_policy: ExchangeSessionPolicy = DEFAULT_US_EQUITY_SESSION_POLICY

    def __post_init__(self) -> None:
        if type(self.session_date) is not date:
            raise ValidationPlanError("exchange-session boundary must be a date")
        if not isinstance(cast(object, self.session_policy), ExchangeSessionPolicy):
            raise ValidationPlanError("exchange-session boundary policy is invalid")
        try:
            resolve_exchange_session(self.session_date, self.session_policy)
        except ValueError as error:
            raise ValidationPlanError(
                "exchange-session boundary must identify an actual configured "
                "exchange session"
            ) from error

    @property
    def axis(self) -> BoundaryAxis:
        return BoundaryAxis.EXCHANGE_SESSION

    def to_primitive(self) -> PrimitiveMapping:
        policy = self.session_policy.to_primitive()
        return {
            "axis": self.axis.value,
            "session": self.session_date.isoformat(),
            "session_policy": {
                "configuration_id": configuration_identity(policy),
                "configuration": policy,
            },
        }


@dataclass(frozen=True, slots=True)
class TimestampBoundary:
    """One timezone-aware instant normalized to UTC for validation membership."""

    timestamp: datetime

    def __post_init__(self) -> None:
        value = cast(object, self.timestamp)
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ValidationPlanError(
                "timestamp boundary must be a timezone-aware datetime"
            )
        object.__setattr__(self, "timestamp", value.astimezone(UTC))

    @property
    def axis(self) -> BoundaryAxis:
        return BoundaryAxis.TIMESTAMP

    def to_primitive(self) -> PrimitiveMapping:
        return {"axis": self.axis.value, "timestamp": self.timestamp.isoformat()}


type ValidationBoundary = ExchangeSessionBoundary | TimestampBoundary


def boundary_axis(boundary: ValidationBoundary) -> BoundaryAxis:
    """Return the typed ordering domain for a validation boundary."""
    return boundary.axis


def boundary_value(boundary: ValidationBoundary) -> date | datetime:
    """Return the comparable normalized value for a validation boundary."""
    if isinstance(boundary, ExchangeSessionBoundary):
        return boundary.session_date
    return boundary.timestamp


def boundaries_share_semantics(
    left: ValidationBoundary, right: ValidationBoundary
) -> bool:
    """Return whether two boundaries can safely participate in one partition."""
    if type(left) is not type(right):
        return False
    if isinstance(left, ExchangeSessionBoundary) and isinstance(
        right, ExchangeSessionBoundary
    ):
        return left.session_policy == right.session_policy
    return True


@dataclass(frozen=True, slots=True)
class ValidationInterval:
    """A closed chronological interval over sessions or UTC timestamps."""

    start: ValidationBoundary
    end: ValidationBoundary

    def __post_init__(self) -> None:
        if not boundaries_share_semantics(self.start, self.end):
            raise ValidationPlanError(
                "validation interval boundaries must use one temporal axis and policy"
            )
        if boundary_value(self.start) > boundary_value(self.end):
            raise ValidationPlanError(
                "validation interval start must not follow its inclusive end"
            )

    @property
    def axis(self) -> BoundaryAxis:
        return boundary_axis(self.start)

    def contains(self, boundary: ValidationBoundary) -> bool:
        if not boundaries_share_semantics(self.start, boundary):
            raise ValidationPlanError(
                "observation boundary does not match validation interval semantics"
            )
        value = boundary_value(boundary)
        return boundary_value(self.start) <= value <= boundary_value(self.end)

    def precedes(self, other: "ValidationInterval") -> bool:
        if not boundaries_share_semantics(self.start, other.start):
            raise ValidationPlanError(
                "validation intervals do not share temporal semantics"
            )
        return boundary_value(self.end) < boundary_value(other.start)

    def overlaps(self, other: "ValidationInterval") -> bool:
        if not boundaries_share_semantics(self.start, other.start):
            raise ValidationPlanError(
                "validation intervals do not share temporal semantics"
            )
        return not (
            boundary_value(self.end) < boundary_value(other.start)
            or boundary_value(other.end) < boundary_value(self.start)
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "bounds": "closed_inclusive",
            "start": self.start.to_primitive(),
            "end": self.end.to_primitive(),
        }


class PartitionRole(StrEnum):
    """Scientific role of one immutable validation window."""

    DEVELOPMENT = "development_training"
    SELECTION = "validation_selection"
    WALK_FORWARD_TEST = "walk_forward_test"
    FINAL_HOLDOUT = "final_holdout"


@dataclass(frozen=True, slots=True)
class TimeframeWarmUpRequirement:
    """Context-only source-bar count expressed in one exact timeframe."""

    timeframe: Timeframe
    observations: int

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.timeframe), Timeframe):
            raise ValidationPlanError("warm-up source timeframe is invalid")
        observations = cast(object, self.observations)
        if (
            isinstance(observations, bool)
            or not isinstance(observations, int)
            or observations < 0
        ):
            raise ValidationPlanError(
                "timeframe warm-up observations must be a non-negative integer"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "timeframe": {
                "configuration_id": self.timeframe.configuration_id,
                "configuration": self.timeframe.to_primitive(),
            },
            "observations": self.observations,
        }


@dataclass(frozen=True, slots=True)
class ValidationWindow:
    """One identity-bearing interval and its non-selecting warm-up requirement."""

    name: str
    role: PartitionRole
    interval: ValidationInterval
    warm_up_observations: int = 0
    schema_version: str = VALIDATION_WINDOW_SCHEMA_VERSION
    warm_up_by_timeframe: tuple[TimeframeWarmUpRequirement, ...] = ()

    def __post_init__(self) -> None:
        _validated_text(self.name, "validation window name")
        if not isinstance(cast(object, self.role), PartitionRole):
            raise ValidationPlanError("validation window role is invalid")
        if not isinstance(cast(object, self.interval), ValidationInterval):
            raise ValidationPlanError("validation window interval is invalid")
        warm_up = cast(object, self.warm_up_observations)
        if isinstance(warm_up, bool) or not isinstance(warm_up, int) or warm_up < 0:
            raise ValidationPlanError(
                "validation warm-up observations must be a non-negative integer"
            )
        requirements_value = cast(object, self.warm_up_by_timeframe)
        if not isinstance(requirements_value, tuple) or any(
            not isinstance(item, TimeframeWarmUpRequirement)
            for item in cast(tuple[object, ...], requirements_value)
        ):
            raise ValidationPlanError(
                "validation timeframe warm-up requirements are invalid"
            )
        requirements = tuple(
            sorted(
                self.warm_up_by_timeframe,
                key=lambda item: item.timeframe.configuration_id,
            )
        )
        identifiers = tuple(item.timeframe.configuration_id for item in requirements)
        if len(identifiers) != len(set(identifiers)):
            raise ValidationPlanError(
                "validation timeframe warm-up requirements must be unique"
            )
        if warm_up and requirements:
            raise ValidationPlanError(
                "validation window cannot mix scalar and timeframe-specific warm-up"
            )
        if self.schema_version != VALIDATION_WINDOW_SCHEMA_VERSION:
            raise ValidationPlanError("validation window schema version is unsupported")
        object.__setattr__(self, "warm_up_by_timeframe", requirements)

    def warm_up_observations_for(self, source_timeframe: Timeframe | None) -> int:
        """Resolve a source-specific count, failing on ambiguous multi-source use."""
        if (
            self.interval.axis is BoundaryAxis.EXCHANGE_SESSION
            and source_timeframe is not None
            and isinstance(source_timeframe.interval, IntradayInterval)
        ):
            raise ValidationPlanError(
                "intraday source observations require timestamp validation boundaries"
            )
        if not self.warm_up_by_timeframe:
            return self.warm_up_observations
        if source_timeframe is None:
            raise ValidationPlanError(
                "source timeframe is required for timeframe-specific warm-up"
            )
        for requirement in self.warm_up_by_timeframe:
            if requirement.timeframe == source_timeframe:
                return requirement.observations
        raise ValidationPlanError(
            "validation window has no warm-up requirement for the source timeframe"
        )

    def _identity_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "role": self.role.value,
            "interval": self.interval.to_primitive(),
            "warm_up": {
                "observations": self.warm_up_observations,
                "by_timeframe": [
                    item.to_primitive() for item in self.warm_up_by_timeframe
                ],
                "purpose": "indicator_context_only",
                "eligible_for_selection": False,
            },
        }

    @property
    def window_id(self) -> str:
        return configuration_identity(self._identity_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {"window_id": self.window_id, **self._identity_primitive()}


class TrainingWindowMode(StrEnum):
    """How explicit development windows advance between walk-forward folds."""

    EXPANDING = "expanding"
    ROLLING = "rolling"


@dataclass(frozen=True, slots=True)
class ValidationFold:
    """One explicit development/selection/test definition, without execution."""

    name: str
    development: ValidationWindow
    test: ValidationWindow
    selection: ValidationWindow | None = None

    def __post_init__(self) -> None:
        _validated_text(self.name, "validation fold name")
        if not isinstance(
            cast(object, self.development), ValidationWindow
        ) or not isinstance(cast(object, self.test), ValidationWindow):
            raise ValidationPlanError("validation fold windows are invalid")
        if self.selection is not None and not isinstance(
            cast(object, self.selection), ValidationWindow
        ):
            raise ValidationPlanError("validation fold selection window is invalid")
        if self.development.role is not PartitionRole.DEVELOPMENT:
            raise ValidationPlanError(
                "validation fold development window has the wrong role"
            )
        if self.test.role is not PartitionRole.WALK_FORWARD_TEST:
            raise ValidationPlanError("validation fold test window has the wrong role")
        protected = self.test
        if self.selection is not None:
            if self.selection.role is not PartitionRole.SELECTION:
                raise ValidationPlanError(
                    "validation fold selection window has the wrong role"
                )
            if not self.development.interval.precedes(self.selection.interval):
                raise ValidationPlanError(
                    "development and validation/selection intervals overlap or "
                    "are not chronological"
                )
            protected = self.selection
            if not self.selection.interval.precedes(self.test.interval):
                raise ValidationPlanError(
                    "validation/selection and test intervals overlap or are not "
                    "chronological"
                )
        elif not self.development.interval.precedes(self.test.interval):
            raise ValidationPlanError(
                "development and test intervals overlap or are not chronological"
            )
        if self.development.interval.axis is not protected.interval.axis:
            raise ValidationPlanError("validation fold mixes temporal axes")

    @property
    def next_protected_window(self) -> ValidationWindow:
        """Return the first interval that development outcomes must not enter."""
        return self.selection or self.test

    def _identity_primitive(self) -> PrimitiveMapping:
        return {
            "name": self.name,
            "development": self.development.to_primitive(),
            "selection": (
                None if self.selection is None else self.selection.to_primitive()
            ),
            "test": self.test.to_primitive(),
        }

    @property
    def fold_id(self) -> str:
        return configuration_identity(self._identity_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {"fold_id": self.fold_id, **self._identity_primitive()}


@dataclass(frozen=True, slots=True)
class FinalHoldout:
    """Reserved final interval; QF-8 never consumes or evaluates it."""

    window: ValidationWindow
    reservation_purpose: str

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.window), ValidationWindow):
            raise ValidationPlanError("final holdout window is invalid")
        if self.window.role is not PartitionRole.FINAL_HOLDOUT:
            raise ValidationPlanError("final holdout window has the wrong role")
        _validated_text(self.reservation_purpose, "holdout reservation purpose")

    def _identity_primitive(self) -> PrimitiveMapping:
        return {
            "window": self.window.to_primitive(),
            "reservation": {
                "reserved": True,
                "purpose": self.reservation_purpose,
                "consumption": "outside_qf8_scope",
            },
        }

    @property
    def holdout_id(self) -> str:
        return configuration_identity(self._identity_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {"holdout_id": self.holdout_id, **self._identity_primitive()}


@dataclass(frozen=True, slots=True)
class TemporalOffset:
    """A non-negative session count or exact elapsed duration."""

    axis: BoundaryAxis
    exchange_sessions: int | None = None
    elapsed: timedelta | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.axis), BoundaryAxis):
            raise ValidationPlanError("temporal offset axis is invalid")
        if self.axis is BoundaryAxis.EXCHANGE_SESSION:
            sessions = cast(object, self.exchange_sessions)
            if (
                isinstance(sessions, bool)
                or not isinstance(sessions, int)
                or sessions < 0
                or self.elapsed is not None
            ):
                raise ValidationPlanError(
                    "exchange-session offset requires one non-negative session count"
                )
            return
        elapsed = cast(object, self.elapsed)
        if (
            not isinstance(elapsed, timedelta)
            or elapsed < timedelta(0)
            or self.exchange_sessions is not None
        ):
            raise ValidationPlanError(
                "timestamp offset requires one non-negative elapsed duration"
            )

    @classmethod
    def sessions(cls, count: int) -> "TemporalOffset":
        return cls(BoundaryAxis.EXCHANGE_SESSION, exchange_sessions=count)

    @classmethod
    def duration(cls, elapsed: timedelta) -> "TemporalOffset":
        return cls(BoundaryAxis.TIMESTAMP, elapsed=elapsed)

    def to_primitive(self) -> PrimitiveMapping:
        if self.axis is BoundaryAxis.EXCHANGE_SESSION:
            return {
                "axis": self.axis.value,
                "exchange_sessions": self.exchange_sessions,
            }
        assert self.elapsed is not None
        return {
            "axis": self.axis.value,
            "elapsed_microseconds": _duration_microseconds(self.elapsed),
        }

    @property
    def magnitude(self) -> int:
        """Return comparable units within this offset's declared axis."""
        if self.axis is BoundaryAxis.EXCHANGE_SESSION:
            assert self.exchange_sessions is not None
            return self.exchange_sessions
        assert self.elapsed is not None
        return _duration_microseconds(self.elapsed)


@dataclass(frozen=True, slots=True)
class PurgePolicy:
    """Maximum future-label reach plus explicit additional embargo."""

    label_horizon: TemporalOffset
    embargo: TemporalOffset

    def __post_init__(self) -> None:
        if not isinstance(
            cast(object, self.label_horizon), TemporalOffset
        ) or not isinstance(cast(object, self.embargo), TemporalOffset):
            raise ValidationPlanError("purge policy offsets are invalid")
        if self.label_horizon.axis is not self.embargo.axis:
            raise ValidationPlanError(
                "label horizon and embargo must use the same temporal axis"
            )

    @property
    def axis(self) -> BoundaryAxis:
        return self.label_horizon.axis

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "label_horizon": self.label_horizon.to_primitive(),
            "embargo": self.embargo.to_primitive(),
            "boundary_rule": (
                "purge_when_label_horizon_plus_embargo_reaches_or_crosses_"
                "protected_start"
            ),
        }


class ConfiguredComponent(Protocol):
    """Existing QuantForge component with stable versioned configuration."""

    name: str
    implementation_version: str

    @property
    def configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...


class SessionOutcomeComponent(ConfiguredComponent, Protocol):
    """Existing outcome component with an exchange-session future horizon."""

    required_future_sessions: int


class TimestampOutcomeComponent(ConfiguredComponent, Protocol):
    """Existing outcome component with an exact elapsed future horizon."""

    required_future_duration: timedelta


@dataclass(frozen=True, slots=True)
class ConfigurationReference:
    """Immutable reference to one existing versioned component configuration."""

    component_type: str
    name: str
    implementation_version: str
    configuration_id: str
    configuration_snapshot: PrimitiveMappingSnapshot

    def __post_init__(self) -> None:
        _validated_text(self.component_type, "component type")
        _validated_text(self.name, "component name")
        _validated_text(self.implementation_version, "component implementation version")
        _validated_hash(self.configuration_id, "component configuration ID")
        if (
            configuration_identity(self.configuration_snapshot.to_primitive())
            != self.configuration_id
        ):
            raise ValidationPlanError(
                "component configuration does not match its configuration ID"
            )

    @classmethod
    def capture(
        cls,
        component_type: str,
        name: str,
        implementation_version: str,
        configuration: PrimitiveMapping,
        *,
        configuration_id: str | None = None,
    ) -> "ConfigurationReference":
        snapshot = PrimitiveMappingSnapshot.capture(configuration)
        expected_id = configuration_identity(snapshot.to_primitive())
        if configuration_id is not None and configuration_id != expected_id:
            raise ValidationPlanError(
                "captured component configuration does not match its declared ID"
            )
        return cls(
            component_type,
            name,
            implementation_version,
            expected_id,
            snapshot,
        )

    @classmethod
    def capture_component(
        cls,
        component_type: str,
        component: ConfiguredComponent,
    ) -> "ConfigurationReference":
        """Capture an existing rule, strategy, outcome, evaluator, or policy."""
        return cls.capture(
            component_type,
            component.name,
            component.implementation_version,
            component.configuration(),
            configuration_id=component.configuration_id,
        )

    @classmethod
    def capture_aggregation_policy(
        cls,
        policy: AggregationPolicy,
    ) -> "ConfigurationReference":
        """Capture the exact typed aggregation policy recorded by QF-14."""
        if not isinstance(cast(object, policy), AggregationPolicy):
            raise ValidationPlanError(
                "aggregation provenance requires a typed aggregation policy"
            )
        return cls.capture(
            "aggregation_policy",
            policy.policy_name,
            policy.policy_version,
            policy.to_primitive(),
            configuration_id=policy.configuration_id,
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "component_type": self.component_type,
            "name": self.name,
            "implementation_version": self.implementation_version,
            "configuration_id": self.configuration_id,
            "configuration": self.configuration_snapshot.to_primitive(),
        }


@dataclass(frozen=True, slots=True, init=False)
class OutcomeProvenance:
    """One outcome configuration bound to its component's future reach."""

    configuration: ConfigurationReference
    future_horizon: TemporalOffset

    def __init__(self) -> None:
        raise TypeError(
            "OutcomeProvenance must be captured from a typed outcome component"
        )

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.configuration), ConfigurationReference):
            raise ValidationPlanError("outcome configuration reference is invalid")
        if not isinstance(cast(object, self.future_horizon), TemporalOffset):
            raise ValidationPlanError("outcome future horizon is invalid")

    @property
    def configuration_id(self) -> str:
        return self.configuration.configuration_id

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "configuration": self.configuration.to_primitive(),
            "future_horizon": self.future_horizon.to_primitive(),
        }

    @classmethod
    def capture_exchange_sessions(
        cls,
        outcome: SessionOutcomeComponent,
    ) -> "OutcomeProvenance":
        """Capture an existing QF-11-style session-indexed outcome labeler."""
        sessions = cast(object, outcome.required_future_sessions)
        if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 0:
            raise ValidationPlanError(
                "outcome required future sessions must be a non-negative integer"
            )
        return cls._capture_component(
            outcome,
            TemporalOffset.sessions(sessions),
        )

    @classmethod
    def capture_timestamp(
        cls,
        outcome: TimestampOutcomeComponent,
    ) -> "OutcomeProvenance":
        """Capture an outcome component with an exact elapsed future horizon."""
        duration = cast(object, outcome.required_future_duration)
        if not isinstance(duration, timedelta) or duration < timedelta(0):
            raise ValidationPlanError(
                "outcome required future duration must be a non-negative timedelta"
            )
        return cls._capture_component(outcome, TemporalOffset.duration(duration))

    @classmethod
    def _capture_component(
        cls,
        outcome: ConfiguredComponent,
        future_horizon: TemporalOffset,
    ) -> "OutcomeProvenance":
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "configuration",
            ConfigurationReference.capture_component("outcome_labeler", outcome),
        )
        object.__setattr__(instance, "future_horizon", future_horizon)
        instance.__post_init__()
        return instance


class IndicatorComponent(ConfiguredComponent, Protocol):
    """Configured indicator accepted by the provenance capture adapter."""

    @property
    def warm_up_observations(self) -> int: ...


class ResearchRuleComponent(ConfiguredComponent, Protocol):
    """Configured prediction rule or trading strategy with causal history needs."""

    @property
    def required_indicators(self) -> tuple[Indicator, ...]: ...

    @property
    def warm_up_observations(self) -> int: ...


@dataclass(frozen=True, slots=True, init=False)
class IndicatorProvenance:
    """Exact normalized indicator configuration and resolved backend provenance."""

    name: str
    implementation_version: str
    configuration_id: str
    configuration_snapshot: PrimitiveMappingSnapshot
    backend_identity: IndicatorBackendIdentity | None
    legacy_native_configuration: bool
    warm_up_observations: int
    source_timeframe: Timeframe

    def __init__(self) -> None:
        raise TypeError(
            "IndicatorProvenance must be captured from a typed indicator component"
        )

    def __post_init__(self) -> None:
        _validated_text(self.name, "indicator name")
        _validated_text(self.implementation_version, "indicator implementation version")
        _validated_hash(self.configuration_id, "indicator configuration ID")
        if (
            configuration_identity(self.configuration_snapshot.to_primitive())
            != self.configuration_id
        ):
            raise ValidationPlanError(
                "indicator configuration does not match its configuration ID"
            )
        if self.backend_identity is not None and not isinstance(
            cast(object, self.backend_identity), IndicatorBackendIdentity
        ):
            raise ValidationPlanError("indicator backend identity is invalid")
        if not isinstance(cast(object, self.legacy_native_configuration), bool):
            raise ValidationPlanError(
                "legacy-native indicator marker must be a boolean"
            )
        warm_up = cast(object, self.warm_up_observations)
        if isinstance(warm_up, bool) or not isinstance(warm_up, int) or warm_up < 1:
            raise ValidationPlanError(
                "indicator warm-up observations must be a positive integer"
            )
        if not isinstance(cast(object, self.source_timeframe), Timeframe):
            raise ValidationPlanError("indicator source timeframe is invalid")

    @property
    def required_context_observations(self) -> int:
        """Return preceding rows needed before a window's first study row."""
        return self.warm_up_observations - 1

    @classmethod
    def capture(
        cls,
        indicator: IndicatorComponent,
        source_timeframe: Timeframe,
    ) -> "IndicatorProvenance":
        if not isinstance(cast(object, source_timeframe), Timeframe):
            raise ValidationPlanError("captured indicator source timeframe is invalid")
        configuration = indicator.configuration()
        snapshot = PrimitiveMappingSnapshot.capture(configuration)
        configuration_id = indicator.configuration_id
        if configuration_identity(snapshot.to_primitive()) != configuration_id:
            raise ValidationPlanError(
                "captured indicator configuration does not match its declared ID"
            )
        backend = cast(
            IndicatorBackendIdentity | None,
            getattr(indicator, "backend_identity", None),
        )
        legacy = cast(
            bool,
            getattr(indicator, "uses_legacy_native_configuration", False),
        )
        warm_up = cast(object, indicator.warm_up_observations)
        if isinstance(warm_up, bool) or not isinstance(warm_up, int) or warm_up < 1:
            raise ValidationPlanError(
                "captured indicator warm-up observations must be a positive integer"
            )
        instance = object.__new__(cls)
        object.__setattr__(instance, "name", indicator.name)
        object.__setattr__(
            instance,
            "implementation_version",
            indicator.implementation_version,
        )
        object.__setattr__(instance, "configuration_id", configuration_id)
        object.__setattr__(instance, "configuration_snapshot", snapshot)
        object.__setattr__(instance, "backend_identity", backend)
        object.__setattr__(instance, "legacy_native_configuration", legacy)
        object.__setattr__(instance, "warm_up_observations", warm_up)
        object.__setattr__(instance, "source_timeframe", source_timeframe)
        instance.__post_init__()
        return instance

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "name": self.name,
            "implementation_version": self.implementation_version,
            "configuration_id": self.configuration_id,
            "configuration": self.configuration_snapshot.to_primitive(),
            "backend": (
                None
                if self.backend_identity is None
                else self.backend_identity.to_primitive()
            ),
            "legacy_native_configuration": self.legacy_native_configuration,
            "warm_up": {
                "observations_required_for_first_result": self.warm_up_observations,
                "required_pre_window_context": self.required_context_observations,
            },
            "source_timeframe": {
                "configuration_id": self.source_timeframe.configuration_id,
                "configuration": self.source_timeframe.to_primitive(),
            },
        }


@dataclass(frozen=True, slots=True)
class IndicatorTimeframeBinding:
    """One rule-required indicator configuration on one exact source timeframe."""

    timeframe_configuration_id: str
    indicator_configuration_id: str

    def __post_init__(self) -> None:
        _validated_hash(
            self.timeframe_configuration_id,
            "indicator binding timeframe configuration ID",
        )
        _validated_hash(
            self.indicator_configuration_id,
            "indicator binding indicator configuration ID",
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "timeframe_configuration_id": self.timeframe_configuration_id,
            "indicator_configuration_id": self.indicator_configuration_id,
        }


def _rule_context_provenance(
    configuration: PrimitiveMapping,
) -> tuple[str | None, tuple[IndicatorTimeframeBinding, ...]]:
    context_value = cast(object, configuration.get("context_requirements"))
    if context_value is None:
        return None, ()
    if not isinstance(context_value, dict):
        raise ValidationPlanError("research rule context requirements are invalid")
    context = cast(PrimitiveMapping, context_value)
    primary_value = cast(object, context.get("primary"))
    if not isinstance(primary_value, dict):
        raise ValidationPlanError(
            "research rule primary timeframe requirement is invalid"
        )
    primary = cast(PrimitiveMapping, primary_value)
    timeframe_value = cast(object, primary.get("timeframe"))
    if not isinstance(timeframe_value, dict):
        raise ValidationPlanError("research rule primary timeframe is invalid")
    timeframe = cast(PrimitiveMapping, timeframe_value)
    configuration_id = cast(object, timeframe.get("configuration_id"))
    primary_timeframe_id = _validated_hash(
        configuration_id,
        "research rule primary timeframe configuration ID",
    )
    contextual_value = cast(object, context.get("contextual"))
    if not isinstance(contextual_value, list):
        raise ValidationPlanError(
            "research rule contextual timeframe requirements are invalid"
        )
    requirements: tuple[object, ...] = (
        cast(object, primary_value),
        *cast(list[object], contextual_value),
    )
    bindings: set[tuple[str, str]] = set()
    for requirement_value in requirements:
        if not isinstance(requirement_value, dict):
            raise ValidationPlanError("research rule timeframe requirement is invalid")
        requirement = cast(PrimitiveMapping, requirement_value)
        requirement_timeframe_value = cast(object, requirement.get("timeframe"))
        indicators_value = cast(object, requirement.get("indicators"))
        if not isinstance(requirement_timeframe_value, dict) or not isinstance(
            indicators_value, list
        ):
            raise ValidationPlanError(
                "research rule timeframe indicator requirements are invalid"
            )
        requirement_timeframe = cast(PrimitiveMapping, requirement_timeframe_value)
        timeframe_id = _validated_hash(
            cast(object, requirement_timeframe.get("configuration_id")),
            "research rule indicator timeframe configuration ID",
        )
        for indicator_requirement_value in cast(list[object], indicators_value):
            if not isinstance(indicator_requirement_value, dict):
                raise ValidationPlanError(
                    "research rule indicator requirement is invalid"
                )
            indicator_requirement = cast(PrimitiveMapping, indicator_requirement_value)
            indicator_value = cast(object, indicator_requirement.get("indicator"))
            if not isinstance(indicator_value, dict):
                raise ValidationPlanError(
                    "research rule indicator configuration is invalid"
                )
            indicator = cast(PrimitiveMapping, indicator_value)
            indicator_id = _validated_hash(
                cast(object, indicator.get("configuration_id")),
                "research rule bound indicator configuration ID",
            )
            bindings.add((timeframe_id, indicator_id))
    return primary_timeframe_id, tuple(
        IndicatorTimeframeBinding(timeframe_id, indicator_id)
        for timeframe_id, indicator_id in sorted(bindings)
    )


@dataclass(frozen=True, slots=True, init=False)
class ResearchRuleProvenance:
    """Rule or strategy configuration bound to its causal history requirements."""

    configuration: ConfigurationReference
    warm_up_observations: int
    warm_up_timeframe_configuration_id: str | None
    required_indicator_configuration_ids: tuple[str, ...]
    required_indicator_bindings: tuple[IndicatorTimeframeBinding, ...]

    def __init__(self) -> None:
        raise TypeError(
            "ResearchRuleProvenance must be captured from a typed rule or strategy"
        )

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.configuration), ConfigurationReference):
            raise ValidationPlanError(
                "research rule configuration reference is invalid"
            )
        if self.configuration.component_type not in {
            "prediction_rule",
            "trading_strategy",
        }:
            raise ValidationPlanError("research rule semantic type is invalid")
        warm_up = cast(object, self.warm_up_observations)
        if isinstance(warm_up, bool) or not isinstance(warm_up, int) or warm_up < 1:
            raise ValidationPlanError(
                "research rule warm-up observations must be a positive integer"
            )
        if self.warm_up_timeframe_configuration_id is not None:
            _validated_hash(
                self.warm_up_timeframe_configuration_id,
                "research rule warm-up timeframe configuration ID",
            )
        indicator_ids_value = cast(object, self.required_indicator_configuration_ids)
        if not isinstance(indicator_ids_value, tuple):
            raise ValidationPlanError(
                "research rule required indicator configuration IDs are invalid"
            )
        untyped_indicator_ids = cast(tuple[object, ...], indicator_ids_value)
        if any(not isinstance(item, str) for item in untyped_indicator_ids):
            raise ValidationPlanError(
                "research rule required indicator configuration IDs are invalid"
            )
        indicator_ids = cast(tuple[str, ...], indicator_ids_value)
        for configuration_id in indicator_ids:
            _validated_hash(
                configuration_id,
                "research rule required indicator configuration ID",
            )
        if tuple(sorted(set(indicator_ids))) != indicator_ids:
            raise ValidationPlanError(
                "research rule required indicator configuration IDs must be unique "
                "and ordered"
            )
        bindings_value = cast(object, self.required_indicator_bindings)
        if not isinstance(bindings_value, tuple) or any(
            not isinstance(item, IndicatorTimeframeBinding)
            for item in cast(tuple[object, ...], bindings_value)
        ):
            raise ValidationPlanError(
                "research rule required indicator bindings are invalid"
            )
        bindings = cast(tuple[IndicatorTimeframeBinding, ...], bindings_value)
        ordered_bindings = tuple(
            sorted(
                set(bindings),
                key=lambda item: (
                    item.timeframe_configuration_id,
                    item.indicator_configuration_id,
                ),
            )
        )
        if bindings != ordered_bindings:
            raise ValidationPlanError(
                "research rule required indicator bindings must be unique and ordered"
            )
        if self.warm_up_timeframe_configuration_id is not None and {
            item.indicator_configuration_id for item in bindings
        } != set(indicator_ids):
            raise ValidationPlanError(
                "research rule context indicator bindings must exactly match its "
                "required indicators"
            )

    @property
    def component_type(self) -> str:
        return self.configuration.component_type

    @property
    def configuration_id(self) -> str:
        return self.configuration.configuration_id

    @property
    def implementation_version(self) -> str:
        return self.configuration.implementation_version

    @property
    def required_context_observations(self) -> int:
        """Return preceding rows needed before a window's first study row."""
        return self.warm_up_observations - 1

    @classmethod
    def capture_prediction(
        cls,
        rule: ResearchRuleComponent,
    ) -> "ResearchRuleProvenance":
        """Capture a component that identifies itself as a prediction strategy."""
        return cls._capture("prediction_rule", "prediction_strategy", rule)

    @classmethod
    def capture_trading(
        cls,
        strategy: ResearchRuleComponent,
    ) -> "ResearchRuleProvenance":
        """Capture a component that identifies itself as a trading strategy."""
        return cls._capture("trading_strategy", "strategy", strategy)

    @classmethod
    def _capture(
        cls,
        component_type: str,
        configured_component_type: str,
        rule: ResearchRuleComponent,
    ) -> "ResearchRuleProvenance":
        configuration = rule.configuration()
        if configuration.get("component_type") != configured_component_type:
            raise ValidationPlanError(
                f"{component_type} provenance requires a component whose canonical "
                f"configuration type is {configured_component_type!r}"
            )
        required_indicators_value = cast(object, rule.required_indicators)
        if not isinstance(required_indicators_value, tuple):
            raise ValidationPlanError(
                "research rule required indicators must be an immutable tuple"
            )
        required_indicators = cast(tuple[Indicator, ...], required_indicators_value)
        required_ids_values: list[str] = []
        for indicator in required_indicators:
            indicator_configuration = indicator.configuration()
            indicator_id = indicator.configuration_id
            if configuration_identity(indicator_configuration) != indicator_id:
                raise ValidationPlanError(
                    "research rule required indicator configuration identity is invalid"
                )
            indicator_warm_up = cast(object, indicator.warm_up_observations)
            if (
                isinstance(indicator_warm_up, bool)
                or not isinstance(indicator_warm_up, int)
                or indicator_warm_up < 1
            ):
                raise ValidationPlanError(
                    "research rule required indicator warm-up is invalid"
                )
            required_ids_values.append(indicator_id)
        required_ids = tuple(sorted(required_ids_values))
        if len(set(required_ids)) != len(required_ids):
            raise ValidationPlanError(
                "research rule required indicators must be unique"
            )
        warm_up = cast(object, rule.warm_up_observations)
        if isinstance(warm_up, bool) or not isinstance(warm_up, int) or warm_up < 1:
            raise ValidationPlanError(
                "captured research rule warm-up observations must be a positive integer"
            )
        primary_timeframe_id, required_bindings = _rule_context_provenance(
            configuration
        )
        if primary_timeframe_id is not None and {
            item.indicator_configuration_id for item in required_bindings
        } != set(required_ids):
            raise ValidationPlanError(
                "research rule context indicator bindings must exactly match its "
                "required indicators"
            )
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "configuration",
            ConfigurationReference.capture(
                component_type,
                rule.name,
                rule.implementation_version,
                configuration,
                configuration_id=rule.configuration_id,
            ),
        )
        object.__setattr__(instance, "warm_up_observations", warm_up)
        object.__setattr__(
            instance,
            "warm_up_timeframe_configuration_id",
            primary_timeframe_id,
        )
        object.__setattr__(
            instance,
            "required_indicator_configuration_ids",
            required_ids,
        )
        object.__setattr__(instance, "required_indicator_bindings", required_bindings)
        instance.__post_init__()
        return instance

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "configuration": self.configuration.to_primitive(),
            "warm_up": {
                "observations_required_for_first_result": self.warm_up_observations,
                "required_pre_window_context": self.required_context_observations,
                "source_timeframe_configuration_id": (
                    self.warm_up_timeframe_configuration_id
                ),
            },
            "required_indicator_configuration_ids": list(
                self.required_indicator_configuration_ids
            ),
            "required_indicator_bindings": [
                item.to_primitive() for item in self.required_indicator_bindings
            ],
        }


@dataclass(frozen=True, slots=True, init=False)
class BacktestProvenance:
    """Complete typed execution, cost, sizing, and accounting provenance."""

    configuration: ConfigurationReference

    def __init__(self) -> None:
        raise TypeError(
            "BacktestProvenance must be captured from a typed backtest configuration"
        )

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.configuration), ConfigurationReference):
            raise ValidationPlanError("backtest configuration reference is invalid")
        if self.configuration.component_type != "backtest_configuration":
            raise ValidationPlanError("backtest provenance semantic type is invalid")

    @property
    def configuration_id(self) -> str:
        return self.configuration.configuration_id

    @classmethod
    def capture(
        cls,
        configuration: "BacktestConfig",
    ) -> "BacktestProvenance":
        from quantforge.backtesting import BacktestConfig

        if type(configuration) is not BacktestConfig:
            raise ValidationPlanError(
                "backtest provenance requires an existing validated BacktestConfig"
            )
        engine_version = _validated_text(
            cast(object, configuration.engine_version),
            "backtest engine version",
        )
        result_schema_version = _validated_text(
            cast(object, configuration.result_schema_version),
            "backtest result schema version",
        )
        primitive = configuration.to_primitive()
        if primitive.get("engine_version") != engine_version:
            raise ValidationPlanError(
                "backtest configuration engine version does not match its provenance"
            )
        if primitive.get("result_schema_version") != result_schema_version:
            raise ValidationPlanError(
                "backtest result schema version does not match its provenance"
            )
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "configuration",
            ConfigurationReference.capture(
                "backtest_configuration",
                "quantforge_backtest",
                engine_version,
                primitive,
            ),
        )
        instance.__post_init__()
        return instance

    def to_primitive(self) -> PrimitiveMapping:
        return self.configuration.to_primitive()


@dataclass(frozen=True, slots=True, init=False)
class DatasetProvenance:
    """Fixed dataset fingerprint plus optional verified QF-14 family bindings."""

    dataset_fingerprint: str
    dataset_ids: tuple[str, ...]
    family_references: tuple[DatasetFamilyReference, ...] = ()
    dataset_family: DatasetFamily | None = None
    standalone_timeframe: Timeframe | None = None
    market_data_metadata: DatasetMetadata | None = None

    def __init__(self) -> None:
        raise TypeError(
            "DatasetProvenance must be captured from a MarketDataset or DatasetFamily"
        )

    def __post_init__(self) -> None:
        _validated_hash(self.dataset_fingerprint, "dataset fingerprint")
        if not self.dataset_ids or any(not value for value in self.dataset_ids):
            raise ValidationPlanError("dataset provenance requires dataset IDs")
        ordered_ids = tuple(sorted(set(self.dataset_ids)))
        if len(ordered_ids) != len(self.dataset_ids):
            raise ValidationPlanError("dataset provenance IDs must be unique")
        if any(
            not isinstance(item, DatasetFamilyReference)
            for item in cast(tuple[object, ...], self.family_references)
        ):
            raise ValidationPlanError("dataset family references are invalid")
        references = tuple(
            sorted(self.family_references, key=lambda item: item.dataset_id)
        )
        if len({item.dataset_id for item in references}) != len(references):
            raise ValidationPlanError("dataset family references must be unique")
        if references:
            if self.market_data_metadata is not None:
                raise ValidationPlanError(
                    "family-backed provenance cannot define standalone market metadata"
                )
            if self.standalone_timeframe is not None:
                raise ValidationPlanError(
                    "family-backed provenance cannot define a standalone timeframe"
                )
            family = cast(object, self.dataset_family)
            if not isinstance(family, DatasetFamily):
                raise ValidationPlanError(
                    "dataset family references require the complete dataset family"
                )
            family_ids = {item.family_id for item in references}
            source_ids = {item.canonical_source_snapshot_id for item in references}
            if len(family_ids) != 1 or len(source_ids) != 1:
                raise ValidationPlanError(
                    "dataset provenance cannot mix family or canonical source IDs"
                )
            if set(ordered_ids) != {item.dataset_id for item in references}:
                raise ValidationPlanError(
                    "dataset IDs must exactly match dataset family references"
                )
            try:
                expected_references = tuple(
                    family.reference(dataset_id) for dataset_id in ordered_ids
                )
            except ValueError as error:
                raise ValidationPlanError(
                    "dataset IDs must be recorded in the supplied dataset family"
                ) from error
            if references != expected_references:
                raise ValidationPlanError(
                    "dataset family references must match the supplied family manifest"
                )
        elif self.dataset_family is not None:
            raise ValidationPlanError(
                "dataset family cannot be supplied without family references"
            )
        elif not isinstance(cast(object, self.standalone_timeframe), Timeframe):
            raise ValidationPlanError(
                "standalone dataset provenance requires its canonical timeframe"
            )
        elif (
            self.market_data_metadata is None
            or ordered_ids != (self.market_data_metadata.dataset_id,)
            or self.dataset_fingerprint != self.market_data_metadata.data_sha256
        ):
            raise ValidationPlanError(
                "standalone dataset provenance requires its validated market metadata"
            )
        object.__setattr__(self, "dataset_ids", ordered_ids)
        object.__setattr__(self, "family_references", references)

    @property
    def family_manifest_id(self) -> str | None:
        """Return the verified exact-family manifest identity, when applicable."""
        if self.dataset_family is None:
            return None
        return self.dataset_family.manifest_id

    @classmethod
    def from_market_dataset(cls, dataset: MarketDataset) -> "DatasetProvenance":
        """Capture one validated QF-3 dataset without retrofitting QF-14 identity."""
        validate_market_dataset(dataset)
        try:
            timezone_name = resolve_exchange_timezone_name(dataset.metadata.calendar)
            timeframe = Timeframe(
                SessionInterval(),
                ExchangeSessionPolicy(
                    calendar_name=dataset.metadata.calendar,
                    timezone_name=timezone_name,
                ),
            )
        except TimeframeValidationError as error:
            raise ValidationPlanError(
                "standalone dataset metadata cannot define a canonical timeframe"
            ) from error
        return cls._create(
            dataset.metadata.data_sha256,
            (dataset.metadata.dataset_id,),
            standalone_timeframe=timeframe,
            market_data_metadata=dataset.metadata,
        )

    @classmethod
    def from_dataset_family(
        cls,
        family: DatasetFamily,
        dataset_ids: tuple[str, ...],
    ) -> "DatasetProvenance":
        """Capture selected artifact identity from an immutable family manifest."""
        try:
            references = tuple(
                family.reference(dataset_id) for dataset_id in dataset_ids
            )
        except ValueError as error:
            raise ValidationPlanError(
                "dataset IDs must be recorded in the supplied dataset family"
            ) from error
        ordered_references = tuple(
            sorted(references, key=lambda reference: reference.dataset_id)
        )
        dataset_fingerprint = configuration_identity(
            {
                "dataset_family_manifest_id": family.manifest_id,
                "selected_dataset_references": [
                    reference.to_primitive(include_feed_scope=True)
                    for reference in ordered_references
                ],
            }
        )
        return cls._create(
            dataset_fingerprint,
            dataset_ids,
            references,
            family,
        )

    @classmethod
    def _create(
        cls,
        dataset_fingerprint: str,
        dataset_ids: tuple[str, ...],
        family_references: tuple[DatasetFamilyReference, ...] = (),
        dataset_family: DatasetFamily | None = None,
        standalone_timeframe: Timeframe | None = None,
        market_data_metadata: DatasetMetadata | None = None,
    ) -> "DatasetProvenance":
        instance = object.__new__(cls)
        object.__setattr__(instance, "dataset_fingerprint", dataset_fingerprint)
        object.__setattr__(instance, "dataset_ids", dataset_ids)
        object.__setattr__(instance, "family_references", family_references)
        object.__setattr__(instance, "dataset_family", dataset_family)
        object.__setattr__(instance, "standalone_timeframe", standalone_timeframe)
        object.__setattr__(instance, "market_data_metadata", market_data_metadata)
        instance.__post_init__()
        return instance

    def to_primitive(self) -> PrimitiveMapping:
        references = cast(
            list[Primitive],
            [
                item.to_primitive(include_feed_scope=True)
                for item in self.family_references
            ],
        )
        family: PrimitiveMapping | None = None
        if references:
            assert self.dataset_family is not None
            family = {
                "family_id": self.family_references[0].family_id,
                "manifest_id": self.family_manifest_id,
                "canonical_source_snapshot_id": (
                    self.family_references[0].canonical_source_snapshot_id
                ),
                "references": references,
                "manifest": self.dataset_family.to_manifest(),
            }
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_ids": list(self.dataset_ids),
            "dataset_family": family,
            "market_data_metadata": (
                None
                if self.market_data_metadata is None
                else cast(
                    PrimitiveMapping,
                    serialize_metadata_values(asdict(self.market_data_metadata)),
                )
            ),
            "standalone_timeframe": (
                None
                if self.standalone_timeframe is None
                else {
                    "configuration_id": self.standalone_timeframe.configuration_id,
                    "configuration": self.standalone_timeframe.to_primitive(),
                }
            ),
        }


def _validate_context_feed_scopes(
    configuration: PrimitiveMapping,
    references: tuple[DatasetFamilyReference, ...],
) -> None:
    """Reconcile every captured context requirement with selected lineage."""
    context = cast(PrimitiveMapping | None, configuration.get("context_requirements"))
    if context is None:
        return
    # Rule capture already validates this shape, including indicator-free inputs.
    requirements = (
        cast(PrimitiveMapping, context["primary"]),
        *cast(list[PrimitiveMapping], context["contextual"]),
    )
    for requirement in requirements:
        timeframe = cast(PrimitiveMapping, requirement["timeframe"])
        selected_references = tuple(
            reference
            for reference in references
            if reference.timeframe_configuration_id == timeframe["configuration_id"]
        )
        if not selected_references or any(
            reference.feed_scope.to_primitive() != requirement.get("feed_scope")
            for reference in selected_references
        ):
            raise ValidationPlanError(
                "research rule context feed scope must match selected dataset "
                f"family provenance for timeframe: {timeframe['configuration_id']}"
            )


class ResearchStudyType(StrEnum):
    """Validation consumers that share partitions without sharing result models."""

    PREDICTION = "prediction"
    TRADING_BACKTEST = "trading_backtest"


@dataclass(frozen=True, slots=True)
class ResearchEnvironment:
    """One fixed scientific environment shared by every plan window."""

    study_type: ResearchStudyType
    dataset: DatasetProvenance
    timeframes: tuple[Timeframe, ...]
    research_rule: ResearchRuleProvenance
    aggregation_policies: tuple[ConfigurationReference, ...] = ()
    indicators: tuple[IndicatorProvenance, ...] = ()
    outcomes: tuple[OutcomeProvenance, ...] = ()
    execution: BacktestProvenance | None = None
    schema_version: str = RESEARCH_ENVIRONMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.study_type), ResearchStudyType):
            raise ValidationPlanError("research study type is invalid")
        if not isinstance(cast(object, self.dataset), DatasetProvenance):
            raise ValidationPlanError("research dataset provenance is invalid")
        if not isinstance(cast(object, self.research_rule), ResearchRuleProvenance):
            raise ValidationPlanError("research rule provenance is invalid")
        expected_rule_type = (
            "prediction_rule"
            if self.study_type is ResearchStudyType.PREDICTION
            else "trading_strategy"
        )
        if self.research_rule.component_type != expected_rule_type:
            raise ValidationPlanError(
                f"{self.study_type.value} research requires a "
                f"{expected_rule_type} configuration reference"
            )
        if self.execution is not None and not isinstance(
            cast(object, self.execution), BacktestProvenance
        ):
            raise ValidationPlanError(
                "research execution provenance must capture a complete backtest "
                "configuration"
            )
        if (
            self.study_type is ResearchStudyType.TRADING_BACKTEST
            and self.execution is None
        ):
            raise ValidationPlanError(
                "trading/backtest research requires execution provenance"
            )
        if (
            self.study_type is ResearchStudyType.TRADING_BACKTEST
            and self.dataset.market_data_metadata is not None
        ):
            from quantforge.backtesting.config import DividendPolicy
            from quantforge.backtesting.errors import InvalidMarketDataError
            from quantforge.backtesting.validation import (
                validate_backtest_dataset_metadata,
            )

            assert self.execution is not None
            execution = (
                self.execution.configuration.configuration_snapshot.to_primitive()
            )
            try:
                validate_backtest_dataset_metadata(
                    self.dataset.market_data_metadata,
                    dividend_policy=DividendPolicy(
                        cast(str, execution["dividend_policy"])
                    ),
                )
            except InvalidMarketDataError as error:
                raise ValidationPlanError(
                    f"dataset is incompatible with trading execution: {error}"
                ) from error
        if not self.timeframes or any(
            not isinstance(item, Timeframe)
            for item in cast(tuple[object, ...], self.timeframes)
        ):
            raise ValidationPlanError(
                "research environment requires configured timeframes"
            )
        ordered_timeframes = tuple(
            sorted(self.timeframes, key=lambda item: item.configuration_id)
        )
        if len({item.configuration_id for item in ordered_timeframes}) != len(
            ordered_timeframes
        ):
            raise ValidationPlanError("research timeframes must be unique")
        configured_timeframe_ids = {
            item.configuration_id for item in ordered_timeframes
        }
        if self.dataset.family_references:
            available_timeframe_ids = {
                item.timeframe_configuration_id
                for item in self.dataset.family_references
            }
        else:
            assert self.dataset.standalone_timeframe is not None
            available_timeframe_ids = {
                self.dataset.standalone_timeframe.configuration_id
            }
        if available_timeframe_ids != configured_timeframe_ids:
            raise ValidationPlanError(
                "configured research timeframes must exactly match dataset provenance"
            )
        _validate_context_feed_scopes(
            self.research_rule.configuration.configuration_snapshot.to_primitive(),
            self.dataset.family_references,
        )
        ordered_aggregations = _ordered_unique_references(
            self.aggregation_policies, "aggregation policies"
        )
        expected_aggregations: tuple[ConfigurationReference, ...] = ()
        if self.dataset.dataset_family is not None:
            selected_ids = set(self.dataset.dataset_ids)
            if any(
                item.dataset_id in selected_ids and not item.is_canonical_source
                for item in self.dataset.dataset_family.datasets
            ):
                expected_aggregations = (
                    ConfigurationReference.capture_aggregation_policy(
                        self.dataset.dataset_family.aggregation_policy
                    ),
                )
        if ordered_aggregations != expected_aggregations:
            raise ValidationPlanError(
                "research aggregation provenance must exactly match the selected "
                "dataset lineage"
            )
        ordered_indicators = tuple(
            sorted(
                self.indicators,
                key=lambda item: (
                    item.source_timeframe.configuration_id,
                    item.configuration_id,
                ),
            )
        )
        indicator_bindings = {
            (item.source_timeframe.configuration_id, item.configuration_id)
            for item in ordered_indicators
        }
        if len(indicator_bindings) != len(ordered_indicators):
            raise ValidationPlanError("research indicators must be unique")
        if any(
            item.source_timeframe.configuration_id not in configured_timeframe_ids
            for item in ordered_indicators
        ):
            raise ValidationPlanError(
                "indicator source timeframes must be configured in the research "
                "environment"
            )
        required_bindings = {
            (
                item.timeframe_configuration_id,
                item.indicator_configuration_id,
            )
            for item in self.research_rule.required_indicator_bindings
        }
        configured_indicator_ids = {item[1] for item in indicator_bindings}
        missing_rule_indicators = (
            required_bindings.difference(indicator_bindings)
            if required_bindings
            else set(
                self.research_rule.required_indicator_configuration_ids
            ).difference(configured_indicator_ids)
        )
        if missing_rule_indicators:
            raise ValidationPlanError(
                "research environment must include every source-timeframe indicator "
                "binding required by its rule or strategy"
            )
        rule_timeframe_id = self.research_rule.warm_up_timeframe_configuration_id
        if rule_timeframe_id is None:
            if len(ordered_timeframes) != 1:
                raise ValidationPlanError(
                    "multi-timeframe research rules must identify the source "
                    "timeframe for their warm-up"
                )
        elif rule_timeframe_id not in configured_timeframe_ids:
            raise ValidationPlanError(
                "research rule warm-up timeframe must be configured in the "
                "research environment"
            )
        if any(
            not isinstance(item, OutcomeProvenance)
            for item in cast(tuple[object, ...], self.outcomes)
        ):
            raise ValidationPlanError("research outcomes are invalid")
        ordered_outcomes = tuple(
            sorted(self.outcomes, key=lambda item: item.configuration_id)
        )
        if len({item.configuration_id for item in ordered_outcomes}) != len(
            ordered_outcomes
        ):
            raise ValidationPlanError("research outcomes must be unique")
        if self.study_type is ResearchStudyType.PREDICTION and not ordered_outcomes:
            raise ValidationPlanError(
                "prediction research requires outcome configuration provenance"
            )
        if self.schema_version != RESEARCH_ENVIRONMENT_SCHEMA_VERSION:
            raise ValidationPlanError(
                "research environment schema version is unsupported"
            )
        object.__setattr__(self, "timeframes", ordered_timeframes)
        object.__setattr__(self, "aggregation_policies", ordered_aggregations)
        object.__setattr__(self, "indicators", ordered_indicators)
        object.__setattr__(self, "outcomes", ordered_outcomes)

    def _identity_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "study_type": self.study_type.value,
            "dataset": self.dataset.to_primitive(),
            "timeframes": [
                {
                    "configuration_id": item.configuration_id,
                    "configuration": item.to_primitive(),
                }
                for item in self.timeframes
            ],
            "aggregation_policies": [
                item.to_primitive() for item in self.aggregation_policies
            ],
            "indicators": [item.to_primitive() for item in self.indicators],
            "research_rule": self.research_rule.to_primitive(),
            "outcomes": [item.to_primitive() for item in self.outcomes],
            "execution": (
                None if self.execution is None else self.execution.to_primitive()
            ),
        }

    @property
    def environment_id(self) -> str:
        return configuration_identity(self._identity_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {"environment_id": self.environment_id, **self._identity_primitive()}


def _ordered_unique_references(
    references: tuple[ConfigurationReference, ...], label: str
) -> tuple[ConfigurationReference, ...]:
    if any(
        not isinstance(item, ConfigurationReference)
        for item in cast(tuple[object, ...], references)
    ):
        raise ValidationPlanError(f"research {label} are invalid")
    ordered = tuple(sorted(references, key=lambda item: item.configuration_id))
    if len({item.configuration_id for item in ordered}) != len(ordered):
        raise ValidationPlanError(f"research {label} must be unique")
    return ordered


@dataclass(frozen=True, slots=True)
class ValidationPlan:
    """A deterministic set of explicit folds and one untouched final holdout."""

    name: str
    environment: ResearchEnvironment
    folds: tuple[ValidationFold, ...]
    final_holdout: FinalHoldout
    purge_policy: PurgePolicy
    training_window_mode: TrainingWindowMode
    schema_version: str = VALIDATION_PLAN_SCHEMA_VERSION
    _axis: BoundaryAxis = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validated_text(self.name, "validation plan name")
        if not self.folds or any(
            not isinstance(item, ValidationFold)
            for item in cast(tuple[object, ...], self.folds)
        ):
            raise ValidationPlanError("validation plan requires explicit folds")
        if not isinstance(cast(object, self.environment), ResearchEnvironment):
            raise ValidationPlanError("validation plan environment is invalid")
        if not isinstance(cast(object, self.final_holdout), FinalHoldout):
            raise ValidationPlanError("validation plan final holdout is invalid")
        if not isinstance(cast(object, self.purge_policy), PurgePolicy):
            raise ValidationPlanError("validation plan purge policy is invalid")
        if not isinstance(cast(object, self.training_window_mode), TrainingWindowMode):
            raise ValidationPlanError("validation training-window mode is invalid")
        if self.schema_version != VALIDATION_PLAN_SCHEMA_VERSION:
            raise ValidationPlanError("validation plan schema version is unsupported")
        axis = self.folds[0].development.interval.axis
        windows = tuple(
            window
            for fold in self.folds
            for window in (
                fold.development,
                *((fold.selection,) if fold.selection is not None else ()),
                fold.test,
            )
        )
        if any(window.interval.axis is not axis for window in windows) or (
            self.final_holdout.window.interval.axis is not axis
        ):
            raise ValidationPlanError("validation plan cannot mix temporal axes")
        if self.purge_policy.axis is not axis:
            raise ValidationPlanError(
                "validation purge policy does not match the plan temporal axis"
            )
        if axis is BoundaryAxis.EXCHANGE_SESSION and any(
            isinstance(timeframe.interval, IntradayInterval)
            for timeframe in self.environment.timeframes
        ):
            raise ValidationPlanError(
                "intraday source observations require timestamp validation boundaries"
            )
        self._validate_research_warm_up((*windows, self.final_holdout.window))
        self._validate_outcome_horizon(axis)
        if axis is BoundaryAxis.EXCHANGE_SESSION:
            first = cast(ExchangeSessionBoundary, windows[0].interval.start)
            policy = first.session_policy
            for window in (*windows, self.final_holdout.window):
                start = cast(ExchangeSessionBoundary, window.interval.start)
                if start.session_policy != policy:
                    raise ValidationPlanError(
                        "validation plan cannot mix exchange-session policies"
                    )
            if any(
                item.session_policy != policy for item in self.environment.timeframes
            ):
                raise ValidationPlanError(
                    "validation boundaries and research timeframes must share one "
                    "exchange-session policy"
                )
        self._validate_fold_progression()
        if any(
            not window.interval.precedes(self.final_holdout.window.interval)
            for window in windows
        ):
            raise ValidationPlanError(
                "final holdout must follow and not overlap every research window"
            )
        object.__setattr__(self, "_axis", axis)

    def _validate_research_warm_up(
        self,
        windows: tuple[ValidationWindow, ...],
    ) -> None:
        required_context = {
            timeframe.configuration_id: 0 for timeframe in self.environment.timeframes
        }
        for indicator in self.environment.indicators:
            timeframe_id = indicator.source_timeframe.configuration_id
            required_context[timeframe_id] = max(
                required_context[timeframe_id],
                indicator.required_context_observations,
            )
        rule_timeframe_id = (
            self.environment.research_rule.warm_up_timeframe_configuration_id
        )
        if rule_timeframe_id is None:
            assert len(self.environment.timeframes) == 1
            rule_timeframe_id = self.environment.timeframes[0].configuration_id
        required_context[rule_timeframe_id] = max(
            required_context[rule_timeframe_id],
            self.environment.research_rule.required_context_observations,
        )
        for window in windows:
            if len(required_context) == 1 and not window.warm_up_by_timeframe:
                timeframe_id, required = next(iter(required_context.items()))
                provided = window.warm_up_observations
                if provided < required:
                    raise ValidationPlanError(
                        f"validation window {window.name!r} requires at least "
                        f"{required} warm-up observations for timeframe "
                        f"{timeframe_id}"
                    )
                continue
            provided_by_timeframe = {
                item.timeframe.configuration_id: item.observations
                for item in window.warm_up_by_timeframe
            }
            if set(provided_by_timeframe) != set(required_context):
                raise ValidationPlanError(
                    f"validation window {window.name!r} must define warm-up for "
                    "every configured source timeframe"
                )
            for timeframe_id, required in required_context.items():
                if provided_by_timeframe[timeframe_id] < required:
                    raise ValidationPlanError(
                        f"validation window {window.name!r} requires at least "
                        f"{required} warm-up observations for timeframe "
                        f"{timeframe_id}"
                    )

    def _validate_outcome_horizon(self, axis: BoundaryAxis) -> None:
        outcomes = self.environment.outcomes
        if not outcomes:
            if self.purge_policy.label_horizon.magnitude != 0:
                raise ValidationPlanError(
                    "a plan without outcomes must use a zero label horizon"
                )
            return
        if any(item.future_horizon.axis is not axis for item in outcomes):
            raise ValidationPlanError(
                "outcome horizons must use the validation plan temporal axis"
            )
        maximum = max(item.future_horizon.magnitude for item in outcomes)
        if self.purge_policy.label_horizon.magnitude != maximum:
            raise ValidationPlanError(
                "purge label horizon must equal the maximum configured outcome horizon"
            )

    @property
    def axis(self) -> BoundaryAxis:
        return self._axis

    def _validate_fold_progression(self) -> None:
        for previous, current in pairwise(self.folds):
            if not previous.test.interval.precedes(current.test.interval):
                raise ValidationPlanError(
                    "walk-forward test intervals must be chronological and disjoint"
                )
            previous_start = boundary_value(previous.development.interval.start)
            current_start = boundary_value(current.development.interval.start)
            previous_end = boundary_value(previous.development.interval.end)
            current_end = boundary_value(current.development.interval.end)
            if self.training_window_mode is TrainingWindowMode.EXPANDING:
                if current_start != previous_start or current_end <= previous_end:
                    raise ValidationPlanError(
                        "expanding development windows must keep their start and "
                        "advance their end"
                    )
            elif current_start <= previous_start or current_end <= previous_end:
                raise ValidationPlanError(
                    "rolling development windows must advance both start and end"
                )

    def _identity_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "training_window_mode": self.training_window_mode.value,
            "environment": self.environment.to_primitive(),
            "purge_policy": self.purge_policy.to_primitive(),
            "folds": [fold.to_primitive() for fold in self.folds],
            "final_holdout": self.final_holdout.to_primitive(),
            "execution_scope": (
                "partition_definition_only; walk_forward_execution_and_holdout_"
                "consumption_are_outside_qf8"
            ),
        }

    @property
    def plan_id(self) -> str:
        return configuration_identity(self._identity_primitive())

    def to_manifest(self) -> PrimitiveMapping:
        return {"plan_id": self.plan_id, **self._identity_primitive()}
