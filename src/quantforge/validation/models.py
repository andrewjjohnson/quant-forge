"""Immutable, study-neutral validation-plan contracts."""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from typing import Protocol, cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import (
    DatasetFamily,
    DatasetFamilyReference,
    MarketDataset,
    validate_market_dataset,
)
from quantforge.indicators import IndicatorBackendIdentity
from quantforge.timeframes import (
    DEFAULT_US_EQUITY_SESSION_POLICY,
    ExchangeSessionPolicy,
    Timeframe,
    resolve_exchange_session,
)
from quantforge.validation.errors import ValidationPlanError

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
class ValidationWindow:
    """One identity-bearing interval and its non-selecting warm-up requirement."""

    name: str
    role: PartitionRole
    interval: ValidationInterval
    warm_up_observations: int = 0
    schema_version: str = VALIDATION_WINDOW_SCHEMA_VERSION

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
        if self.schema_version != VALIDATION_WINDOW_SCHEMA_VERSION:
            raise ValidationPlanError("validation window schema version is unsupported")

    def _identity_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "role": self.role.value,
            "interval": self.interval.to_primitive(),
            "warm_up": {
                "observations": self.warm_up_observations,
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

    @property
    def required_context_observations(self) -> int:
        """Return preceding rows needed before a window's first study row."""
        return self.warm_up_observations - 1

    @classmethod
    def capture(cls, indicator: IndicatorComponent) -> "IndicatorProvenance":
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
        }


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Fixed dataset fingerprint plus optional verified QF-14 family bindings."""

    dataset_fingerprint: str
    dataset_ids: tuple[str, ...]
    family_references: tuple[DatasetFamilyReference, ...] = ()
    dataset_family: DatasetFamily | None = None

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
        return cls(
            dataset.metadata.data_sha256,
            (dataset.metadata.dataset_id,),
        )

    @classmethod
    def from_dataset_family(
        cls,
        dataset_fingerprint: str,
        family: DatasetFamily,
        dataset_ids: tuple[str, ...],
    ) -> "DatasetProvenance":
        """Capture compact references plus the exact immutable family manifest."""
        return cls(
            dataset_fingerprint,
            dataset_ids,
            tuple(family.reference(dataset_id) for dataset_id in dataset_ids),
            family,
        )

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
        }


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
    research_rule: ConfigurationReference
    aggregation_policies: tuple[ConfigurationReference, ...] = ()
    indicators: tuple[IndicatorProvenance, ...] = ()
    outcomes: tuple[OutcomeProvenance, ...] = ()
    execution: ConfigurationReference | None = None
    schema_version: str = RESEARCH_ENVIRONMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.study_type), ResearchStudyType):
            raise ValidationPlanError("research study type is invalid")
        if not isinstance(cast(object, self.dataset), DatasetProvenance):
            raise ValidationPlanError("research dataset provenance is invalid")
        if not isinstance(cast(object, self.research_rule), ConfigurationReference):
            raise ValidationPlanError("research rule reference is invalid")
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
            cast(object, self.execution), ConfigurationReference
        ):
            raise ValidationPlanError("research execution reference is invalid")
        if (
            self.study_type is ResearchStudyType.TRADING_BACKTEST
            and self.execution is None
        ):
            raise ValidationPlanError(
                "trading/backtest research requires execution provenance"
            )
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
        referenced_timeframe_ids = {
            item.timeframe_configuration_id for item in self.dataset.family_references
        }
        if not referenced_timeframe_ids.issubset(configured_timeframe_ids):
            raise ValidationPlanError(
                "dataset family references must match configured research timeframes"
            )
        ordered_aggregations = _ordered_unique_references(
            self.aggregation_policies, "aggregation policies"
        )
        ordered_indicators = tuple(
            sorted(self.indicators, key=lambda item: item.configuration_id)
        )
        if len({item.configuration_id for item in ordered_indicators}) != len(
            ordered_indicators
        ):
            raise ValidationPlanError("research indicators must be unique")
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
        self._validate_indicator_warm_up((*windows, self.final_holdout.window))
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

    def _validate_indicator_warm_up(
        self,
        windows: tuple[ValidationWindow, ...],
    ) -> None:
        required_context = max(
            (
                indicator.required_context_observations
                for indicator in self.environment.indicators
            ),
            default=0,
        )
        for window in windows:
            if window.warm_up_observations < required_context:
                raise ValidationPlanError(
                    f"validation window {window.name!r} requires at least "
                    f"{required_context} indicator warm-up observations"
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
