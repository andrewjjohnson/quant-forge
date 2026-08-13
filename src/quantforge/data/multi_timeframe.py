"""Leakage-safe as-of alignment across compatible dataset timeframes."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data.developing_bars import (
    DevelopingBar,
    DevelopingBarValidationError,
    reconstruct_developing_bar_as_of,
)
from quantforge.data.exceptions import CacheError, ValidationError
from quantforge.data.identity import canonical_json_bytes
from quantforge.data.intraday import IntradayBar
from quantforge.data.intraday_aggregation import AggregatedIntradayDataset
from quantforge.data.intraday_ingestion import (
    IntradayDataset,
    IntradayMarketDataCache,
)
from quantforge.data.intraday_validation import IntradayCoverageInterval
from quantforge.data.lineage import (
    DatasetFamily,
    DatasetFamilyReference,
    SourceConsistencyValidation,
    validate_source_consistency,
)
from quantforge.data.session_aggregation import (
    AggregatedSessionBar,
    AggregatedSessionDataset,
)
from quantforge.timeframes import BarCompletion, DevelopingBarExposure, Timeframe

MULTI_TIMEFRAME_CONTEXT_SCHEMA_VERSION = "1"
_CONTEXT_FAMILY_COMPOSITION_POLICY_NAME = "quantforge_context_artifact_set"
_CONTEXT_FAMILY_COMPOSITION_POLICY_VERSION = "1"
_ARTIFACT_FAMILY_MANIFEST_IDS_KEY = "artifact_family_manifest_ids"


class MultiTimeframeContextError(ValueError):
    """Base domain error for multi-timeframe context use."""


class MultiTimeframeContextValidationError(MultiTimeframeContextError):
    """Context inputs are inconsistent or cannot be aligned safely."""


class UndeclaredTimeframeError(MultiTimeframeContextError):
    """A consumer requested a timeframe that the context did not declare."""


class UnavailableTimeframeError(MultiTimeframeContextError):
    """A declared timeframe has no completed bar available as of the decision."""


class ContextCompletionPolicy(StrEnum):
    """Which bar completion states a context may expose."""

    COMPLETED_BARS_ONLY = "completed_bars_only"
    DEVELOPING_BAR_AS_OF = "developing_bar_as_of"


class ContextAvailability(StrEnum):
    """Availability of one declared timeframe at the decision timestamp."""

    AVAILABLE = "available"
    STALE = "stale"
    MISSING = "missing"


type ArtifactBar = IntradayBar | AggregatedSessionBar
type ContextBar = ArtifactBar | DevelopingBar


def _utc_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise MultiTimeframeContextValidationError(
            f"{field_name} must be a timezone-aware datetime"
        )
    return value.astimezone(UTC)


def _duration_microseconds(value: timedelta) -> int:
    return (value.days * 86_400 + value.seconds) * 1_000_000 + value.microseconds


def _timeframe_primitive(timeframe: Timeframe) -> PrimitiveMapping:
    return {
        "configuration_id": timeframe.configuration_id,
        "configuration": timeframe.to_primitive(),
    }


def _bar_sort_key(bar: ContextBar) -> tuple[datetime, datetime, str]:
    return (bar.end_timestamp, bar.start_timestamp, bar.bar_id)


@dataclass(frozen=True, slots=True)
class _DevelopingSourceEvidence:
    expected_intervals: tuple[IntradayCoverageInterval, ...]
    request_start_timestamp: datetime
    request_end_timestamp: datetime


@dataclass(frozen=True, slots=True)
class ContextTimeframeRequirement:
    """One declared contextual timeframe and its optional freshness limit."""

    timeframe: Timeframe
    maximum_age: timedelta | None = None

    def __post_init__(self) -> None:
        timeframe = cast(object, self.timeframe)
        if not isinstance(timeframe, Timeframe):
            raise MultiTimeframeContextValidationError(
                "context requirement timeframe is invalid"
            )
        if self.maximum_age is not None and self.maximum_age <= timedelta(0):
            raise MultiTimeframeContextValidationError(
                "context maximum age must be positive when specified"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "timeframe": _timeframe_primitive(self.timeframe),
            "maximum_age_microseconds": (
                None
                if self.maximum_age is None
                else _duration_microseconds(self.maximum_age)
            ),
        }


@dataclass(frozen=True, slots=True, init=False)
class TimeframeBarSeries:
    """Validated dataset artifact projected as immutable bars for alignment.

    Use one of the public ``from_*_dataset`` constructors. Directly pairing a
    family reference with unbound bars is intentionally unsupported.
    """

    dataset_reference: DatasetFamilyReference
    timeframe: Timeframe
    bars: tuple[ArtifactBar, ...]
    _developing_source_evidence: _DevelopingSourceEvidence | None

    @classmethod
    def _from_validated_artifact(
        cls,
        dataset_reference: DatasetFamilyReference,
        timeframe: Timeframe,
        bars: tuple[ArtifactBar, ...],
        *,
        developing_source_evidence: _DevelopingSourceEvidence | None = None,
    ) -> "TimeframeBarSeries":
        """Construct after a concrete dataset artifact has passed validation."""
        reference_value = cast(object, dataset_reference)
        timeframe_value = cast(object, timeframe)
        bars_value = cast(object, bars)
        if not isinstance(reference_value, DatasetFamilyReference):
            raise MultiTimeframeContextValidationError(
                "timeframe series dataset reference is invalid"
            )
        if not isinstance(timeframe_value, Timeframe):
            raise MultiTimeframeContextValidationError(
                "timeframe series timeframe is invalid"
            )
        if (
            reference_value.timeframe_configuration_id
            != timeframe_value.configuration_id
        ):
            raise MultiTimeframeContextValidationError(
                "timeframe series does not match its dataset reference"
            )
        if not isinstance(bars_value, tuple):
            raise MultiTimeframeContextValidationError(
                "timeframe series bars must be a tuple"
            )
        untyped_bars = cast(tuple[object, ...], bars_value)
        if any(
            not isinstance(bar, (IntradayBar, AggregatedSessionBar))
            for bar in untyped_bars
        ):
            raise MultiTimeframeContextValidationError(
                "timeframe series contains an unsupported bar"
            )
        typed_bars = cast(tuple[ArtifactBar, ...], untyped_bars)
        if any(bar.timeframe != timeframe_value for bar in typed_bars):
            raise MultiTimeframeContextValidationError(
                "timeframe series contains a bar from another timeframe"
            )
        ordered = tuple(sorted(typed_bars, key=_bar_sort_key))
        boundary_keys = tuple(
            (bar.start_timestamp, bar.end_timestamp, bar.completion) for bar in ordered
        )
        if len(set(boundary_keys)) != len(boundary_keys):
            raise MultiTimeframeContextValidationError(
                "timeframe series contains duplicate bar boundaries"
            )
        for previous, current in pairwise(ordered):
            if current.start_timestamp < previous.end_timestamp:
                raise MultiTimeframeContextValidationError(
                    "timeframe series contains overlapping bars"
                )
        instance = object.__new__(cls)
        object.__setattr__(instance, "dataset_reference", reference_value)
        object.__setattr__(instance, "timeframe", timeframe_value)
        object.__setattr__(instance, "bars", ordered)
        object.__setattr__(
            instance, "_developing_source_evidence", developing_source_evidence
        )
        return instance

    @classmethod
    def from_source_dataset(
        cls,
        dataset: IntradayDataset,
        *,
        family: DatasetFamily,
        cache: IntradayMarketDataCache,
    ) -> "TimeframeBarSeries":
        """Reload, validate, and bind one canonical QF-15/QF-16 dataset."""
        validated_dataset = _validated_cached_source_artifact(dataset, family, cache)
        return cls._from_validated_artifact(
            family.reference(validated_dataset.metadata.dataset_id),
            validated_dataset.request.timeframe,
            validated_dataset.bars,
            developing_source_evidence=_developing_source_evidence(validated_dataset),
        )

    @classmethod
    def from_aggregated_intraday_dataset(
        cls,
        dataset: AggregatedIntradayDataset,
        *,
        family: DatasetFamily | None = None,
    ) -> "TimeframeBarSeries":
        """Validate and bind one QF-18 derived intraday dataset."""
        dataset_value = cast(object, dataset)
        if not isinstance(dataset_value, AggregatedIntradayDataset):
            raise MultiTimeframeContextValidationError(
                "derived intraday series requires an aggregated dataset"
            )
        try:
            dataset_value.validate()
            reference = _reference_for_derived_artifact(
                artifact_family=dataset_value.dataset_family,
                context_family=family,
                dataset_id=dataset_value.metadata.dataset_id,
                timeframe=dataset_value.request.timeframe,
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise MultiTimeframeContextValidationError(
                f"derived intraday dataset validation failed: {error}"
            ) from error
        return cls._from_validated_artifact(
            reference, dataset_value.request.timeframe, dataset_value.bars
        )

    @classmethod
    def from_aggregated_session_dataset(
        cls,
        dataset: AggregatedSessionDataset,
        *,
        family: DatasetFamily | None = None,
    ) -> "TimeframeBarSeries":
        """Validate and bind one QF-19 derived daily or weekly dataset."""
        dataset_value = cast(object, dataset)
        if not isinstance(dataset_value, AggregatedSessionDataset):
            raise MultiTimeframeContextValidationError(
                "session series requires an aggregated dataset"
            )
        try:
            dataset_value.validate()
            reference = _reference_for_derived_artifact(
                artifact_family=dataset_value.dataset_family,
                context_family=family,
                dataset_id=dataset_value.metadata.dataset_id,
                timeframe=dataset_value.metadata.target_timeframe,
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise MultiTimeframeContextValidationError(
                f"session dataset validation failed: {error}"
            ) from error
        return cls._from_validated_artifact(
            reference, dataset_value.metadata.target_timeframe, dataset_value.bars
        )


def _reference_for_derived_artifact(
    *,
    artifact_family: DatasetFamily,
    context_family: DatasetFamily | None,
    dataset_id: str,
    timeframe: Timeframe,
) -> DatasetFamilyReference:
    family = artifact_family if context_family is None else context_family
    if context_family is not None and (
        context_family.canonical_source_snapshot_id
        != artifact_family.canonical_source_snapshot_id
        or context_family.canonical_symbol != artifact_family.canonical_symbol
        or context_family.provider_name != artifact_family.provider_name
        or context_family.feed_scope != artifact_family.feed_scope
        or context_family.adjustment_basis != artifact_family.adjustment_basis
        or context_family.source_timeframe != artifact_family.source_timeframe
    ):
        raise MultiTimeframeContextValidationError(
            "derived dataset does not match the supplied context family"
        )
    if context_family is not None and context_family != artifact_family:
        _validate_composed_family_binding(context_family, artifact_family)
    reference = family.reference(dataset_id)
    if reference.timeframe_configuration_id != timeframe.configuration_id:
        raise MultiTimeframeContextValidationError(
            "derived dataset timeframe does not match the supplied context family"
        )
    return reference


def _validate_composed_family_binding(
    context_family: DatasetFamily,
    artifact_family: DatasetFamily,
) -> None:
    policy = context_family.aggregation_policy
    policy_primitive = policy.to_primitive()
    configuration = cast(
        PrimitiveMapping,
        policy_primitive["configuration"],
    )
    manifest_ids = cast(
        object,
        configuration.get(_ARTIFACT_FAMILY_MANIFEST_IDS_KEY),
    )
    if (
        policy.policy_name != _CONTEXT_FAMILY_COMPOSITION_POLICY_NAME
        or policy.policy_version != _CONTEXT_FAMILY_COMPOSITION_POLICY_VERSION
        or not isinstance(manifest_ids, list)
        or artifact_family.manifest_id not in manifest_ids
    ):
        raise MultiTimeframeContextValidationError(
            "supplied context family does not bind the derived artifact family manifest"
        )


def _validated_cached_source_artifact(
    dataset: IntradayDataset,
    family: DatasetFamily,
    cache: IntradayMarketDataCache,
) -> IntradayDataset:
    dataset_value = cast(object, dataset)
    family_value = cast(object, family)
    cache_value = cast(object, cache)
    if not isinstance(dataset_value, IntradayDataset):
        raise MultiTimeframeContextValidationError(
            "source series requires an intraday dataset"
        )
    if not isinstance(family_value, DatasetFamily):
        raise MultiTimeframeContextValidationError(
            "source series requires a dataset family"
        )
    if not isinstance(cache_value, IntradayMarketDataCache):
        raise MultiTimeframeContextValidationError(
            "source series requires an intraday market data cache"
        )
    try:
        metadata = dataset_value.metadata
        validated_dataset = cache_value.load(metadata.dataset_id, dataset_value.request)
        if validated_dataset != dataset_value:
            raise MultiTimeframeContextValidationError(
                "source dataset does not match its immutable cache artifact"
            )
        if (
            family_value.canonical_source_snapshot_id != metadata.dataset_id
            or family_value.canonical_symbol != dataset_value.request.symbol
            or family_value.provider_name != metadata.provider_name
            or family_value.feed_scope != dataset_value.request.feed_scope
            or family_value.adjustment_basis != dataset_value.request.adjustment_basis
            or family_value.source_timeframe != dataset_value.request.timeframe
        ):
            raise MultiTimeframeContextValidationError(
                "source dataset does not match its family identity"
            )
        family_value.reference(metadata.dataset_id)
        return validated_dataset
    except MultiTimeframeContextValidationError:
        raise
    except (CacheError, TypeError, ValueError) as error:
        raise MultiTimeframeContextValidationError(
            f"source dataset immutable cache validation failed: {error}"
        ) from error


def _developing_source_evidence(
    dataset: IntradayDataset,
) -> _DevelopingSourceEvidence:
    expected: list[IntradayCoverageInterval] = []
    bars_by_session: dict[date, list[IntradayBar]] = {}
    for bar in dataset.bars:
        if bar.completion is not BarCompletion.DEVELOPING:
            bars_by_session.setdefault(bar.session_date, []).append(bar)
    for session in dataset.quality_report.sessions:
        unexpected = {
            (
                interval.start_timestamp,
                interval.end_timestamp,
                interval.completion,
            )
            for interval in session.unexpected_intervals
        }
        observed = (
            IntradayCoverageInterval(
                bar.session_date,
                bar.start_timestamp,
                bar.end_timestamp,
                bar.completion,
            )
            for bar in bars_by_session.get(session.session_date, ())
            if (bar.start_timestamp, bar.end_timestamp, bar.completion)
            not in unexpected
        )
        expected.extend((*observed, *session.missing_intervals))
    return _DevelopingSourceEvidence(
        tuple(
            sorted(
                expected,
                key=lambda item: (
                    item.start_timestamp,
                    item.end_timestamp,
                    item.completion.value,
                ),
            )
        ),
        dataset.request.start_timestamp,
        dataset.request.end_timestamp,
    )


@dataclass(frozen=True, slots=True, init=False)
class TimeframeContext:
    """As-of view and audit metadata for one declared timeframe."""

    requirement: ContextTimeframeRequirement
    dataset_reference: DatasetFamilyReference | None
    availability: ContextAvailability
    bars: tuple[ContextBar, ...]
    latest_completed_bar_timestamp: datetime | None
    age: timedelta | None

    @classmethod
    def _from_aligned_series(
        cls,
        *,
        requirement: ContextTimeframeRequirement,
        dataset_reference: DatasetFamilyReference | None,
        availability: ContextAvailability,
        bars: tuple[ContextBar, ...],
        latest_completed_bar_timestamp: datetime | None,
        age: timedelta | None,
    ) -> "TimeframeContext":
        """Construct only after the builder has aligned a validated series."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "requirement", requirement)
        object.__setattr__(instance, "dataset_reference", dataset_reference)
        object.__setattr__(instance, "availability", availability)
        object.__setattr__(instance, "bars", bars)
        object.__setattr__(
            instance,
            "latest_completed_bar_timestamp",
            latest_completed_bar_timestamp,
        )
        object.__setattr__(instance, "age", age)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        requirement = cast(object, self.requirement)
        reference = cast(object, self.dataset_reference)
        availability = cast(object, self.availability)
        bars = cast(object, self.bars)
        if not isinstance(requirement, ContextTimeframeRequirement):
            raise MultiTimeframeContextValidationError(
                "aligned timeframe requirement is invalid"
            )
        if reference is not None and not isinstance(reference, DatasetFamilyReference):
            raise MultiTimeframeContextValidationError(
                "aligned timeframe dataset reference is invalid"
            )
        if not isinstance(availability, ContextAvailability):
            raise MultiTimeframeContextValidationError(
                "aligned timeframe availability is invalid"
            )
        if not isinstance(bars, tuple):
            raise MultiTimeframeContextValidationError(
                "aligned timeframe bars must be a tuple"
            )
        untyped_bars = cast(tuple[object, ...], bars)
        if any(
            not isinstance(bar, (IntradayBar, AggregatedSessionBar, DevelopingBar))
            for bar in untyped_bars
        ):
            raise MultiTimeframeContextValidationError(
                "aligned timeframe contains an unsupported bar"
            )
        typed_bars = cast(tuple[ContextBar, ...], untyped_bars)
        if any(bar.timeframe != requirement.timeframe for bar in typed_bars):
            raise MultiTimeframeContextValidationError(
                "aligned timeframe contains a bar from another timeframe"
            )
        if typed_bars != tuple(sorted(typed_bars, key=_bar_sort_key)):
            raise MultiTimeframeContextValidationError(
                "aligned timeframe bars must be ordered"
            )
        if reference is not None and (
            reference.timeframe_configuration_id
            != requirement.timeframe.configuration_id
        ):
            raise MultiTimeframeContextValidationError(
                "aligned timeframe does not match its dataset reference"
            )
        completed_bars = tuple(
            bar for bar in typed_bars if bar.completion is not BarCompletion.DEVELOPING
        )
        latest_completed_timestamp = (
            None if not completed_bars else completed_bars[-1].end_timestamp
        )
        if self.latest_completed_bar_timestamp != latest_completed_timestamp:
            raise MultiTimeframeContextValidationError(
                "latest completed timestamp does not match aligned bars"
            )
        if not typed_bars:
            if self.age is not None or availability is not ContextAvailability.MISSING:
                raise MultiTimeframeContextValidationError(
                    "empty aligned timeframe must be explicitly missing"
                )
            return
        if self.age is None or self.age < timedelta(0):
            raise MultiTimeframeContextValidationError(
                "available aligned timeframe requires a nonnegative age"
            )
        maximum_age = requirement.maximum_age
        expected_availability = (
            ContextAvailability.STALE
            if maximum_age is not None and self.age > maximum_age
            else ContextAvailability.AVAILABLE
        )
        if availability is not expected_availability:
            raise MultiTimeframeContextValidationError(
                "aligned timeframe availability disagrees with its age"
            )

    @property
    def timeframe(self) -> Timeframe:
        return self.requirement.timeframe

    @property
    def dataset_id(self) -> str | None:
        if self.dataset_reference is None:
            return None
        return self.dataset_reference.dataset_id

    @property
    def latest_bar(self) -> ContextBar | None:
        return self.bars[-1] if self.bars else None

    @property
    def completion(self) -> BarCompletion | None:
        latest = self.latest_bar
        return None if latest is None else latest.completion

    def to_primitive(self) -> PrimitiveMapping:
        latest = self.latest_bar
        primitive: PrimitiveMapping = {
            "requirement": self.requirement.to_primitive(),
            "dataset_reference": (
                None
                if self.dataset_reference is None
                else self.dataset_reference.to_primitive()
            ),
            "dataset_id": self.dataset_id,
            "bar_interval": self.timeframe.interval.to_primitive(),
            "availability": self.availability.value,
            "latest_completed_bar_timestamp": (
                None
                if self.latest_completed_bar_timestamp is None
                else self.latest_completed_bar_timestamp.isoformat()
            ),
            "latest_completion_state": (
                None if latest is None else latest.completion.value
            ),
            "age_microseconds": (
                None if self.age is None else _duration_microseconds(self.age)
            ),
            "visible_bar_ids": [bar.bar_id for bar in self.bars],
        }
        if isinstance(latest, DevelopingBar):
            primitive["developing_bar"] = {
                "bar_id": latest.bar_id,
                "bar": latest.to_primitive(),
            }
        return primitive


@dataclass(frozen=True, slots=True, init=False)
class MultiTimeframeContext:
    """Deterministic completed-bar views synchronized to one decision timestamp."""

    as_of: datetime
    primary_timeframe: Timeframe
    required_timeframes: tuple[ContextTimeframeRequirement, ...]
    completion_policy: ContextCompletionPolicy
    source_consistency: SourceConsistencyValidation
    timeframes: tuple[TimeframeContext, ...]
    schema_version: str = MULTI_TIMEFRAME_CONTEXT_SCHEMA_VERSION

    @classmethod
    def _from_aligned_timeframes(
        cls,
        *,
        as_of: datetime,
        primary_timeframe: Timeframe,
        required_timeframes: tuple[ContextTimeframeRequirement, ...],
        completion_policy: ContextCompletionPolicy,
        source_consistency: SourceConsistencyValidation,
        timeframes: tuple[TimeframeContext, ...],
    ) -> "MultiTimeframeContext":
        """Construct only from builder-produced, artifact-bound views."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "as_of", as_of)
        object.__setattr__(instance, "primary_timeframe", primary_timeframe)
        object.__setattr__(instance, "required_timeframes", required_timeframes)
        object.__setattr__(instance, "completion_policy", completion_policy)
        object.__setattr__(instance, "source_consistency", source_consistency)
        object.__setattr__(instance, "timeframes", timeframes)
        object.__setattr__(
            instance, "schema_version", MULTI_TIMEFRAME_CONTEXT_SCHEMA_VERSION
        )
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        decision_timestamp = _utc_timestamp(self.as_of, "context as-of")
        object.__setattr__(self, "as_of", decision_timestamp)
        if self.schema_version != MULTI_TIMEFRAME_CONTEXT_SCHEMA_VERSION:
            raise MultiTimeframeContextValidationError(
                "multi-timeframe context schema is invalid"
            )
        completion_policy = cast(object, self.completion_policy)
        source_consistency = cast(object, self.source_consistency)
        timeframes = cast(object, self.timeframes)
        required = cast(object, self.required_timeframes)
        if not isinstance(completion_policy, ContextCompletionPolicy):
            raise MultiTimeframeContextValidationError(
                "context completion policy is invalid"
            )
        if not isinstance(source_consistency, SourceConsistencyValidation):
            raise MultiTimeframeContextValidationError(
                "context source-consistency evidence is invalid"
            )
        if not isinstance(required, tuple):
            raise MultiTimeframeContextValidationError(
                "context requirements must be a tuple"
            )
        typed_required = cast(tuple[ContextTimeframeRequirement, ...], required)
        ordered_requirements = _validate_declared_timeframes(
            self.primary_timeframe, typed_required
        )
        if ordered_requirements != typed_required:
            raise MultiTimeframeContextValidationError(
                "context requirements must use deterministic order"
            )
        if not isinstance(timeframes, tuple):
            raise MultiTimeframeContextValidationError(
                "context aligned timeframes are invalid"
            )
        untyped_timeframes = cast(tuple[object, ...], timeframes)
        if any(
            not isinstance(aligned, TimeframeContext) for aligned in untyped_timeframes
        ):
            raise MultiTimeframeContextValidationError(
                "context aligned timeframes are invalid"
            )
        typed_timeframes = cast(tuple[TimeframeContext, ...], untyped_timeframes)
        expected_requirements = (
            ContextTimeframeRequirement(self.primary_timeframe),
            *ordered_requirements,
        )
        if tuple(aligned.requirement for aligned in typed_timeframes) != (
            expected_requirements
        ):
            raise MultiTimeframeContextValidationError(
                "aligned timeframes do not match declared requirements"
            )
        developing_bars = tuple(
            (aligned, bar)
            for aligned in typed_timeframes
            for bar in aligned.bars
            if isinstance(bar, DevelopingBar)
        )
        if any(
            bar.end_timestamp > decision_timestamp
            for aligned in typed_timeframes
            for bar in aligned.bars
        ) or any(
            aligned.latest_bar is not None
            and aligned.age != decision_timestamp - aligned.latest_bar.end_timestamp
            for aligned in typed_timeframes
        ):
            raise MultiTimeframeContextValidationError(
                "aligned timeframe exposes a bar unavailable at context as-of"
            )
        if completion_policy is ContextCompletionPolicy.COMPLETED_BARS_ONLY:
            if developing_bars:
                raise MultiTimeframeContextValidationError(
                    "completed-bars-only context exposes a developing bar"
                )
        else:
            if any(
                bar.as_of != decision_timestamp
                or aligned.timeframe == self.primary_timeframe
                or aligned.bars[-1] is not bar
                for aligned, bar in developing_bars
            ) or any(
                sum(isinstance(bar, DevelopingBar) for bar in aligned.bars) > 1
                for aligned in typed_timeframes
            ):
                raise MultiTimeframeContextValidationError(
                    "developing bars must be current, contextual, and unique"
                )
        references = tuple(
            aligned.dataset_reference
            for aligned in typed_timeframes
            if aligned.dataset_reference is not None
        )
        if not references:
            raise MultiTimeframeContextValidationError(
                "context requires at least one dataset-family reference"
            )
        try:
            expected_consistency = validate_source_consistency(references)
        except ValueError as error:
            raise MultiTimeframeContextValidationError(str(error)) from error
        if expected_consistency != source_consistency:
            raise MultiTimeframeContextValidationError(
                "context source-consistency evidence does not match its datasets"
            )

    def _for_timeframe(self, timeframe: Timeframe) -> TimeframeContext:
        timeframe_value = cast(object, timeframe)
        if not isinstance(timeframe_value, Timeframe):
            raise UndeclaredTimeframeError("requested timeframe is invalid")
        configuration_id = timeframe_value.configuration_id
        for aligned in self.timeframes:
            if aligned.timeframe.configuration_id == configuration_id:
                return aligned
        raise UndeclaredTimeframeError(
            "timeframe was not declared by this multi-timeframe context: "
            f"{configuration_id}"
        )

    def metadata_for(self, timeframe: Timeframe) -> TimeframeContext:
        """Return audit metadata for a declared timeframe, including missing state."""
        return self._for_timeframe(timeframe)

    def bars_for(self, timeframe: Timeframe) -> tuple[ContextBar, ...]:
        """Return visible completed bars or fail for a declared unavailable view."""
        aligned = self._for_timeframe(timeframe)
        if aligned.availability is ContextAvailability.MISSING:
            raise UnavailableTimeframeError(
                "no completed bar is available for declared timeframe: "
                f"{aligned.timeframe.configuration_id}"
            )
        return aligned.bars

    def latest_bar_for(self, timeframe: Timeframe) -> ContextBar:
        """Return the latest visible completed bar for a declared timeframe."""
        bars = self.bars_for(timeframe)
        return bars[-1]

    def _identity_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "artifact_type": "multi_timeframe_context",
            "as_of": self.as_of.isoformat(),
            "primary_timeframe": _timeframe_primitive(self.primary_timeframe),
            "required_timeframes": [
                requirement.to_primitive() for requirement in self.required_timeframes
            ],
            "completion_policy": self.completion_policy.value,
            "source_consistency": self.source_consistency.to_primitive(),
            "timeframes": [aligned.to_primitive() for aligned in self.timeframes],
        }

    @property
    def context_id(self) -> str:
        """Return the identity of the exact causally visible context."""
        return configuration_identity(self._identity_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        """Return the stable audit representation without copying OHLCV payloads."""
        return {"context_id": self.context_id, **self._identity_primitive()}

    def serialize(self) -> bytes:
        """Serialize the context as canonical sorted JSON bytes."""
        return canonical_json_bytes(self.to_primitive())


def _validate_declared_timeframes(
    primary_timeframe: Timeframe,
    required_timeframes: tuple[ContextTimeframeRequirement, ...],
) -> tuple[ContextTimeframeRequirement, ...]:
    primary_value = cast(object, primary_timeframe)
    if not isinstance(primary_value, Timeframe):
        raise MultiTimeframeContextValidationError("primary timeframe is invalid")
    untyped_requirements = cast(tuple[object, ...], required_timeframes)
    if any(
        not isinstance(requirement, ContextTimeframeRequirement)
        for requirement in untyped_requirements
    ):
        raise MultiTimeframeContextValidationError(
            "required timeframes contain an invalid requirement"
        )
    requirements = cast(tuple[ContextTimeframeRequirement, ...], untyped_requirements)
    primary_id = primary_timeframe.configuration_id
    requirement_ids = tuple(
        requirement.timeframe.configuration_id for requirement in requirements
    )
    if primary_id in requirement_ids:
        raise MultiTimeframeContextValidationError(
            "primary timeframe cannot also be a contextual requirement"
        )
    if len(set(requirement_ids)) != len(requirement_ids):
        raise MultiTimeframeContextValidationError(
            "required contextual timeframes must be unique"
        )
    declared = (
        primary_timeframe,
        *(requirement.timeframe for requirement in requirements),
    )
    if any(
        timeframe.developing_bar_exposure is not DevelopingBarExposure.EXCLUDE
        for timeframe in declared
    ):
        raise MultiTimeframeContextValidationError(
            "completed-bars-only context requires completed-only timeframes"
        )
    if any(
        timeframe.session_policy != primary_timeframe.session_policy
        for timeframe in declared[1:]
    ):
        raise MultiTimeframeContextValidationError(
            "declared timeframes use incompatible exchange session policies"
        )
    return tuple(sorted(requirements, key=lambda item: item.timeframe.configuration_id))


def build_multi_timeframe_context(
    *,
    as_of: datetime,
    primary_timeframe: Timeframe,
    required_timeframes: tuple[ContextTimeframeRequirement, ...],
    series: tuple[TimeframeBarSeries, ...],
    completion_policy: ContextCompletionPolicy = (
        ContextCompletionPolicy.COMPLETED_BARS_ONLY
    ),
) -> MultiTimeframeContext:
    """Build one immutable as-of context from common-family bar series.

    Completed-bars-only is the default. The explicit developing policy appends
    one reconstructed contextual bar using only validated primary source bars
    whose explicit end is at or before ``as_of``.
    """
    decision_timestamp = _utc_timestamp(as_of, "context as-of")
    ordered_requirements = _validate_declared_timeframes(
        primary_timeframe, required_timeframes
    )
    completion_policy_value = cast(object, completion_policy)
    if not isinstance(completion_policy_value, ContextCompletionPolicy):
        raise MultiTimeframeContextValidationError(
            "context completion policy is invalid"
        )
    completion_policy = completion_policy_value
    series_value = cast(object, series)
    if not isinstance(series_value, tuple) or not series_value:
        raise MultiTimeframeContextValidationError(
            "context requires at least one dataset-family series"
        )
    untyped_series = cast(tuple[object, ...], series_value)
    if any(not isinstance(item, TimeframeBarSeries) for item in untyped_series):
        raise MultiTimeframeContextValidationError(
            "context contains an invalid timeframe series"
        )
    typed_series = cast(tuple[TimeframeBarSeries, ...], untyped_series)
    by_timeframe = {item.timeframe.configuration_id: item for item in typed_series}
    if len(by_timeframe) != len(typed_series):
        raise MultiTimeframeContextValidationError(
            "context contains duplicate timeframe series"
        )
    developing_source = by_timeframe.get(primary_timeframe.configuration_id)
    developing_evidence = (
        None
        if developing_source is None
        else developing_source._developing_source_evidence  # pyright: ignore[reportPrivateUsage]
    )
    if completion_policy is ContextCompletionPolicy.DEVELOPING_BAR_AS_OF:
        if developing_source is None or developing_evidence is None:
            raise MultiTimeframeContextValidationError(
                "developing-bar mode requires the validated canonical source "
                "series as the primary timeframe"
            )
        if not (
            developing_evidence.request_start_timestamp
            <= decision_timestamp
            <= developing_evidence.request_end_timestamp
        ):
            raise MultiTimeframeContextValidationError(
                "context as-of falls outside the developing source request"
            )

    primary_requirement = ContextTimeframeRequirement(primary_timeframe)
    declared_requirements = (primary_requirement, *ordered_requirements)
    declared_ids = {
        requirement.timeframe.configuration_id for requirement in declared_requirements
    }
    unexpected_ids = set(by_timeframe) - declared_ids
    if unexpected_ids:
        raise MultiTimeframeContextValidationError(
            f"context series contains an undeclared timeframe: {min(unexpected_ids)}"
        )

    try:
        source_consistency = validate_source_consistency(
            item.dataset_reference for item in typed_series
        )
    except ValueError as error:
        raise MultiTimeframeContextValidationError(str(error)) from error

    aligned_timeframes: list[TimeframeContext] = []
    for requirement in declared_requirements:
        timeframe_id = requirement.timeframe.configuration_id
        input_series = by_timeframe.get(timeframe_id)
        completed_visible: tuple[ContextBar, ...] = (
            ()
            if input_series is None
            else tuple(
                bar
                for bar in input_series.bars
                if bar.completion is not BarCompletion.DEVELOPING
                and bar.end_timestamp <= decision_timestamp
            )
        )
        latest_completed_timestamp = (
            None if not completed_visible else completed_visible[-1].end_timestamp
        )
        visible = completed_visible
        if (
            completion_policy is ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
            and requirement.timeframe != primary_timeframe
        ):
            assert developing_source is not None
            assert developing_evidence is not None
            source_bars = tuple(
                bar for bar in developing_source.bars if isinstance(bar, IntradayBar)
            )
            try:
                developing_bar = reconstruct_developing_bar_as_of(
                    as_of=decision_timestamp,
                    target_timeframe=requirement.timeframe,
                    source_timeframe=developing_source.timeframe,
                    source_bars=source_bars,
                    expected_source_intervals=(developing_evidence.expected_intervals),
                    source_dataset_reference=(developing_source.dataset_reference),
                )
            except DevelopingBarValidationError as error:
                raise MultiTimeframeContextValidationError(str(error)) from error
            if developing_bar is not None:
                visible = (*visible, developing_bar)
        latest = visible[-1] if visible else None
        latest_timestamp = None if latest is None else latest.end_timestamp
        age = (
            None if latest_timestamp is None else decision_timestamp - latest_timestamp
        )
        if latest is None:
            availability = ContextAvailability.MISSING
        elif (
            requirement.maximum_age is not None
            and cast(timedelta, age) > requirement.maximum_age
        ):
            availability = ContextAvailability.STALE
        else:
            availability = ContextAvailability.AVAILABLE
        aligned_timeframes.append(
            TimeframeContext._from_aligned_series(  # pyright: ignore[reportPrivateUsage]
                requirement=requirement,
                dataset_reference=(
                    None if input_series is None else input_series.dataset_reference
                ),
                availability=availability,
                bars=visible,
                latest_completed_bar_timestamp=latest_completed_timestamp,
                age=age,
            )
        )

    return MultiTimeframeContext._from_aligned_timeframes(  # pyright: ignore[reportPrivateUsage]
        as_of=decision_timestamp,
        primary_timeframe=primary_timeframe,
        required_timeframes=ordered_requirements,
        completion_policy=completion_policy,
        source_consistency=source_consistency,
        timeframes=tuple(aligned_timeframes),
    )


__all__ = [
    "MULTI_TIMEFRAME_CONTEXT_SCHEMA_VERSION",
    "ContextAvailability",
    "ContextBar",
    "ContextCompletionPolicy",
    "ContextTimeframeRequirement",
    "MultiTimeframeContext",
    "MultiTimeframeContextError",
    "MultiTimeframeContextValidationError",
    "TimeframeBarSeries",
    "TimeframeContext",
    "UnavailableTimeframeError",
    "UndeclaredTimeframeError",
    "build_multi_timeframe_context",
]
