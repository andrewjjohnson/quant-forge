"""Leakage-safe completed-bar alignment across compatible dataset timeframes."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import canonical_json_bytes, sha256_hex
from quantforge.data.intraday import IntradayBar, IntradayBarBatch
from quantforge.data.intraday_aggregation import AggregatedIntradayDataset
from quantforge.data.intraday_ingestion import (
    INTRADAY_DATASET_SCHEMA_VERSION,
    IntradayDataset,
)
from quantforge.data.intraday_validation import (
    IntradayValidationMode,
    validate_intraday_coverage,
)
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


class ContextAvailability(StrEnum):
    """Availability of one declared timeframe at the decision timestamp."""

    AVAILABLE = "available"
    STALE = "stale"
    MISSING = "missing"


type ContextBar = IntradayBar | AggregatedSessionBar


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
    bars: tuple[ContextBar, ...]

    @classmethod
    def _from_validated_artifact(
        cls,
        dataset_reference: DatasetFamilyReference,
        timeframe: Timeframe,
        bars: tuple[ContextBar, ...],
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
        typed_bars = cast(tuple[ContextBar, ...], untyped_bars)
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
        return instance

    @classmethod
    def from_source_dataset(
        cls,
        dataset: IntradayDataset,
        *,
        family: DatasetFamily,
    ) -> "TimeframeBarSeries":
        """Validate and bind one canonical QF-15/QF-16 source dataset."""
        _validate_source_artifact(dataset, family)
        return cls._from_validated_artifact(
            family.reference(dataset.metadata.dataset_id),
            dataset.request.timeframe,
            dataset.bars,
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
    reference = family.reference(dataset_id)
    if reference.timeframe_configuration_id != timeframe.configuration_id:
        raise MultiTimeframeContextValidationError(
            "derived dataset timeframe does not match the supplied context family"
        )
    return reference


def _validate_source_artifact(dataset: IntradayDataset, family: DatasetFamily) -> None:
    dataset_value = cast(object, dataset)
    family_value = cast(object, family)
    if not isinstance(dataset_value, IntradayDataset):
        raise MultiTimeframeContextValidationError(
            "source series requires an intraday dataset"
        )
    if not isinstance(family_value, DatasetFamily):
        raise MultiTimeframeContextValidationError(
            "source series requires a dataset family"
        )
    try:
        batch = IntradayBarBatch(dataset_value.request, dataset_value.bars)
        metadata = dataset_value.metadata
        report = validate_intraday_coverage(
            batch, mode=IntradayValidationMode.DIAGNOSTIC
        )
        if metadata.schema_version != INTRADAY_DATASET_SCHEMA_VERSION:
            raise MultiTimeframeContextValidationError(
                "source dataset metadata schema is invalid"
            )
        if (
            metadata.request_id != dataset_value.request.request_id
            or metadata.batch_id != batch.batch_id
            or metadata.bar_count != len(batch.bars)
            or metadata.data_sha256 != sha256_hex(batch.serialize())
            or metadata.quality_report != report
        ):
            raise MultiTimeframeContextValidationError(
                "source dataset metadata does not match its bars"
            )
        if any(
            bar.provenance.provider_name != metadata.provider_name
            or bar.provenance.provider_symbol != metadata.provider_symbol
            or bar.provenance.adapter_version != metadata.adapter_version
            or bar.provenance.source_snapshot_id not in metadata.raw_snapshot_ids
            for bar in batch.bars
        ):
            raise MultiTimeframeContextValidationError(
                "source dataset bar provenance does not match its metadata"
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
    except MultiTimeframeContextValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise MultiTimeframeContextValidationError(
            f"source dataset validation failed: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class TimeframeContext:
    """As-of view and audit metadata for one declared timeframe."""

    requirement: ContextTimeframeRequirement
    dataset_reference: DatasetFamilyReference | None
    availability: ContextAvailability
    bars: tuple[ContextBar, ...]
    latest_completed_bar_timestamp: datetime | None
    age: timedelta | None

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
            not isinstance(bar, (IntradayBar, AggregatedSessionBar))
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
        latest_timestamp = None if not typed_bars else typed_bars[-1].end_timestamp
        if self.latest_completed_bar_timestamp != latest_timestamp:
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
        return {
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


@dataclass(frozen=True, slots=True)
class MultiTimeframeContext:
    """Deterministic completed-bar views synchronized to one decision timestamp."""

    as_of: datetime
    primary_timeframe: Timeframe
    required_timeframes: tuple[ContextTimeframeRequirement, ...]
    completion_policy: ContextCompletionPolicy
    source_consistency: SourceConsistencyValidation
    timeframes: tuple[TimeframeContext, ...]
    schema_version: str = MULTI_TIMEFRAME_CONTEXT_SCHEMA_VERSION

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
        if completion_policy is not ContextCompletionPolicy.COMPLETED_BARS_ONLY:
            raise MultiTimeframeContextValidationError(
                "only completed-bars-only context is implemented"
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
        if any(
            bar.completion is BarCompletion.DEVELOPING
            or bar.end_timestamp > decision_timestamp
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

    A bar is visible only when it is terminal and its explicit end timestamp is
    at or before ``as_of``. Future or developing inputs are ignored rather than
    being truncated, reconstructed, or filled.
    """
    decision_timestamp = _utc_timestamp(as_of, "context as-of")
    ordered_requirements = _validate_declared_timeframes(
        primary_timeframe, required_timeframes
    )
    if completion_policy is not ContextCompletionPolicy.COMPLETED_BARS_ONLY:
        raise MultiTimeframeContextValidationError(
            "only completed-bars-only context is implemented"
        )
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
        visible = (
            ()
            if input_series is None
            else tuple(
                bar
                for bar in input_series.bars
                if bar.completion is not BarCompletion.DEVELOPING
                and bar.end_timestamp <= decision_timestamp
            )
        )
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
            TimeframeContext(
                requirement=requirement,
                dataset_reference=(
                    None if input_series is None else input_series.dataset_reference
                ),
                availability=availability,
                bars=visible,
                latest_completed_bar_timestamp=latest_timestamp,
                age=age,
            )
        )

    return MultiTimeframeContext(
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
