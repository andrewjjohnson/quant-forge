"""Provider-neutral dataset-family lineage and source-consistency contracts."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.identity import canonical_json_bytes
from quantforge.data.models import AdjustmentMode
from quantforge.timeframes import Timeframe

DATASET_FAMILY_SCHEMA_VERSION = "1"


class DatasetFamilyValidationError(ValueError):
    """A dataset-family manifest or lineage graph is internally inconsistent."""


class MixedDatasetFamilyError(DatasetFamilyValidationError):
    """A context silently combines datasets without common-source provenance."""


def _validated_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetFamilyValidationError(f"{field_name} must be a nonempty string")
    if value != value.strip():
        raise DatasetFamilyValidationError(
            f"{field_name} cannot contain leading or trailing whitespace"
        )
    return value


class FeedCoverage(StrEnum):
    """Provider-neutral market coverage represented by one source feed."""

    CONSOLIDATED = "consolidated"
    SINGLE_VENUE = "single_venue"
    PROVIDER_DEFINED = "provider_defined"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FeedScope:
    """Coverage of source observations without a provider response dependency."""

    coverage: FeedCoverage
    market_center: str | None = None
    provider_scope: str | None = None

    def __post_init__(self) -> None:
        coverage = cast(object, self.coverage)
        if not isinstance(coverage, FeedCoverage):
            raise DatasetFamilyValidationError("feed coverage is invalid")
        if self.coverage is FeedCoverage.CONSOLIDATED:
            if self.market_center is not None or self.provider_scope is not None:
                raise DatasetFamilyValidationError(
                    "consolidated feed scope cannot name a venue or provider scope"
                )
            return
        if self.coverage is FeedCoverage.UNKNOWN:
            if self.market_center is not None or self.provider_scope is not None:
                raise DatasetFamilyValidationError(
                    "unknown feed scope cannot name a venue or provider scope"
                )
            return
        if self.coverage is FeedCoverage.SINGLE_VENUE:
            _validated_text(self.market_center, "feed market center")
            if self.provider_scope is not None:
                raise DatasetFamilyValidationError(
                    "single-venue feed scope cannot name a provider-defined scope"
                )
            return
        _validated_text(self.provider_scope, "provider-defined feed scope")
        if self.market_center is not None:
            raise DatasetFamilyValidationError(
                "provider-defined feed scope cannot also name a market center"
            )

    @classmethod
    def consolidated(cls) -> "FeedScope":
        """Return a feed that combines all covered market centers."""
        return cls(FeedCoverage.CONSOLIDATED)

    @classmethod
    def single_venue(cls, market_center: str) -> "FeedScope":
        """Return a feed restricted to one named market center, such as IEX."""
        return cls(FeedCoverage.SINGLE_VENUE, market_center=market_center)

    @classmethod
    def iex_only(cls) -> "FeedScope":
        """Return the provider-neutral representation of IEX-only observations."""
        return cls.single_venue("IEX")

    @classmethod
    def provider_defined(cls, provider_scope: str) -> "FeedScope":
        """Return an explicit provider-defined scope with no canonical analogue."""
        return cls(FeedCoverage.PROVIDER_DEFINED, provider_scope=provider_scope)

    @classmethod
    def unknown(cls) -> "FeedScope":
        """Return an explicitly unknown feed rather than assuming consolidation."""
        return cls(FeedCoverage.UNKNOWN)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "coverage": self.coverage.value,
            "market_center": self.market_center,
            "provider_scope": self.provider_scope,
        }


@dataclass(frozen=True, slots=True)
class AdjustmentBasis:
    """Price, volume, and corporate-action basis of the canonical source."""

    adjustment_mode: AdjustmentMode
    ohlc_basis: str
    volume_basis: str
    corporate_action_policy: str
    adjusted_fields_used: bool

    def __post_init__(self) -> None:
        adjustment_mode = cast(object, self.adjustment_mode)
        if not isinstance(adjustment_mode, AdjustmentMode):
            raise DatasetFamilyValidationError("adjustment mode is invalid")
        _validated_text(self.ohlc_basis, "OHLC basis")
        _validated_text(self.volume_basis, "volume basis")
        _validated_text(self.corporate_action_policy, "corporate-action policy")
        adjusted_fields_used = cast(object, self.adjusted_fields_used)
        if not isinstance(adjusted_fields_used, bool):
            raise DatasetFamilyValidationError(
                "adjusted-fields-used flag must be a boolean"
            )
        expected_basis = (
            "raw_provider"
            if self.adjustment_mode is AdjustmentMode.UNADJUSTED
            else "split_adjusted"
        )
        if self.ohlc_basis != expected_basis or self.volume_basis != expected_basis:
            raise DatasetFamilyValidationError(
                "adjustment mode is inconsistent with the OHLC or volume basis"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "adjustment_mode": self.adjustment_mode.value,
            "ohlc_basis": self.ohlc_basis,
            "volume_basis": self.volume_basis,
            "corporate_action_policy": self.corporate_action_policy,
            "adjusted_fields_used": self.adjusted_fields_used,
        }


@dataclass(frozen=True, slots=True, init=False)
class AggregationPolicy:
    """Stable reference to aggregation semantics implemented by a later boundary."""

    policy_name: str
    policy_version: str
    _configuration: PrimitiveMappingSnapshot = field(repr=False)

    def __init__(
        self,
        policy_name: str,
        policy_version: str,
        configuration: PrimitiveMapping,
    ) -> None:
        object.__setattr__(
            self, "policy_name", _validated_text(policy_name, "aggregation policy name")
        )
        object.__setattr__(
            self,
            "policy_version",
            _validated_text(policy_version, "aggregation policy version"),
        )
        try:
            snapshot = PrimitiveMappingSnapshot.capture(configuration)
        except (TypeError, ValueError) as error:
            raise DatasetFamilyValidationError(
                "aggregation policy configuration must be canonical JSON primitives"
            ) from error
        object.__setattr__(self, "_configuration", snapshot)

    def to_primitive(self) -> PrimitiveMapping:
        """Return a detached, complete policy reference without aggregating bars."""
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "configuration": self._configuration.to_primitive(),
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.to_primitive())


@dataclass(frozen=True, slots=True)
class DatasetLineage:
    """One source or derived dataset in a single-parent lineage graph."""

    dataset_id: str
    timeframe: Timeframe
    canonical_source_snapshot_id: str
    parent_dataset_id: str | None
    child_dataset_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validated_text(self.dataset_id, "dataset ID")
        _validated_text(
            self.canonical_source_snapshot_id, "canonical source snapshot ID"
        )
        timeframe = cast(object, self.timeframe)
        if not isinstance(timeframe, Timeframe):
            raise DatasetFamilyValidationError("dataset timeframe is invalid")
        if self.parent_dataset_id is not None:
            _validated_text(self.parent_dataset_id, "parent dataset ID")
        child_ids = cast(object, self.child_dataset_ids)
        if not isinstance(child_ids, tuple):
            raise DatasetFamilyValidationError("child dataset IDs must be a tuple")
        untyped_child_ids = cast(tuple[object, ...], child_ids)
        for child_id in untyped_child_ids:
            _validated_text(child_id, "child dataset ID")
        typed_child_ids = cast(tuple[str, ...], untyped_child_ids)
        if len(set(typed_child_ids)) != len(typed_child_ids):
            raise DatasetFamilyValidationError("child dataset IDs must be unique")
        object.__setattr__(self, "child_dataset_ids", tuple(sorted(typed_child_ids)))

        is_source = self.dataset_id == self.canonical_source_snapshot_id
        if is_source and self.parent_dataset_id is not None:
            raise DatasetFamilyValidationError(
                "canonical source snapshot cannot have a parent dataset"
            )
        if not is_source and self.parent_dataset_id is None:
            raise DatasetFamilyValidationError(
                "every derived dataset must name a parent dataset"
            )
        if self.parent_dataset_id == self.dataset_id:
            raise DatasetFamilyValidationError("dataset lineage contains a cycle")
        if self.dataset_id in self.child_dataset_ids:
            raise DatasetFamilyValidationError("dataset lineage contains a cycle")

    @property
    def is_canonical_source(self) -> bool:
        return self.dataset_id == self.canonical_source_snapshot_id

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "dataset_id": self.dataset_id,
            "role": (
                "canonical_source_snapshot"
                if self.is_canonical_source
                else "derived_dataset"
            ),
            "timeframe": {
                "configuration_id": self.timeframe.configuration_id,
                "configuration": self.timeframe.to_primitive(),
            },
            "canonical_source_snapshot_id": self.canonical_source_snapshot_id,
            "parent_dataset_id": self.parent_dataset_id,
            "child_dataset_ids": list(self.child_dataset_ids),
        }


@dataclass(frozen=True, slots=True)
class DatasetFamilyReference:
    """Compact lineage reference carried by one dataset in a study context."""

    family_id: str
    dataset_id: str
    canonical_source_snapshot_id: str
    timeframe_configuration_id: str
    feed_scope: FeedScope

    def __post_init__(self) -> None:
        _validated_text(self.family_id, "dataset family ID")
        _validated_text(self.dataset_id, "dataset ID")
        _validated_text(
            self.canonical_source_snapshot_id, "canonical source snapshot ID"
        )
        _validated_text(self.timeframe_configuration_id, "timeframe configuration ID")
        if not isinstance(cast(object, self.feed_scope), FeedScope):
            raise DatasetFamilyValidationError(
                "dataset family reference feed scope is invalid"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "family_id": self.family_id,
            "dataset_id": self.dataset_id,
            "canonical_source_snapshot_id": self.canonical_source_snapshot_id,
            "timeframe_configuration_id": self.timeframe_configuration_id,
            "feed_scope": self.feed_scope.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class DatasetFamily:
    """Canonical source metadata and the complete derived-dataset lineage DAG."""

    canonical_symbol: str
    provider_name: str
    feed_scope: FeedScope
    adjustment_basis: AdjustmentBasis
    aggregation_policy: AggregationPolicy
    canonical_source_snapshot_id: str
    datasets: tuple[DatasetLineage, ...]
    schema_version: str = DATASET_FAMILY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        symbol = _validated_text(self.canonical_symbol, "canonical symbol")
        if symbol != symbol.upper():
            raise DatasetFamilyValidationError(
                "canonical symbol must use uppercase representation"
            )
        _validated_text(self.provider_name, "provider name")
        _validated_text(
            self.canonical_source_snapshot_id, "canonical source snapshot ID"
        )
        feed_scope = cast(object, self.feed_scope)
        adjustment_basis = cast(object, self.adjustment_basis)
        aggregation_policy = cast(object, self.aggregation_policy)
        if not isinstance(feed_scope, FeedScope):
            raise DatasetFamilyValidationError("feed scope is invalid")
        if not isinstance(adjustment_basis, AdjustmentBasis):
            raise DatasetFamilyValidationError("adjustment basis is invalid")
        if not isinstance(aggregation_policy, AggregationPolicy):
            raise DatasetFamilyValidationError("aggregation policy is invalid")
        if self.schema_version != DATASET_FAMILY_SCHEMA_VERSION:
            raise DatasetFamilyValidationError(
                f"dataset-family schema {DATASET_FAMILY_SCHEMA_VERSION} is required"
            )
        datasets = cast(object, self.datasets)
        if not isinstance(datasets, tuple) or not datasets:
            raise DatasetFamilyValidationError(
                "dataset family must contain at least its canonical source"
            )
        untyped_datasets = cast(tuple[object, ...], datasets)
        if any(not isinstance(item, DatasetLineage) for item in untyped_datasets):
            raise DatasetFamilyValidationError("dataset lineage entry is invalid")
        typed_datasets = cast(tuple[DatasetLineage, ...], untyped_datasets)
        ordered = tuple(sorted(typed_datasets, key=lambda item: item.dataset_id))
        object.__setattr__(self, "datasets", ordered)
        self._validate_lineage()

    def _validate_lineage(self) -> None:
        by_id = {item.dataset_id: item for item in self.datasets}
        if len(by_id) != len(self.datasets):
            raise DatasetFamilyValidationError("dataset IDs must be unique in a family")
        source = by_id.get(self.canonical_source_snapshot_id)
        if source is None or not source.is_canonical_source:
            raise DatasetFamilyValidationError(
                "family must contain its canonical source snapshot"
            )
        if any(
            item.canonical_source_snapshot_id != self.canonical_source_snapshot_id
            for item in self.datasets
        ):
            raise DatasetFamilyValidationError(
                "every dataset must point to the family canonical source snapshot"
            )
        for item in self.datasets:
            if (
                item.parent_dataset_id is not None
                and item.parent_dataset_id not in by_id
            ):
                raise DatasetFamilyValidationError(
                    f"lineage parent is not in the family: {item.parent_dataset_id}"
                )

        for item in self.datasets:
            path: set[str] = set()
            current = item
            while current.dataset_id != self.canonical_source_snapshot_id:
                if current.dataset_id in path:
                    raise DatasetFamilyValidationError(
                        "dataset lineage contains a cycle"
                    )
                path.add(current.dataset_id)
                parent_id = cast(str, current.parent_dataset_id)
                current = by_id[parent_id]

        expected_children: dict[str, list[str]] = {
            dataset_id: [] for dataset_id in by_id
        }
        for item in self.datasets:
            if item.parent_dataset_id is not None:
                expected_children[item.parent_dataset_id].append(item.dataset_id)
        for item in self.datasets:
            expected = tuple(sorted(expected_children[item.dataset_id]))
            if item.child_dataset_ids != expected:
                raise DatasetFamilyValidationError(
                    f"child dataset IDs disagree with parent links: {item.dataset_id}"
                )

    @property
    def source_timeframe(self) -> Timeframe:
        return next(
            item.timeframe
            for item in self.datasets
            if item.dataset_id == self.canonical_source_snapshot_id
        )

    def _canonical_source_primitive(self) -> PrimitiveMapping:
        session_policy = self.source_timeframe.session_policy
        return {
            "snapshot_id": self.canonical_source_snapshot_id,
            "symbol": self.canonical_symbol,
            "provider": self.provider_name,
            "feed_scope": self.feed_scope.to_primitive(),
            "source_interval": self.source_timeframe.interval.to_primitive(),
            "session_scope": session_policy.scope.value,
            "exchange_calendar": session_policy.calendar_name,
            "exchange_timezone": session_policy.timezone_name,
            "timeframe": {
                "configuration_id": self.source_timeframe.configuration_id,
                "configuration": self.source_timeframe.to_primitive(),
            },
            "adjustment_basis": self.adjustment_basis.to_primitive(),
            "aggregation_policy": {
                "configuration_id": self.aggregation_policy.configuration_id,
                "configuration": self.aggregation_policy.to_primitive(),
            },
        }

    def _family_identity_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "canonical_source": self._canonical_source_primitive(),
            "source_consistency": {
                "required_policy": "common_canonical_source",
                "external_bar_validation_policy": None,
            },
        }

    @property
    def family_id(self) -> str:
        """Return the deterministic identity shared by every family member."""
        return configuration_identity(self._family_identity_primitive())

    def _manifest_primitive(self) -> PrimitiveMapping:
        return {
            **self._family_identity_primitive(),
            "family_id": self.family_id,
            "lineage": [item.to_primitive() for item in self.datasets],
        }

    @property
    def manifest_id(self) -> str:
        """Return the deterministic identity of the exact recorded lineage graph."""
        return configuration_identity(self._manifest_primitive())

    def to_manifest(self) -> PrimitiveMapping:
        """Return the complete embeddable dataset-family manifest."""
        return {"manifest_id": self.manifest_id, **self._manifest_primitive()}

    def serialize_manifest(self) -> bytes:
        """Serialize the manifest as canonical sorted JSON bytes."""
        return canonical_json_bytes(self.to_manifest())

    def reference(self, dataset_id: str) -> DatasetFamilyReference:
        """Create validated provenance for one dataset in this family."""
        requested_id = _validated_text(dataset_id, "dataset ID")
        try:
            lineage = next(
                item for item in self.datasets if item.dataset_id == requested_id
            )
        except StopIteration as error:
            raise DatasetFamilyValidationError(
                f"dataset is not recorded in family lineage: {requested_id}"
            ) from error
        return DatasetFamilyReference(
            family_id=self.family_id,
            dataset_id=lineage.dataset_id,
            canonical_source_snapshot_id=self.canonical_source_snapshot_id,
            timeframe_configuration_id=lineage.timeframe.configuration_id,
            feed_scope=self.feed_scope,
        )


class SourceConsistencyMode(StrEnum):
    """How a set of dataset references established source consistency."""

    COMMON_DATASET_FAMILY = "common_dataset_family"
    EXTERNALLY_VALIDATED = "externally_validated"


@dataclass(frozen=True, slots=True)
class SourceConsistencyValidation:
    """Serializable evidence returned after a context passes source validation."""

    mode: SourceConsistencyMode
    family_id: str | None
    external_validation_policy_id: str | None

    def __post_init__(self) -> None:
        if self.mode is SourceConsistencyMode.COMMON_DATASET_FAMILY:
            _validated_text(self.family_id, "dataset family ID")
            if self.external_validation_policy_id is not None:
                raise DatasetFamilyValidationError(
                    "common-family validation cannot name an external policy"
                )
            return
        if self.mode is not SourceConsistencyMode.EXTERNALLY_VALIDATED:
            raise DatasetFamilyValidationError("source-consistency mode is invalid")
        _validated_text(
            self.external_validation_policy_id, "external validation policy ID"
        )
        if self.family_id is not None:
            raise DatasetFamilyValidationError(
                "external validation cannot claim one common family"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "mode": self.mode.value,
            "family_id": self.family_id,
            "external_validation_policy_id": self.external_validation_policy_id,
        }


class ExternalBarValidationPolicy(Protocol):
    """Extension point for a future explicit external-bar validation policy.

    QuantForge intentionally ships no implementation in QF-14. A later policy
    must expose deterministic configuration and either return normally after
    validation or raise when the supplied references are incompatible.
    """

    @property
    def configuration_id(self) -> str: ...

    def to_primitive(self) -> PrimitiveMapping: ...

    def validate(self, references: tuple[DatasetFamilyReference, ...]) -> None: ...


def validate_source_consistency(
    references: Iterable[DatasetFamilyReference],
    *,
    external_bar_validation_policy: ExternalBarValidationPolicy | None = None,
) -> SourceConsistencyValidation:
    """Reject mixed-family contexts unless an explicit external policy validates."""
    fixed_references = tuple(references)
    if not fixed_references:
        raise DatasetFamilyValidationError(
            "source consistency requires at least one dataset reference"
        )
    untyped_references = cast(tuple[object, ...], fixed_references)
    if any(
        not isinstance(reference, DatasetFamilyReference)
        for reference in untyped_references
    ):
        raise DatasetFamilyValidationError("dataset family reference is invalid")
    typed_references = cast(tuple[DatasetFamilyReference, ...], untyped_references)

    family_ids = {reference.family_id for reference in typed_references}
    source_snapshot_ids = {
        reference.canonical_source_snapshot_id for reference in typed_references
    }
    if external_bar_validation_policy is None:
        if len(family_ids) != 1 or len(source_snapshot_ids) != 1:
            raise MixedDatasetFamilyError(
                "mixed dataset families require an explicit validated external-bar "
                "policy"
            )
        return SourceConsistencyValidation(
            SourceConsistencyMode.COMMON_DATASET_FAMILY,
            next(iter(family_ids)),
            None,
        )

    try:
        before = PrimitiveMappingSnapshot.capture(
            external_bar_validation_policy.to_primitive()
        )
        policy_id_value = cast(object, external_bar_validation_policy.configuration_id)
        if (
            not isinstance(policy_id_value, str)
            or not policy_id_value
            or configuration_identity(before.to_primitive()) != policy_id_value
        ):
            raise DatasetFamilyValidationError(
                "external-bar validation policy identity is inconsistent"
            )
        policy_id = policy_id_value
        external_bar_validation_policy.validate(typed_references)
        if (
            PrimitiveMappingSnapshot.capture(
                external_bar_validation_policy.to_primitive()
            )
            != before
            or external_bar_validation_policy.configuration_id != policy_id
        ):
            raise DatasetFamilyValidationError(
                "external-bar validation policy mutated during validation"
            )
    except DatasetFamilyValidationError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise DatasetFamilyValidationError(
            "external-bar validation policy is invalid"
        ) from error
    return SourceConsistencyValidation(
        SourceConsistencyMode.EXTERNALLY_VALIDATED,
        None,
        policy_id,
    )


__all__ = [
    "DATASET_FAMILY_SCHEMA_VERSION",
    "AdjustmentBasis",
    "AggregationPolicy",
    "DatasetFamily",
    "DatasetFamilyReference",
    "DatasetFamilyValidationError",
    "DatasetLineage",
    "ExternalBarValidationPolicy",
    "FeedCoverage",
    "FeedScope",
    "MixedDatasetFamilyError",
    "SourceConsistencyMode",
    "SourceConsistencyValidation",
    "validate_source_consistency",
]
