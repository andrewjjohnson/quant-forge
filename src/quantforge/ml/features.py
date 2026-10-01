"""Explicit, versioned causal feature schemas for event ML datasets (QF-67).

Model inputs are never discovered. Each column is declared by an
``EventFeatureDefinition`` that names one persisted causal value of the
generated signal (``signal["features"]``), its approved scalar type, its
missing-value policy, its causal availability and, where material, its source
timeframe. Values are read only from that causal mapping; outcome, evaluation,
temporal-resolution, provenance and partition records are never consulted.

Names that denote future outcomes, availability or ambiguity, identifiers,
hashes, provenance or partitions are refused as feature names and as source
fields, even if a rule happened to persist a value under such a name.

Numeric encoding (documented, lossless):

- ``decimal``: the exact canonical decimal string persisted by QF-11
  (``decimal_to_primitive``); never parsed into binary floating point for
  storage or identity. ``EventDataset.numeric_feature_rows`` offers an explicit
  IEEE-754 binary64 view (``float(Decimal(text))``, correctly rounded).
- ``integer``: a Python/Arrow 64-bit integer (booleans are not integers).
- ``boolean``: a boolean.

Missing values are explicit nulls. ``reject`` fails the build; ``preserve_null``
keeps the null. Nothing is imputed, zero-filled, forward-filled or scaled.
"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.ml.errors import EventFeatureSchemaError
from quantforge.timeframes import Timeframe

EVENT_FEATURE_SCHEMA_CONTRACT_VERSION = "1"
_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
_MAXIMUM_NAME_LENGTH = 64

type FeatureValue = str | int | bool | None


class FeatureValueType(StrEnum):
    """Approved scalar types; anything else is refused."""

    DECIMAL = "decimal"
    INTEGER = "integer"
    BOOLEAN = "boolean"


class MissingValuePolicy(StrEnum):
    """Null handling; there is deliberately no imputation policy."""

    REJECT = "reject"
    PRESERVE_NULL = "preserve_null"


class CausalAvailability(StrEnum):
    """When a value becomes known; only decision-time values are admitted."""

    DECISION_TIMESTAMP = "known_at_decision_timestamp_from_completed_inputs"


class FeatureSource(StrEnum):
    """Where a value is read from; only the persisted causal signal mapping."""

    PERSISTED_SIGNAL_FEATURE = "persisted_generated_signal_feature"


# Reserved dataset columns: metadata, partition membership and target.
RESERVED_COLUMN_NAMES = frozenset(
    {
        "row_index",
        "source_observation_id",
        "source_index",
        "decision_timestamp",
        "signal_session",
        "decision_sequence",
        "signal_index",
        "context_id",
        "prediction_study_id",
        "direction",
        "disposition",
        "partition_role",
        "fold_id",
        "fold_index",
        "target",
        "target_status",
        "target_source_value",
        "target_outcome_id",
    }
)

# Fields of persisted QF-11 rows and QF-46/QF-47/QF-49 (and legacy QF-7/QF-11)
# outcome, evaluation and resolution records, plus provenance and partition
# envelopes. None of these may become a model input under any name.
FORBIDDEN_FEATURE_NAMES = RESERVED_COLUMN_NAMES | frozenset(
    {
        # Row and outcome/evaluation envelopes.
        "prediction",
        "features",
        "outcome",
        "evaluation",
        "values",
        "temporal_resolution",
        "request",
        "row",
        "rows",
        "signal",
        "signals",
        # QF-46 resolution and QF-49 endpoint values.
        "anchor_kind",
        "horizon_kind",
        "elapsed_duration_microseconds",
        "requested_target_timestamp",
        "expected_observation_timestamp",
        "resolved_observation_timestamp",
        "status",
        "available",
        "reference_price",
        "reference_price_convention",
        "future_price_convention",
        "outcome_price",
        "raw_return",
        "source_reference",
        # QF-47 path, excursion and target/stop values.
        "endpoint_status",
        "endpoint_available",
        "path_start_timestamp",
        "path_end_timestamp",
        "missing_observation_timestamp",
        "unavailable_reason",
        "future_ranges",
        "mfe_percentage",
        "mae_percentage",
        "mfe_timestamp",
        "mae_timestamp",
        "stop_percentage",
        "same_bar_conflict_policy",
        "label",
        "stop_level",
        "event_timestamp",
        "ambiguous_timestamp",
        "ambiguous_high",
        "ambiguous_low",
        # Legacy QF-7/QF-11 session outcomes and evaluations.
        "next_open",
        "overnight_gap_percentage",
        "gap_size_percentage",
        "signed_prediction_return",
        "signed_return",
        "direction_correct",
        "baseline_correct",
        "correct",
        "accuracy",
        "excursion",
        "first_touch",
        # Provenance, partition, holdout and experiment identifiers.
        "role",
        "lineage",
        "plan",
        "window",
        "schedule",
        "manifest",
        "provenance",
        "dataset_fingerprint",
    }
)
_FORBIDDEN_PREFIXES = (
    "outcome",
    "evaluation",
    "evaluator",
    "target",
    "label",
    "future",
    "forward",
    "fold",
    "partition",
    "holdout",
    "selection",
    "experiment",
    "result",
    "provenance",
    "lineage",
    "mfe",
    "mae",
)
_FORBIDDEN_SUFFIXES = (
    "_id",
    "_ids",
    "_sha256",
    "_fingerprint",
    "_hash",
    "_digest",
    "_timestamp",
)


def require_causal_feature_name(name: object, *, label: str = "feature name") -> str:
    """Return ``name`` if it is a safe causal model-input name, else fail closed."""
    if not isinstance(name, str) or not name:
        raise EventFeatureSchemaError(f"{label} must be nonempty text")
    if len(name) > _MAXIMUM_NAME_LENGTH or _NAME.fullmatch(name) is None:
        raise EventFeatureSchemaError(
            f"{label} {name!r} must be lowercase snake_case of at most "
            f"{_MAXIMUM_NAME_LENGTH} characters"
        )
    tokens = name.split("_")
    if (
        name in FORBIDDEN_FEATURE_NAMES
        or tokens[0] in _FORBIDDEN_PREFIXES
        or any(name.endswith(suffix) for suffix in _FORBIDDEN_SUFFIXES)
    ):
        raise EventFeatureSchemaError(
            f"{label} {name!r} denotes a future outcome, availability, identifier, "
            "provenance or partition field and cannot be a model input"
        )
    return name


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EventFeatureSchemaError(f"{label} must be nonempty text")
    return value


@dataclass(frozen=True, slots=True)
class EventFeatureDefinition:
    """One explicitly approved causal model input.

    ``source_field`` names a key of the persisted generated signal's causal
    ``features`` mapping (for QF-45, e.g. ``current_fast``). ``timeframe`` is
    the source timeframe when it is material; it must be declared by the plan.
    """

    name: str
    source_field: str
    value_type: FeatureValueType
    missing_values: MissingValuePolicy
    unit: str
    description: str
    timeframe: Timeframe | None = None
    version: str = "1"
    availability: CausalAvailability = CausalAvailability.DECISION_TIMESTAMP
    source: FeatureSource = FeatureSource.PERSISTED_SIGNAL_FEATURE

    def __post_init__(self) -> None:
        require_causal_feature_name(self.name)
        require_causal_feature_name(self.source_field, label="feature source field")
        for value, kind, label in (
            (self.value_type, FeatureValueType, "value type"),
            (self.missing_values, MissingValuePolicy, "missing-value policy"),
            (self.availability, CausalAvailability, "causal availability"),
            (self.source, FeatureSource, "feature source"),
        ):
            if type(cast(object, value)) is not kind:
                raise EventFeatureSchemaError(
                    f"feature {self.name!r} has an unsupported {label}"
                )
        _text(self.unit, "feature unit")
        _text(self.description, "feature description")
        _text(self.version, "feature definition version")
        if self.timeframe is not None and not isinstance(
            cast(object, self.timeframe), Timeframe
        ):
            raise EventFeatureSchemaError("feature timeframe must be a Timeframe")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source.value,
            "source_field": self.source_field,
            "value_type": self.value_type.value,
            "missing_values": self.missing_values.value,
            "availability": self.availability.value,
            "unit": self.unit,
            "description": self.description,
            "timeframe": None
            if self.timeframe is None
            else {
                "configuration_id": self.timeframe.configuration_id,
                "configuration": self.timeframe.to_primitive(),
            },
        }

    @property
    def definition_id(self) -> str:
        return configuration_identity(self.to_primitive())

    @classmethod
    def from_primitive(cls, value: Primitive) -> "EventFeatureDefinition":
        if not isinstance(value, dict):
            raise EventFeatureSchemaError("feature definition must be a mapping")
        if frozenset(value) != _DEFINITION_FIELDS:
            raise EventFeatureSchemaError("feature definition fields are invalid")
        timeframe: Timeframe | None = None
        raw_timeframe = value["timeframe"]
        try:
            if raw_timeframe is not None:
                if not isinstance(raw_timeframe, dict) or not isinstance(
                    raw_timeframe.get("configuration"), dict
                ):
                    raise ValueError("timeframe reference")
                timeframe = Timeframe.from_primitive(
                    cast(PrimitiveMapping, raw_timeframe["configuration"])
                )
            definition = cls(
                name=cast(str, value["name"]),
                source_field=cast(str, value["source_field"]),
                value_type=FeatureValueType(cast(str, value["value_type"])),
                missing_values=MissingValuePolicy(cast(str, value["missing_values"])),
                unit=cast(str, value["unit"]),
                description=cast(str, value["description"]),
                timeframe=timeframe,
                version=cast(str, value["version"]),
                availability=CausalAvailability(cast(str, value["availability"])),
                source=FeatureSource(cast(str, value["source"])),
            )
        except (ValueError, TypeError, KeyError) as error:
            if isinstance(error, EventFeatureSchemaError):
                raise
            raise EventFeatureSchemaError("feature definition is invalid") from error
        if (
            definition.to_primitive()
            != {k: v for k, v in value.items() if k != "definition_id"}
            or value["definition_id"] != definition.definition_id
        ):
            raise EventFeatureSchemaError("feature definition identity is inconsistent")
        return definition

    def value(self, features: PrimitiveMapping) -> FeatureValue:
        """Return this feature's exact causal value from a persisted mapping."""
        if self.source_field not in features:
            raise EventFeatureSchemaError(
                f"persisted observation has no causal feature {self.source_field!r}; "
                "the feature schema does not match this population"
            )
        raw = features[self.source_field]
        if raw is None:
            if self.missing_values is MissingValuePolicy.REJECT:
                raise EventFeatureSchemaError(
                    f"required feature {self.name!r} is missing for an observation"
                )
            return None
        if self.value_type is FeatureValueType.DECIMAL:
            return canonical_decimal(raw, f"feature {self.name!r}")
        if self.value_type is FeatureValueType.INTEGER:
            if type(raw) is not int:
                raise EventFeatureSchemaError(
                    f"feature {self.name!r} must be an integer scalar"
                )
            return raw
        if type(raw) is not bool:
            raise EventFeatureSchemaError(f"feature {self.name!r} must be a boolean")
        return raw


_DEFINITION_FIELDS = frozenset(
    {
        "name",
        "version",
        "source",
        "source_field",
        "value_type",
        "missing_values",
        "availability",
        "unit",
        "description",
        "timeframe",
        "definition_id",
    }
)


def canonical_decimal(raw: object, label: str) -> str:
    """Require an exact, finite, canonically rendered decimal string."""
    if not isinstance(raw, str):
        raise EventFeatureSchemaError(
            f"{label} must be an exact decimal string, never a float or structure"
        )
    try:
        parsed = Decimal(raw)
    except InvalidOperation as error:
        raise EventFeatureSchemaError(f"{label} is not a decimal") from error
    if not parsed.is_finite() or decimal_to_primitive(parsed) != raw:
        raise EventFeatureSchemaError(f"{label} is not a canonical finite decimal")
    return raw


@dataclass(frozen=True, slots=True)
class EventFeatureSchema:
    """Ordered explicit allowlist of model inputs; column order is declared order."""

    name: str
    version: str
    features: tuple[EventFeatureDefinition, ...]

    def __post_init__(self) -> None:
        _text(self.name, "feature schema name")
        _text(self.version, "feature schema version")
        if not isinstance(cast(object, self.features), tuple) or not self.features:
            raise EventFeatureSchemaError(
                "feature schemas require a nonempty tuple of definitions"
            )
        if any(
            type(cast(object, item)) is not EventFeatureDefinition
            for item in self.features
        ):
            raise EventFeatureSchemaError(
                "feature schemas accept only EventFeatureDefinition values"
            )
        names = [item.name for item in self.features]
        if len(set(names)) != len(names):
            raise EventFeatureSchemaError("feature names must be unique")
        sources = [item.source_field for item in self.features]
        if len(set(sources)) != len(sources):
            raise EventFeatureSchemaError(
                "each persisted causal field may enter the schema only once"
            )

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.features)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "contract_version": EVENT_FEATURE_SCHEMA_CONTRACT_VERSION,
            "name": self.name,
            "version": self.version,
            "features": [
                {**item.to_primitive(), "definition_id": item.definition_id}
                for item in self.features
            ],
            "selection": "explicit_allowlist_in_declared_order",
            "missing_values": "explicit_null_never_imputed",
            "decimal_encoding": "exact_canonical_decimal_string",
        }

    @property
    def schema_id(self) -> str:
        return configuration_identity(self.to_primitive())

    @classmethod
    def from_primitive(cls, value: Primitive) -> "EventFeatureSchema":
        if not isinstance(value, dict):
            raise EventFeatureSchemaError("feature schema must be a mapping")
        raw_features = value.get("features")
        if value.get(
            "contract_version"
        ) != EVENT_FEATURE_SCHEMA_CONTRACT_VERSION or not isinstance(
            raw_features, list
        ):
            raise EventFeatureSchemaError("unsupported feature schema contract")
        schema = cls(
            name=cast(str, value.get("name")),
            version=cast(str, value.get("version")),
            features=tuple(
                EventFeatureDefinition.from_primitive(item) for item in raw_features
            ),
        )
        if schema.to_primitive() != value:
            raise EventFeatureSchemaError("feature schema fields are inconsistent")
        return schema

    def require_plan_timeframes(self, timeframes: tuple[Timeframe, ...]) -> None:
        """Every declared source timeframe must be one the plan declares."""
        declared = {item.configuration_id for item in timeframes}
        for item in self.features:
            if (
                item.timeframe is not None
                and item.timeframe.configuration_id not in declared
            ):
                raise EventFeatureSchemaError(
                    f"feature {item.name!r} names a timeframe the plan does not declare"
                )

    def values(self, features: PrimitiveMapping) -> tuple[FeatureValue, ...]:
        return tuple(item.value(features) for item in self.features)


__all__ = [
    "EVENT_FEATURE_SCHEMA_CONTRACT_VERSION",
    "FORBIDDEN_FEATURE_NAMES",
    "RESERVED_COLUMN_NAMES",
    "CausalAvailability",
    "EventFeatureDefinition",
    "EventFeatureSchema",
    "FeatureSource",
    "FeatureValue",
    "FeatureValueType",
    "MissingValuePolicy",
    "canonical_decimal",
    "require_causal_feature_name",
]
