"""Typed QF-7 signal snapshots, schema fields, and dataset results."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    PrimitiveScalar,
    decimal_to_primitive,
)
from quantforge.prediction.errors import (
    InvalidPredictionOutputError,
    SignalFeaturePersistenceError,
)
from quantforge.prediction.models import PredictionDirection, PredictionMarketData

FEATURE_SCHEMA_VERSION = "1"
OUTCOME_SCHEMA_VERSION = "1"
FEATURE_DATASET_ENGINE_VERSION = "8"


class SignalDisposition(StrEnum):
    """Stable classification of one logical signal opportunity."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    OVERLAPPING = "overlapping"
    BLOCKED = "blocked"


class SchemaFieldCategory(StrEnum):
    """Temporal and semantic category used by the schema manifest."""

    IDENTITY = "identity"
    CONTEMPORANEOUS_FEATURE = "contemporaneous_feature"
    FUTURE_OUTCOME = "future_outcome"
    DISPOSITION = "disposition"


type FeatureScalar = Decimal | PrimitiveScalar


@dataclass(frozen=True, slots=True)
class SignalFeatureValue:
    """One typed causal feature value; ``None`` means explicitly unavailable."""

    name: str
    value: FeatureScalar

    def __post_init__(self) -> None:
        if not self.name:
            raise InvalidPredictionOutputError("signal feature names are required")
        if isinstance(self.value, Decimal) and not self.value.is_finite():
            raise InvalidPredictionOutputError(
                "signal feature decimals must be finite or unavailable"
            )

    def primitive_value(self) -> PrimitiveScalar:
        if isinstance(self.value, Decimal):
            return decimal_to_primitive(self.value)
        return self.value


@dataclass(frozen=True, slots=True)
class SchemaField:
    """Machine-readable definition of one flattened analytics column."""

    name: str
    category: SchemaFieldCategory
    data_type: str
    unit: str
    nullable: bool
    calculation_or_source: str
    temporal_availability: str

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.name,
                self.data_type,
                self.unit,
                self.calculation_or_source,
                self.temporal_availability,
            )
        ):
            raise InvalidPredictionOutputError(
                "schema fields require names, types, units, sources, and timing"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "calculation_or_source": self.calculation_or_source,
            "category": self.category.value,
            "data_type": self.data_type,
            "field_name": self.name,
            "nullable": self.nullable,
            "temporal_availability": self.temporal_availability,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class SignalFeatureSchema:
    """Versioned deterministic flattened feature/outcome schema."""

    feature_schema_version: str
    outcome_schema_version: str
    fields: tuple[SchemaField, ...]

    def __post_init__(self) -> None:
        names = tuple(field.name for field in self.fields)
        if (
            not self.feature_schema_version
            or not self.outcome_schema_version
            or not names
            or len(names) != len(set(names))
        ):
            raise InvalidPredictionOutputError(
                "signal-feature schemas require versions and unique fields"
            )

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "feature_schema_version": self.feature_schema_version,
            "fields": [field.to_primitive() for field in self.fields],
            "outcome_schema_version": self.outcome_schema_version,
        }


@dataclass(frozen=True, slots=True)
class SignalFeatureCandidate:
    """One fixed QF-11 prediction record before any future outcome is attached."""

    symbol: str
    signal_session: date
    strategy_id: str
    strategy_implementation_version: str
    strategy_configuration_id: str
    source_rule_id: str
    source_rule_implementation_version: str
    source_rule_configuration_id: str
    strategy_parameters: PrimitiveMappingSnapshot
    disposition: SignalDisposition
    reason_codes: tuple[str, ...]
    explanation: str | None
    direction: PredictionDirection | None
    selected_rule_reason: str | None
    matched_rule_reasons: tuple[str, ...]
    strategy_features: tuple[SignalFeatureValue, ...]
    contextual_features: tuple[SignalFeatureValue, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.symbol,
                self.strategy_id,
                self.strategy_implementation_version,
                self.strategy_configuration_id,
                self.source_rule_id,
                self.source_rule_implementation_version,
                self.source_rule_configuration_id,
            )
        ):
            raise InvalidPredictionOutputError(
                "signal-feature candidates require complete rule identity"
            )
        if not self.reason_codes or any(not code for code in self.reason_codes):
            raise InvalidPredictionOutputError(
                "signal-feature candidates require stable reason codes"
            )
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise InvalidPredictionOutputError(
                "signal-feature reason codes must not contain duplicates"
            )
        if any(not reason for reason in self.matched_rule_reasons):
            raise InvalidPredictionOutputError("matched rule reasons must be nonempty")
        if self.disposition is SignalDisposition.ACCEPTED and (
            self.direction is None or not self.selected_rule_reason
        ):
            raise InvalidPredictionOutputError(
                "accepted signal-feature candidates require a direction and selected "
                "rule reason"
            )
        strategy_names = tuple(feature.name for feature in self.strategy_features)
        contextual_names = tuple(feature.name for feature in self.contextual_features)
        names = (*strategy_names, *contextual_names)
        if (
            strategy_names != tuple(sorted(strategy_names))
            or contextual_names != tuple(sorted(contextual_names))
            or len(names) != len(set(names))
        ):
            raise InvalidPredictionOutputError(
                "strategy and contextual features must be sorted and globally unique"
            )

    def parameters_primitive(self) -> PrimitiveMapping:
        return self.strategy_parameters.to_primitive()

    def features_primitive(self) -> PrimitiveMapping:
        return {
            feature.name: feature.primitive_value()
            for feature in (*self.strategy_features, *self.contextual_features)
        }

    def prediction_primitive(self) -> PrimitiveMapping:
        return {
            "direction": None if self.direction is None else self.direction.value,
            "disposition": self.disposition.value,
            "disposition_explanation": self.explanation,
            "matched_rule_reasons": list(self.matched_rule_reasons),
            "reason_codes": list(self.reason_codes),
            "selected_rule_reason": self.selected_rule_reason,
            "source_rule_configuration_id": self.source_rule_configuration_id,
            "source_rule_id": self.source_rule_id,
            "source_rule_implementation_version": (
                self.source_rule_implementation_version
            ),
        }


@dataclass(frozen=True, slots=True)
class SignalFeatureCandidateOutput:
    """QF-11 rule output containing one record per logical candidate session."""

    strategy_id: str
    strategy_configuration_id: str
    dataset_id: str
    signals: tuple[SignalFeatureCandidate, ...]
    contract_version: str = "1"


@dataclass(frozen=True, slots=True)
class SignalFeatureRow:
    """One flattened identity + causal snapshot + future outcome row."""

    row_id: str
    candidate_id: str
    signal_session: date
    disposition: SignalDisposition
    primitive_snapshot: PrimitiveMappingSnapshot

    @classmethod
    def capture(cls, values: PrimitiveMapping) -> "SignalFeatureRow":
        row_id = values.get("row_id")
        candidate_id = values.get("candidate_id")
        signal_session = values.get("signal_session")
        disposition = values.get("signal_disposition")
        if not all(
            isinstance(value, str) and value
            for value in (row_id, candidate_id, signal_session, disposition)
        ):
            raise SignalFeaturePersistenceError(
                "signal-feature rows require identity, session, and disposition"
            )
        try:
            return cls(
                cast(str, row_id),
                cast(str, candidate_id),
                date.fromisoformat(cast(str, signal_session)),
                SignalDisposition(cast(str, disposition)),
                PrimitiveMappingSnapshot.capture(values),
            )
        except ValueError as error:
            raise SignalFeaturePersistenceError(
                "signal-feature row session or disposition is invalid"
            ) from error

    def to_primitive(self) -> PrimitiveMapping:
        return self.primitive_snapshot.to_primitive()


@dataclass(frozen=True, slots=True)
class SignalFeatureSummary:
    """Deterministic counts that distinguish every supported disposition."""

    candidate_count: int
    accepted_count: int
    rejected_count: int
    blocked_count: int
    overlapping_count: int

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "accepted_count": self.accepted_count,
            "blocked_count": self.blocked_count,
            "candidate_count": self.candidate_count,
            "overlapping_count": self.overlapping_count,
            "rejected_count": self.rejected_count,
        }


@dataclass(frozen=True, slots=True)
class SignalFeatureDatasetResult:
    """Completed deterministic QF-7 dataset and its QF-3/QF-11 provenance."""

    dataset_id: str
    engine_version: str
    market_data: PredictionMarketData
    configuration_snapshot: PrimitiveMappingSnapshot
    schema: SignalFeatureSchema
    rows: tuple[SignalFeatureRow, ...]
    summary: SignalFeatureSummary
    prediction_study_ids: tuple[str, ...]
    limitations: tuple[str, ...]

    @property
    def configuration(self) -> PrimitiveMapping:
        return self.configuration_snapshot.to_primitive()

    def manifest_primitive(self) -> PrimitiveMapping:
        return {
            "component": "quantforge_signal_feature_dataset",
            "configuration": self.configuration,
            "dataset_id": self.dataset_id,
            "engine_version": self.engine_version,
            "feature_outcome_boundary": (
                "candidate dispositions and causal features are fixed before any "
                "QF-11 outcome labeler is invoked"
            ),
            "limitations": list(self.limitations),
            "market_data": self.market_data.to_primitive(),
            "prediction_study_ids": list(self.prediction_study_ids),
            "record_counts": self.summary.to_primitive(),
            "status": "complete",
        }

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "manifest": self.manifest_primitive(),
            "rows": cast(list[Primitive], [row.to_primitive() for row in self.rows]),
            "schema": self.schema.to_primitive(),
            "summary": self.summary.to_primitive(),
        }


def summarize_dispositions(
    rows: tuple[SignalFeatureRow, ...],
) -> SignalFeatureSummary:
    """Count logical candidates exactly once by their fixed disposition."""
    return SignalFeatureSummary(
        candidate_count=len(rows),
        accepted_count=sum(
            row.disposition is SignalDisposition.ACCEPTED for row in rows
        ),
        rejected_count=sum(
            row.disposition is SignalDisposition.REJECTED for row in rows
        ),
        blocked_count=sum(row.disposition is SignalDisposition.BLOCKED for row in rows),
        overlapping_count=sum(
            row.disposition is SignalDisposition.OVERLAPPING for row in rows
        ),
    )
