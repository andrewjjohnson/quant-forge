"""Deterministic conditional event ML datasets from verified observations (QF-67).

One row is one persisted generated signal of one frozen strategy population.
Rows are assembled from QF-64 coverage receipts: every scheduled decision is
counted, but no-trigger receipts are never expanded into rows, negatives or
rich objects. Features come from an explicit ``EventFeatureSchema``; the target
from a bound ``ForwardReturnBinaryTarget``; partition membership from verified
sources. Nothing here trains, scales, imputes or selects features.

Ordering (documented, stable, traversal independent): decision timestamp (UTC),
then fold index in plan order (the final holdout last), then role (development,
selection, test, holdout), then signal index, then source observation ID.
"""

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.ml.errors import (
    EventDatasetError,
    EventDatasetIntegrityError,
    EventFeatureSchemaError,
    EventHoldoutError,
    EventPartitionError,
    EventPopulationError,
    EventTargetError,
)
from quantforge.ml.features import (
    EventFeatureSchema,
    FeatureValue,
    FeatureValueType,
)
from quantforge.ml.sources import (
    FOLD_ROLES,
    ROLE_ORDER,
    DispositionPolicy,
    EventPopulation,
    EventSourceWindow,
    HoldoutIsolation,
    consumed_holdout_source,
    held_isolation,
    load_study_event_sources,
    reject_non_authoritative,
    universe_candidate,
    workspace_ledger,
)
from quantforge.ml.targets import (
    BoundEventTarget,
    ForwardReturnBinaryTarget,
    TargetLabel,
)
from quantforge.oos import HoldoutEvaluation
from quantforge.prediction.window_coverage import PredictionWindowObservation
from quantforge.validation import PartitionRole, TimestampBoundary, ValidationPlan
from quantforge.validation.errors import ValidationPlanError

EVENT_DATASET_COMPONENT = "quantforge_event_ml_dataset"
EVENT_DATASET_SCHEMA_VERSION = "1"
ROW_ORDERING = (
    "decision_timestamp_utc, fold_index_in_plan_order_final_holdout_last, "
    "partition_role(development,selection,test,final_holdout), signal_index, "
    "source_observation_id"
)
_LAST_FOLD = 1 << 30


@dataclass(frozen=True, slots=True)
class EventRow:
    """One generated signal: causal features, its label and its membership."""

    source_observation_id: str
    source_index: int
    decision_timestamp: datetime
    signal_session: date
    decision_sequence: int
    signal_index: int
    context_id: str | None
    prediction_study_id: str
    direction: str | None
    disposition: str | None
    partition_role: PartitionRole
    fold_id: str | None
    fold_index: int | None
    features: tuple[FeatureValue, ...]
    label: TargetLabel

    def sort_key(self) -> tuple[datetime, int, int, int, str]:
        return (
            self.decision_timestamp,
            _LAST_FOLD if self.fold_index is None else self.fold_index,
            ROLE_ORDER[self.partition_role],
            self.signal_index,
            self.source_observation_id,
        )

    def to_primitive(self, row_index: int) -> PrimitiveMapping:
        """Canonical logical row; its stream hash enters dataset identity."""
        return {
            "row_index": row_index,
            "source_observation_id": self.source_observation_id,
            "source_index": self.source_index,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "signal_session": self.signal_session.isoformat(),
            "decision_sequence": self.decision_sequence,
            "signal_index": self.signal_index,
            "context_id": self.context_id,
            "prediction_study_id": self.prediction_study_id,
            "direction": self.direction,
            "disposition": self.disposition,
            "partition_role": self.partition_role.value,
            "fold_id": self.fold_id,
            "fold_index": self.fold_index,
            "features": cast(list[Primitive], list(self.features)),
            "target": {
                "value": self.label.value,
                "status": self.label.status,
                "source_value": self.label.source_value,
                "outcome_id": self.label.outcome_id,
            },
        }


def logical_rows_sha256(rows: Sequence[EventRow]) -> str:
    """SHA-256 of the canonical JSON lines of the ordered logical rows."""
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        digest.update(
            PrimitiveMappingSnapshot.capture(
                row.to_primitive(index)
            ).canonical_json.encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EventPartitionMembership:
    """Verified chronological membership of one row; never a model input."""

    row_index: int
    role: PartitionRole
    fold_id: str | None
    fold_index: int | None


@dataclass(frozen=True, slots=True)
class EventTargetColumn:
    """The configured target: ``None`` values are unavailable, never negatives."""

    name: str
    values: tuple[bool | None, ...]
    statuses: tuple[str, ...]
    source_values: tuple[str | None, ...]


def label_summary(rows: Sequence[EventRow]) -> PrimitiveMapping:
    """Row and label-availability counts, overall and per (role, fold)."""

    def summarize(items: Sequence[EventRow]) -> PrimitiveMapping:
        unavailable = Counter(
            row.label.status for row in items if row.label.value is None
        )
        return {
            "rows": len(items),
            "positive": sum(row.label.value is True for row in items),
            "negative": sum(row.label.value is False for row in items),
            "unavailable": sum(unavailable.values()),
            "unavailable_by_status": dict(sorted(unavailable.items())),
        }

    groups: dict[tuple[str, str], list[EventRow]] = {}
    for row in rows:
        key = (row.partition_role.value, row.fold_id or "")
        groups.setdefault(key, []).append(row)
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            ROLE_ORDER[PartitionRole(item[0][0])],
            _LAST_FOLD if item[1][0].fold_index is None else item[1][0].fold_index,
        ),
    )
    return {
        "overall": summarize(rows),
        "by_partition": [
            {
                "partition_role": role,
                "fold_id": fold or None,
                **summarize(items),
            }
            for (role, fold), items in ordered
        ],
        "dispositions": dict(
            sorted(Counter(str(row.disposition) for row in rows).items())
        ),
        "directions": dict(sorted(Counter(str(row.direction) for row in rows).items())),
    }


@dataclass(frozen=True, slots=True)
class EventDataset:
    """A verified conditional event ML dataset (built in memory or read back).

    Consumers obtain ``feature_columns`` (ordered model inputs), ``target``
    and ``partition_membership`` without any prediction-window internals.
    """

    dataset_id: str
    scientific: PrimitiveMappingSnapshot
    summaries: PrimitiveMappingSnapshot
    provenance: PrimitiveMappingSnapshot
    feature_schema: EventFeatureSchema
    target_definition: ForwardReturnBinaryTarget
    rows: tuple[EventRow, ...]

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.feature_schema.column_names

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def feature_values(self, name: str) -> tuple[FeatureValue, ...]:
        """Exact persisted values of one feature column (decimals as text)."""
        try:
            index = self.feature_columns.index(name)
        except ValueError as error:
            raise EventFeatureSchemaError(f"{name!r} is not a model feature") from error
        return tuple(row.features[index] for row in self.rows)

    def numeric_feature_rows(self) -> tuple[tuple[float | None, ...], ...]:
        """Explicit binary64 view in ``feature_columns`` order.

        Decimals use ``float(Decimal(text))`` (correctly rounded, half-even);
        integers convert exactly within 2**53; booleans become 1.0/0.0; nulls
        stay ``None``. The stored exact values are unchanged.
        """
        types = tuple(item.value_type for item in self.feature_schema.features)
        return tuple(
            tuple(
                None
                if value is None
                else float(Decimal(cast(str, value)))
                if kind is FeatureValueType.DECIMAL
                else float(cast(int | bool, value))
                for value, kind in zip(row.features, types, strict=True)
            )
            for row in self.rows
        )

    @property
    def target(self) -> EventTargetColumn:
        return EventTargetColumn(
            self.target_definition.name,
            tuple(row.label.value for row in self.rows),
            tuple(row.label.status for row in self.rows),
            tuple(row.label.source_value for row in self.rows),
        )

    @property
    def partition_membership(self) -> tuple[EventPartitionMembership, ...]:
        return tuple(
            EventPartitionMembership(
                index, row.partition_role, row.fold_id, row.fold_index
            )
            for index, row in enumerate(self.rows)
        )

    def rows_for(
        self, role: PartitionRole, *, fold_id: str | None = None
    ) -> tuple[int, ...]:
        """Row indices of one role (and fold); the final holdout has no fold."""
        return tuple(
            index
            for index, row in enumerate(self.rows)
            if row.partition_role is role
            and (fold_id is None or row.fold_id == fold_id)
        )

    def manifest(self) -> PrimitiveMapping:
        return {
            "component": EVENT_DATASET_COMPONENT,
            "schema_version": EVENT_DATASET_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "scientific": self.scientific.to_primitive(),
            "summaries": self.summaries.to_primitive(),
            "provenance": self.provenance.to_primitive(),
            "interpretation": (
                "Rows exist only where the declared strategy configuration "
                "triggered; no-trigger decisions are not negatives. Conditional "
                "on this population only. Not evidence of an edge, robustness "
                "or profitability; holdout rows appear only after explicit "
                "ledger consumption."
            ),
        }


def column_layout(schema: EventFeatureSchema) -> PrimitiveMapping:
    """Versioned physical column layout bound into scientific identity."""
    metadata = [
        ("row_index", "int64", False),
        ("source_observation_id", "string", False),
        ("source_index", "int32", False),
        ("decision_timestamp", "timestamp_us_utc", False),
        ("signal_session", "date32", False),
        ("decision_sequence", "int64", False),
        ("signal_index", "int32", False),
        ("context_id", "string", True),
        ("prediction_study_id", "string", False),
        ("direction", "string", True),
        ("disposition", "string", True),
    ]
    partition = [
        ("partition_role", "string", False),
        ("fold_id", "string", True),
        ("fold_index", "int32", True),
    ]
    target = [
        ("target", "boolean", True),
        ("target_status", "string", False),
        ("target_source_value", "string", True),
        ("target_outcome_id", "string", False),
    ]
    features = [
        (
            item.name,
            "string"
            if item.value_type is FeatureValueType.DECIMAL
            else (
                "int64" if item.value_type is FeatureValueType.INTEGER else "boolean"
            ),
            True,
        )
        for item in schema.features
    ]

    def entries(group: str, items: list[tuple[str, str, bool]]) -> list[Primitive]:
        return [
            {"name": name, "group": group, "arrow_type": kind, "nullable": nullable}
            for name, kind, nullable in items
        ]

    return {
        "layout_version": "1",
        "columns": [
            *entries("metadata", metadata),
            *entries("partition", partition),
            *entries("feature", features),
            *entries("target", target),
        ],
    }


def _utc(value: datetime) -> datetime:
    if value.utcoffset() != timedelta(0):
        raise EventDatasetIntegrityError("decision timestamps must be UTC")
    return value.astimezone(UTC)


def _observation_id(
    scientific_window_id: str,
    observation: PredictionWindowObservation,
    signal: PrimitiveMapping,
    row: PrimitiveMapping | None,
) -> str:
    return configuration_identity(
        {
            "component": "quantforge_event_observation",
            "version": "1",
            "scientific_window_id": scientific_window_id,
            "decision_sequence": observation.sequence,
            "signal_index": observation.signal_index,
            "decision_timestamp": _utc(observation.decision_timestamp).isoformat(),
            "context_id": observation.context_id,
            "prediction_study_id": observation.prediction_study_id,
            "signal": signal,
            "row_id": None if row is None else row.get("row_id"),
        }
    )


def _source_rows(
    index: int,
    source: EventSourceWindow,
    *,
    plan: ValidationPlan,
    population: EventPopulation,
    schema: EventFeatureSchema,
    target: BoundEventTarget,
    isolation: HoldoutIsolation,
) -> tuple[list[EventRow], PrimitiveMapping]:
    """Rows of one verified window; no-trigger receipts are only counted."""
    membership = plan.prediction_membership
    assert membership is not None
    if source.candidate != population.candidate:
        raise EventPopulationError("source window belongs to another candidate")
    population.require_window(source.reader)
    retained = source.membership.to_primitive().get("retained_decision_count")
    if retained is not None and retained != source.reader.decision_count:
        raise EventPartitionError("window schedule differs from its QF-8 membership")
    statuses: Counter[str] = Counter()
    rows: list[EventRow] = []
    generated = excluded = 0
    window_id = source.scientific_window_id  # Hash the shared scope once.
    identity = population.signal_identity()
    for receipt in source.reader.iterate_decision_receipts():
        statuses[receipt.status] += 1
        decision = _utc(receipt.decision_timestamp)
        if not source.window.interval.contains(TimestampBoundary(decision)):
            raise EventPartitionError("scheduled decision escapes its plan window")
        if receipt.decision is None:
            continue  # A no-trigger (or bare) receipt is coverage, never a row.
        for observation in receipt.observations():
            generated += 1
            signal = observation.signal()
            values = population.require_signal(signal, identity)
            if not population.includes(values.get("disposition")):
                excluded += 1
                continue
            row = observation.row()
            prediction = cast(PrimitiveMapping, signal["prediction"])
            features = signal.get("features")
            if not isinstance(features, dict):
                raise EventDatasetIntegrityError("generated signal has no features")
            if row is not None and (
                row.get("features") != features or row.get("prediction") != prediction
            ):
                raise EventDatasetIntegrityError(
                    "labeled row differs from its generated signal"
                )
            stated = values.get("decision_timestamp")
            if stated is not None and stated != decision.isoformat():
                raise EventDatasetIntegrityError(
                    "signal decision timestamp differs from its schedule"
                )
            try:
                session = date.fromisoformat(cast(str, prediction["signal_session"]))
                expected_session = membership.session_for(decision)
            except (KeyError, TypeError, ValueError, ValidationPlanError) as error:
                raise EventPartitionError(
                    "observation session is outside the plan schedule"
                ) from error
            if session != expected_session:
                raise EventPartitionError("observation session differs from the plan")
            isolation.require(source.role, decision, session)
            direction, disposition = values.get("direction"), values.get("disposition")
            rows.append(
                EventRow(
                    source_observation_id=_observation_id(
                        window_id, observation, signal, row
                    ),
                    source_index=index,
                    decision_timestamp=decision,
                    signal_session=session,
                    decision_sequence=observation.sequence,
                    signal_index=observation.signal_index,
                    context_id=observation.context_id,
                    prediction_study_id=observation.prediction_study_id,
                    direction=None if direction is None else str(direction),
                    disposition=None if disposition is None else str(disposition),
                    partition_role=source.role,
                    fold_id=source.fold_id,
                    fold_index=source.fold_index,
                    features=schema.values(cast(PrimitiveMapping, features)),
                    label=target.label(row, decision),
                )
            )
    schedule = source.reader.evidence.schedule.decision_timestamps
    coverage: PrimitiveMapping = {
        "decision_statuses": dict(sorted(statuses.items())),
        "generated_signals": generated,
        "excluded_by_disposition": excluded,
        "rows": len(rows),
        "first_scheduled_decision": _utc(schedule[0]).isoformat() if schedule else None,
        "last_scheduled_decision": _utc(schedule[-1]).isoformat() if schedule else None,
    }
    return rows, coverage


def assemble_event_dataset(
    sources: Sequence[EventSourceWindow],
    *,
    plan: ValidationPlan,
    population: EventPopulation,
    feature_schema: EventFeatureSchema,
    target: ForwardReturnBinaryTarget,
    isolation: HoldoutIsolation,
    exclusions: Sequence[PrimitiveMappingSnapshot] = (),
    study_id: str | None = None,
) -> EventDataset:
    """Assemble verified sources into one deterministic dataset (in memory)."""
    for item in cast(Sequence[object], sources):
        reject_non_authoritative(item)
        if type(item) is not EventSourceWindow:
            raise EventDatasetError(
                "event datasets accept only verified EventSourceWindow sources"
            )
    if type(cast(object, feature_schema)) is not EventFeatureSchema:
        raise EventFeatureSchemaError("an explicit EventFeatureSchema is required")
    if type(cast(object, target)) is not ForwardReturnBinaryTarget:
        raise EventTargetError("an explicit supported target is required")
    if not sources:
        raise EventPartitionError("event datasets require at least one source window")
    feature_schema.require_plan_timeframes(plan.environment.timeframes)
    bound = target.bind(population.definition, plan)
    ordered = sorted(
        sources,
        key=lambda item: (
            _LAST_FOLD if item.fold_index is None else item.fold_index,
            ROLE_ORDER[item.role],
        ),
    )
    keys = [(item.fold_id, item.role) for item in ordered]
    windows = [item.scientific_window_id for item in ordered]
    if len(set(keys)) != len(keys) or len(set(windows)) != len(windows):
        raise EventPartitionError("duplicate source windows or partition roles")
    rows: list[EventRow] = []
    source_records: list[Primitive] = []
    for index, source in enumerate(ordered):
        if source.role is PartitionRole.FINAL_HOLDOUT:
            expected = plan.final_holdout.window
        else:
            fold = plan.folds[cast(int, source.fold_index)]
            if fold.fold_id != source.fold_id:
                raise EventPartitionError("source fold differs from the plan")
            expected = (
                fold.development
                if source.role is PartitionRole.DEVELOPMENT
                else fold.selection
                if source.role is PartitionRole.SELECTION
                else fold.test
            )
        if expected is None or expected.window_id != source.window.window_id:
            raise EventPartitionError("source role/window differs from the plan")
        window_rows, coverage = _source_rows(
            index,
            source,
            plan=plan,
            population=population,
            schema=feature_schema,
            target=bound,
            isolation=isolation,
        )
        rows.extend(window_rows)
        source_records.append(
            {"source_index": index, **source.to_primitive(), **coverage}
        )
    identities = [row.source_observation_id for row in rows]
    if len(set(identities)) != len(identities):
        raise EventPartitionError("duplicate source observation IDs")
    events = [(row.fold_id, row.decision_timestamp, row.signal_index) for row in rows]
    if len(set(events)) != len(events):
        raise EventPartitionError(
            "one fold holds the same decision event in more than one role"
        )
    rows.sort(key=EventRow.sort_key)
    roles = sorted({item.role for item in ordered}, key=ROLE_ORDER.__getitem__)
    scientific: PrimitiveMapping = {
        "component": EVENT_DATASET_COMPONENT,
        "schema_version": EVENT_DATASET_SCHEMA_VERSION,
        "population": population.to_primitive(),
        "population_id": population.population_id,
        "feature_schema": feature_schema.to_primitive(),
        "feature_schema_id": feature_schema.schema_id,
        "target": bound.to_primitive(),
        "target_configuration_id": bound.configuration_id,
        "partition_plan": {
            "plan_id": plan.plan_id,
            "final_holdout_id": plan.final_holdout.holdout_id,
            "roles": [role.value for role in roles],
            "sources": source_records,
            "excluded_sources": [item.to_primitive() for item in exclusions],
            "features_exclude_partition_metadata": True,
        },
        "columns": column_layout(feature_schema),
        "rows": {
            "row_count": len(rows),
            "ordering": ROW_ORDERING,
            "logical_rows_sha256": logical_rows_sha256(rows),
        },
    }
    provenance: PrimitiveMapping = {
        "study_id": study_id,
        "sources": [item.physical.to_primitive() for item in ordered],
        "note": "physical references; outside scientific identity",
    }
    return EventDataset(
        configuration_identity(scientific),
        PrimitiveMappingSnapshot.capture(scientific),
        PrimitiveMappingSnapshot.capture(label_summary(rows)),
        PrimitiveMappingSnapshot.capture(provenance),
        feature_schema,
        target,
        tuple(rows),
    )


def build_event_dataset(
    *,
    plan: ValidationPlan,
    study_path: Path,
    workspace: Path,
    combination_id: str,
    feature_schema: EventFeatureSchema,
    target: ForwardReturnBinaryTarget,
    disposition_policy: DispositionPolicy = DispositionPolicy.ACCEPTED_ONLY,
    roles: frozenset[PartitionRole] = frozenset(FOLD_ROLES),
    final_holdout: HoldoutEvaluation | None = None,
) -> EventDataset:
    """Build one population's dataset from a verified QF-39 prediction study.

    ``roles`` selects fold roles. Final-holdout rows are added only when
    ``final_holdout`` names an evaluation that the workspace's permanent
    ledger already records as consumed; building never consumes a holdout.
    """
    reject_non_authoritative(study_path, label="study path")
    reject_non_authoritative(final_holdout, label="holdout evaluation")
    if PartitionRole.FINAL_HOLDOUT in roles:
        raise EventHoldoutError(
            "final-holdout rows require final_holdout= with a consumed evaluation"
        )
    ledger = workspace_ledger(workspace)
    holdout = None
    if final_holdout is not None:
        if type(cast(object, final_holdout)) is not HoldoutEvaluation:
            raise EventHoldoutError("final_holdout must be a HoldoutEvaluation")
        with canonical_preparation():
            holdout = consumed_holdout_source(
                ledger,
                final_holdout,
                plan=plan,
                study_id=study_path.name,
                candidate=universe_candidate(final_holdout.source, combination_id),
            )
    metadata = plan.environment.outcome_dataset.market_data_metadata
    if metadata is None:
        raise EventPopulationError("event datasets require canonical market metadata")
    # One QF-65 load session: every window's canonical lineage is authenticated
    # once and reused only for content-identical objects (operational only).
    with (
        canonical_preparation(),
        held_isolation(ledger, plan, metadata.canonical_symbol) as isolation,
    ):
        study = load_study_event_sources(
            plan, study_path, combination_id=combination_id, roles=roles
        )
        candidate = universe_candidate(study.source, combination_id)
        population = EventPopulation.capture(plan, candidate, disposition_policy)
        sources = (*study.windows, *(() if holdout is None else (holdout,)))
        return assemble_event_dataset(
            sources,
            plan=plan,
            population=population,
            feature_schema=feature_schema,
            target=target,
            isolation=isolation,
            exclusions=study.exclusions,
            study_id=study.source.study_id,
        )


__all__ = [
    "EVENT_DATASET_COMPONENT",
    "EVENT_DATASET_SCHEMA_VERSION",
    "ROW_ORDERING",
    "EventDataset",
    "EventPartitionMembership",
    "EventRow",
    "EventTargetColumn",
    "assemble_event_dataset",
    "build_event_dataset",
    "column_layout",
    "label_summary",
    "logical_rows_sha256",
]
