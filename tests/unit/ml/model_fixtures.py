"""QF-68 fixtures: synthetic QF-67 datasets bound to a real QF-8 timestamp plan.

The plan is the QF-48 timestamp fixture (5m primary bars, 30m outcome, 35m
purged label horizon). Each fold occupies one session, in New York time:

- fold 0 (session 4) and fold 1 (session 5): development 10:00-11:55,
  selection 12:00-12:55, walk-forward test 13:00-13:55;
- final holdout: session 5, 14:00-15:55.

Events sit only on QF-8 retained membership: development at 10:00-11:20 (the
reach 11:20 + 35m ends before 12:00), selection at 12:00-12:20 and fold-1 test
at 13:00-13:20. Every dataset is exported and re-read through QF-67's offline
validator, so the model layer consumes exactly what a QF-67 reader returns.
These are contract fixtures, never research evidence.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.ml import (
    ROW_ORDERING,
    BoundEventTarget,
    EventDataset,
    EventFeatureDefinition,
    EventFeatureSchema,
    EventRow,
    FeatureValueType,
    ForwardReturnBinaryTarget,
    MissingValuePolicy,
    TargetLabel,
    export_event_dataset,
)
from quantforge.ml.dataset import column_layout, label_summary, logical_rows_sha256
from quantforge.ml.features import FeatureValue
from quantforge.ml.modeling import ModelConfiguration
from quantforge.ml.sources import ROLE_ORDER
from quantforge.validation import PartitionRole, ValidationPlan
from tests.unit.helpers import SESSIONS
from tests.unit.walk_forward.timestamp_fixtures import instant, timestamp_fixture

FOLD_SESSION = (4, 5)
HOLDOUT_SESSION = 5
TRAINING_CLOCKS = tuple(f"{10 + m // 60}:{m % 60:02d}" for m in range(0, 81, 5))
SELECTION_CLOCKS = ("12:00", "12:05", "12:10", "12:15", "12:20")
TEST_CLOCKS = ("13:00", "13:05", "13:10", "13:15", "13:20")
HOLDOUT_CLOCKS = ("14:00", "14:30", "15:00", "15:20")

SCHEMA = EventFeatureSchema(
    "qf68_fixture_features",
    "1",
    (
        EventFeatureDefinition(
            "momentum",
            "momentum",
            FeatureValueType.DECIMAL,
            MissingValuePolicy.PRESERVE_NULL,
            "ratio",
            "synthetic causal momentum",
        ),
        EventFeatureDefinition(
            "range_width",
            "range_width",
            FeatureValueType.DECIMAL,
            MissingValuePolicy.PRESERVE_NULL,
            "price",
            "synthetic causal range",
        ),
        EventFeatureDefinition(
            "bar_count",
            "bar_count",
            FeatureValueType.INTEGER,
            MissingValuePolicy.PRESERVE_NULL,
            "bars",
            "synthetic causal count",
        ),
        EventFeatureDefinition(
            "gap_up",
            "gap_up",
            FeatureValueType.BOOLEAN,
            MissingValuePolicy.PRESERVE_NULL,
            "flag",
            "synthetic causal flag",
        ),
    ),
)
REVERSED_SCHEMA = EventFeatureSchema(
    SCHEMA.name, SCHEMA.version, tuple(reversed(SCHEMA.features))
)


def configuration(**options: object) -> ModelConfiguration:
    """Small-sample fixture configuration (explicit minimums)."""
    values: dict[str, object] = {
        "name": "qf68_fixture_logistic",
        "version": "1",
        "minimum_training_observations": 4,
        **options,
    }
    return ModelConfiguration(**values)  # pyright: ignore[reportArgumentType]


def plan_fixture(root: Path, *, embargo: timedelta = timedelta(0)) -> ValidationPlan:
    config, _ = timestamp_fixture(root, embargo=embargo)
    return config.plan


@dataclass(frozen=True)
class Event:
    """One synthetic generated signal and its persisted label."""

    role: PartitionRole
    fold: int | None
    clock: str
    features: tuple[FeatureValue, ...]
    label: bool | None
    signal: int = 0
    status: str = "available"

    @property
    def session(self) -> int:
        return HOLDOUT_SESSION if self.fold is None else FOLD_SESSION[self.fold]


def features(index: int) -> tuple[FeatureValue, ...]:
    """Deterministic varied causal values (canonical decimal text)."""
    momentum = Decimal(index % 7 - 3) / 10
    width = Decimal(1 + (index * 3) % 5) / 4
    return (
        decimal_to_primitive(momentum),
        decimal_to_primitive(width),
        index % 4 + 1,
        index % 3 == 0,
    )


def label_for(index: int) -> bool:
    """Mostly follows momentum, with deterministic exceptions (both classes)."""
    return (index % 7 - 3 > 0) != (index % 5 == 0)


def events(
    role: PartitionRole,
    fold: int | None,
    clocks: Sequence[str],
    *,
    signals: int = 2,
    offset: int = 0,
) -> list[Event]:
    output: list[Event] = []
    for position, clock in enumerate(clocks):
        for signal in range(signals):
            index = offset + position * signals + signal
            output.append(
                Event(role, fold, clock, features(index), label_for(index), signal)
            )
    return output


def standard_events(
    fold: int = 0, *, holdout: bool = False, selection: bool = True
) -> list[Event]:
    """Training, selection and test events of one fold (plus the holdout)."""
    output = [
        *events(PartitionRole.DEVELOPMENT, fold, TRAINING_CLOCKS),
        *(
            events(PartitionRole.SELECTION, fold, SELECTION_CLOCKS, offset=100)
            if selection
            else []
        ),
        *events(PartitionRole.WALK_FORWARD_TEST, fold, TEST_CLOCKS, offset=200),
    ]
    if holdout:
        output.extend(
            events(PartitionRole.FINAL_HOLDOUT, None, HOLDOUT_CLOCKS, offset=300)
        )
    return output


def change(
    items: Iterable[Event], role: PartitionRole, **values: object
) -> list[Event]:
    """Replace fields of every event of one role."""
    return [
        replace(item, **values) if item.role is role else item  # pyright: ignore[reportArgumentType]
        for item in items
    ]


def _label(item: Event, outcome: str, threshold: Decimal) -> TargetLabel:
    if item.label is None:
        status = "session_overflow" if item.status == "available" else item.status
        return TargetLabel(None, status, None, outcome)
    value = threshold + (Decimal("0.01") if item.label else Decimal("-0.01"))
    return TargetLabel(item.label, "available", decimal_to_primitive(value), outcome)


def build_dataset(
    plan: ValidationPlan,
    items: Sequence[Event],
    *,
    schema: EventFeatureSchema = SCHEMA,
    population: str = "qf68-population-a",
    excluded_sources: Sequence[PrimitiveMapping] = (),
    window_overrides: dict[tuple[int | None, PartitionRole], str] | None = None,
    consumption_request_id: str = "synthetic-unconsumed-request",
    target: ForwardReturnBinaryTarget | None = None,
    outcome_reach: timedelta = timedelta(minutes=35),
) -> EventDataset:
    """An in-memory QF-67 dataset with the documented scientific layout."""
    target = target or ForwardReturnBinaryTarget()
    bound = BoundEventTarget(
        target,
        outcome_configuration_id="qf68-fixture-outcome",
        evaluator_configuration_id="qf68-fixture-evaluator",
        outcome_reach=outcome_reach,
    )
    keys = sorted(
        {(item.fold, item.role) for item in items},
        key=lambda key: (1 << 30 if key[0] is None else key[0], ROLE_ORDER[key[1]]),
    )
    overrides = window_overrides or {}
    sources: list[Primitive] = []
    for index, (fold, role) in enumerate(keys):
        if fold is None:
            window = plan.final_holdout.window
            fold_id = None
            membership: PrimitiveMapping = {
                "membership_selection_id": "synthetic-holdout-membership",
                "purge_result_id": None,
                "retained_decision_count": 24,
                "holdout_id": plan.final_holdout.holdout_id,
                "consumption_request_id": consumption_request_id,
                "tail_policy": "outcome_horizon_remains_inside_holdout",
            }
        else:
            plan_fold = plan.folds[fold]
            window = {
                PartitionRole.DEVELOPMENT: plan_fold.development,
                PartitionRole.SELECTION: plan_fold.selection,
                PartitionRole.WALK_FORWARD_TEST: plan_fold.test,
            }[role]
            assert window is not None
            fold_id = plan_fold.fold_id
            membership = {
                "membership_selection_id": f"synthetic-{fold}-{role.value}",
                "purge_result_id": None,
                "retained_decision_count": 24,
            }
        sources.append(
            {
                "source_index": index,
                "role": role.value,
                "fold_id": fold_id,
                "fold_index": fold,
                "qf8_window_id": overrides.get((fold, role), window.window_id),
                "membership": membership,
                "candidate_combination_id": population,
                "scientific_window_id": configuration_identity(
                    {"fixture": "qf68", "fold": fold, "role": role.value}
                ),
                "schedule_id": "synthetic-schedule",
                "scheduled_decisions": 24,
            }
        )
    rows: list[EventRow] = []
    for item in items:
        stamp = instant(item.session, item.clock)
        identity = configuration_identity(
            {
                "fixture": "qf68_event",
                "population": population,
                "role": item.role.value,
                "fold": item.fold,
                "clock": item.clock,
                "signal": item.signal,
            }
        )
        fold_id = None if item.fold is None else plan.folds[item.fold].fold_id
        rows.append(
            EventRow(
                source_observation_id=identity,
                source_index=keys.index((item.fold, item.role)),
                decision_timestamp=stamp,
                signal_session=SESSIONS[item.session],
                decision_sequence=int((stamp.timestamp() // 300) % 100_000),
                signal_index=item.signal,
                context_id=None,
                prediction_study_id="synthetic-study",
                direction="up",
                disposition="accepted",
                partition_role=item.role,
                fold_id=fold_id,
                fold_index=item.fold,
                features=item.features,
                label=_label(item, f"outcome-{identity[:16]}", target.threshold),
            )
        )
    rows.sort(key=EventRow.sort_key)
    population_record: PrimitiveMapping = {
        "contract_version": "1",
        "fixture": "qf68 synthetic population",
        "plan_id": plan.plan_id,
        "candidate": {"combination_id": population},
    }
    roles = sorted({role for _, role in keys}, key=ROLE_ORDER.__getitem__)
    scientific: PrimitiveMapping = {
        "component": "quantforge_event_ml_dataset",
        "schema_version": "1",
        "population": population_record,
        "population_id": configuration_identity(population_record),
        "feature_schema": schema.to_primitive(),
        "feature_schema_id": schema.schema_id,
        "target": bound.to_primitive(),
        "target_configuration_id": bound.configuration_id,
        "partition_plan": {
            "plan_id": plan.plan_id,
            "final_holdout_id": plan.final_holdout.holdout_id,
            "roles": [role.value for role in roles],
            "sources": sources,
            "excluded_sources": list(excluded_sources),
            "features_exclude_partition_metadata": True,
        },
        "columns": column_layout(schema),
        "rows": {
            "row_count": len(rows),
            "ordering": ROW_ORDERING,
            "logical_rows_sha256": logical_rows_sha256(rows),
        },
    }
    return EventDataset(
        configuration_identity(scientific),
        PrimitiveMappingSnapshot.capture(scientific),
        PrimitiveMappingSnapshot.capture(label_summary(rows)),
        PrimitiveMappingSnapshot.capture({"fixture": "qf68 synthetic dataset"}),
        schema,
        target,
        tuple(rows),
    )


def write_dataset(
    plan: ValidationPlan, items: Sequence[Event], root: Path, **options: object
) -> Path:
    """Export through QF-67 (which validates before publishing)."""
    dataset = build_dataset(plan, items, **options)  # pyright: ignore[reportArgumentType]
    return export_event_dataset(dataset, root)


def session_date(fold: int | None) -> date:
    return SESSIONS[HOLDOUT_SESSION if fold is None else FOLD_SESSION[fold]]
