"""QF-8 fold chronology and QF-67 dataset binding for event models (QF-68).

A model is bound to exactly one QF-8 fold of the plan that built its dataset.
Its training rows are that fold's ``development_training`` rows only; the
fold's selection, walk-forward test and the plan's final holdout are scored,
never fitted. Roles are never reassigned and there is no random split.

Training labels must be available before the fold's first protected instant,
using the existing QF-8 purge rule on the plan's label horizon (the maximum
configured outcome reach, which covers the dataset target's reach) plus the
embargo::

    decision + label_horizon + embargo < protected_start   (equality is leakage)

``protected_start`` is the start of the fold's selection window, or of its
test window when the fold declares no selection window. The same reach must
also end before the final holdout. Timestamp ordering alone is insufficient,
so a violating training row fails closed (QF-67 builds would have purged it).
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    configuration_identity,
)
from quantforge.ml.dataset import EventDataset, logical_rows_sha256
from quantforge.ml.modeling.errors import (
    ModelArtifactIntegrityError,
    ModelBindingError,
    ModelLeakageError,
)
from quantforge.validation import (
    PartitionRole,
    TimestampBoundary,
    ValidationPlan,
    ValidationWindow,
)

_MICROSECOND = timedelta(microseconds=1)


def parse_utc_instant(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ModelArtifactIntegrityError(f"{label} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ModelArtifactIntegrityError(f"{label} is not a timestamp") from error
    if parsed.utcoffset() != timedelta(0) or parsed.isoformat() != value:
        raise ModelArtifactIntegrityError(f"{label} must be canonical UTC")
    return parsed.astimezone(UTC)


def parse_microseconds(value: object, label: str) -> timedelta:
    if type(value) is not int or value < 0:
        raise ModelArtifactIntegrityError(f"{label} must be whole microseconds")
    return timedelta(microseconds=value)


@dataclass(frozen=True, slots=True)
class WindowBounds:
    """One plan window's closed UTC interval and QF-8 identity."""

    role: PartitionRole
    window_id: str
    start: datetime
    end: datetime

    @classmethod
    def from_window(cls, window: ValidationWindow) -> "WindowBounds":
        start, end = window.interval.start, window.interval.end
        if not isinstance(start, TimestampBoundary) or not isinstance(
            end, TimestampBoundary
        ):
            raise ModelBindingError("event models require timestamp plan windows")
        return cls(window.role, window.window_id, start.timestamp, end.timestamp)

    def contains(self, instant: datetime) -> bool:
        return self.start <= instant <= self.end

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "role": self.role.value,
            "window_id": self.window_id,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "bounds": "closed_inclusive_utc",
        }

    @classmethod
    def from_primitive(cls, value: Primitive) -> "WindowBounds":
        if not isinstance(value, dict):
            raise ModelArtifactIntegrityError("window bounds must be a mapping")
        try:
            bounds = cls(
                PartitionRole(cast(str, value["role"])),
                cast(str, value["window_id"]),
                parse_utc_instant(value["start"], "window start"),
                parse_utc_instant(value["end"], "window end"),
            )
        except (KeyError, ValueError) as error:
            if isinstance(error, ModelArtifactIntegrityError):
                raise
            raise ModelArtifactIntegrityError("invalid window bounds") from error
        if bounds.to_primitive() != value or bounds.start > bounds.end:
            raise ModelArtifactIntegrityError("window bounds are inconsistent")
        return bounds


@dataclass(frozen=True, slots=True)
class FoldChronology:
    """The plan-derived boundaries one model is bound to (persisted)."""

    plan_id: str
    fold_id: str
    fold_index: int
    development: WindowBounds
    selection: WindowBounds | None
    test: WindowBounds
    holdout: WindowBounds
    holdout_id: str
    label_horizon: timedelta
    embargo: timedelta
    outcome_reach: timedelta

    def __post_init__(self) -> None:
        roles = (
            (self.development, PartitionRole.DEVELOPMENT),
            (self.test, PartitionRole.WALK_FORWARD_TEST),
            (self.holdout, PartitionRole.FINAL_HOLDOUT),
            *(
                ()
                if self.selection is None
                else ((self.selection, PartitionRole.SELECTION),)
            ),
        )
        if any(bounds.role is not role for bounds, role in roles):
            raise ModelBindingError("fold chronology roles are inconsistent")
        ordered = [
            self.development,
            *(() if self.selection is None else (self.selection,)),
            self.test,
            self.holdout,
        ]
        if any(left.end >= right.start for left, right in pairwise(ordered)):
            raise ModelBindingError("fold windows are not strictly chronological")
        if self.embargo < timedelta(0) or self.outcome_reach <= timedelta(0):
            raise ModelBindingError("fold chronology offsets are invalid")
        if self.outcome_reach > self.label_horizon:
            raise ModelLeakageError(
                "the dataset target's outcome reach exceeds the plan's purged label "
                "horizon"
            )

    @property
    def protected_start(self) -> datetime:
        """The first instant that training outcomes must not reach."""
        return (self.selection or self.test).start

    def bounds(self, role: PartitionRole) -> WindowBounds | None:
        return {
            PartitionRole.DEVELOPMENT: self.development,
            PartitionRole.SELECTION: self.selection,
            PartitionRole.WALK_FORWARD_TEST: self.test,
            PartitionRole.FINAL_HOLDOUT: self.holdout,
        }[role]

    def label_reach(self, decision: datetime) -> datetime:
        """When a training label is safely usable: horizon plus embargo."""
        return decision + self.label_horizon + self.embargo

    def require_training_decision(self, decision: datetime) -> None:
        """A training row must be inside development and resolve before cutoff."""
        if not self.development.contains(decision):
            raise ModelLeakageError(
                "training row is outside its fold's development window"
            )
        reach = self.label_reach(decision)
        if reach >= self.protected_start:
            raise ModelLeakageError(
                "training label reach plus embargo reaches the fold's protected "
                "window (QF-8 purge rule); timestamp order alone is insufficient"
            )
        if reach >= self.holdout.start:
            raise ModelLeakageError("training label reach reaches the final holdout")

    def require_scored_decision(self, role: PartitionRole, decision: datetime) -> None:
        bounds = self.bounds(role)
        if bounds is None or not bounds.contains(decision):
            raise ModelLeakageError(
                f"{role.value} row is outside its plan window for this fold"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "contract_version": "1",
            "plan_id": self.plan_id,
            "fold_id": self.fold_id,
            "fold_index": self.fold_index,
            "development": self.development.to_primitive(),
            "selection": None
            if self.selection is None
            else self.selection.to_primitive(),
            "test": self.test.to_primitive(),
            "final_holdout": self.holdout.to_primitive(),
            "final_holdout_id": self.holdout_id,
            "label_horizon_microseconds": self.label_horizon // _MICROSECOND,
            "embargo_microseconds": self.embargo // _MICROSECOND,
            "target_outcome_reach_microseconds": self.outcome_reach // _MICROSECOND,
            "training_role": PartitionRole.DEVELOPMENT.value,
            "training_cutoff": self.protected_start.isoformat(),
            "training_label_rule": (
                "decision + label_horizon + embargo < training_cutoff "
                "(equality is leakage)"
            ),
        }

    @classmethod
    def from_primitive(cls, value: Primitive) -> "FoldChronology":
        if not isinstance(value, dict):
            raise ModelArtifactIntegrityError("fold chronology must be a mapping")
        try:
            selection = value["selection"]
            chronology = cls(
                plan_id=cast(str, value["plan_id"]),
                fold_id=cast(str, value["fold_id"]),
                fold_index=cast(int, value["fold_index"]),
                development=WindowBounds.from_primitive(value["development"]),
                selection=None
                if selection is None
                else WindowBounds.from_primitive(selection),
                test=WindowBounds.from_primitive(value["test"]),
                holdout=WindowBounds.from_primitive(value["final_holdout"]),
                holdout_id=cast(str, value["final_holdout_id"]),
                label_horizon=parse_microseconds(
                    value["label_horizon_microseconds"], "label horizon"
                ),
                embargo=parse_microseconds(value["embargo_microseconds"], "embargo"),
                outcome_reach=parse_microseconds(
                    value["target_outcome_reach_microseconds"], "outcome reach"
                ),
            )
        except (KeyError, ModelBindingError) as error:
            raise ModelArtifactIntegrityError("invalid fold chronology") from error
        if chronology.to_primitive() != value:
            raise ModelArtifactIntegrityError("fold chronology is inconsistent")
        return chronology


def fold_chronology(
    plan: ValidationPlan, fold_id: str, *, outcome_reach: timedelta
) -> FoldChronology:
    """Derive one fold's boundaries from the exact QF-8 plan (never caller-built)."""
    if type(cast(object, plan)) is not ValidationPlan:
        raise ModelBindingError("event models require the dataset's QF-8 plan")
    matches = [
        (index, fold)
        for index, fold in enumerate(plan.folds)
        if fold.fold_id == fold_id
    ]
    if len(matches) != 1:
        raise ModelBindingError("fold_id is not a fold of this plan")
    index, fold = matches[0]
    horizon = plan.purge_policy.label_horizon.elapsed
    embargo = plan.purge_policy.embargo.elapsed
    if horizon is None or embargo is None:
        raise ModelBindingError("event models require a timestamp purge policy")
    return FoldChronology(
        plan_id=plan.plan_id,
        fold_id=fold.fold_id,
        fold_index=index,
        development=WindowBounds.from_window(fold.development),
        selection=None
        if fold.selection is None
        else WindowBounds.from_window(fold.selection),
        test=WindowBounds.from_window(fold.test),
        holdout=WindowBounds.from_window(plan.final_holdout.window),
        holdout_id=plan.final_holdout.holdout_id,
        label_horizon=horizon,
        embargo=embargo,
        outcome_reach=outcome_reach,
    )


@dataclass(frozen=True, slots=True)
class DatasetBinding:
    """The exact QF-67 identities a model is bound to.

    ``non_holdout_rows_sha256`` is the QF-67 logical-row digest of every row
    outside the final holdout. A dataset rebuilt after explicit holdout
    consumption keeps it, so a model frozen before consumption can be applied
    to that dataset's holdout rows without any other row changing.
    """

    dataset_id: str
    population_id: str
    feature_schema_id: str
    feature_columns: tuple[str, ...]
    target_configuration_id: str
    plan_id: str
    final_holdout_id: str
    outcome_reach: timedelta
    non_holdout_rows_sha256: str

    @classmethod
    def capture(cls, dataset: EventDataset) -> "DatasetBinding":
        scientific = dataset.scientific.to_primitive()
        try:
            plan = cast(PrimitiveMapping, scientific["partition_plan"])
            target = cast(PrimitiveMapping, scientific["target"])
            binding = cls(
                dataset_id=dataset.dataset_id,
                population_id=cast(str, scientific["population_id"]),
                feature_schema_id=dataset.feature_schema.schema_id,
                feature_columns=dataset.feature_columns,
                target_configuration_id=cast(
                    str, scientific["target_configuration_id"]
                ),
                plan_id=cast(str, plan["plan_id"]),
                final_holdout_id=cast(str, plan["final_holdout_id"]),
                outcome_reach=timedelta(
                    microseconds=cast(int, target["outcome_reach_microseconds"])
                ),
                non_holdout_rows_sha256=logical_rows_sha256(
                    [
                        row
                        for row in dataset.rows
                        if row.partition_role is not PartitionRole.FINAL_HOLDOUT
                    ]
                ),
            )
        except (KeyError, TypeError) as error:
            raise ModelBindingError(
                "dataset scientific identity is incomplete"
            ) from error
        if configuration_identity(scientific) != dataset.dataset_id:
            raise ModelBindingError("dataset identity differs from its content")
        return binding

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "dataset_id": self.dataset_id,
            "population_id": self.population_id,
            "feature_schema_id": self.feature_schema_id,
            "feature_columns": list(self.feature_columns),
            "target_configuration_id": self.target_configuration_id,
            "plan_id": self.plan_id,
            "final_holdout_id": self.final_holdout_id,
            "target_outcome_reach_microseconds": self.outcome_reach // _MICROSECOND,
            "non_holdout_rows_sha256": self.non_holdout_rows_sha256,
        }

    @classmethod
    def from_primitive(cls, value: Primitive) -> "DatasetBinding":
        if not isinstance(value, dict):
            raise ModelArtifactIntegrityError("dataset binding must be a mapping")
        try:
            columns = value["feature_columns"]
            if not isinstance(columns, list) or not all(
                isinstance(item, str) for item in columns
            ):
                raise ModelArtifactIntegrityError("feature columns are invalid")
            binding = cls(
                dataset_id=cast(str, value["dataset_id"]),
                population_id=cast(str, value["population_id"]),
                feature_schema_id=cast(str, value["feature_schema_id"]),
                feature_columns=tuple(cast(list[str], columns)),
                target_configuration_id=cast(str, value["target_configuration_id"]),
                plan_id=cast(str, value["plan_id"]),
                final_holdout_id=cast(str, value["final_holdout_id"]),
                outcome_reach=parse_microseconds(
                    value["target_outcome_reach_microseconds"], "outcome reach"
                ),
                non_holdout_rows_sha256=cast(str, value["non_holdout_rows_sha256"]),
            )
        except KeyError as error:
            raise ModelArtifactIntegrityError("invalid dataset binding") from error
        if binding.to_primitive() != value:
            raise ModelArtifactIntegrityError("dataset binding is inconsistent")
        return binding

    def require_compatible(self, other: "DatasetBinding") -> None:
        """Same population, schema/order, target, plan and non-holdout rows."""
        for name in (
            "population_id",
            "feature_schema_id",
            "feature_columns",
            "target_configuration_id",
            "plan_id",
            "final_holdout_id",
            "outcome_reach",
        ):
            if getattr(self, name) != getattr(other, name):
                raise ModelBindingError(
                    f"dataset {name.replace('_', ' ')} differs from the model binding"
                )
        if self.non_holdout_rows_sha256 != other.non_holdout_rows_sha256:
            raise ModelBindingError(
                "dataset rows outside the final holdout differ from the model's "
                "training dataset"
            )


def require_fold_membership(
    dataset: EventDataset, chronology: FoldChronology, binding: DatasetBinding
) -> None:
    """Every source and row of the fold (and holdout) matches the plan windows.

    ``binding`` is ``DatasetBinding.capture(dataset)``, computed once by the caller.
    """
    if binding.dataset_id != dataset.dataset_id:
        raise ModelBindingError("dataset binding was captured from another dataset")
    if binding.plan_id != chronology.plan_id:
        raise ModelBindingError("dataset was built under another QF-8 plan")
    if binding.final_holdout_id != chronology.holdout_id:
        raise ModelBindingError("dataset final holdout differs from the plan")
    if binding.outcome_reach != chronology.outcome_reach:
        raise ModelBindingError("dataset target reach differs from the model binding")
    plan = cast(PrimitiveMapping, dataset.scientific.to_primitive()["partition_plan"])
    for source in cast(list[PrimitiveMapping], plan["sources"]):
        try:
            role = PartitionRole(cast(str, source["role"]))
        except (KeyError, ValueError) as error:
            raise ModelBindingError("dataset source role is invalid") from error
        same_fold = source.get("fold_id") == chronology.fold_id
        if role is not PartitionRole.FINAL_HOLDOUT and not same_fold:
            if source.get("fold_index") == chronology.fold_index:
                raise ModelBindingError("dataset fold identity differs from the plan")
            continue
        bounds = chronology.bounds(role)
        if (
            bounds is None
            or source.get("qf8_window_id") != bounds.window_id
            or (same_fold and source.get("fold_index") != chronology.fold_index)
        ):
            raise ModelBindingError(
                f"dataset {role.value} source differs from the plan's window"
            )
    for row in dataset.rows:
        if row.partition_role is PartitionRole.FINAL_HOLDOUT:
            chronology.require_scored_decision(
                row.partition_role, row.decision_timestamp
            )
        elif row.fold_id == chronology.fold_id:
            if row.fold_index != chronology.fold_index:
                raise ModelBindingError("row fold index differs from the plan")
            if row.partition_role is PartitionRole.DEVELOPMENT:
                chronology.require_training_decision(row.decision_timestamp)
            else:
                chronology.require_scored_decision(
                    row.partition_role, row.decision_timestamp
                )


__all__ = [
    "DatasetBinding",
    "FoldChronology",
    "WindowBounds",
    "fold_chronology",
    "parse_microseconds",
    "parse_utc_instant",
    "require_fold_membership",
]
