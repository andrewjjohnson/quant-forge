"""Chronological, training-only fitting of one event model (QF-68).

``fit_event_model`` validates a persisted QF-67 dataset offline, binds it to
one fold of its exact QF-8 plan, selects that fold's ``development_training``
rows (never selection, test or holdout rows), excludes unusable rows under an
explicit recorded policy, fits preprocessing and the estimator on exactly the
same fitting rows and returns an unfrozen ``FittedEventModel``.

Data limitations return a ``TrainingResult`` with an explicit status and no
model; contract violations (binding, leakage, non-finite inputs) raise.

Identities (all SHA-256 of canonical JSON):

- ``ModelConfiguration.configuration_id``: the fixed scientific choices only;
- ``model_configuration_id``: those choices bound to the dataset, population,
  feature schema and order, target, plan fold chronology and runtime versions;
- ``fitted_state_id``: the fitting membership, preprocessing state, estimator
  state and baseline only, so rows outside training cannot change it;
- ``model_id``: the complete fitted model (binding plus fitted state).
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.ml.artifact import read_event_dataset
from quantforge.ml.dataset import EventDataset
from quantforge.ml.features import EventFeatureSchema
from quantforge.ml.modeling.chronology import (
    DatasetBinding,
    FoldChronology,
    fold_chronology,
    parse_utc_instant,
    require_fold_membership,
)
from quantforge.ml.modeling.configuration import (
    MissingFeaturePolicy,
    ModelConfiguration,
    runtime_versions,
)
from quantforge.ml.modeling.errors import (
    ModelArtifactIntegrityError,
    ModelBindingError,
    ModelConfigurationError,
    ModelInputError,
)
from quantforge.ml.modeling.estimator import LinearModelState, fit_logistic_regression
from quantforge.ml.modeling.preprocessing import (
    NumericRow,
    PreprocessingState,
    finite,
    fit_preprocessing,
)
from quantforge.ml.modeling.status import TrainingLimitationError, TrainingStatus
from quantforge.ml.sources import reject_non_authoritative
from quantforge.ml.targets import ForwardReturnBinaryTarget
from quantforge.validation import PartitionRole, ValidationPlan

EVENT_MODEL_CONTRACT_VERSION = "1"
INTERPRETATION = (
    "Conditional on the dataset's strategy population (rows exist only where "
    "it triggered). Descriptive classification evidence on a raw-return target "
    "without costs; not evidence of an edge, robustness or profitability."
)
_VERIFIED = object()


def load_event_dataset(path: Path) -> EventDataset:
    """Validate a persisted QF-67 dataset offline before any model use."""
    reject_non_authoritative(path, label="dataset path")
    if not isinstance(cast(object, path), Path):
        raise ModelBindingError("event models read persisted QF-67 dataset paths")
    return read_event_dataset(path)


def numeric_rows(dataset: EventDataset) -> tuple[NumericRow, ...]:
    """The QF-67 explicit binary64 view in model column order."""
    return dataset.numeric_feature_rows()


@dataclass(frozen=True, slots=True)
class BoundModelConfiguration:
    """A fixed configuration bound to one dataset, fold and runtime."""

    configuration: ModelConfiguration
    dataset: DatasetBinding
    feature_schema: PrimitiveMappingSnapshot
    target: PrimitiveMappingSnapshot
    chronology: FoldChronology
    runtime: PrimitiveMappingSnapshot

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "contract_version": EVENT_MODEL_CONTRACT_VERSION,
            "configuration": self.configuration.to_primitive(),
            "configuration_id": self.configuration.configuration_id,
            "dataset": self.dataset.to_primitive(),
            "feature_schema": self.feature_schema.to_primitive(),
            "target": self.target.to_primitive(),
            "chronology": self.chronology.to_primitive(),
            "runtime": self.runtime.to_primitive(),
            "observation_unit": "qf67_event_row",
            "interpretation": INTERPRETATION,
        }

    @property
    def model_configuration_id(self) -> str:
        return configuration_identity(self.to_primitive())

    @classmethod
    def from_primitive(cls, value: Primitive) -> "BoundModelConfiguration":
        if not isinstance(value, dict):
            raise ModelArtifactIntegrityError("model configuration must be a mapping")
        try:
            schema = cast(PrimitiveMapping, value["feature_schema"])
            target = cast(PrimitiveMapping, value["target"])
            bound = cls(
                ModelConfiguration.from_primitive(
                    cast(PrimitiveMapping, value["configuration"])
                ),
                DatasetBinding.from_primitive(value["dataset"]),
                PrimitiveMappingSnapshot.capture(schema),
                PrimitiveMappingSnapshot.capture(target),
                FoldChronology.from_primitive(value["chronology"]),
                PrimitiveMappingSnapshot.capture(
                    cast(PrimitiveMapping, value["runtime"])
                ),
            )
            parsed_schema = EventFeatureSchema.from_primitive(schema)
            ForwardReturnBinaryTarget.from_primitive(
                cast(PrimitiveMapping, target["target"])
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ModelArtifactIntegrityError):
                raise
            raise ModelArtifactIntegrityError("invalid model configuration") from error
        binding, chronology = bound.dataset, bound.chronology
        if (
            bound.to_primitive() != value
            or parsed_schema.schema_id != binding.feature_schema_id
            or parsed_schema.column_names != binding.feature_columns
            or configuration_identity(target) != binding.target_configuration_id
            or chronology.plan_id != binding.plan_id
            or chronology.holdout_id != binding.final_holdout_id
            or chronology.outcome_reach != binding.outcome_reach
        ):
            raise ModelArtifactIntegrityError("model configuration is inconsistent")
        return bound


@dataclass(frozen=True, slots=True)
class TrainingExclusion:
    """A development row of the fold that did not enter fitting, and why."""

    source_observation_id: str
    decision_timestamp: datetime
    reason: str
    detail: str

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "source_observation_id": self.source_observation_id,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class TrainingMembership:
    """The exact observations that fitted preprocessing and the estimator."""

    training_rows: int
    fitting: tuple[tuple[str, datetime], ...]
    exclusions: tuple[TrainingExclusion, ...]
    positives: int
    dataset_source_exclusions: tuple[PrimitiveMappingSnapshot, ...]

    @property
    def fitting_observation_ids(self) -> tuple[str, ...]:
        return tuple(identity for identity, _ in self.fitting)

    @property
    def negatives(self) -> int:
        return len(self.fitting) - self.positives

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "training_role": PartitionRole.DEVELOPMENT.value,
            "training_rows": self.training_rows,
            "fitting_observation_count": len(self.fitting),
            "fitting_positives": self.positives,
            "fitting_negatives": self.negatives,
            "fitting_observations": [
                {
                    "source_observation_id": identity,
                    "decision_timestamp": stamp.isoformat(),
                }
                for identity, stamp in self.fitting
            ],
            "preprocessing_fit_rows": "identical_to_estimator_fit_rows",
            "excluded_by_reason": dict(
                sorted(Counter(item.reason for item in self.exclusions).items())
            ),
            "exclusions": [item.to_primitive() for item in self.exclusions],
            "dataset_source_exclusions": [
                item.to_primitive() for item in self.dataset_source_exclusions
            ],
            "ordering": "qf67_dataset_row_order",
        }

    @classmethod
    def from_primitive(
        cls, value: Primitive, chronology: FoldChronology
    ) -> "TrainingMembership":
        if not isinstance(value, dict):
            raise ModelArtifactIntegrityError("training membership must be a mapping")
        try:
            membership = cls(
                training_rows=cast(int, value["training_rows"]),
                fitting=tuple(
                    (
                        cast(str, item["source_observation_id"]),
                        parse_utc_instant(
                            item["decision_timestamp"], "fitting timestamp"
                        ),
                    )
                    for item in cast(
                        list[PrimitiveMapping], value["fitting_observations"]
                    )
                ),
                exclusions=tuple(
                    TrainingExclusion(
                        cast(str, item["source_observation_id"]),
                        parse_utc_instant(
                            item["decision_timestamp"], "exclusion timestamp"
                        ),
                        cast(str, item["reason"]),
                        cast(str, item["detail"]),
                    )
                    for item in cast(list[PrimitiveMapping], value["exclusions"])
                ),
                positives=cast(int, value["fitting_positives"]),
                dataset_source_exclusions=tuple(
                    PrimitiveMappingSnapshot.capture(item)
                    for item in cast(
                        list[PrimitiveMapping], value["dataset_source_exclusions"]
                    )
                ),
            )
        except (KeyError, TypeError) as error:
            raise ModelArtifactIntegrityError("invalid training membership") from error
        identities = [
            *membership.fitting_observation_ids,
            *(item.source_observation_id for item in membership.exclusions),
        ]
        if (
            membership.to_primitive() != value
            or len(set(identities)) != len(identities)
            or membership.training_rows != len(identities)
            or not 0 <= membership.positives <= len(membership.fitting)
        ):
            raise ModelArtifactIntegrityError("training membership is inconsistent")
        for _, decision in membership.fitting:
            chronology.require_training_decision(decision)
        return membership


@dataclass(frozen=True, slots=True)
class EventModelState:
    """Complete fitted scientific content shared by fitted and frozen models."""

    bound: BoundModelConfiguration
    membership: TrainingMembership
    preprocessing: PreprocessingState
    estimator: LinearModelState

    @property
    def baseline_probability(self) -> float:
        """Constant training prevalence over the fitting rows (binary64)."""
        return self.membership.positives / len(self.membership.fitting)

    @property
    def configuration(self) -> ModelConfiguration:
        return self.bound.configuration

    @property
    def chronology(self) -> FoldChronology:
        return self.bound.chronology

    def _fitted(self) -> PrimitiveMapping:
        return {
            "training_membership": self.membership.to_primitive(),
            "preprocessing": {
                "state": self.preprocessing.to_primitive(),
                "state_id": self.preprocessing.state_id,
            },
            "estimator": {
                "configuration": self.configuration.estimator.to_primitive(),
                "state": self.estimator.to_primitive(),
                "state_id": self.estimator.state_id,
            },
            "baseline": {
                "kind": "constant_training_prevalence",
                "probability": self.baseline_probability,
                "fitting_positives": self.membership.positives,
                "fitting_observations": len(self.membership.fitting),
            },
        }

    @property
    def fitted_state_id(self) -> str:
        return configuration_identity(self._fitted())

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "model_configuration": self.bound.to_primitive(),
            "model_configuration_id": self.bound.model_configuration_id,
            **self._fitted(),
            "fitted_state_id": self.fitted_state_id,
        }

    @property
    def model_id(self) -> str:
        return configuration_identity(self.to_primitive())

    @classmethod
    def from_primitive(cls, value: Primitive) -> "EventModelState":
        if not isinstance(value, dict):
            raise ModelArtifactIntegrityError("model state must be a mapping")
        try:
            bound = BoundModelConfiguration.from_primitive(value["model_configuration"])
            preprocessing = cast(PrimitiveMapping, value["preprocessing"])
            estimator = cast(PrimitiveMapping, value["estimator"])
            baseline = cast(PrimitiveMapping, value["baseline"])
            state = cls(
                bound,
                TrainingMembership.from_primitive(
                    value["training_membership"], bound.chronology
                ),
                PreprocessingState.from_primitive(preprocessing["state"]),
                LinearModelState.from_primitive(estimator["state"]),
            )
            finite(baseline["probability"], "baseline probability")
        except (KeyError, TypeError) as error:
            raise ModelArtifactIntegrityError("invalid model state") from error
        columns = bound.dataset.feature_columns
        if (
            state.to_primitive() != value
            or state.preprocessing.columns != columns
            or state.estimator.features != columns
            or state.preprocessing.configuration != bound.configuration.preprocessing
            or not 0 < state.membership.positives < len(state.membership.fitting)
            or any(
                state.estimator.coefficients[index] != 0.0
                for index, item in enumerate(state.preprocessing.features)
                if item.constant_in_training
            )
        ):
            raise ModelArtifactIntegrityError("model state is inconsistent")
        return state


@dataclass(frozen=True, slots=True)
class FittedEventModel:
    """A fitted but **unfrozen** model: selection scoring and freezing only.

    Out-of-sample and holdout inference refuse it; they require the frozen
    envelope written by ``freeze_event_model`` and read by ``read_event_model``.
    """

    state: EventModelState
    _token: object = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self._token is not _VERIFIED:
            raise ModelBindingError("fitted models are created only by fit_event_model")

    @property
    def model_id(self) -> str:
        return self.state.model_id


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The explicit outcome of one training attempt."""

    status: TrainingStatus
    detail: str
    bound: BoundModelConfiguration
    membership: TrainingMembership
    model: FittedEventModel | None

    @property
    def fitted(self) -> bool:
        return self.status is TrainingStatus.FITTED

    @property
    def model_configuration_id(self) -> str:
        return self.bound.model_configuration_id

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "status": self.status.value,
            "detail": self.detail,
            "model_configuration_id": self.model_configuration_id,
            "model_configuration": self.bound.to_primitive(),
            "training_membership": self.membership.to_primitive(),
            "model_id": None if self.model is None else self.model.model_id,
        }


def _dataset_exclusions(
    dataset: EventDataset, fold_id: str
) -> tuple[PrimitiveMappingSnapshot, ...]:
    plan = cast(PrimitiveMapping, dataset.scientific.to_primitive()["partition_plan"])
    return tuple(
        PrimitiveMappingSnapshot.capture(item)
        for item in cast(list[PrimitiveMapping], plan["excluded_sources"])
        if item.get("fold_id") == fold_id
        and item.get("role") == PartitionRole.DEVELOPMENT.value
    )


def bind_model_configuration(
    dataset: EventDataset,
    *,
    plan: ValidationPlan,
    fold_id: str,
    configuration: ModelConfiguration,
) -> BoundModelConfiguration:
    """Bind a fixed configuration to the dataset and one fold of its plan."""
    if type(cast(object, configuration)) is not ModelConfiguration:
        raise ModelConfigurationError("an explicit ModelConfiguration is required")
    binding = DatasetBinding.capture(dataset)
    chronology = fold_chronology(plan, fold_id, outcome_reach=binding.outcome_reach)
    require_fold_membership(dataset, chronology, binding)
    scientific = dataset.scientific.to_primitive()
    return BoundModelConfiguration(
        configuration,
        binding,
        PrimitiveMappingSnapshot.capture(dataset.feature_schema.to_primitive()),
        PrimitiveMappingSnapshot.capture(cast(PrimitiveMapping, scientific["target"])),
        chronology,
        PrimitiveMappingSnapshot.capture(runtime_versions()),
    )


def _membership(
    dataset: EventDataset,
    chronology: FoldChronology,
    policy: MissingFeaturePolicy,
    numeric: tuple[NumericRow, ...],
) -> tuple[TrainingMembership, list[int]]:
    """Training rows of the fold and the subset that fits, in dataset order."""
    columns = dataset.feature_columns
    fitting: list[int] = []
    exclusions: list[TrainingExclusion] = []
    training = dataset.rows_for(PartitionRole.DEVELOPMENT, fold_id=chronology.fold_id)
    for index in training:
        row = dataset.rows[index]
        values = numeric[index]
        for name, value in zip(columns, values, strict=True):
            if value is not None and not math.isfinite(value):
                raise ModelInputError(
                    f"training feature {name!r} is not finite binary64"
                )
        missing = [
            name for name, value in zip(columns, values, strict=True) if value is None
        ]
        if row.label.value is None:
            reason, detail = "target_unavailable", row.label.status
        elif missing and policy is MissingFeaturePolicy.EXCLUDE_OBSERVATION:
            reason, detail = "null_feature_value", ",".join(missing)
        else:
            fitting.append(index)
            continue
        exclusions.append(
            TrainingExclusion(
                row.source_observation_id, row.decision_timestamp, reason, detail
            )
        )
    membership = TrainingMembership(
        training_rows=len(training),
        fitting=tuple(
            (
                dataset.rows[index].source_observation_id,
                dataset.rows[index].decision_timestamp,
            )
            for index in fitting
        ),
        exclusions=tuple(exclusions),
        positives=sum(dataset.rows[index].label.value is True for index in fitting),
        dataset_source_exclusions=_dataset_exclusions(dataset, chronology.fold_id),
    )
    return membership, fitting


def fit_event_model(
    dataset: Path,
    *,
    plan: ValidationPlan,
    fold_id: str,
    configuration: ModelConfiguration,
) -> TrainingResult:
    """Fit one fold's model on its permitted training rows only.

    ``dataset`` is a persisted QF-67 dataset directory (validated offline) and
    ``plan`` the exact QF-8 plan it was built under. Training uses the fold's
    development rows whose label reach plus embargo ends before the fold's
    protected window; any other row only fails binding checks or is scored
    later. Selection, test and holdout values never reach fitted state.
    """
    loaded = load_event_dataset(dataset)
    bound = bind_model_configuration(
        loaded, plan=plan, fold_id=fold_id, configuration=configuration
    )
    numeric = numeric_rows(loaded)
    membership, fitting = _membership(
        loaded,
        bound.chronology,
        configuration.preprocessing.missing_features,
        numeric,
    )

    def failed(status: TrainingStatus, detail: str) -> TrainingResult:
        return TrainingResult(status, detail, bound, membership, None)

    if membership.training_rows == 0:
        reasons = sorted(
            {
                str(item.to_primitive().get("reason"))
                for item in membership.dataset_source_exclusions
            }
        )
        return failed(
            TrainingStatus.NO_TRAINING_OBSERVATIONS,
            "the fold has no development_training rows"
            + (f" (dataset exclusions: {', '.join(reasons)})" if reasons else ""),
        )
    if len(fitting) < configuration.minimum_training_observations:
        return failed(
            TrainingStatus.INSUFFICIENT_TRAINING_OBSERVATIONS,
            f"{len(fitting)} fitting rows < minimum "
            f"{configuration.minimum_training_observations}",
        )
    if membership.positives in (0, len(fitting)):
        return failed(
            TrainingStatus.SINGLE_CLASS_TRAINING_LABELS,
            "fitting labels contain one class only",
        )
    rows = [numeric[index] for index in fitting]
    labels = [loaded.rows[index].label.value is True for index in fitting]
    try:
        preprocessing = fit_preprocessing(
            loaded.feature_columns, rows, configuration.preprocessing
        )
        design: list[list[float]] = []
        for index, values in zip(fitting, rows, strict=True):
            transformed = preprocessing.transform(
                values, f"training row {loaded.rows[index].source_observation_id}"
            )
            assert transformed is not None  # Fitting rows are scorable by policy.
            design.append(list(transformed.values))
        estimator = fit_logistic_regression(
            loaded.feature_columns,
            preprocessing.informative,
            design,
            labels,
            configuration.estimator,
        )
    except TrainingLimitationError as failure:
        return failed(failure.status, failure.detail)
    state = EventModelState(bound, membership, preprocessing, estimator)
    return TrainingResult(
        TrainingStatus.FITTED,
        f"fitted on {len(fitting)} development rows",
        bound,
        membership,
        FittedEventModel(state, _VERIFIED),
    )


__all__ = [
    "EVENT_MODEL_CONTRACT_VERSION",
    "BoundModelConfiguration",
    "EventModelState",
    "FittedEventModel",
    "TrainingExclusion",
    "TrainingMembership",
    "TrainingResult",
    "bind_model_configuration",
    "fit_event_model",
    "load_event_dataset",
    "numeric_rows",
]
