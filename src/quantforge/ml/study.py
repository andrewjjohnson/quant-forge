"""Frozen conditional ML smoke-study lifecycle over QF-67/QF-68 (QF-69).

A study composes the existing APIs in a fixed chronological order; it adds no
estimator, feature, search or threshold:

1. ``freeze_specification`` persists every choice (population, windows, feature
   schema, target, model configuration, minimums, buckets, evaluation and
   holdout policy) before any outcome or model is inspected. Every later stage
   rebuilds the specification from code and refuses to run if it differs.
2. ``build_dataset``: QF-67 builds and exports the verified event dataset.
3. ``train_and_freeze``: QF-68 fits one fold's development rows only; the
   frozen class-count floor is applied; selection predictions are exported;
   the model is frozen and re-read through ``read_event_model``.
4. ``evaluate_out_of_sample``: requires the persisted freeze record; scores
   walk-forward test rows with the re-read frozen model and exports them.
5. ``verify_reproduction``: rebuilds the dataset, refits, rescores and
   regenerates both prediction sets offline; identities must be identical.
6. ``report`` and ``pre_holdout_gate`` read persisted artifacts only.

Each stage writes one immutable fingerprinted record under ``stages/`` bound to
the specification ID and to the identities of the earlier stages, so a stage can
never run ahead of its predecessors, and a repeated run must reproduce it
exactly. Data limitations (for example too few training rows) end the study
with an explicit recorded status; nothing is relaxed, extended or swapped.

The study never reserves, consumes or reads a final holdout. The pre-holdout
gate requires the plan's holdout to be RESERVED / UNCONSUMED in the permanent
ledger, as reported by the caller from the normal QF-40 contracts.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.ml.artifact import (
    MANIFEST_FILE,
    ROWS_PARQUET,
    export_event_dataset,
    read_event_dataset,
)
from quantforge.ml.dataset import EventDataset, build_event_dataset
from quantforge.ml.features import CausalAvailability, EventFeatureSchema
from quantforge.ml.modeling import (
    MODEL_FILE,
    FrozenEventModel,
    ModelConfiguration,
    TrainingStatus,
    export_prediction_set,
    fit_event_model,
    freeze_event_model,
    predict_out_of_sample,
    predict_selection,
    read_event_model,
    read_prediction_set,
    reproduce_prediction_set,
)
from quantforge.ml.sources import DispositionPolicy, reject_non_authoritative
from quantforge.ml.study_report import (
    base_strategy,
    prediction_partition,
    source_coverage,
)
from quantforge.ml.targets import ForwardReturnBinaryTarget
from quantforge.validation import PartitionRole, ValidationPlan
from quantforge.walk_forward.models import (
    CandidateConfiguration,
    WalkForwardPersistenceError,
)
from quantforge.walk_forward.persistence import read_record, write_record

STUDY_COMPONENT = "quantforge_conditional_ml_study"
STUDY_SCHEMA_VERSION = "1"
SPECIFICATION_FILE = "specification.json"
STAGE_DIRECTORY = "stages"
STAGES = ("dataset", "training", "freeze", "out_of_sample", "reproduction")
# Study-level status when QF-68 fitted but a class floor was not met.
INSUFFICIENT_CLASS_OBSERVATIONS = "insufficient_training_class_observations"
LIFECYCLE = (
    "specification frozen before outcome or model inspection",
    "QF-67 dataset built and exported",
    "QF-68 fit on the fold's development rows only",
    "frozen class-count floor applied",
    "selection predictions exported",
    "model frozen and re-read before any out-of-sample inference",
    "walk-forward test predictions exported from the frozen model",
    "deterministic rebuild/refit and offline reproduction",
    "report and pre-holdout gate from persisted artifacts only",
    "final holdout stays reserved and unconsumed",
)


class ConditionalStudyError(ValueError):
    """A study stage ran out of order, against changed choices or bad evidence."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ConditionalStudySpecification:
    """Every scientific choice of one study, frozen before research runs.

    ``design`` holds the study-specific declarations (strategy, source,
    windows, exposure history, planning evidence); the typed fields are what
    the lifecycle enforces. The study is probability-only: a class threshold
    (a filter) would need retained-subset reporting and is refused here.
    """

    name: str
    version: str
    plan: ValidationPlan
    fold_id: str
    qf39_study_id: str
    candidate: CandidateConfiguration
    feature_schema: EventFeatureSchema
    target: ForwardReturnBinaryTarget
    disposition_policy: DispositionPolicy
    model: ModelConfiguration
    minimum_training_class_observations: int
    score_buckets: int
    design: PrimitiveMappingSnapshot

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise ConditionalStudyError("study name and version are required")
        fold = next((f for f in self.plan.folds if f.fold_id == self.fold_id), None)
        if fold is None:
            raise ConditionalStudyError("study fold is not in the validation plan")
        if fold.selection is None:
            raise ConditionalStudyError(
                "a conditional ML study needs development, selection and test roles"
            )
        if self.model.class_threshold is not None:
            raise ConditionalStudyError(
                "the smoke study is probability-only; a frozen filter threshold "
                "is not supported"
            )
        for value, label in (
            (self.minimum_training_class_observations, "class floor"),
            (self.score_buckets, "score bucket count"),
        ):
            if type(value) is not int or value < 1:
                raise ConditionalStudyError(f"{label} must be a positive integer")
        if any(
            item.availability is not CausalAvailability.DECISION_TIMESTAMP
            for item in self.feature_schema.features
        ):
            raise ConditionalStudyError("every feature must be causal")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "component": STUDY_COMPONENT,
            "schema_version": STUDY_SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "plan_id": self.plan.plan_id,
            "final_holdout_id": self.plan.final_holdout.holdout_id,
            "fold_id": self.fold_id,
            "qf39_study_id": self.qf39_study_id,
            "population": {
                "candidate": self.candidate.to_primitive(),
                "disposition_policy": self.disposition_policy.value,
            },
            "feature_schema": self.feature_schema.to_primitive(),
            "feature_schema_id": self.feature_schema.schema_id,
            "target": self.target.to_primitive(),
            "target_id": self.target.target_id,
            "model_configuration": self.model.to_primitive(),
            "model_configuration_id": self.model.configuration_id,
            "minimums": {
                "minimum_training_observations": (
                    self.model.minimum_training_observations
                ),
                "minimum_training_class_observations": (
                    self.minimum_training_class_observations
                ),
                "minimum_evaluation_observations": (
                    self.model.minimum_evaluation_observations
                ),
                "insufficient_partitions": (
                    "recorded with status and reason; no rule, date, minimum or "
                    "model change afterwards"
                ),
            },
            "evaluation": {
                "mode": "probability_only",
                "class_threshold": None,
                "score_buckets": {
                    "count": self.score_buckets,
                    "binning": "equal_width_on_unit_interval",
                    "assignment": "min(floor(p * B), B - 1)",
                    "frozen": "before fitting; never changed after viewing OOS",
                },
                "metrics": [
                    "log_loss",
                    "brier_score",
                    "roc_auc_where_defined",
                    "calibration_in_score_buckets",
                ],
                "baseline": (
                    "constant training prevalence, evaluated on exactly the "
                    "model's labeled scored rows"
                ),
                "base_strategy": "unfiltered 30m raw-return summaries per partition",
                "independence": (
                    "overlapping intraday labels are not independent samples; "
                    "no significance claims"
                ),
            },
            "holdout_policy": {
                "ledger": "workspace permanent ledger (reports/holdout-ledger)",
                "reservation": "normal QF-40 reservation before research",
                "pre_holdout_gate": "stop with the holdout RESERVED / UNCONSUMED",
                "consumption": (
                    "never by this study; only a later explicitly authorized "
                    "QF-40 step, with no retuning afterwards"
                ),
            },
            "lifecycle": list(LIFECYCLE),
            "design": self.design.to_primitive(),
        }

    @property
    def specification_id(self) -> str:
        return configuration_identity(self.to_primitive())


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """The recorded training decision and, if frozen, the model's path."""

    status: str
    detail: str
    model_path: Path | None

    @property
    def frozen(self) -> bool:
        return self.model_path is not None


class ConditionalStudy:
    """Run one frozen specification's stages in their only permitted order."""

    def __init__(
        self,
        specification: ConditionalStudySpecification,
        *,
        root: Path,
        study_path: Path,
        workspace: Path,
    ) -> None:
        for value, label in (
            (root, "study output"),
            (study_path, "QF-39 study path"),
            (workspace, "workspace"),
        ):
            reject_non_authoritative(value, label=label)
        if study_path.name != specification.qf39_study_id:
            raise ConditionalStudyError("QF-39 study differs from the specification")
        self.specification = specification
        self.root = Path(root)
        self.study_path = Path(study_path)
        self.workspace = Path(workspace)

    # -- records -----------------------------------------------------------

    @property
    def specification_path(self) -> Path:
        return self.root / SPECIFICATION_FILE

    def freeze_specification(self) -> str:
        """Persist the specification immutably; an existing one must be equal."""
        try:
            write_record(
                self.specification_path,
                self.specification.to_primitive(),
                immutable=True,
            )
        except WalkForwardPersistenceError as error:
            raise ConditionalStudyError(
                "the persisted study specification differs; the frozen design "
                "cannot change"
            ) from error
        return self.specification.specification_id

    def _require_specification(self) -> str:
        if not self.specification_path.exists():
            raise ConditionalStudyError(
                "the study specification must be frozen before research stages"
            )
        try:
            persisted = read_record(self.specification_path)
        except WalkForwardPersistenceError as error:
            raise ConditionalStudyError("study specification is corrupt") from error
        if persisted != self.specification.to_primitive():
            raise ConditionalStudyError(
                "the frozen study specification differs from this design"
            )
        return self.specification.specification_id

    def _stage_path(self, name: str) -> Path:
        return self.root / STAGE_DIRECTORY / f"{STAGES.index(name) + 1:02d}-{name}.json"

    def stage(self, name: str) -> PrimitiveMapping | None:
        """A recorded stage, validated against its envelope and specification."""
        path = self._stage_path(name)
        if not path.exists():
            return None
        try:
            record = read_record(path)
        except WalkForwardPersistenceError as error:
            raise ConditionalStudyError(f"{name} stage record is corrupt") from error
        if (
            record.get("stage") != name
            or record.get("specification_id") != self.specification.specification_id
        ):
            raise ConditionalStudyError(f"{name} stage belongs to another study")
        return record

    def _require_stage(self, name: str) -> PrimitiveMapping:
        record = self.stage(name)
        if record is None:
            raise ConditionalStudyError(f"the {name} stage has not been recorded")
        return record

    def _record(self, name: str, payload: PrimitiveMapping) -> PrimitiveMapping:
        record: PrimitiveMapping = {
            "stage": name,
            "specification_id": self.specification.specification_id,
            **payload,
        }
        try:
            write_record(self._stage_path(name), record, immutable=True)
        except WalkForwardPersistenceError as error:
            raise ConditionalStudyError(
                f"the {name} stage differs from its recorded result"
            ) from error
        return record

    def _path(self, relative: Primitive) -> Path:
        if not isinstance(relative, str):
            raise ConditionalStudyError("stage path is missing")
        return self.root / relative

    # -- stages ------------------------------------------------------------

    def _build(self) -> EventDataset:
        specification = self.specification
        dataset = build_event_dataset(
            plan=specification.plan,
            study_path=self.study_path,
            workspace=self.workspace,
            combination_id=specification.candidate.combination_id,
            feature_schema=specification.feature_schema,
            target=specification.target,
            disposition_policy=specification.disposition_policy,
        )
        population = cast(
            PrimitiveMapping, dataset.scientific.to_primitive()["population"]
        )
        if (
            population.get("candidate") != specification.candidate.to_primitive()
            or dataset.feature_schema.schema_id
            != specification.feature_schema.schema_id
            or dataset.target_definition != specification.target
        ):
            raise ConditionalStudyError("dataset differs from the frozen population")
        return dataset

    def dataset_path(self) -> Path:
        record = self._require_stage("dataset")
        return self._path(record["path"])

    def build_dataset(self) -> Path:
        """QF-67 build and export; a recorded dataset is re-validated instead."""
        self._require_specification()
        record = self.stage("dataset")
        if record is None:
            dataset = self._build()
            path = export_event_dataset(
                dataset, self.root / "datasets", include_csv=True
            )
            record = self._record(
                "dataset",
                {
                    "dataset_id": dataset.dataset_id,
                    "population_id": dataset.scientific.to_primitive()["population_id"],
                    "path": path.relative_to(self.root).as_posix(),
                    "files": {
                        name: _sha256(path / name)
                        for name in (MANIFEST_FILE, ROWS_PARQUET)
                    },
                },
            )
        path = self._path(record["path"])
        if read_event_dataset(path).dataset_id != record["dataset_id"]:
            raise ConditionalStudyError("persisted dataset differs from its stage")
        return path

    def train_and_freeze(self) -> TrainingOutcome:
        """Fit development rows, apply the class floor, export selection, freeze.

        The fitted state is persisted only through ``freeze_event_model``; a
        model that misses a frozen floor is discarded unscored and unfrozen.
        """
        self._require_specification()
        specification = self.specification
        dataset_stage = self._require_stage("dataset")
        dataset = self.build_dataset()
        result = fit_event_model(
            dataset,
            plan=specification.plan,
            fold_id=specification.fold_id,
            configuration=specification.model,
        )
        membership = result.membership
        status, detail = result.status.value, result.detail
        floor = specification.minimum_training_class_observations
        if result.fitted and min(membership.positives, membership.negatives) < floor:
            status = INSUFFICIENT_CLASS_OBSERVATIONS
            detail = (
                f"{membership.positives} positive / {membership.negatives} negative "
                f"fitting rows; each class needs at least {floor}"
            )
        frozen = status == TrainingStatus.FITTED.value and result.model is not None
        training = self._record(
            "training",
            {
                "dataset_id": dataset_stage["dataset_id"],
                "status": status,
                "detail": detail,
                "qf68_status": result.status.value,
                "model_configuration_id": result.model_configuration_id,
                "fitted_state_id": None
                if result.model is None
                else result.model.state.fitted_state_id,
                "model_id": None
                if result.model is None or not frozen
                else result.model.model_id,
                "membership": {
                    "training_rows": membership.training_rows,
                    "fitting_observations": len(membership.fitting),
                    "fitting_positives": membership.positives,
                    "fitting_negatives": membership.negatives,
                    "excluded_by_reason": cast(
                        PrimitiveMapping,
                        membership.to_primitive()["excluded_by_reason"],
                    ),
                    "dataset_source_exclusions": [
                        item.to_primitive()
                        for item in membership.dataset_source_exclusions
                    ],
                },
            },
        )
        if not frozen:
            return TrainingOutcome(status, detail, None)
        assert result.model is not None
        selection = predict_selection(result.model, dataset)
        selection_path = export_prediction_set(selection, self.root / "predictions")
        model_path = freeze_event_model(
            result.model, dataset=dataset, output_root=self.root / "models"
        )
        model = read_event_model(model_path)
        if (
            model.model_id != training["model_id"]
            or model.selection_prediction_set_id != selection.prediction_set_id
        ):
            raise ConditionalStudyError("frozen model differs from its training")
        self._record(
            "freeze",
            {
                "dataset_id": dataset_stage["dataset_id"],
                "model_id": model.model_id,
                "fitted_state_id": model.state.fitted_state_id,
                "model_configuration_id": model.state.bound.model_configuration_id,
                "model_path": model_path.relative_to(self.root).as_posix(),
                "model_sha256": _sha256(model_path / MODEL_FILE),
                "selection_prediction_set_id": selection.prediction_set_id,
                "selection_path": selection_path.relative_to(self.root).as_posix(),
                "lifecycle": "frozen before any walk-forward test inference",
            },
        )
        return TrainingOutcome(status, detail, model_path)

    def frozen_model(self) -> FrozenEventModel:
        """The persisted frozen model; out-of-sample work requires its record."""
        self._require_specification()
        record = self._require_stage("freeze")
        path = self._path(record["model_path"])
        if _sha256(path / MODEL_FILE) != record["model_sha256"]:
            raise ConditionalStudyError("frozen model envelope changed after freezing")
        model = read_event_model(path)
        if model.model_id != record["model_id"]:
            raise ConditionalStudyError("frozen model differs from its freeze record")
        return model

    def evaluate_out_of_sample(self) -> Path:
        """Score walk-forward test rows with the re-read frozen model only."""
        model = self.frozen_model()
        freeze = self._require_stage("freeze")
        dataset = self.dataset_path()
        predictions = predict_out_of_sample(model, dataset)
        path = export_prediction_set(predictions, self.root / "predictions")
        self._record(
            "out_of_sample",
            {
                "dataset_id": freeze["dataset_id"],
                "model_id": model.model_id,
                "prediction_set_id": predictions.prediction_set_id,
                "path": path.relative_to(self.root).as_posix(),
                "rows": len(predictions.records),
            },
        )
        return path

    def verify_reproduction(self) -> PrimitiveMapping:
        """Rebuild, refit, rescore and regenerate offline; all must be identical."""
        model = self.frozen_model()
        dataset_stage = self._require_stage("dataset")
        freeze = self._require_stage("freeze")
        oos = self._require_stage("out_of_sample")
        dataset = self.dataset_path()
        rebuilt = self._build()
        rebuilt_path = export_event_dataset(
            rebuilt, self.root / "reproduction" / "datasets", include_csv=True
        )
        refit = fit_event_model(
            dataset,
            plan=self.specification.plan,
            fold_id=self.specification.fold_id,
            configuration=self.specification.model,
        )
        if refit.model is None:
            raise ConditionalStudyError("refit did not reproduce a fitted model")
        refrozen = freeze_event_model(
            refit.model,
            dataset=dataset,
            output_root=self.root / "reproduction" / "models",
        )
        selection_path = self._path(freeze["selection_path"])
        oos_path = self._path(oos["path"])
        rescored = {
            "selection": predict_selection(refit.model, dataset).prediction_set_id,
            "out_of_sample": predict_out_of_sample(model, dataset).prediction_set_id,
        }
        offline = {
            role: reproduce_prediction_set(path, model=model, dataset=dataset)
            for role, path in (
                ("selection", selection_path),
                ("out_of_sample", oos_path),
            )
        }
        checks: PrimitiveMapping = {
            "dataset_rebuild_identical": rebuilt.dataset_id
            == dataset_stage["dataset_id"],
            "dataset_files_identical": {
                name: _sha256(rebuilt_path / name)
                for name in (MANIFEST_FILE, ROWS_PARQUET)
            }
            == dataset_stage["files"],
            "model_configuration_identical": refit.model_configuration_id
            == freeze["model_configuration_id"],
            "fitted_state_identical": refit.model.state.fitted_state_id
            == freeze["fitted_state_id"],
            "model_identical": refit.model.model_id == freeze["model_id"],
            "model_envelope_identical": _sha256(refrozen / MODEL_FILE)
            == freeze["model_sha256"],
            "selection_rescore_identical": rescored["selection"]
            == freeze["selection_prediction_set_id"],
            "out_of_sample_rescore_identical": rescored["out_of_sample"]
            == oos["prediction_set_id"],
            "offline_reproduction_exact": all(item.exact for item in offline.values()),
        }
        record = self._record(
            "reproduction",
            {
                "dataset_id": dataset_stage["dataset_id"],
                "model_id": freeze["model_id"],
                "checks": checks,
                "offline": {
                    role: {
                        "prediction_set_id": item.prediction_set_id,
                        "records": item.records,
                        "maximum_absolute_difference": item.maximum_absolute_difference,
                        "exact": item.exact,
                    }
                    for role, item in offline.items()
                },
                "network": "not used; persisted artifacts and local study evidence",
            },
        )
        if not all(value is True for value in checks.values()):
            raise ConditionalStudyError("the study did not reproduce deterministically")
        return record

    # -- report and gate ---------------------------------------------------

    def report(self) -> PrimitiveMapping:
        """Descriptive results from persisted, re-validated artifacts only."""
        specification = self.specification
        self._require_specification()
        dataset_stage = self._require_stage("dataset")
        training = self._require_stage("training")
        dataset = read_event_dataset(self.dataset_path())
        fold = specification.fold_id
        report: PrimitiveMapping = {
            "component": "quantforge_conditional_ml_study_report",
            "schema_version": STUDY_SCHEMA_VERSION,
            "specification_id": specification.specification_id,
            "specification": specification.to_primitive(),
            "artifacts": {
                "qf39_study_id": specification.qf39_study_id,
                "plan_id": specification.plan.plan_id,
                "final_holdout_id": specification.plan.final_holdout.holdout_id,
                "dataset_id": dataset_stage["dataset_id"],
                "population_id": dataset_stage["population_id"],
                "dataset_path": dataset_stage["path"],
            },
            "events": {
                "sources": source_coverage(dataset),
                "excluded_sources": cast(
                    PrimitiveMapping,
                    dataset.scientific.to_primitive()["partition_plan"],
                )["excluded_sources"],
                "development": base_strategy(dataset, PartitionRole.DEVELOPMENT, fold),
                "selection": base_strategy(dataset, PartitionRole.SELECTION, fold),
            },
            "training": {
                key: training[key]
                for key in ("status", "detail", "qf68_status", "membership")
            },
            "stages": {
                name: record
                for name in STAGES
                if (record := self.stage(name)) is not None
            },
        }
        if self.stage("freeze") is None:
            report["out_of_sample"] = {
                "status": "not_evaluated",
                "reason": f"training ended with {training['status']}",
            }
            return report
        model = self.frozen_model()
        freeze = self._require_stage("freeze")
        oos_stage = self._require_stage("out_of_sample")
        selection = read_prediction_set(self._path(freeze["selection_path"]))
        oos = read_prediction_set(self._path(oos_stage["path"]))
        buckets = specification.score_buckets
        state = model.state.to_primitive()
        artifacts = cast(PrimitiveMapping, report["artifacts"])
        artifacts.update(
            {
                "model_configuration_id": freeze["model_configuration_id"],
                "fitted_state_id": freeze["fitted_state_id"],
                "model_id": freeze["model_id"],
                "model_path": freeze["model_path"],
                "selection_prediction_set_id": selection.prediction_set_id,
                "out_of_sample_prediction_set_id": oos.prediction_set_id,
            }
        )
        report["model"] = {
            "preprocessing": cast(PrimitiveMapping, state["preprocessing"])["state"],
            "estimator": cast(PrimitiveMapping, state["estimator"])["state"],
            "baseline": state["baseline"],
        }
        report["selection"] = prediction_partition(selection, dataset, buckets)
        # Out-of-sample analysis exists only behind the persisted freeze record.
        cast(PrimitiveMapping, report["events"])["walk_forward_test"] = base_strategy(
            dataset, PartitionRole.WALK_FORWARD_TEST, fold
        )
        report["out_of_sample"] = prediction_partition(oos, dataset, buckets)
        return report

    def pre_holdout_gate(
        self,
        *,
        leakage_audit: PrimitiveMapping,
        artifact_validation: PrimitiveMapping,
        holdout_state: str,
    ) -> PrimitiveMapping:
        """Record every pre-holdout check; consumption is never performed here."""
        self._require_specification()
        training = self._require_stage("training")
        frozen = self.stage("freeze") is not None

        def passed(condition: bool, detail: Primitive) -> PrimitiveMapping:
            return {"status": "passed" if condition else "failed", "detail": detail}

        def not_applicable() -> PrimitiveMapping:
            return {
                "status": "not_applicable",
                "detail": f"training ended with {training['status']}",
            }

        dataset = read_event_dataset(self.dataset_path())
        items: dict[str, PrimitiveMapping] = {
            "dataset_validation": passed(
                dataset.dataset_id == self._require_stage("dataset")["dataset_id"],
                f"read_event_dataset validated {dataset.dataset_id}",
            ),
            "feature_leakage_audit": passed(
                leakage_audit.get("status") == "passed", leakage_audit
            ),
        }
        if frozen:
            reproduction = self._require_stage("reproduction")
            checks = cast(PrimitiveMapping, reproduction["checks"])
            model = self.frozen_model()
            oos = read_prediction_set(
                self._path(self._require_stage("out_of_sample")["path"])
            )
            test_rows = dataset.rows_for(
                PartitionRole.WALK_FORWARD_TEST, fold_id=self.specification.fold_id
            )
            items.update(
                {
                    "deterministic_training": passed(
                        all(
                            checks[key] is True
                            for key in (
                                "model_configuration_identical",
                                "fitted_state_identical",
                                "model_identical",
                                "model_envelope_identical",
                            )
                        ),
                        "refit reproduced configuration, fitted state and model",
                    ),
                    "persisted_frozen_model": passed(
                        model.model_id == training["model_id"],
                        f"read_event_model validated {model.model_id}",
                    ),
                    "out_of_sample_predictions_and_metrics": passed(
                        oos.model_id == model.model_id
                        and [r.row_index for r in oos.records] == list(test_rows),
                        f"{len(oos.records)} walk-forward test predictions persisted",
                    ),
                    "offline_reproduction": passed(
                        checks["offline_reproduction_exact"] is True
                        and checks["selection_rescore_identical"] is True
                        and checks["out_of_sample_rescore_identical"] is True,
                        reproduction["offline"],
                    ),
                }
            )
        else:
            for name in (
                "deterministic_training",
                "persisted_frozen_model",
                "out_of_sample_predictions_and_metrics",
                "offline_reproduction",
            ):
                items[name] = not_applicable()
        items["artifact_manifest_validation"] = passed(
            artifact_validation.get("status") == "passed", artifact_validation
        )
        items["holdout_state"] = passed(
            holdout_state == "reserved_unconsumed",
            f"permanent ledger reports {holdout_state}",
        )
        statuses = {cast(str, item["status"]) for item in items.values()}
        if "failed" in statuses:
            outcome = "PRE_HOLDOUT_BLOCKED"
        elif statuses == {"passed"}:
            outcome = "PRE_HOLDOUT_COMPLETE"
        else:
            outcome = "PRE_HOLDOUT_COMPLETE_INSUFFICIENT_TRAINING"
        gate: PrimitiveMapping = {
            "specification_id": self.specification.specification_id,
            "outcome": outcome,
            "holdout": {
                "final_holdout_id": self.specification.plan.final_holdout.holdout_id,
                "state": holdout_state,
                "consumed_by_this_study": False,
            },
            "items": cast(PrimitiveMapping, items),
            "next_step": (
                "manual review; any holdout consumption is a separate, explicitly "
                "authorized QF-40 step with no retuning afterwards"
            ),
        }
        try:
            write_record(self.root / "pre-holdout-gate.json", gate)
        except WalkForwardPersistenceError as error:
            raise ConditionalStudyError("cannot persist the gate record") from error
        return gate


def schema_leakage_audit(
    dataset: EventDataset,
    *,
    declared_contemporaneous: frozenset[str],
    row_audit: Callable[[EventDataset], PrimitiveMapping],
) -> PrimitiveMapping:
    """Static schema causality plus a study-specific row recomputation audit."""
    schema = dataset.feature_schema
    target_columns = {"target", "target_status", "target_source_value"}
    checks: PrimitiveMapping = {
        "every_feature_known_at_decision_from_completed_inputs": all(
            item.availability is CausalAvailability.DECISION_TIMESTAMP
            for item in schema.features
        ),
        "sources_are_declared_contemporaneous_rule_features": {
            item.source_field for item in schema.features
        }
        <= declared_contemporaneous,
        "features_exclude_target_columns": not set(schema.column_names)
        & target_columns,
        "features_exclude_partition_metadata": cast(
            PrimitiveMapping, dataset.scientific.to_primitive()["partition_plan"]
        )["features_exclude_partition_metadata"]
        is True,
    }
    rows = row_audit(dataset)
    ok = all(value is True for value in checks.values()) and rows.get("status") == (
        "passed"
    )
    return {
        "status": "passed" if ok else "failed",
        "schema": checks,
        "rows": rows,
    }


__all__ = [
    "INSUFFICIENT_CLASS_OBSERVATIONS",
    "SPECIFICATION_FILE",
    "STAGES",
    "ConditionalStudy",
    "ConditionalStudyError",
    "ConditionalStudySpecification",
    "TrainingOutcome",
    "schema_leakage_audit",
]
