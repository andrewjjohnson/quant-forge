"""Immutable QF-39 orchestration records; evaluator payloads remain distinct."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.validation import ResearchStudyType, ValidationPlan


class WalkForwardError(ValueError):
    """A study cannot safely select or evaluate the declared configuration."""


class WalkForwardPersistenceError(WalkForwardError):
    """Persisted state is incomplete, incompatible, or corrupt."""


class SelectionPolicy(StrEnum):
    """Select one candidate using the existing eligible objective order."""

    BEST_ELIGIBLE = "best_eligible"
    FIRST_STABLE = "first_stable_nonisolated_in_objective_order"


class FoldStatus(StrEnum):
    PENDING = "pending"
    SELECTING = "selecting"
    SELECTION_FROZEN = "selection_frozen"
    EVALUATING_TEST = "evaluating_test"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CandidateConfiguration:
    """An existing grid combination and its exact executable definition."""

    combination_id: str
    parameters: PrimitiveMappingSnapshot
    definition: PrimitiveMappingSnapshot

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "combination_id": self.combination_id,
            "parameters": self.parameters.to_primitive(),
            "definition": self.definition.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class CandidateUniverse:
    """Finite allowed configurations, captured from a QF-6 or QF-32 grid."""

    study_type: ResearchStudyType
    candidates: tuple[CandidateConfiguration, ...]
    grid_definition: PrimitiveMappingSnapshot

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.candidates), tuple):
            raise WalkForwardError("candidate universe must be immutable")
        ids = tuple(item.combination_id for item in self.candidates)
        if len(set(ids)) != len(ids):
            raise WalkForwardError("candidate identities must be unique")

    def candidate(self, combination_id: str) -> CandidateConfiguration:
        for candidate in self.candidates:
            if candidate.combination_id == combination_id:
                return candidate
        raise WalkForwardError("selected candidate is outside the declared universe")

    @property
    def universe_id(self) -> str:
        return configuration_identity(self.to_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": "1",
            "study_type": self.study_type.value,
            "candidates": [item.to_primitive() for item in self.candidates],
            "grid_definition": self.grid_definition.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    name: str
    plan: ValidationPlan
    selection_policy: SelectionPolicy = SelectionPolicy.BEST_ELIGIBLE
    minimum_training_observations: int = 2
    minimum_test_observations: int = 1
    continue_on_failure: bool = True
    retry_failed: bool = False

    def __post_init__(self) -> None:
        if not self.name or not isinstance(
            cast(object, self.selection_policy), SelectionPolicy
        ):
            raise WalkForwardError("study name and typed selection policy are required")
        for count in (
            self.minimum_training_observations,
            self.minimum_test_observations,
        ):
            if type(count) is not int or count < 1:
                raise WalkForwardError("minimum observation counts must be positive")
        if (
            type(self.continue_on_failure) is not bool
            or type(self.retry_failed) is not bool
        ):
            raise WalkForwardError("failure policies must be boolean")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": "1",
            "name": self.name,
            "plan": self.plan.to_manifest(),
            "selection_policy": self.selection_policy.value,
            "selection_partition": "selection_if_present_else_development",
            "minimum_training_observations": self.minimum_training_observations,
            "minimum_test_observations": self.minimum_test_observations,
            "continue_on_failure": self.continue_on_failure,
            "retry_failed": self.retry_failed,
        }


@dataclass(frozen=True, slots=True)
class SelectionEvidence:
    candidate: CandidateConfiguration
    trial_id: str
    grid_study_id: str
    evidence: PrimitiveMappingSnapshot


@dataclass(frozen=True, slots=True)
class FrozenSelection:
    """Detached complete selection persisted before any test callback."""

    snapshot: PrimitiveMappingSnapshot

    @property
    def selection_id(self) -> str:
        return configuration_identity(self.snapshot.to_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {"selection_id": self.selection_id, **self.snapshot.to_primitive()}


@dataclass(frozen=True, slots=True)
class PredictionOOSArtifact:
    """Original ordered QF-42/QF-11 evidence, never portfolio accounting."""

    selection_id: str
    window_result_id: str
    snapshot: PrimitiveMappingSnapshot
    kind: str = field(default="prediction", init=False)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "kind": self.kind,
            "selection_id": self.selection_id,
            "result_id": self.window_result_id,
            "result": self.snapshot.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class BacktestOOSArtifact:
    """Original QF-5 result and integrity-checked tabular export."""

    selection_id: str
    run_id: str
    snapshot: PrimitiveMappingSnapshot
    export_location: str
    export_fingerprint: str
    kind: str = field(default="backtest", init=False)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "kind": self.kind,
            "selection_id": self.selection_id,
            "result_id": self.run_id,
            "result": self.snapshot.to_primitive(),
            "export_location": self.export_location,
            "export_fingerprint": self.export_fingerprint,
        }


type OOSArtifact = PredictionOOSArtifact | BacktestOOSArtifact

# QF-69: the frozen candidate's development window, beside its fold's state.
DEVELOPMENT_EVIDENCE_FILE = "development.json"
DEVELOPMENT_EVIDENCE_DIRECTORY = "development"


def partition_membership_identity(part: PrimitiveMapping) -> PrimitiveMapping:
    """Scientific QF-8 identities of one persisted role membership."""
    member = cast(PrimitiveMapping, part["membership"])
    purge = part.get("purge")
    timestamps = part.get("evaluation_timestamps")
    return {
        "qf8_window_id": cast(PrimitiveMapping, part["window"])["window_id"],
        "membership_selection_id": member["selection_id"],
        "purge_result_id": None
        if purge is None
        else cast(PrimitiveMapping, purge)["result_id"],
        "retained_decision_count": len(cast(list[object], timestamps))
        if isinstance(timestamps, list)
        else None,
    }


@dataclass(frozen=True, slots=True)
class DevelopmentEvidence:
    """The frozen candidate's in-sample window on its fold's development role.

    QF-39 executes its QF-32 trials on the selection window when a fold has
    one, so development is otherwise never executed. This record binds one
    additional execution of the **frozen** candidate on the exact development
    membership frozen in ``selection.json``. It never influences selection or
    test evaluation and is never OOS evidence.
    """

    study_id: str
    plan_id: str
    fold_id: str
    selection_id: str
    candidate: CandidateConfiguration
    membership: PrimitiveMappingSnapshot
    window_result_id: str
    window: PrimitiveMappingSnapshot

    def artifact(self) -> PredictionOOSArtifact:
        """The window reference in QF-39's existing prediction-artifact shape."""
        return PredictionOOSArtifact(
            self.selection_id, self.window_result_id, self.window
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": "1",
            "component": "quantforge_walk_forward_development_evidence",
            "study_id": self.study_id,
            "plan_id": self.plan_id,
            "fold_id": self.fold_id,
            "selection_id": self.selection_id,
            "role": "development_training",
            "candidate": self.candidate.to_primitive(),
            "membership": self.membership.to_primitive(),
            "window_result_id": self.window_result_id,
            "window": self.window.to_primitive(),
            "interpretation": (
                "in-sample evidence of the frozen candidate on the frozen "
                "development membership; executed after selection was frozen, "
                "never used for selection and never out-of-sample"
            ),
        }

    @classmethod
    def from_primitive(cls, value: PrimitiveMapping) -> "DevelopmentEvidence":
        try:
            candidate = cast(PrimitiveMapping, value["candidate"])
            evidence = cls(
                cast(str, value["study_id"]),
                cast(str, value["plan_id"]),
                cast(str, value["fold_id"]),
                cast(str, value["selection_id"]),
                CandidateConfiguration(
                    cast(str, candidate["combination_id"]),
                    PrimitiveMappingSnapshot.capture(
                        cast(PrimitiveMapping, candidate["parameters"])
                    ),
                    PrimitiveMappingSnapshot.capture(
                        cast(PrimitiveMapping, candidate["definition"])
                    ),
                ),
                PrimitiveMappingSnapshot.capture(
                    cast(PrimitiveMapping, value["membership"])
                ),
                cast(str, value["window_result_id"]),
                PrimitiveMappingSnapshot.capture(
                    cast(PrimitiveMapping, value["window"])
                ),
            )
        except (KeyError, TypeError) as error:
            raise WalkForwardPersistenceError(
                "invalid development evidence record"
            ) from error
        if evidence.to_primitive() != value:
            raise WalkForwardPersistenceError("unsupported development evidence record")
        return evidence


@dataclass(frozen=True, slots=True)
class FoldResult:
    fold_id: str
    status: FoldStatus
    selection: FrozenSelection | None
    artifact: OOSArtifact | None
    failures: tuple[PrimitiveMappingSnapshot, ...]


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    study_id: str
    folds: tuple[FoldResult, ...]


class WalkForwardEvaluator(Protocol):
    """Only selection and typed OOS evaluation are shared across the adapters."""

    @property
    def universe(self) -> CandidateUniverse: ...

    def configuration(self) -> PrimitiveMapping: ...

    def validate(self, plan: ValidationPlan) -> None: ...

    def select(
        self, config: WalkForwardConfig, fold_index: int, output_root: Path
    ) -> SelectionEvidence: ...

    def membership(
        self, config: WalkForwardConfig, fold_index: int
    ) -> PrimitiveMapping: ...

    def evaluate(
        self,
        config: WalkForwardConfig,
        fold_index: int,
        selection: FrozenSelection,
        output_root: Path,
    ) -> OOSArtifact: ...

    def validate_artifact(
        self,
        config: WalkForwardConfig,
        fold_index: int,
        selection: FrozenSelection,
        artifact: OOSArtifact,
        output_root: Path,
    ) -> None: ...
