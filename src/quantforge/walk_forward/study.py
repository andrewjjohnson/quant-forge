"""Ordered selection/freeze/test orchestration; no combined OOS calculations."""

from pathlib import Path
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.walk_forward._adapters import frozen_candidate
from quantforge.walk_forward.models import (
    BacktestOOSArtifact,
    FoldResult,
    FoldStatus,
    FrozenSelection,
    OOSArtifact,
    PredictionOOSArtifact,
    WalkForwardConfig,
    WalkForwardError,
    WalkForwardEvaluator,
    WalkForwardPersistenceError,
    WalkForwardResult,
)
from quantforge.walk_forward.persistence import read_record, write_record


def _load_artifact(record: PrimitiveMapping) -> OOSArtifact:
    snapshot = PrimitiveMappingSnapshot.capture(
        cast(PrimitiveMapping, record["result"])
    )
    selection_id = cast(str, record["selection_id"])
    result_id = cast(str, record["result_id"])
    if record["kind"] == "prediction":
        return PredictionOOSArtifact(selection_id, result_id, snapshot)
    if record["kind"] == "backtest":
        return BacktestOOSArtifact(
            selection_id,
            result_id,
            snapshot,
            cast(str, record["export_location"]),
            cast(str, record["export_fingerprint"]),
        )
    raise WalkForwardPersistenceError("unknown OOS artifact type")


class WalkForwardStudy:
    """One writer, explicit QF-8 fold order, atomic freeze before OOS execution."""

    def __init__(
        self,
        config: WalkForwardConfig,
        evaluator: WalkForwardEvaluator,
        output_root: Path,
    ) -> None:
        evaluator.validate(config.plan)
        self.config = config
        self.evaluator = evaluator
        self._config = PrimitiveMappingSnapshot.capture(config.to_primitive())
        self._adapter = PrimitiveMappingSnapshot.capture(evaluator.configuration())
        self._universe = evaluator.universe
        if (
            self._adapter.to_primitive().get("universe")
            != self._universe.to_primitive()
        ):
            raise WalkForwardError("adapter changed after candidate universe capture")
        self._identity = PrimitiveMappingSnapshot.capture(
            {
                "component": "quantforge_walk_forward",
                "engine_version": "1",
                "configuration": self._config.to_primitive(),
                "adapter": self._adapter.to_primitive(),
                "candidate_universe_id": self._universe.universe_id,
            }
        )
        self.study_id = configuration_identity(self._identity.to_primitive())
        self.study_path = Path(output_root) / self.study_id

    def _unchanged(self) -> None:
        if (
            PrimitiveMappingSnapshot.capture(self.config.to_primitive()) != self._config
            or PrimitiveMappingSnapshot.capture(self.evaluator.configuration())
            != self._adapter
            or self.evaluator.universe != self._universe
        ):
            raise WalkForwardError(
                "walk-forward configuration changed during execution"
            )

    def run(self) -> WalkForwardResult:
        return self._execute(resume=False)

    def resume(self) -> WalkForwardResult:
        return self._execute(resume=True)

    def _execute(self, *, resume: bool) -> WalkForwardResult:
        self._unchanged()
        manifest_path = self.study_path / "manifest.json"
        if manifest_path.exists() and not resume:
            raise WalkForwardPersistenceError("study already exists; use resume")
        write_record(manifest_path, self._identity.to_primitive(), immutable=True)
        results: list[FoldResult] = []
        for index, fold in enumerate(self.config.plan.folds):
            result = self._fold(index, fold.fold_id)
            results.append(result)
            if (
                result.status is FoldStatus.FAILED
                and not self.config.continue_on_failure
            ):
                break
        return WalkForwardResult(self.study_id, tuple(results))

    def _load_selection(
        self, path: Path, fold_id: str, membership: PrimitiveMapping | None
    ) -> FrozenSelection | None:
        if not path.exists():
            return None
        record = read_record(path)
        selection_id = record.pop("selection_id")
        selection = FrozenSelection(PrimitiveMappingSnapshot.capture(record))
        if (
            selection.selection_id != selection_id
            or record.get("study_id") != self.study_id
            or record.get("fold_id") != fold_id
            or record.get("plan_id") != self.config.plan.plan_id
            or record.get("candidate_universe_id") != self._universe.universe_id
            or record.get("study_definition") != self._identity.to_primitive()
            or (membership is not None and record.get("membership") != membership)
        ):
            raise WalkForwardPersistenceError("frozen selection identity mismatch")
        frozen_candidate(self._universe, selection)
        return selection

    def _fold(self, index: int, fold_id: str) -> FoldResult:
        root = self.study_path / "folds" / fold_id
        state_path, selection_path = root / "state.json", root / "selection.json"
        if state_path.exists():
            state = read_record(state_path)
            if (
                state.get("study_id") != self.study_id
                or state.get("fold_id") != fold_id
            ):
                raise WalkForwardPersistenceError(
                    "incompatible persisted fold identity"
                )
        else:
            if selection_path.exists() or (root / "oos.json").exists():
                raise WalkForwardPersistenceError(
                    "orphaned fold artifacts have no execution state"
                )
            state: PrimitiveMapping = {
                "study_id": self.study_id,
                "fold_id": fold_id,
                "status": FoldStatus.PENDING.value,
                "selection_id": None,
                "artifact_id": None,
                "failures": [],
            }
            write_record(state_path, state)
        try:
            status = FoldStatus(cast(str, state["status"]))
            failures = tuple(
                PrimitiveMappingSnapshot.capture(item)
                for item in cast(list[PrimitiveMapping], state["failures"])
            )
        except (ValueError, KeyError, TypeError) as error:
            raise WalkForwardPersistenceError("invalid fold state") from error
        selection = self._load_selection(selection_path, fold_id, None)
        frozen_states = (
            FoldStatus.SELECTION_FROZEN,
            FoldStatus.EVALUATING_TEST,
            FoldStatus.COMPLETED,
        )
        if (status in frozen_states and selection is None) or (
            state["selection_id"] is not None
            and (selection is None or state["selection_id"] != selection.selection_id)
        ):
            raise WalkForwardPersistenceError("fold state lost its frozen selection")
        if status is FoldStatus.FAILED and not self.config.retry_failed:
            return FoldResult(fold_id, status, selection, None, failures)
        if status is FoldStatus.COMPLETED:
            assert selection is not None
            permitted = self.evaluator.membership(self.config, index)
            self._load_selection(selection_path, fold_id, permitted)
            record = read_record(root / "oos.json")
            if configuration_identity(record) != state["artifact_id"]:
                raise WalkForwardPersistenceError(
                    "completed OOS artifact fingerprint mismatch"
                )
            artifact = _load_artifact(record)
            self.evaluator.validate_artifact(
                self.config, index, selection, artifact, root / "test"
            )
            self._unchanged()
            return FoldResult(fold_id, status, selection, artifact, failures)

        stage = "membership"
        try:
            permitted = self.evaluator.membership(self.config, index)
            self._unchanged()
            selection = self._load_selection(selection_path, fold_id, permitted)
            if selection is None:
                stage = "selection"
                state["status"] = FoldStatus.SELECTING.value
                write_record(state_path, state)
                evidence = self.evaluator.select(self.config, index, root / "selection")
                self._unchanged()
                if (
                    self._universe.candidate(evidence.candidate.combination_id)
                    != evidence.candidate
                ):
                    raise WalkForwardError(
                        "selection returned a configuration outside the universe"
                    )
                selection = FrozenSelection(
                    PrimitiveMappingSnapshot.capture(
                        {
                            "schema_version": "1",
                            "study_id": self.study_id,
                            "study_definition": self._identity.to_primitive(),
                            "plan_id": self.config.plan.plan_id,
                            "fold_id": fold_id,
                            "candidate_universe_id": self._universe.universe_id,
                            "candidate": evidence.candidate.to_primitive(),
                            "selected_trial_id": evidence.trial_id,
                            "selection_grid_study_id": evidence.grid_study_id,
                            "selection_evidence": evidence.evidence.to_primitive(),
                            "membership": permitted,
                        }
                    )
                )
                stage = "freeze_persistence"
                write_record(selection_path, selection.to_primitive(), immutable=True)
            state["selection_id"] = selection.selection_id
            state["status"] = FoldStatus.SELECTION_FROZEN.value
            write_record(state_path, state)
            # Both the immutable snapshot and its state are durable BEFORE test.
            stage = "test"
            state["status"] = FoldStatus.EVALUATING_TEST.value
            write_record(state_path, state)
            artifact = self.evaluator.evaluate(
                self.config, index, selection, root / "test"
            )
            self._unchanged()
            if self._load_selection(selection_path, fold_id, permitted) != selection:
                raise WalkForwardError("test execution changed the frozen selection")
            self.evaluator.validate_artifact(
                self.config, index, selection, artifact, root / "test"
            )
            self._unchanged()
            stage = "artifact_persistence"
            write_record(root / "oos.json", artifact.to_primitive(), immutable=True)
            state["artifact_id"] = configuration_identity(artifact.to_primitive())
            state["status"] = FoldStatus.COMPLETED.value
            write_record(state_path, state)
            return FoldResult(
                fold_id, FoldStatus.COMPLETED, selection, artifact, failures
            )
        except Exception as error:
            # A fold is an execution boundary. Keep sanitized diagnostics and all
            # prior attempts, then apply the explicit continue/retry policy.
            failure = PrimitiveMappingSnapshot.capture(
                {
                    "attempt": len(failures) + 1,
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "message": str(error)
                    if type(error) is WalkForwardError
                    else "evaluator or persistence operation failed",
                }
            )
            failures = (*failures, failure)
            state["status"] = FoldStatus.FAILED.value
            state["failures"] = [item.to_primitive() for item in failures]
            state["artifact_id"] = None
            write_record(state_path, state)
            return FoldResult(fold_id, FoldStatus.FAILED, selection, None, failures)
