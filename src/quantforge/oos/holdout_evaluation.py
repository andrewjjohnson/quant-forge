"""QF-40 boundary adapter; execution stays in QF-42/QF-43 and QF-11/QF-5."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.data import MarketDataset
from quantforge.oos._records import OOSIntegrityError
from quantforge.oos.common import provenance
from quantforge.oos.models import OOSSource
from quantforge.validation import (
    ValidationWindow,
    WindowObservationSelection,
    select_window_observations,
)
from quantforge.walk_forward import BacktestEvaluator, PredictionEvaluator
from quantforge.walk_forward.models import FrozenSelection, OOSArtifact
from quantforge.walk_forward.partitions import (
    observation_keys,
    project_dataset,
)


@dataclass(frozen=True, slots=True)
class HoldoutPartition:
    """Final reservation membership; no invented QF-8 following-window purge."""

    window: ValidationWindow
    membership: WindowObservationSelection
    dataset: MarketDataset
    sessions: tuple[date, ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "window": self.window.to_primitive(),
            "membership": self.membership.to_primitive(),
            "purge": None,
            "evaluation_sessions": [session.isoformat() for session in self.sessions],
            "bounded_dataset_id": self.dataset.metadata.dataset_id,
            "bounded_data_sha256": self.dataset.metadata.data_sha256,
        }


@dataclass(frozen=True, slots=True)
class HoldoutEvaluation:
    """Exact executable adapter and frozen source selection, with no selector API."""

    source: OOSSource
    evaluator: PredictionEvaluator | BacktestEvaluator
    selection: FrozenSelection
    permitted: HoldoutPartition
    adapter_snapshot: PrimitiveMappingSnapshot

    @classmethod
    def prepare(
        cls,
        source: OOSSource,
        evaluator: PredictionEvaluator | BacktestEvaluator,
        *,
        selection_fold_id: str,
    ) -> "HoldoutEvaluation":
        provenance(source)
        evaluator.validate(source.plan)
        definition = source.definition.to_primitive()
        adapter = PrimitiveMappingSnapshot.capture(evaluator.configuration())
        if adapter.to_primitive() != definition["adapter"]:
            raise OOSIntegrityError(
                "holdout evaluator differs from original research configuration"
            )
        fold = next((f for f in source.folds if f.fold_id == selection_fold_id), None)
        if fold is None or fold.selection is None:
            raise OOSIntegrityError(
                "holdout requires an existing QF-39 frozen selection"
            )
        plan, dataset = source.plan, evaluator.dataset
        window = plan.final_holdout.window
        keys = observation_keys(dataset, plan)
        member = select_window_observations(
            window,
            keys,
            source=dataset,
            source_timeframe=plan.environment.outcome_dataset.standalone_timeframe,
        )
        # All outcomes must end INSIDE the reserved interval. Its final horizon
        # supplies labels only and cannot create decisions needing unseen data.
        horizon = plan.purge_policy.label_horizon.exchange_sessions or 0
        retained = (
            member.study_observations[:-horizon]
            if horizon
            else member.study_observations
        )
        if not retained:
            raise OOSIntegrityError(
                "holdout is too short for its configured outcome horizon"
            )
        start_key = (member.warm_up_context or retained)[0]
        start = keys.index(start_key)
        end = keys.index(member.study_observations[-1])
        sessions = tuple(dataset.bars[keys.index(key)].session_date for key in retained)
        permitted = HoldoutPartition(
            window,
            member,
            project_dataset(
                dataset,
                dataset.bars[start].session_date,
                dataset.bars[end].session_date,
            ),
            sessions,
        )
        return cls(source, evaluator, fold.selection, permitted, adapter)

    def configuration(self) -> PrimitiveMapping:
        return {
            "schema_version": "1",
            "operation": "evaluate_frozen_final_holdout",
            "lineage_id": self.source.lineage_id,
            "lineage": self.source.lineage.to_primitive(),
            "plan_id": self.source.plan.plan_id,
            "study_id": self.source.study_id,
            "study_definition": self.source.definition.to_primitive(),
            "holdout": self.source.plan.final_holdout.to_primitive(),
            "frozen_selection": self.selection.to_primitive(),
            "evaluation_membership": self.permitted.to_primitive(),
            "tail_policy": "outcome_horizon_remains_inside_holdout; no_tail_decisions",
        }

    def validate(self) -> None:
        self.evaluator.validate(self.source.plan)
        if (
            PrimitiveMappingSnapshot.capture(self.evaluator.configuration())
            != self.adapter_snapshot
            or self.adapter_snapshot.to_primitive()
            != self.source.definition.to_primitive()["adapter"]
        ):
            raise OOSIntegrityError("holdout executable configuration changed")
        prepared = HoldoutEvaluation.prepare(
            self.source,
            self.evaluator,
            selection_fold_id=str(self.selection.snapshot.to_primitive()["fold_id"]),
        )
        if prepared.configuration() != self.configuration():
            raise OOSIntegrityError("holdout frozen configuration or boundary changed")

    def _evaluate(self, output_root: Path) -> OOSArtifact:
        result = self.evaluator.evaluate_partition(
            self.source.plan, self.permitted, self.selection, output_root
        )
        self.validate()
        self.evaluator.validate_partition_artifact(
            self.source.plan, self.permitted, self.selection, result, output_root
        )
        return result
