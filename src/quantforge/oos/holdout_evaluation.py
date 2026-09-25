"""QF-40 boundary adapter; execution stays in QF-42/QF-43 and QF-11/QF-5."""

import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.data import MarketDataset
from quantforge.oos._records import OOSIntegrityError, mapping
from quantforge.oos.common import provenance
from quantforge.oos.models import OOSSource
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.validation import (
    PredictionMembershipSource,
    TimestampBoundary,
    ValidationWindow,
    WindowObservationSelection,
    select_window_observations,
)
from quantforge.walk_forward import BacktestEvaluator, PredictionEvaluator
from quantforge.walk_forward.models import (
    FrozenSelection,
    OOSArtifact,
    PredictionOOSArtifact,
)
from quantforge.walk_forward.partitions import (
    observation_keys,
    prediction_metadata_prefix,
    project_dataset,
)


@dataclass(frozen=True, slots=True)
class HoldoutPartition:
    """Final reservation membership; no invented QF-8 following-window purge."""

    window: ValidationWindow
    membership: WindowObservationSelection
    dataset: MarketDataset
    sessions: tuple[date, ...]
    decision_timestamps: tuple[datetime, ...] = ()
    prediction_membership: PredictionMembershipSource | None = None

    def to_primitive(self) -> PrimitiveMapping:
        primitive: PrimitiveMapping = {
            "window": self.window.to_primitive(),
            "membership": self.membership.to_primitive(),
            "purge": None,
            "evaluation_sessions": [session.isoformat() for session in self.sessions],
            "bounded_dataset_id": self.dataset.metadata.dataset_id,
            "bounded_data_sha256": self.dataset.metadata.data_sha256,
        }
        if self.prediction_membership is not None:
            primitive["prediction_membership"] = (
                self.prediction_membership.to_primitive()
            )
            primitive["evaluation_timestamps"] = [
                t.isoformat() for t in self.decision_timestamps
            ]
        return primitive


@dataclass(frozen=True, slots=True)
class HoldoutEvaluation:
    """Exact executable adapter and frozen source selection, with no selector API."""

    source: OOSSource
    evaluator: PredictionEvaluator | BacktestEvaluator
    selection: FrozenSelection
    permitted: HoldoutPartition
    adapter_snapshot: PrimitiveMappingSnapshot
    finalized_prediction_window: Path | None = None

    @classmethod
    def prepare(
        cls,
        source: OOSSource,
        evaluator: PredictionEvaluator | BacktestEvaluator,
        *,
        selection_fold_id: str,
        finalized_prediction_window: Path | None = None,
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
            source=plan.prediction_membership or dataset,
            source_timeframe=(
                plan.prediction_membership.schedule.primary_timeframe
                if plan.prediction_membership is not None
                else plan.environment.outcome_dataset.standalone_timeframe
            ),
        )
        # All outcomes must end INSIDE the reserved interval. Its final horizon
        # supplies labels only and cannot create decisions needing unseen data.
        horizon = plan.purge_policy.label_horizon
        if horizon.elapsed is not None:
            end_timestamp = cast(TimestampBoundary, window.interval.end).timestamp
            final_decision_timestamp = end_timestamp - horizon.elapsed
            retained = tuple(
                observation
                for observation in member.study_observations
                if cast(TimestampBoundary, observation).timestamp
                <= final_decision_timestamp
            )
        else:
            horizon_sessions = horizon.exchange_sessions
            assert horizon_sessions is not None
            retained = (
                member.study_observations[:-horizon_sessions]
                if horizon_sessions
                else member.study_observations
            )
        if not retained:
            raise OOSIntegrityError(
                "holdout is too short for its configured outcome horizon"
            )
        minimum = mapping(definition["configuration"])["minimum_test_observations"]
        if type(minimum) is not int or minimum < 1:
            raise OOSIntegrityError("invalid minimum_test_observations in source study")
        if len(retained) < minimum:
            raise OOSIntegrityError(
                f"holdout has {len(retained)} observations after excluding the outcome "
                f"horizon; minimum_test_observations requires {minimum}"
            )
        if plan.prediction_membership is not None:
            timestamps = tuple(
                cast(TimestampBoundary, key).timestamp for key in retained
            )
            permitted = HoldoutPartition(
                window,
                member,
                prediction_metadata_prefix(dataset, timestamps[0]),
                tuple(plan.prediction_membership.session_for(t) for t in timestamps),
                timestamps,
                plan.prediction_membership,
            )
            return cls(
                source,
                evaluator,
                fold.selection,
                permitted,
                adapter,
                finalized_prediction_window,
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
        return cls(
            source,
            evaluator,
            fold.selection,
            permitted,
            adapter,
            finalized_prediction_window,
        )

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
            finalized_prediction_window=self.finalized_prediction_window,
        )
        if prepared.configuration() != self.configuration():
            raise OOSIntegrityError("holdout frozen configuration or boundary changed")

    def _finalized_artifact(self, path: Path) -> PredictionOOSArtifact:
        if path.name != "prediction-window.jsonl" or not isinstance(
            self.evaluator, PredictionEvaluator
        ):
            raise OOSIntegrityError(
                "finalized holdout requires a compact prediction window"
            )
        reader = PredictionWindowReader.open(path)
        if reader.schema_version != "2":
            raise OOSIntegrityError("finalized holdout must use compact schema")
        artifact = PredictionOOSArtifact(
            self.selection.selection_id,
            str(reader.header()["window_result_id"]),
            PrimitiveMappingSnapshot.capture(
                {"schema_version": "2", "path": path.name, "header": reader.header()}
            ),
        )
        self.evaluator.validate_partition_artifact(
            self.source.plan, self.permitted, self.selection, artifact, path.parent
        )
        return artifact

    def _evaluate(self, output_root: Path) -> OOSArtifact:
        if self.finalized_prediction_window is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            destination = output_root / "prediction-window.jsonl"
            # Once published, the ledger's copy is authoritative, including
            # recovery before result.json was written. The external import path
            # is operational input and may no longer exist on a compatible retry.
            artifact = self._finalized_artifact(
                destination
                if destination.exists()
                else self.finalized_prediction_window
            )
            if not destination.exists():
                descriptor, name = tempfile.mkstemp(
                    prefix=".holdout-window-", dir=output_root
                )
                temporary = Path(name)
                try:
                    with (
                        os.fdopen(descriptor, "wb") as output,
                        self.finalized_prediction_window.open("rb") as source,
                    ):
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    # Validate the copied bytes before immutable publication.
                    reader = PredictionWindowReader.open(temporary)
                    reader.verify_integrity()
                    if reader.header() != artifact.snapshot.to_primitive()["header"]:
                        raise OOSIntegrityError(
                            "finalized holdout changed while copying"
                        )
                    os.link(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            self.evaluator.validate_partition_artifact(
                self.source.plan, self.permitted, self.selection, artifact, output_root
            )
            descriptor = os.open(output_root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return artifact
        result = self.evaluator.evaluate_partition(
            self.source.plan, self.permitted, self.selection, output_root
        )
        self.validate()
        self.evaluator.validate_partition_artifact(
            self.source.plan, self.permitted, self.selection, result, output_root
        )
        return result
