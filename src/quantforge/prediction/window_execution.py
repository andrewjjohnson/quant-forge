"""Incremental QF-42 execution using QF-11 and QF-55 contracts unchanged."""

from datetime import datetime
from pathlib import Path

from quantforge.configuration import PrimitiveMapping
from quantforge.data.models import DatasetMetadata
from quantforge.prediction.contracts import (
    EvaluationValuesT,
    OutcomeValuesT,
    PredictionRecordT,
    PredictionStudy,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.study import (
    PredictionStudyDatasetSession,
    _capture_study_configuration,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.window import (
    PredictionDecisionSchedule,
    PredictionWindowContextProvider,
    _capture_window_identity,  # pyright: ignore[reportPrivateUsage]
    iter_prediction_window_decisions,
)
from quantforge.prediction.window_compact import CompactPredictionWindowDecision
from quantforge.prediction.window_compact_validation import (
    PredictionWindowDecisionValidator,
)
from quantforge.prediction.window_incremental import IncrementalPredictionWindowWriter
from quantforge.prediction.window_reader import PredictionWindowReader


class PredictionWindowDecisionError(InvalidPredictionOutputError):
    """Credential-free decision identity for durable trial/fold diagnostics."""

    def __init__(
        self, *, sequence: int, timestamp: datetime, window_id: str, cause_type: str
    ) -> None:
        self.safe_message = (
            f"historical decision sequence={sequence} "
            f"timestamp={timestamp.isoformat()} "
            f"window={window_id} failed ({cause_type})"
        )
        super().__init__(self.safe_message)


def run_incremental_prediction_window_in_session(
    prepared: PredictionStudyDatasetSession,
    study: PredictionStudy[PredictionRecordT, OutcomeValuesT, EvaluationValuesT],
    *,
    path: Path,
    schedule: PredictionDecisionSchedule,
    context_provider: PredictionWindowContextProvider,
    dataset_family_fingerprint: str,
    context_environment: PrimitiveMapping,
    indicator_backend_environment: PrimitiveMapping | None = None,
    canonical_metadata: DatasetMetadata | None = None,
) -> PredictionWindowReader:
    """Resume verified work, execute/validate/compact/commit/release each suffix item.

    Historical caches are deliberately not retained across decisions on this
    path. QF-11 still computes and validates the normal normalized indicators.
    Independent canonical metadata is used only by provenance verification.
    """
    identity = _capture_window_identity(
        prepared,
        _capture_study_configuration(study),
        schedule=schedule,
        dataset_family_fingerprint=dataset_family_fingerprint,
        context_environment=context_environment,
        indicator_backend_environment=indicator_backend_environment,
    )
    validator = PredictionWindowDecisionValidator(
        expected_identity=identity,
        schedule=schedule,
        outcome_sessions=tuple(prepared.bar_indexes),
        strategy_parameters=study.strategy.parameters.to_primitive(),
        canonical_metadata=canonical_metadata,
    )
    writer = IncrementalPredictionWindowWriter.open(path, validator=validator)
    if writer.finalized:
        return writer.finalize()
    decisions = iter_prediction_window_decisions(
        prepared,
        study,
        schedule=schedule,
        context_provider=context_provider,
        dataset_family_fingerprint=dataset_family_fingerprint,
        start_sequence=writer.completed_count,
    )
    for sequence in range(writer.completed_count, len(schedule.decision_timestamps)):
        try:
            decision = next(decisions)
            compact = CompactPredictionWindowDecision.from_embedded(
                decision.to_primitive(),
                sequence=sequence,
                evidence=writer.evidence,
            )
            writer.append(compact)
            del decision, compact
        except Exception as error:
            # Preserve cause for callers; durable grid diagnostics remain sanitized.
            raise PredictionWindowDecisionError(
                sequence=sequence,
                timestamp=schedule.decision_timestamps[sequence],
                window_id=writer.evidence.window_id,
                cause_type=type(error).__name__,
            ) from error
    del decisions
    return writer.finalize()
