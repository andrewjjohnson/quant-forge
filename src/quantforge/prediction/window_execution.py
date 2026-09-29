"""Incremental QF-42 execution using QF-11 and QF-55 contracts unchanged."""

from datetime import datetime
from pathlib import Path
from typing import Any

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.data.models import DatasetMetadata
from quantforge.data.prepared_prediction_views import PreparedProjectionRegistry
from quantforge.prediction.contracts import (
    EvaluationValuesT,
    OutcomeValuesT,
    PredictionRecordT,
    PredictionStudy,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.study import (
    PredictionStudyDatasetSession,
    PredictionStudyResult,
    _capture_study_configuration,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.window import (
    PredictionDecisionSchedule,
    PredictionWindowContextProvider,
    PredictionWindowDecision,
    _capture_window_identity,  # pyright: ignore[reportPrivateUsage]
    iter_prediction_window_decisions,
    iter_prediction_window_results,
)
from quantforge.prediction.window_compact import (
    COMPACT_PREDICTION_WINDOW_SCHEMA_VERSION,
    CompactDecisionReceipt,
    CompactPredictionWindowDecision,
)
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


def sparse_decision_receipt(
    sequence: int,
    timestamp: datetime,
    result: PredictionStudyResult[Any, Any, Any],
    *,
    writer: IncrementalPredictionWindowWriter,
    validator: PredictionWindowDecisionValidator,
) -> CompactDecisionReceipt:
    """Schema 4 record for one executed decision (QF-64).

    Decisions with generated signals, and skipped decisions with their rejected
    context, keep the complete normalized decision and are fully validated on
    append. An ordinary ``no_prediction`` decision builds no decision snapshot:
    its context is decoded once, bound to the schedule, scope and both
    identities, and only the identities are persisted.
    """
    snapshot = result.prediction_context_snapshot
    if snapshot is None:
        raise InvalidPredictionOutputError(
            "historical results require QF-28 context evidence"
        )
    context = snapshot.to_primitive()
    if result.signals or context.get("status") == "skipped":
        decision = PredictionWindowDecision(timestamp, result)
        return CompactDecisionReceipt.with_decision(
            CompactPredictionWindowDecision.from_embedded(
                decision.to_primitive(),
                sequence=sequence,
                evidence=writer.evidence,
                membership=writer.membership,
            )
        )
    context_id = validator.validate_no_prediction(
        context,
        sequence,
        prediction_study_id=result.study_id,
        record_counts={
            "generated_predictions": result.generated_prediction_count,
            "labeled_rows": len(result.rows),
            "unavailable_outcomes": result.unavailable_outcome_count,
        },
    )
    return CompactDecisionReceipt.no_prediction(
        sequence=sequence, context_id=context_id, prediction_study_id=result.study_id
    )


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
    projection_registry: PreparedProjectionRegistry | None = None,
    projection_scope: PrimitiveMappingSnapshot | None = None,
    schema_version: str = COMPACT_PREDICTION_WINDOW_SCHEMA_VERSION,
) -> PredictionWindowReader:
    """Resume verified work, execute/validate/compact/commit/release each suffix item.

    Historical caches are deliberately not retained across decisions on this
    path. QF-11 still computes and validates the normal normalized indicators.
    Independent canonical metadata is used only by provenance verification.
    ``schema_version`` "3" persists QF-62 normalized membership and "4" QF-64
    sparse decision receipts; execution and every scientific result are
    identical for all physical representations.
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
        projection_registry=projection_registry,
        projection_scope=projection_scope,
    )
    writer = IncrementalPredictionWindowWriter.open(
        path, validator=validator, schema_version=schema_version
    )
    if writer.finalized:
        return writer.finalize()
    if writer.evidence.sparse_decisions:
        results = iter_prediction_window_results(
            prepared,
            study,
            schedule=schedule,
            context_provider=context_provider,
            dataset_family_fingerprint=dataset_family_fingerprint,
            start_sequence=writer.completed_count,
        )
        for sequence in range(
            writer.completed_count, len(schedule.decision_timestamps)
        ):
            try:
                timestamp, result = next(results)
                receipt = sparse_decision_receipt(
                    sequence, timestamp, result, writer=writer, validator=validator
                )
                del result
                writer.append(receipt)
                del receipt
            except Exception as error:
                raise PredictionWindowDecisionError(
                    sequence=sequence,
                    timestamp=schedule.decision_timestamps[sequence],
                    window_id=writer.evidence.window_id,
                    cause_type=type(error).__name__,
                ) from error
        del results
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
                membership=writer.membership,
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
