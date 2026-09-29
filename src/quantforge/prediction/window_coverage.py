"""QF-64 decision coverage and research observations for every window version.

Coverage answers whether each authoritative QF-42 decision executed and with
which disposition; observations are the generated signals a decision produced.
Readers derive both views from embedded (1), full compact (2, 3) and sparse (4)
windows, so consumers need not know the physical representation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_compact import (
    CompactDecisionReceipt,
    CompactPredictionWindowDecision,
)
from quantforge.prediction.window_encoding import mapping


@dataclass(frozen=True, slots=True)
class PredictionWindowObservation:
    """One generated signal (prediction, causal features) and its labeled row.

    ``row`` is absent when the outcome was unavailable. ``observation_index``
    orders observations across the window; ``signal_index`` within a decision.
    Values are detached copies of the persisted, integrity-checked evidence.
    """

    observation_index: int
    sequence: int
    signal_index: int
    decision_timestamp: datetime
    context_id: str | None
    prediction_study_id: str
    signal_snapshot: PrimitiveMappingSnapshot
    row_snapshot: PrimitiveMappingSnapshot | None

    def signal(self) -> PrimitiveMapping:
        return self.signal_snapshot.to_primitive()

    def row(self) -> PrimitiveMapping | None:
        return None if self.row_snapshot is None else self.row_snapshot.to_primitive()


@dataclass(frozen=True, slots=True)
class PredictionDecisionReceipt:
    """Coverage of one scheduled decision, independent of physical version.

    ``status`` is the QF-42 disposition (``evaluated``, ``no_prediction`` or
    ``skipped``). ``decision`` is the complete normalized decision when the
    artifact retains one: always for versions 1-3, and for version 4 only for
    evaluated and skipped decisions. ``record`` is the persisted record itself.
    """

    sequence: int
    decision_timestamp: datetime
    status: str
    context_id: str | None
    prediction_study_id: str
    observation_start: int
    observation_count: int
    decision: CompactPredictionWindowDecision | None
    record: CompactPredictionWindowDecision | CompactDecisionReceipt

    def observations(self) -> tuple[PredictionWindowObservation, ...]:
        """Generated signals in order, each matched to its labeled row, if any."""
        if self.decision is None:
            return ()
        record = self.decision.to_primitive()
        study_rows = mapping(record["prediction_study"])["rows"]
        if not isinstance(study_rows, list):
            raise InvalidPredictionOutputError("decision rows are missing")
        rows: dict[str, PrimitiveMapping] = {}
        for item in study_rows:
            row = mapping(item)
            if "prediction" not in row or "features" not in row:
                raise InvalidPredictionOutputError("decision row is incomplete")
            key = {"prediction": row["prediction"], "features": row["features"]}
            rows[configuration_identity(key)] = row
        observations: list[PredictionWindowObservation] = []
        for index, item in enumerate(
            cast(list[Primitive], record["generated_signals"])
        ):
            signal = mapping(item)
            row = rows.get(configuration_identity(signal))
            observations.append(
                PredictionWindowObservation(
                    observation_index=self.observation_start + index,
                    sequence=self.sequence,
                    signal_index=index,
                    decision_timestamp=self.decision_timestamp,
                    context_id=self.context_id,
                    prediction_study_id=self.prediction_study_id,
                    signal_snapshot=PrimitiveMappingSnapshot.capture(signal),
                    row_snapshot=None
                    if row is None
                    else PrimitiveMappingSnapshot.capture(row),
                )
            )
        return tuple(observations)


def decision_receipt(
    record: PrimitiveMapping,
    decision: CompactPredictionWindowDecision,
    *,
    decision_timestamp: datetime,
    observation_start: int,
) -> PredictionDecisionReceipt:
    """Coverage derived from a checked version 1-3 decision record."""
    return PredictionDecisionReceipt(
        sequence=cast(int, record["sequence"]),
        decision_timestamp=decision_timestamp,
        status=cast(str, record["status"]),
        context_id=cast(str | None, record["context_id"]),
        prediction_study_id=cast(str, record["prediction_study_id"]),
        observation_start=observation_start,
        observation_count=len(cast(list[Primitive], record["generated_signals"])),
        decision=decision,
        record=decision,
    )


def sparse_decision_receipt(
    receipt: CompactDecisionReceipt,
    decision: CompactPredictionWindowDecision | None,
    *,
    decision_timestamp: datetime,
    observation_start: int,
) -> PredictionDecisionReceipt:
    """Coverage of a checked version 4 receipt, with its accepted evidence."""
    return PredictionDecisionReceipt(
        sequence=receipt.sequence,
        decision_timestamp=decision_timestamp,
        status=receipt.status,
        context_id=receipt.context_id,
        prediction_study_id=receipt.prediction_study_id,
        observation_start=observation_start,
        observation_count=receipt.observation_count,
        decision=decision,
        record=receipt,
    )


__all__ = [
    "PredictionDecisionReceipt",
    "PredictionWindowObservation",
    "decision_receipt",
    "sparse_decision_receipt",
]
