"""Consume QF-46 requests in prediction execution; no outcome calculations."""

from datetime import datetime

from quantforge.configuration import PrimitiveMapping
from quantforge.data import TimeframeBarSeries
from quantforge.prediction.contracts import PredictionRecord
from quantforge.prediction.errors import InvalidPredictionConfigurationError
from quantforge.prediction.outcome_resolution import (
    OutcomeEvaluationRequest,
    OutcomeResolution,
    resolve_future_observation,
)
from quantforge.prediction.outcome_temporal import OutcomeTemporalConfiguration


def decision_timestamp(signal: PredictionRecord) -> datetime | None:
    """Read the original captured anchor, never infer it from a session date."""
    timestamp = signal.prediction_primitive().get("decision_timestamp")
    if timestamp is None:
        return None
    if not isinstance(timestamp, str):
        raise InvalidPredictionConfigurationError(
            "decision timestamp must be serialized"
        )
    result = datetime.fromisoformat(timestamp)
    if result.utcoffset() is None:
        raise InvalidPredictionConfigurationError("decision timestamp must be aware")
    return result


def source_provenance(
    source: TimeframeBarSeries, temporal: OutcomeTemporalConfiguration
) -> PrimitiveMapping:
    if (
        source.timeframe != temporal.observation_timeframe
        or not source.dataset_family_manifest_id
    ):
        raise InvalidPredictionConfigurationError(
            "outcome source timeframe/lineage differs"
        )
    return {
        "source_reference": source.dataset_reference.to_primitive(
            include_feed_scope=True
        ),
        "family_manifest_id": source.dataset_family_manifest_id,
    }


def bounded_outcome_source(
    source: TimeframeBarSeries, request: OutcomeEvaluationRequest
) -> tuple[TimeframeBarSeries, OutcomeResolution]:
    """Bound labeler inputs, but classify availability using full source coverage."""
    decision = request.anchor.decision_timestamp
    assert decision is not None
    end = decision + request.temporal_configuration.required_future_duration
    session_bars = tuple(
        bar
        for bar in source.bars
        if getattr(bar, "session_date", None) == request.anchor.signal_session
    )
    preceding = tuple(bar for bar in session_bars if bar.end_timestamp <= decision)
    bars = (
        *preceding[-1:],
        *(bar for bar in session_bars if decision < bar.end_timestamp <= end),
    )
    bounded = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        source.dataset_reference,
        source.timeframe,
        bars,
        dataset_family_manifest_id=source.dataset_family_manifest_id,
    )
    return bounded, resolve_future_observation(request, source)
