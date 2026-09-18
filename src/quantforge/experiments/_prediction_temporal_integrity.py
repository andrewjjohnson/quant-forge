"""Read exact elapsed anchors and context evidence without running research."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._json import ManifestError, mapping, text

if TYPE_CHECKING:
    from quantforge.prediction.outcome_temporal import OutcomeTemporalConfiguration


def elapsed_temporal_configuration(
    manifest: PrimitiveMapping,
) -> OutcomeTemporalConfiguration | None:
    from quantforge.prediction.outcome_temporal import (
        ElapsedDurationHorizon,
        outcome_temporal_configuration,
    )

    labeler = mapping(mapping(manifest.get("configuration")).get("outcome_labeler"))
    try:
        temporal = outcome_temporal_configuration(
            mapping(labeler.get("configuration")),
            required_future_sessions=cast(
                int | None, labeler.get("required_future_sessions")
            ),
        )
    except ValueError as error:
        raise ManifestError("prediction temporal configuration is invalid") from error
    return temporal if isinstance(temporal.horizon, ElapsedDurationHorizon) else None


def prediction_observation_key(
    manifest: PrimitiveMapping, prediction: PrimitiveMapping
) -> tuple[str, str]:
    session = text(prediction.get("signal_session"))
    if elapsed_temporal_configuration(manifest) is None:
        return session, ""
    context = manifest.get("prediction_context")
    anchor = mapping(prediction.get("values")).get("decision_timestamp")
    if isinstance(context, dict) and context.get("status") == "available":
        original = mapping(context.get("source_context")).get("as_of")
        if anchor is not None and anchor != original:
            raise ManifestError("elapsed prediction anchor differs from context")
        anchor = original
    try:
        timestamp = datetime.fromisoformat(text(anchor))
        if timestamp.utcoffset() is None:
            raise ValueError("naive anchor")
    except ValueError as error:
        raise ManifestError(
            "elapsed prediction requires an exact aware anchor"
        ) from error
    return session, timestamp.astimezone(UTC).isoformat()


def primary_context_observations(manifest: PrimitiveMapping) -> int | None:
    """Count primary bars only after the existing context verifier accepts them."""
    from quantforge.experiments._prediction_context_integrity import (
        validate_prediction_context,
    )

    context = manifest.get("prediction_context")
    if not isinstance(context, dict) or context.get("status") != "available":
        return None
    validate_prediction_context(manifest)
    timeframes = context.get("timeframes")
    if not isinstance(timeframes, list) or not timeframes:
        raise ManifestError("elapsed prediction lacks primary context evidence")
    bars = mapping(timeframes[0]).get("visible_bar_ids")
    if not isinstance(bars, list):
        raise ManifestError("elapsed prediction lacks primary warm-up evidence")
    return len(bars)


def validate_elapsed_resolution(
    manifest: PrimitiveMapping, prediction: PrimitiveMapping, outcome: PrimitiveMapping
) -> None:
    from quantforge.prediction.window_validation import (
        _validate_timestamp_resolution,  # pyright: ignore[reportPrivateUsage]
    )

    temporal = elapsed_temporal_configuration(manifest)
    assert temporal is not None
    labeler = mapping(mapping(manifest.get("configuration")).get("outcome_labeler"))
    try:
        _validate_timestamp_resolution(
            outcome,
            prediction,
            labeler,
            temporal,
            prediction_observation_key(manifest, prediction)[1],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError("elapsed prediction resolution is inconsistent") from error
