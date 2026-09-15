"""Bind retained QF-28 context evidence to a QF-42 decision."""

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_context_validation import (
    validate_window_context_snapshot,
    validate_window_source_snapshot,
)


def validate_decision_context(
    decision: PrimitiveMapping,
    window: PrimitiveMapping,
    study: PrimitiveMapping,
) -> bool:
    """Check stored context identities and timing; return whether it was skipped.

    A skipped decision may retain the wrong timestamp/family that caused its
    rejection. Validate that evidence internally without treating it as usable.
    The producer's offline helpers calculate no indicators or market context.
    """
    context = mapping(study.get("prediction_context"))
    requirements = mapping(context.get("requirements"))
    configuration = mapping(window.get("configuration"))
    if context.get("status") not in {"available", "skipped"} or requirements != (
        configuration.get("prediction_context_requirements")
    ):
        raise ManifestError("decision context requirements are inconsistent")
    skipped = context["status"] == "skipped"
    if skipped:
        if requirements.get("failure_policy") != "skip":
            raise ManifestError(
                "skipped decision contradicts its context failure policy"
            )
        text(context.get("reason"))
    source = context.get("source_context")
    if source is None:
        if not skipped or decision.get("context_id") is not None:
            raise ManifestError("decision source context evidence is missing")
        return skipped
    source = mapping(source)
    context_id = configuration_identity(
        {key: value for key, value in source.items() if key != "context_id"}
    )
    if (
        source.get("context_id") != context_id
        or decision.get("context_id") != context_id
    ):
        raise ManifestError("decision source context identity is inconsistent")
    try:
        if skipped:
            validate_window_source_snapshot(source)
        else:
            timestamp = text(decision.get("decision_timestamp"))
            if source.get("as_of") != timestamp or mapping(
                source.get("source_consistency")
            ).get("family_id") != window.get("dataset_family_fingerprint"):
                raise ManifestError(
                    "decision context timing or lineage is inconsistent"
                )
            primary = mapping(mapping(window.get("schedule")).get("primary_timeframe"))
            validate_window_context_snapshot(
                context=context,
                source=source,
                requirements=requirements,
                market_data=mapping(window.get("market_data")),
                primary_timeframe={
                    "configuration": primary,
                    "configuration_id": configuration_identity(primary),
                },
                timestamp=timestamp,
            )
    except (InvalidPredictionOutputError, KeyError) as error:
        raise ManifestError("decision context metadata is inconsistent") from error
    return skipped
