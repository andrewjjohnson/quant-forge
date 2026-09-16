"""Check standalone QF-28 evidence without rebuilding or evaluating context."""

from datetime import date, datetime

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments._prediction_sessions import session_text
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_context_validation import (
    validate_window_context_snapshot,
    validate_window_source_snapshot,
)
from quantforge.prediction.window_timeframe_validation import (
    validate_source_timeframe_definition,
)
from quantforge.timeframes import IntradayInterval, resolve_exchange_session


def validate_prediction_context(manifest: PrimitiveMapping) -> None:
    """Validate available or rejected evidence, including SKIP counts."""
    declared = mapping(manifest.get("configuration")).get(
        "prediction_context_requirements"
    )
    if declared is None and "prediction_context" not in manifest:
        return
    context = mapping(manifest.get("prediction_context"))
    requirements = mapping(context.get("requirements"))
    if context.get("status") not in {"available", "skipped"} or configuration_identity(
        requirements
    ) != configuration_identity(mapping(declared)):
        raise ManifestError(
            "prediction context requirements or status are inconsistent"
        )
    skipped = context["status"] == "skipped"
    if skipped:
        if (
            requirements.get("failure_policy") != "skip"
            or mapping(manifest.get("record_counts")).get("generated_predictions") != 0
        ):
            raise ManifestError(
                "skipped prediction context contradicts policy or counts"
            )
        text(context.get("reason"))
    source = context.get("source_context")
    if source is None:
        if not skipped:
            raise ManifestError("prediction source context evidence is missing")
        return
    source = mapping(source)
    if source.get("context_id") != configuration_identity(
        {key: value for key, value in source.items() if key != "context_id"}
    ):
        raise ManifestError("prediction source context identity is inconsistent")
    try:
        if skipped:
            validate_window_source_snapshot(source)
            return
        primary = mapping(mapping(requirements.get("primary")).get("timeframe"))
        validate_window_context_snapshot(
            context=context,
            source=source,
            requirements=requirements,
            market_data=mapping(manifest.get("market_data")),
            primary_timeframe=primary,
        )
        timeframe = validate_source_timeframe_definition(primary)
        if not isinstance(timeframe.interval, IntradayInterval):
            raise ManifestError("prediction primary timeframe must be intraday")
        aligned = source["timeframes"]
        if not isinstance(aligned, list) or not aligned:
            raise ManifestError("prediction primary timeframe evidence is missing")
        boundary = datetime.fromisoformat(
            text(mapping(aligned[0]).get("latest_completed_bar_timestamp"))
        )
        # Use actual exchange bounds: overnight sessions and midnight closes
        # can carry a different trade date from the bar end's local date.
        session = resolve_exchange_session(
            date.fromisoformat(session_text(context.get("decision_session"))),
            timeframe.session_policy,
        )
        if not session.open_timestamp < boundary <= session.close_timestamp:
            raise ManifestError("prediction context decision session is inconsistent")
    except (InvalidPredictionOutputError, KeyError, ValueError) as error:
        raise ManifestError("prediction context metadata is inconsistent") from error
