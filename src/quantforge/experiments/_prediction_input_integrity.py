"""Bind recorded prediction/feature sources to intraday input provenance."""

from collections.abc import Iterable

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data.exceptions import ValidationError
from quantforge.data.prediction_inputs import (
    INTRADAY_CORPORATE_ACTION_POLICY,
    validate_prediction_provenance,
    validate_prediction_source_reference,
)
from quantforge.experiments._json import ManifestError, mapping


def validate_prediction_input_sources(
    market: PrimitiveMapping,
    *,
    outcome_sources: Iterable[Primitive],
    context: Primitive,
) -> None:
    """Check persisted source semantics without loading bars or running research."""
    if (
        "intraday_provenance" not in market
        and market.get("corporate_action_policy") != INTRADAY_CORPORATE_ACTION_POLICY
    ):
        return
    try:
        provenance = validate_prediction_provenance(market)
        assert provenance is not None
        for source in outcome_sources:
            if source is not None:
                validate_prediction_source_reference(
                    provenance, mapping(mapping(source).get("source_reference"))
                )
        if context is None:
            return
        captured = mapping(context)
        # SKIP deliberately retains rejected evidence for auditability.
        if captured.get("status") != "available":
            return
        source_context = captured.get("source_context")
        if source_context is None:
            return
        timeframes = mapping(source_context).get("timeframes")
        if not isinstance(timeframes, list):
            raise ManifestError("prediction source timeframes are invalid")
        for timeframe in timeframes:
            aligned = mapping(timeframe)
            reference = aligned.get("dataset_reference")
            if reference is None:
                continue
            validate_prediction_source_reference(provenance, mapping(reference))
            requirement = mapping(aligned.get("requirement"))
            definition = mapping(
                mapping(requirement.get("timeframe")).get("configuration")
            )
            session_policy = mapping(definition.get("session_policy"))
            if configuration_identity(session_policy) != provenance.session_policy_id:
                raise ValidationError("prediction input session policy is incompatible")
    except (TypeError, ValueError, ValidationError) as error:
        raise ManifestError(
            f"prediction input provenance is invalid: {error}"
        ) from error
