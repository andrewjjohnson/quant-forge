"""Bind recorded prediction/feature sources to intraday input provenance."""

from collections.abc import Iterable

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import INTRADAY_PREDICTION_DATASET_PREFIX
from quantforge.data.prediction_inputs import (
    INTRADAY_CORPORATE_ACTION_POLICY,
    validate_prediction_provenance,
    validate_prediction_source_reference,
)
from quantforge.experiments._json import ManifestError, mapping
from quantforge.prediction.outcome_temporal import (
    ElapsedDurationHorizon,
    outcome_temporal_configuration,
)
from quantforge.prediction.window_context_validation import (
    validate_window_context_snapshot,
)
from quantforge.prediction.window_timeframe_validation import (
    validate_source_timeframe_definition,
)


def _context_feed_scope(
    context: PrimitiveMapping, feed_scope_id: str
) -> PrimitiveMapping:
    """Validate declared and selected requirements before expanding references."""
    requirements = mapping(context.get("requirements"))
    primary = mapping(requirements.get("primary"))
    contextual = requirements.get("contextual")
    selected = context.get("timeframes")
    if not isinstance(contextual, list) or not isinstance(selected, list):
        raise ValidationError("prediction context feed scope requirements are invalid")
    declarations = (
        primary,
        *(mapping(item) for item in contextual),
        *(mapping(mapping(item).get("requirement")) for item in selected),
    )
    for requirement in declarations:
        feed_scope = requirement.get("feed_scope")
        if (
            not isinstance(feed_scope, dict)
            or configuration_identity(feed_scope) != feed_scope_id
        ):
            raise ValidationError("prediction context feed scope is incompatible")
    return mapping(primary["feed_scope"])


def validate_prediction_input_sources(
    market: PrimitiveMapping,
    *,
    outcome_sources: Iterable[tuple[Primitive, Primitive]],
    context: Primitive,
) -> None:
    """Check (source, labeler configuration) pairs without running research."""
    try:
        provenance = None
        if (
            "intraday_provenance" in market
            or market.get("corporate_action_policy") == INTRADAY_CORPORATE_ACTION_POLICY
            or str(market.get("dataset_id", "")).startswith(
                INTRADAY_PREDICTION_DATASET_PREFIX
            )
        ):
            provenance = validate_prediction_provenance(market)
            assert provenance is not None
        for source, labeler_configuration in outcome_sources:
            # Legacy session labelers and custom feature components may omit
            # typed temporal declarations. Preserve their source-free contract.
            if source is None and (
                not isinstance(labeler_configuration, dict)
                or "temporal_configuration" not in labeler_configuration
            ):
                continue
            temporal = outcome_temporal_configuration(mapping(labeler_configuration))
            if source is None:
                if isinstance(temporal.horizon, ElapsedDurationHorizon):
                    raise ValidationError(
                        "elapsed outcomes require a canonical outcome source"
                    )
                continue
            reference = mapping(mapping(source).get("source_reference"))
            timeframe = temporal.observation_timeframe
            if (
                timeframe is None
                or reference.get("timeframe_configuration_id")
                != timeframe.configuration_id
            ):
                raise ValidationError(
                    "outcome source timeframe differs from its observation timeframe"
                )
            if provenance is not None:
                if (
                    mapping(source).get("family_manifest_id")
                    != (provenance.family_manifest.to_primitive()["manifest_id"])
                ):
                    raise ValidationError(
                        "outcome source family manifest is incompatible"
                    )
                if (
                    configuration_identity(timeframe.session_policy.to_primitive())
                    != provenance.session_policy_id
                ):
                    raise ValidationError(
                        "outcome source session policy is incompatible"
                    )
                validate_prediction_source_reference(provenance, reference)
        if provenance is None or context is None:
            return
        captured = mapping(context)
        # SKIP deliberately retains rejected evidence for auditability.
        if captured.get("status") == "skipped":
            return
        if captured.get("status") != "available":
            raise ValidationError("prediction source context status is invalid")
        source_context = captured.get("source_context")
        if not isinstance(source_context, dict):
            raise ValidationError("prediction source context evidence is missing")
        if source_context.get("context_id") != configuration_identity(
            {key: value for key, value in source_context.items() if key != "context_id"}
        ):
            raise ValidationError("prediction source context identity is inconsistent")
        if (
            source_context.get("dataset_family_manifest_id")
            != (provenance.family_manifest.to_primitive()["manifest_id"])
        ):
            raise ValidationError("prediction context family manifest is incompatible")
        feed_scope = _context_feed_scope(captured, provenance.feed_scope_id)
        timeframes = source_context.get("timeframes")
        if not isinstance(timeframes, list):
            raise ManifestError("prediction source timeframes are invalid")
        for timeframe in timeframes:
            aligned = mapping(timeframe)
            reference = aligned.get("dataset_reference")
            if reference is None:
                raise ValidationError(
                    "prediction context timeframe has no dataset provenance"
                )
            # QF-20's compact reference omits feed scope. Expand it only from
            # declarations already checked against the input's feed identity.
            expanded_reference = dict(mapping(reference))
            expanded_reference.setdefault("feed_scope", feed_scope)
            requirement = mapping(aligned.get("requirement"))
            declared_timeframe = validate_source_timeframe_definition(
                mapping(requirement.get("timeframe"))
            )
            if (
                expanded_reference.get("timeframe_configuration_id")
                != declared_timeframe.configuration_id
            ):
                raise ValidationError(
                    "prediction context source lineage timeframe differs "
                    "from its aligned requirement"
                )
            session_policy = declared_timeframe.session_policy.to_primitive()
            if configuration_identity(session_policy) != provenance.session_policy_id:
                raise ValidationError("prediction input session policy is incompatible")
            validate_prediction_source_reference(provenance, expanded_reference)
        # Hash consistency alone does not prove source/rule semantics agree.
        requirements = mapping(captured.get("requirements"))
        validate_window_context_snapshot(
            context=captured,
            source=source_context,
            requirements=requirements,
            market_data=market,
            primary_timeframe=mapping(
                mapping(requirements.get("primary")).get("timeframe")
            ),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise ManifestError(
            f"prediction input provenance is invalid: {error}"
        ) from error
