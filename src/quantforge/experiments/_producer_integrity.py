"""Validate QF-11/QF-5 identities using their persisted scientific inputs."""

from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments._prediction_row_integrity import validate_prediction_row
from quantforge.experiments._prediction_sessions import recorded_session_indexes
from quantforge.experiments._prediction_temporal_integrity import (
    elapsed_temporal_configuration,
    prediction_observation_key,
)
from quantforge.prediction.outcome_temporal import (
    ExchangeSessionHorizon,
    OutcomeTemporalError,
    outcome_temporal_configuration,
)


def validate_prediction_manifest(manifest: PrimitiveMapping) -> None:
    """Share QF-11 identity, count and context checks across direct/grid inputs."""
    from quantforge.experiments._prediction_context_integrity import (
        validate_prediction_context,
    )

    validate_prediction_identity(manifest)
    validate_prediction_counts(manifest)
    validate_prediction_context(manifest)


def validate_prediction_identity(manifest: PrimitiveMapping) -> None:
    """Use QF-11's original version and optional-context identity semantics."""
    if manifest.get("feature_outcome_boundary") != (
        "prediction signals are fixed before outcome labeling; evaluators "
        "receive only a fixed signal and an already-generated outcome"
    ):
        raise ManifestError("prediction feature/outcome boundary is invalid")
    configuration = mapping(manifest.get("configuration"))
    identity: PrimitiveMapping = {
        "component": "quantforge_prediction_study",
        "engine_version": text(manifest.get("engine_version")),
        "market_data": mapping(manifest.get("market_data")),
        "study_configuration": configuration,
    }
    if "prediction_context" in manifest:
        identity["prediction_context"] = mapping(manifest["prediction_context"])
    if manifest.get("component") != identity["component"] or manifest.get(
        "study_id"
    ) != configuration_identity(identity):
        raise ManifestError("prediction study identity is inconsistent")
    for name in ("prediction_rule", "outcome_labeler", "evaluator"):
        component = mapping(configuration.get(name))
        definition = mapping(component.get("configuration"))
        if (
            component.get("configuration_id") != configuration_identity(definition)
            or component.get("name") != definition.get("component_name")
            or component.get("implementation_version")
            != definition.get("implementation_version")
        ):
            raise ManifestError("prediction study component identity is inconsistent")
    validate_prediction_warm_up(configuration)
    validate_outcome_contract(configuration)


def validate_outcome_contract(configuration: PrimitiveMapping) -> None:
    """Bind labeling wrappers to declarations captured by their components.

    Generic components may omit duplicate declarations in their configuration.
    Preserve their validated wrapper values without constructing a component.
    """
    labeler = mapping(configuration.get("outcome_labeler"))
    definition = mapping(labeler.get("configuration"))
    horizon = labeler.get("required_future_sessions")
    if "temporal_configuration" in definition:
        try:
            temporal = outcome_temporal_configuration(definition)
            if isinstance(temporal.horizon, ExchangeSessionHorizon):
                if type(horizon) is not int or horizon != temporal.horizon.count:
                    raise OutcomeTemporalError(
                        "session wrapper differs from typed horizon"
                    )
            elif "required_future_sessions" in labeler:
                raise OutcomeTemporalError(
                    "elapsed outcomes cannot declare a session count"
                )
            wrapper_temporal = mapping(labeler.get("temporal_configuration"))
            if configuration_identity(wrapper_temporal) != temporal.configuration_id:
                raise OutcomeTemporalError(
                    "typed outcome wrapper differs from configuration"
                )
        except (TypeError, ValueError) as error:
            raise ManifestError(
                f"outcome contract temporal configuration is inconsistent: {error}"
            ) from error
    elif "temporal_configuration" in labeler:
        raise ManifestError("outcome temporal wrapper requires a component declaration")
    elif type(horizon) is not int or horizon < 1:
        raise ManifestError("outcome contract horizon must be a positive integer")
    parameters = mapping(definition.get("parameters", {}))
    for declarations, key in (
        (parameters, "future_sessions"),
        (definition, "required_future_sessions"),
    ):
        if key in declarations and (
            type(declarations[key]) is not int or declarations[key] != horizon
        ):
            raise ManifestError("outcome contract horizon differs from configuration")
    fields = labeler.get("required_market_fields")
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(field, str) or not field for field in fields)
        or fields != sorted(set(cast(list[str], fields)))
        or (
            "required_market_fields" in definition
            and definition["required_market_fields"] != fields
        )
    ):
        raise ManifestError("outcome contract market fields are inconsistent")
    for name, label in (("outcome_labeler", "outcome"), ("evaluator", "evaluation")):
        component = mapping(configuration.get(name))
        schema = component.get("result_schema_version")
        captured = mapping(component.get("configuration"))
        if (
            not isinstance(schema, str)
            or not schema
            or (
                "result_schema_version" in captured
                and captured["result_schema_version"] != schema
            )
        ):
            raise ManifestError(f"{label} contract result schema is inconsistent")


def validate_prediction_warm_up(configuration: PrimitiveMapping) -> None:
    """Reconcile the result wrapper with any captured rule warm-up declaration."""
    rule = mapping(configuration.get("prediction_rule"))
    warm_up = rule.get("warm_up_observations")
    if type(warm_up) is not int or warm_up < 1:
        raise ManifestError("prediction rule warm-up must be a positive integer")
    definition = mapping(rule.get("configuration"))
    if "warm_up_observations" in definition and (
        type(definition["warm_up_observations"]) is not int
        or definition["warm_up_observations"] != warm_up
    ):
        raise ManifestError(
            "prediction rule warm-up differs from captured configuration"
        )


def validate_prediction_counts(manifest: PrimitiveMapping) -> int:
    """Validate declared counts even when result rows are unavailable."""
    counts = mapping(manifest.get("record_counts"))
    fields = ("generated_predictions", "labeled_rows", "unavailable_outcomes")
    for field in fields:
        count = counts.get(field)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ManifestError("prediction record counts must be nonnegative integers")
    labeled = cast(int, counts["labeled_rows"])
    unavailable = cast(int, counts["unavailable_outcomes"])
    if unavailable and elapsed_temporal_configuration(manifest) is not None:
        raise ManifestError("elapsed prediction outcomes require explicit labeled rows")
    if counts["generated_predictions"] != labeled + unavailable:
        raise ManifestError("prediction record counts are internally inconsistent")
    return labeled


def validate_prediction_rows(
    manifest: PrimitiveMapping, rows: object
) -> dict[str, int]:
    """Check row counts and provenance without generating or evaluating results."""
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) for row in cast(list[object], rows)
    ):
        raise ManifestError("prediction result rows must be an array of records")
    if validate_prediction_counts(manifest) != len(cast(list[object], rows)):
        raise ManifestError(
            "prediction record counts are inconsistent with result rows"
        )
    indexes = recorded_session_indexes(mapping(manifest.get("market_data")))
    identifiers = [
        validate_prediction_row(manifest, mapping(row), indexes)
        for row in cast(list[object], rows)
    ]
    if len(set(identifiers)) != len(identifiers):
        raise ManifestError("duplicate prediction row identity")
    observations = [
        prediction_observation_key(manifest, mapping(mapping(row).get("prediction")))
        for row in cast(list[object], rows)
    ]
    if observations != sorted(set(observations)):
        raise ManifestError(
            "prediction rows must have ordered unique signal sessions/anchors"
        )
    return indexes


def validate_backtest_identity(manifest: PrimitiveMapping) -> None:
    """Reproduce QF-5's run-ID inputs, excluding diagnostic/export metadata."""
    market_data = mapping(manifest.get("market_data"))
    strategy = mapping(manifest.get("strategy"))
    configuration = mapping(strategy.get("configuration"))
    try:
        identity: PrimitiveMapping = {
            "component": "quantforge_backtest",
            "engine_version": text(manifest.get("engine_version")),
            "result_schema_version": text(manifest.get("result_schema_version")),
            # QF-4 MarketDataReference plus QF-5's actual-bar fingerprint.
            "market_data": {
                key: market_data[key]
                for key in (
                    "dataset_id",
                    "schema_version",
                    "adjustment_mode",
                    "calendar",
                    "corporate_action_snapshot_id",
                    "bars_fingerprint",
                )
            },
            "strategy": {
                key: strategy[key]
                for key in (
                    "strategy_id",
                    "strategy_implementation_version",
                    "strategy_configuration_id",
                    "configuration",
                )
            },
            "backtest_configuration": mapping(manifest.get("backtest_configuration")),
        }
    except KeyError as error:
        raise ManifestError("backtest run identity inputs are incomplete") from error
    if strategy.get("strategy_configuration_id") != configuration_identity(
        configuration
    ) or manifest.get("run_id") != configuration_identity(identity):
        raise ManifestError("backtest run identity is inconsistent")
    from quantforge.experiments._backtest_manifest_integrity import (
        validate_backtest_manifest,
    )

    validate_backtest_manifest(manifest)
