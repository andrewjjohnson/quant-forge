"""Validate QF-11/QF-5 identities using their persisted scientific inputs."""

from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments._prediction_row_integrity import validate_prediction_row
from quantforge.experiments._prediction_sessions import recorded_session_indexes


def validate_prediction_identity(manifest: PrimitiveMapping) -> None:
    """Use QF-11's original version and optional-context identity semantics."""
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
    sessions = [
        text(mapping(mapping(row).get("prediction")).get("signal_session"))
        for row in cast(list[object], rows)
    ]
    if sessions != sorted(set(sessions)):
        raise ManifestError("prediction rows must have ordered unique signal sessions")
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
