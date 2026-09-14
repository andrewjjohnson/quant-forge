"""Validate QF-11/QF-5 identities using their persisted scientific inputs."""

from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text


def validate_prediction_identity(manifest: PrimitiveMapping) -> None:
    """Use QF-11's original version and optional-context identity semantics."""
    identity: PrimitiveMapping = {
        "component": "quantforge_prediction_study",
        "engine_version": text(manifest.get("engine_version")),
        "market_data": mapping(manifest.get("market_data")),
        "study_configuration": mapping(manifest.get("configuration")),
    }
    if "prediction_context" in manifest:
        identity["prediction_context"] = mapping(manifest["prediction_context"])
    if manifest.get("component") != identity["component"] or manifest.get(
        "study_id"
    ) != configuration_identity(identity):
        raise ManifestError("prediction study identity is inconsistent")


def validate_prediction_rows(manifest: PrimitiveMapping, rows: object) -> None:
    """Reconcile stored row counts; do not generate outcomes or evaluate rows."""
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) for row in cast(list[object], rows)
    ):
        raise ManifestError("prediction result rows must be an array of records")
    counts = mapping(manifest.get("record_counts"))
    fields = ("generated_predictions", "labeled_rows", "unavailable_outcomes")
    for field in fields:
        count = counts.get(field)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ManifestError("prediction record counts must be nonnegative integers")
    labeled = cast(int, counts["labeled_rows"])
    unavailable = cast(int, counts["unavailable_outcomes"])
    if (
        labeled != len(cast(list[object], rows))
        or counts["generated_predictions"] != labeled + unavailable
    ):
        raise ManifestError(
            "prediction record counts are inconsistent with result rows"
        )


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
