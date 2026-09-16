"""Bind QF-40's copied evidence to captured folds without aggregating metrics."""

from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._aggregate_schema import (
    validate_configuration_stability,
    validate_equity_rows,
)
from quantforge.experiments._backtest_summary_integrity import (
    validate_backtest_aggregate_summary,
    validate_backtest_summary_counts,
)
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments._prediction_counts_integrity import (
    validate_prediction_summary_counts,
)
from quantforge.experiments._prediction_summary_integrity import (
    validate_prediction_aggregate_summary,
)
from quantforge.oos.common import completeness as source_completeness
from quantforge.oos.models import OOSSource
from quantforge.validation import ResearchStudyType
from quantforge.walk_forward.models import BacktestOOSArtifact, PredictionOOSArtifact


def _records(value: object) -> list[PrimitiveMapping]:
    if not isinstance(value, list):
        raise ManifestError("OOS aggregate records must be arrays")
    return [mapping(item) for item in cast(list[object], value)]


def _same_records(
    actual: object, expected: list[PrimitiveMapping], description: str
) -> None:
    if [configuration_identity(row) for row in _records(actual)] != [
        configuration_identity(row) for row in expected
    ]:
        raise ManifestError(f"OOS aggregate {description} differ from captured folds")


def _prediction_records(
    artifact: PredictionOOSArtifact, fold_id: str
) -> list[PrimitiveMapping]:
    """Project existing signals/rows and their references, without summarizing."""
    observations: list[PrimitiveMapping] = []
    for decision in _records(artifact.snapshot.to_primitive().get("decisions")):
        rows = {
            configuration_identity(
                {"prediction": row.get("prediction"), "features": row.get("features")}
            ): row
            for row in _records(mapping(decision.get("prediction_study")).get("rows"))
        }
        for signal in _records(decision.get("generated_signals")):
            values = mapping(mapping(signal.get("prediction")).get("values"))
            observations.append(
                {
                    "fold_id": fold_id,
                    "selection_id": artifact.selection_id,
                    "window_result_id": artifact.window_result_id,
                    "decision_timestamp": decision.get("decision_timestamp"),
                    "context_id": decision.get("context_id"),
                    "prediction_study_id": decision.get("prediction_study_id"),
                    "signal": signal,
                    "eligible": values.get("direction") in ("up", "down")
                    and values.get("disposition") in (None, "accepted"),
                    "row": rows.get(configuration_identity(signal)),
                }
            )
    return observations


def validate_aggregate_folds(source: OOSSource, aggregate: PrimitiveMapping) -> None:
    """Check copied payloads and membership; leave aggregate statistics intact."""
    backtest = source.plan.environment.study_type is ResearchStudyType.TRADING_BACKTEST
    if aggregate.get("kind") != (
        "backtest_oos_aggregate" if backtest else "prediction_oos_aggregate"
    ):
        raise ManifestError("OOS aggregate kind differs from captured research family")
    summary = mapping(aggregate.get("summary"))
    # This producer helper projects captured status metadata, not research metrics.
    if configuration_identity(
        mapping(summary.get("completeness"))
    ) != configuration_identity(source_completeness(source)):
        raise ManifestError("OOS aggregate completeness differs from captured folds")
    native: list[PrimitiveMapping] = []
    equity_references: list[PrimitiveMapping] = []
    observations: list[PrimitiveMapping] = []
    windows: list[PrimitiveMapping] = []
    prediction_windows: dict[str, tuple[list[PrimitiveMapping], int]] = {}
    for fold in source.folds:
        window: PrimitiveMapping = {
            "fold_id": fold.fold_id,
            "status": fold.status.value,
        }
        artifact = fold.artifact
        if artifact is None:
            window["performance" if backtest else "summary"] = None
        elif backtest:
            if not isinstance(artifact, BacktestOOSArtifact):
                raise ManifestError("OOS aggregate has incompatible fold artifacts")
            payload = artifact.snapshot.to_primitive()
            manifest = mapping(payload.get("manifest"))
            native.append(
                {
                    "fold_id": fold.fold_id,
                    "selection_id": artifact.selection_id,
                    "result": payload,
                    "export_location": artifact.export_location,
                    "export_fingerprint": artifact.export_fingerprint,
                }
            )
            window["performance"] = manifest.get("performance")
            window["benchmark_performance"] = mapping(manifest.get("benchmark")).get(
                "performance"
            )
            for position, row in enumerate(_records(payload.get("daily_equity"))):
                equity_references.append(
                    {
                        "fold_id": fold.fold_id,
                        "run_id": artifact.run_id,
                        "session": text(row.get("session")),
                        "timestamp_semantics": "exchange_session_close",
                        "window_start": position == 0,
                    }
                )
        else:
            if not isinstance(artifact, PredictionOOSArtifact):
                raise ManifestError("OOS aggregate has incompatible fold artifacts")
            captured = _prediction_records(artifact, fold.fold_id)
            scheduled = len(_records(artifact.snapshot.to_primitive().get("decisions")))
            prediction_windows[fold.fold_id] = captured, scheduled
            observations.extend(captured)
        windows.append(window)
    summary_windows = _records(summary.get("windows"))
    validate_configuration_stability(aggregate.get("stability"), source)
    if backtest:
        validate_backtest_aggregate_summary(aggregate.get("summary"))
        _same_records(aggregate.get("native_windows"), native, "native windows")
        validate_equity_rows(aggregate.get("normalized_equity"))
        fields = ("fold_id", "run_id", "session", "timestamp_semantics", "window_start")
        _same_records(
            [
                {key: row.get(key) for key in fields}
                for row in _records(aggregate.get("normalized_equity"))
            ],
            equity_references,
            "equity references",
        )
        _same_records(summary_windows, windows, "window summaries")
        validate_backtest_summary_counts(summary, windows, len(equity_references))
    else:
        validate_prediction_aggregate_summary(aggregate.get("summary"))
        _same_records(
            aggregate.get("observations"), observations, "prediction observations"
        )
        # Prediction summaries contain calculated metrics, so bind their window
        # membership and availability without evaluating those metrics again.
        projected: list[PrimitiveMapping] = []
        for row in summary_windows:
            reference: PrimitiveMapping = {
                key: row.get(key) for key in ("fold_id", "status")
            }
            if row.get("summary") is None:
                reference["summary"] = None
            else:
                mapping(row["summary"])
            projected.append(reference)
        _same_records(projected, windows, "window summaries")
        fields = mapping(summary["metric_fields"])
        validate_prediction_summary_counts(
            summary,
            observations,
            sum(scheduled for _, scheduled in prediction_windows.values()),
            fields,
        )
        for window in summary_windows:
            if window["summary"] is not None:
                captured, scheduled = prediction_windows[text(window["fold_id"])]
                validate_prediction_summary_counts(
                    mapping(window["summary"]), captured, scheduled, fields
                )
