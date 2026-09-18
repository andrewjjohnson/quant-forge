"""QF-42 snapshot identity and record-count checks over persisted primitives."""

from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments._prediction_row_integrity import validate_prediction_signal
from quantforge.experiments._producer_integrity import (
    validate_outcome_contract,
    validate_prediction_identity,
    validate_prediction_rows,
    validate_prediction_warm_up,
)
from quantforge.experiments._window_context_integrity import validate_decision_context
from quantforge.experiments._window_sessions import scheduled_sessions
from quantforge.prediction.outcome_temporal import (
    ElapsedDurationHorizon,
    outcome_temporal_configuration,
)
from quantforge.prediction.window import (
    _window_record_counts,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.window_validation import (
    _validate_decision,  # pyright: ignore[reportPrivateUsage]
)


def _records(value: object) -> list[PrimitiveMapping]:
    if not isinstance(value, list):
        raise ManifestError("window decision records must be arrays")
    return [mapping(item) for item in cast(list[object], value)]


def _validate_signal_rows(
    signals: list[PrimitiveMapping],
    rows: list[PrimitiveMapping],
    manifest: PrimitiveMapping,
    indexes: dict[str, int],
) -> None:
    signal_ids: set[str] = set()
    sessions: list[str] = []
    for signal in signals:
        mapping(signal.get("features"))
        sessions.append(
            validate_prediction_signal(
                manifest, mapping(signal.get("prediction")), indexes
            )
        )
        signal_id = configuration_identity(signal)
        if signal_id in signal_ids:
            raise ManifestError("window generated signals contain duplicates")
        signal_ids.add(signal_id)
    if sessions != sorted(set(sessions)):
        raise ManifestError("window signals must have ordered unique sessions")
    for row in rows:
        signal_id = configuration_identity(
            {"features": row.get("features"), "prediction": row.get("prediction")}
        )
        if signal_id not in signal_ids:
            raise ManifestError("window row does not match a distinct generated signal")
        signal_ids.remove(signal_id)
    horizon = mapping(
        mapping(manifest.get("configuration")).get("outcome_labeler")
    ).get("required_future_sessions")
    if type(horizon) is not int or horizon < 1:
        raise ManifestError("prediction outcome horizon must be positive")
    if any(
        configuration_identity(signal) in signal_ids
        and indexes[text(mapping(signal.get("prediction")).get("signal_session"))]
        + horizon
        < len(indexes)
        for signal in signals
    ):
        raise ManifestError("window signal is missing an available outcome row")


def validate_window_snapshot(snapshot: PrimitiveMapping) -> None:
    """Check supported schema, complete calendar, identities and stored totals.

    Reuse QF-42's calendar contract and counts over stored signals/rows. No
    context, feature, outcome, prediction or research metric is generated.
    """
    manifest = mapping(snapshot.get("manifest"))
    if manifest.get("component") != "quantforge_prediction_window":
        raise ManifestError("unsupported prediction window component")
    if manifest.get("schema_version") != "1":
        raise ManifestError("unsupported prediction window schema version")
    validate_prediction_warm_up(mapping(manifest.get("configuration")))
    validate_outcome_contract(mapping(manifest.get("configuration")))
    decisions = _records(snapshot.get("decisions"))
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"window_id", "window_result_id", "schedule_id", "record_counts"}
    }
    window_id = configuration_identity(identity)
    if window_id != manifest.get("window_id"):
        raise ManifestError("incompatible QF-42 window configuration")
    if configuration_identity(
        {"window_id": window_id, "decisions": snapshot["decisions"]}
    ) != manifest.get("window_result_id"):
        raise ManifestError("window result identity is inconsistent")
    schedule = mapping(manifest.get("schedule"))
    if manifest.get("schedule_id") != configuration_identity(schedule):
        raise ManifestError("window schedule identity is inconsistent")
    if [
        text(decision.get("decision_timestamp")) for decision in decisions
    ] != schedule.get("decision_timestamps"):
        raise ManifestError("window decisions do not match the declared schedule")
    sessions = scheduled_sessions(schedule)
    # Validate the shapes consumed by the producer's primitive count helper.
    for decision in decisions:
        if text(decision.get("status")) not in {
            "evaluated",
            "skipped",
            "no_prediction",
        }:
            raise ManifestError("invalid window decision status")
        signals = _records(decision.get("generated_signals"))
        study = mapping(decision.get("prediction_study"))
        study_manifest = mapping(study.get("manifest"))
        validate_prediction_identity(study_manifest)
        if (
            study_manifest.get("configuration") != manifest.get("configuration")
            or study_manifest.get("market_data") != manifest.get("market_data")
            or study_manifest.get("engine_version")
            != manifest.get("prediction_engine_version")
            or decision.get("prediction_study_id") != study_manifest.get("study_id")
        ):
            raise ManifestError("decision study does not match its window provenance")
        configuration = mapping(manifest.get("configuration"))
        labeler = mapping(configuration.get("outcome_labeler"))
        temporal = outcome_temporal_configuration(
            mapping(labeler.get("configuration")),
            required_future_sessions=cast(
                int | None, labeler.get("required_future_sessions")
            ),
        )
        if isinstance(temporal.horizon, ElapsedDurationHorizon):
            # Reuse the QF-42 stored-decision verifier, including exact QF-46
            # requests and availability. No context or labels are regenerated.
            timestamp = text(decision["decision_timestamp"])
            rule = mapping(mapping(configuration["prediction_rule"])["configuration"])
            try:
                _validate_decision(
                    decision,
                    identity,
                    timestamp,
                    primary_timeframe={
                        "configuration": schedule["primary_timeframe"],
                        "configuration_id": configuration_identity(
                            mapping(schedule["primary_timeframe"])
                        ),
                    },
                    decision_session=sessions[timestamp],
                    session_indexes={},
                    strategy_parameters=mapping(rule.get("parameters", {})),
                )
            except (ValueError, KeyError, TypeError) as error:
                raise ManifestError(
                    "invalid elapsed prediction decision evidence"
                ) from error
            continue
        indexes = validate_prediction_rows(study_manifest, study.get("rows"))
        rows = _records(study.get("rows"))
        if len(rows) > len(signals):
            raise ManifestError("window rows exceed generated signal records")
        skipped = validate_decision_context(
            decision,
            manifest,
            study_manifest,
            sessions[text(decision["decision_timestamp"])],
        )
        expected_status = (
            "skipped" if skipped else "evaluated" if signals else "no_prediction"
        )
        if decision.get("status") != expected_status or (skipped and (signals or rows)):
            raise ManifestError(
                "window decision status contradicts its context or signals"
            )
        if mapping(study_manifest.get("record_counts")).get(
            "generated_predictions"
        ) != len(signals):
            raise ManifestError("decision record counts differ from generated signals")
        _validate_signal_rows(signals, rows, study_manifest, indexes)
    if configuration_identity(
        {"counts": manifest.get("record_counts")}
    ) != configuration_identity({"counts": _window_record_counts(decisions)}):
        raise ManifestError("window record counts are inconsistent")
