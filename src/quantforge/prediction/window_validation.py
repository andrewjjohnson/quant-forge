"""Offline provenance checks for persisted historical prediction windows."""

from datetime import date, datetime
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.outcome_resolution import (
    OutcomeEvaluationRequest,
    _target_boundaries,  # pyright: ignore[reportPrivateUsage]
)  # pyright: ignore[reportPrivateUsage]
from quantforge.prediction.outcome_temporal import (
    ElapsedDurationHorizon,
    OutcomeAnchor,
    OutcomeTemporalConfiguration,
    outcome_temporal_configuration,
)
from quantforge.prediction.window import (
    PredictionDecisionSchedule,
    _window_record_counts,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.window_context_validation import (
    validate_window_context_snapshot,
    validate_window_source_snapshot,
)


def _validate_generated_signals(
    signals: list[Primitive],
    *,
    configuration: PrimitiveMapping,
    market_data: PrimitiveMapping,
    decision_session: str,
    session_indexes: dict[str, int],
    strategy_parameters: PrimitiveMapping,
) -> None:
    rule = configuration.get("prediction_rule")
    if not isinstance(rule, dict):
        raise InvalidPredictionOutputError("prediction rule configuration is missing")
    warm_up = rule.get("warm_up_observations")
    if not isinstance(warm_up, int) or isinstance(warm_up, bool) or warm_up < 1:
        raise InvalidPredictionOutputError("prediction rule warm-up is invalid")
    labeler = cast(PrimitiveMapping, configuration["outcome_labeler"])
    elapsed = isinstance(
        outcome_temporal_configuration(
            cast(PrimitiveMapping, labeler["configuration"]),
            required_future_sessions=cast(
                int | None, labeler.get("required_future_sessions")
            ),
        ).horizon,
        ElapsedDurationHorizon,
    )
    sessions: list[str] = []
    for signal in signals:
        prediction = signal.get("prediction") if isinstance(signal, dict) else None
        if (
            not isinstance(signal, dict)
            or not isinstance(signal.get("features"), dict)
            or not isinstance(prediction, dict)
            or not isinstance(prediction.get("values"), dict)
        ):
            raise InvalidPredictionOutputError("generated signal payload is incomplete")
        session = prediction.get("signal_session")
        if (
            not isinstance(session, str)
            or (not elapsed and session not in session_indexes)
            or session != decision_session
            or (not elapsed and session_indexes[session] + 1 < warm_up)
            or configuration_identity(
                {
                    key: prediction.get(key)
                    for key in (
                        "symbol",
                        "strategy_id",
                        "strategy_implementation_version",
                        "strategy_configuration_id",
                        "strategy_parameters",
                    )
                }
            )
            != configuration_identity(
                {
                    "symbol": market_data["symbol"],
                    "strategy_id": rule["name"],
                    "strategy_implementation_version": rule["implementation_version"],
                    "strategy_configuration_id": rule["configuration_id"],
                    "strategy_parameters": strategy_parameters,
                }
            )
        ):
            raise InvalidPredictionOutputError("generated signal provenance is invalid")
        sessions.append(session)
    if sessions != sorted(set(sessions)):
        raise InvalidPredictionOutputError(
            "generated signals must be ordered and unique per session"
        )


def _validate_rows(
    rows: list[Primitive],
    signals: list[Primitive],
    configuration: PrimitiveMapping,
    session_indexes: dict[str, int],
    timestamp: str,
) -> None:
    labeler = configuration.get("outcome_labeler")
    if not isinstance(labeler, dict):
        raise InvalidPredictionOutputError("missing outcome labeler")
    horizon = labeler.get("required_future_sessions")
    temporal = outcome_temporal_configuration(
        cast(PrimitiveMapping, labeler["configuration"]),
        required_future_sessions=cast(int | None, horizon),
    )
    elapsed = isinstance(temporal.horizon, ElapsedDurationHorizon)
    if not elapsed and (type(horizon) is not int or horizon < 1):
        raise InvalidPredictionOutputError(
            "outcome horizon must be a positive session count"
        )
    if elapsed and (
        "required_future_sessions" in labeler
        or labeler.get("temporal_configuration") != temporal.to_primitive()
    ):
        raise InvalidPredictionOutputError(
            "elapsed outcome wrapper differs from its temporal contract"
        )
    signal_outcome_indexes: dict[str, int] = {}
    for signal in signals:
        if not isinstance(signal, dict):
            raise InvalidPredictionOutputError("generated signal is not an object")
        signal_id = configuration_identity(signal)
        if signal_id in signal_outcome_indexes:
            raise InvalidPredictionOutputError("generated signals contain duplicates")
        prediction = cast(PrimitiveMapping, signal["prediction"])
        signal_outcome_indexes[signal_id] = (
            -1
            if elapsed
            else session_indexes[cast(str, prediction["signal_session"])]
            + cast(int, horizon)
        )

    for row in rows:
        if not isinstance(row, dict):
            raise InvalidPredictionOutputError("decision row is not an object")
        prediction, features = row.get("prediction"), row.get("features")
        outcome, evaluation = row.get("outcome"), row.get("evaluation")
        if (
            not isinstance(prediction, dict)
            or not isinstance(prediction.get("values"), dict)
            or not isinstance(features, dict)
            or not isinstance(outcome, dict)
            or not isinstance(outcome.get("values"), dict)
            or not isinstance(evaluation, dict)
            or not isinstance(evaluation.get("values"), dict)
        ):
            raise InvalidPredictionOutputError("decision row payload is incomplete")
        row_signal: PrimitiveMapping = {"prediction": prediction, "features": features}
        signal_id = configuration_identity(row_signal)
        if signal_id not in signal_outcome_indexes:
            raise InvalidPredictionOutputError(
                "decision row does not match a distinct generated signal"
            )
        expected_outcome_index = signal_outcome_indexes.pop(signal_id)
        outcome_session = outcome.get("outcome_session")
        if elapsed:
            _validate_timestamp_resolution(
                outcome, prediction, labeler, temporal, timestamp
            )
        elif (
            not isinstance(outcome_session, str)
            or session_indexes.get(outcome_session) != expected_outcome_index
        ):
            raise InvalidPredictionOutputError(
                "outcome session differs from the configured future-session horizon"
            )

        # QF-11 hashes complete outcome/evaluation payloads, excluding their
        # own IDs. Evaluation identity also binds the fixed prediction values;
        # row identity binds the full signal, including contemporaneous features.
        outcome_id = configuration_identity(
            {
                **{key: value for key, value in outcome.items() if key != "outcome_id"},
                "record_type": "prediction_outcome",
            }
        )
        evaluation_id = configuration_identity(
            {
                **{
                    key: value
                    for key, value in evaluation.items()
                    if key != "evaluation_id"
                },
                "prediction": prediction["values"],
                "record_type": "prediction_evaluation",
            }
        )
        row_id = configuration_identity(
            {
                "evaluation_id": evaluation_id,
                "outcome_id": outcome_id,
                "record_type": "prediction_study_row",
                "signal": row_signal,
                "study_id": row["study_id"],
            }
        )
        if (
            outcome.get("outcome_id") != outcome_id
            or evaluation.get("outcome_id") != outcome_id
            or evaluation.get("evaluation_id") != evaluation_id
            or row.get("row_id") != row_id
            or outcome.get("dataset_id") != row["dataset_id"]
            or outcome.get("dataset_fingerprint") != row["dataset_fingerprint"]
            or outcome.get("signal_session") != prediction.get("signal_session")
        ):
            raise InvalidPredictionOutputError(
                "decision row identities are inconsistent"
            )
        for payload, component, prefix, schema_field in (
            (outcome, "outcome_labeler", "outcome", "outcome_result_schema_version"),
            (evaluation, "evaluator", "evaluator", "evaluation_result_schema_version"),
        ):
            configured = configuration.get(component)
            if not isinstance(configured, dict) or configuration_identity(
                {
                    **{
                        field: payload.get(f"{prefix}_{field}")
                        for field in (
                            "name",
                            "implementation_version",
                            "configuration_id",
                        )
                    },
                    "result_schema_version": payload.get(schema_field),
                }
            ) != configuration_identity(
                {
                    field: configured.get(field)
                    for field in (
                        "name",
                        "implementation_version",
                        "configuration_id",
                        "result_schema_version",
                    )
                }
            ):
                raise InvalidPredictionOutputError(
                    "decision row component differs from its study configuration"
                )
    if any(index < len(session_indexes) for index in signal_outcome_indexes.values()):
        raise InvalidPredictionOutputError(
            "generated signal omits an available outcome"
        )


def _validate_decision(
    decision: PrimitiveMapping,
    identity: PrimitiveMapping,
    timestamp: str,
    *,
    primary_timeframe: PrimitiveMapping,
    decision_session: str,
    session_indexes: dict[str, int],
    strategy_parameters: PrimitiveMapping,
) -> None:
    study = decision.get("prediction_study")
    manifest = study.get("manifest") if isinstance(study, dict) else None
    context = manifest.get("prediction_context") if isinstance(manifest, dict) else None
    signals = decision.get("generated_signals")
    if (
        decision.get("decision_timestamp") != timestamp
        or not isinstance(manifest, dict)
        or not isinstance(context, dict)
        or not isinstance(signals, list)
        or manifest.get("component") != "quantforge_prediction_study"
        or configuration_identity(
            {
                "configuration": manifest.get("configuration"),
                "market_data": manifest.get("market_data"),
                "engine_version": manifest.get("engine_version"),
            }
        )
        != configuration_identity(
            {
                "configuration": identity["configuration"],
                "market_data": identity["market_data"],
                "engine_version": identity["prediction_engine_version"],
            }
        )
    ):
        raise InvalidPredictionOutputError(
            "decision provenance differs from its window"
        )
    study_id = configuration_identity(
        {
            "component": "quantforge_prediction_study",
            "engine_version": manifest["engine_version"],
            "market_data": manifest["market_data"],
            "study_configuration": manifest["configuration"],
            "prediction_context": context,
        }
    )
    _validate_decision_record(
        decision,
        identity,
        timestamp,
        study_id=study_id,
        primary_timeframe=primary_timeframe,
        decision_session=decision_session,
        session_indexes=session_indexes,
        strategy_parameters=strategy_parameters,
    )


def _validate_decision_record(
    decision: PrimitiveMapping,
    identity: PrimitiveMapping,
    timestamp: str,
    *,
    study_id: str,
    primary_timeframe: PrimitiveMapping,
    decision_session: str,
    session_indexes: dict[str, int],
    strategy_parameters: PrimitiveMapping,
) -> None:
    """Shared v1/v2 semantic checks; v2 passes no embedded market/configuration.

    The caller verifies the version-specific reference and supplies the original
    QF-11 study identity computed from the authenticated shared scope/context.
    """
    study = decision.get("prediction_study")
    manifest = study.get("manifest") if isinstance(study, dict) else None
    context = manifest.get("prediction_context") if isinstance(manifest, dict) else None
    signals = decision.get("generated_signals")
    if (
        decision.get("decision_timestamp") != timestamp
        or not isinstance(manifest, dict)
        or manifest.get("component") != "quantforge_prediction_study"
        or not isinstance(context, dict)
        or not isinstance(signals, list)
    ):
        raise InvalidPredictionOutputError(
            "decision provenance differs from its window"
        )
    if (
        manifest.get("study_id") != study_id
        or decision.get("prediction_study_id") != study_id
    ):
        raise InvalidPredictionOutputError("decision study identity is inconsistent")
    rows = study.get("rows") if isinstance(study, dict) else None
    market_data = identity["market_data"]
    if (
        not isinstance(rows, list)
        or not isinstance(market_data, dict)
        or any(
            not isinstance(row, dict)
            or row.get("study_id") != study_id
            or row.get("dataset_id") != market_data.get("dataset_id")
            or row.get("dataset_fingerprint") != market_data.get("bars_fingerprint")
            for row in rows
        )
    ):
        raise InvalidPredictionOutputError("decision rows have incompatible provenance")
    configuration = identity["configuration"]
    requirements = context.get("requirements")
    if (
        not isinstance(configuration, dict)
        or not isinstance(requirements, dict)
        or context.get("status") not in ("available", "skipped")
        or configuration_identity({"requirements": requirements})
        != configuration_identity(
            {"requirements": configuration.get("prediction_context_requirements")}
        )
    ):
        raise InvalidPredictionOutputError(
            "decision context requirements are inconsistent"
        )
    _validate_generated_signals(
        signals,
        configuration=configuration,
        market_data=market_data,
        decision_session=decision_session,
        session_indexes=session_indexes,
        strategy_parameters=strategy_parameters,
    )
    _validate_rows(rows, signals, configuration, session_indexes, timestamp)
    if configuration_identity(
        {"counts": manifest.get("record_counts")}
    ) != configuration_identity(
        {
            "counts": {
                "generated_predictions": len(signals),
                "labeled_rows": len(rows),
                "unavailable_outcomes": len(signals) - len(rows),
            }
        }
    ):
        raise InvalidPredictionOutputError("decision record counts are inconsistent")
    skipped = context["status"] == "skipped"
    if not skipped and context.get("decision_session") != decision_session:
        raise InvalidPredictionOutputError(
            "decision session differs from the scheduled exchange session"
        )
    if skipped and requirements.get("failure_policy") != "skip":
        raise InvalidPredictionOutputError(
            "skipped decision contradicts its failure policy"
        )
    source = context.get("source_context")
    if isinstance(source, dict):
        context_id = configuration_identity(
            {key: value for key, value in source.items() if key != "context_id"}
        )
        if (
            source.get("context_id") != context_id
            or decision.get("context_id") != context_id
        ):
            raise InvalidPredictionOutputError(
                "decision source context identity is inconsistent"
            )
        if skipped:
            validate_window_source_snapshot(source)
        else:
            consistency = source.get("source_consistency")
            if (
                source.get("as_of") != timestamp
                or not isinstance(consistency, dict)
                or consistency.get("family_id")
                != identity["dataset_family_fingerprint"]
            ):
                raise InvalidPredictionOutputError(
                    "available context has incompatible provenance"
                )
            validate_window_context_snapshot(
                context=context,
                source=source,
                requirements=requirements,
                market_data=market_data,
                primary_timeframe=primary_timeframe,
                timestamp=timestamp,
            )
    elif source is not None or not skipped or decision.get("context_id") is not None:
        raise InvalidPredictionOutputError(
            "decision source context evidence is invalid"
        )
    expected_status = (
        "skipped" if skipped else "no_prediction" if not signals else "evaluated"
    )
    if decision.get("status") != expected_status or (skipped and (signals or rows)):
        raise InvalidPredictionOutputError("decision status contradicts its context")


def validate_prediction_window_snapshot(
    snapshot: PrimitiveMapping,
    *,
    expected_identity: PrimitiveMappingSnapshot,
    schedule: PredictionDecisionSchedule,
    outcome_sessions: tuple[date, ...],
    strategy_parameters: PrimitiveMapping,
) -> None:
    """Check evidence using validated, ordered dataset sessions and rule parameters."""
    manifest, decisions = snapshot.get("manifest"), snapshot.get("decisions")
    if not isinstance(manifest, dict) or not isinstance(decisions, list):
        raise InvalidPredictionOutputError("window manifest or decisions are missing")
    if (
        manifest.get("component") != "quantforge_prediction_window"
        or manifest.get("schema_version") != "1"
    ):
        raise InvalidPredictionOutputError("unsupported embedded window schema version")
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"window_id", "window_result_id", "schedule_id", "record_counts"}
    }
    window_id = configuration_identity(identity)
    if (
        window_id != manifest.get("window_id")
        or window_id != configuration_identity(expected_identity.to_primitive())
        or manifest.get("schedule_id") != schedule.schedule_id
        or manifest.get("schedule") != schedule.to_primitive()
        or len(decisions) != len(schedule.decision_timestamps)
    ):
        raise InvalidPredictionOutputError(
            "window identity or schedule differs from the candidate"
        )
    session_indexes = {
        session.isoformat(): index for index, session in enumerate(outcome_sessions)
    }
    primary_timeframe: PrimitiveMapping = {
        "configuration_id": schedule.primary_timeframe.configuration_id,
        "configuration": schedule.primary_timeframe.to_primitive(),
    }
    for decision, timestamp, session in zip(
        decisions, schedule.decision_timestamps, schedule.decision_sessions, strict=True
    ):
        if not isinstance(decision, dict):
            raise InvalidPredictionOutputError("window decision is not an object")
        _validate_decision(
            decision,
            identity,
            timestamp.isoformat(),
            primary_timeframe=primary_timeframe,
            decision_session=session.isoformat(),
            session_indexes=session_indexes,
            strategy_parameters=strategy_parameters,
        )
    if configuration_identity(
        {"counts": manifest.get("record_counts")}
    ) != configuration_identity(
        {"counts": _window_record_counts(cast(list[PrimitiveMapping], decisions))}
    ):
        raise InvalidPredictionOutputError("window record counts are inconsistent")
    if configuration_identity(
        {"window_id": window_id, "decisions": decisions}
    ) != manifest.get("window_result_id"):
        raise InvalidPredictionOutputError("window result identity is inconsistent")


def _validate_timestamp_resolution(
    outcome: PrimitiveMapping,
    prediction: PrimitiveMapping,
    labeler: PrimitiveMapping,
    temporal: OutcomeTemporalConfiguration,
    timestamp: str,
) -> None:
    resolution = outcome.get("temporal_resolution")
    if not isinstance(resolution, dict) or not isinstance(
        resolution.get("request"), dict
    ):
        raise InvalidPredictionOutputError(
            "elapsed outcome lacks QF-46 resolution provenance"
        )
    request = cast(PrimitiveMapping, resolution["request"])
    anchor = OutcomeAnchor.from_primitive(cast(PrimitiveMapping, request["anchor"]))
    source = labeler.get("outcome_source")
    if not isinstance(source, dict) or any(
        (
            anchor.decision_timestamp != datetime.fromisoformat(timestamp),
            cast(PrimitiveMapping, prediction["values"]).get(
                "decision_timestamp", timestamp
            )
            != timestamp,
            anchor.signal_session.isoformat() != prediction["signal_session"],
            outcome["outcome_session"] != prediction["signal_session"],
            request.get("temporal_configuration") != temporal.to_primitive(),
            request.get("outcome_configuration_id")
            != outcome["outcome_configuration_id"],
            request.get("dataset_id") != outcome["dataset_id"],
            request.get("dataset_fingerprint") != outcome["dataset_fingerprint"],
            request.get("source_reference") != source.get("source_reference"),
        )
    ):
        raise InvalidPredictionOutputError(
            "elapsed outcome request differs from its exact decision/source"
        )
    typed = OutcomeEvaluationRequest(
        anchor,
        temporal,
        str(request["outcome_configuration_id"]),
        str(request["dataset_id"]),
        str(request["dataset_fingerprint"]),
    )
    target, expected = _target_boundaries(typed)
    available = resolution.get("available")
    status = resolution.get("status")
    if (
        resolution.get("requested_target_timestamp") != target.isoformat()
        or resolution.get("expected_observation_timestamp")
        != (None if expected is None else expected.isoformat())
        or type(available) is not bool
        or available != (status == "available")
        or (expected is None) != (status == "session_overflow")
        or status
        not in {
            "available",
            "session_overflow",
            "missing_required_observation",
            "incomplete_future_data",
            "dataset_end",
        }
        or resolution.get("resolved_observation_timestamp")
        != (expected.isoformat() if available and expected is not None else None)
        or (available and not isinstance(resolution.get("observation_id"), str))
        or (not available and resolution.get("observation_id") is not None)
    ):
        raise InvalidPredictionOutputError(
            "elapsed outcome resolution contradicts QF-46 boundaries/availability"
        )
