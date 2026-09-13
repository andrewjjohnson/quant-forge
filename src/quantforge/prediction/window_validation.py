"""Offline provenance checks for persisted historical prediction windows."""

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window import PredictionDecisionSchedule


def _validate_rows(
    rows: list[Primitive],
    signals: list[Primitive],
    configuration: PrimitiveMapping,
) -> None:
    signal_ids: set[str] = set()
    for signal in signals:
        if not isinstance(signal, dict):
            raise InvalidPredictionOutputError("generated signal is not an object")
        signal_id = configuration_identity(signal)
        if signal_id in signal_ids:
            raise InvalidPredictionOutputError("generated signals contain duplicates")
        signal_ids.add(signal_id)

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
        if signal_id not in signal_ids:
            raise InvalidPredictionOutputError(
                "decision row does not match a distinct generated signal"
            )
        signal_ids.remove(signal_id)

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


def _validate_decision(
    decision: PrimitiveMapping, identity: PrimitiveMapping, timestamp: str
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
    _validate_rows(rows, signals, configuration)
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
        if not skipped:
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
) -> None:
    """Check scientific identity against the requested candidate without execution."""
    manifest, decisions = snapshot.get("manifest"), snapshot.get("decisions")
    if not isinstance(manifest, dict) or not isinstance(decisions, list):
        raise InvalidPredictionOutputError("window manifest or decisions are missing")
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
    for decision, timestamp in zip(
        decisions, schedule.decision_timestamps, strict=True
    ):
        if not isinstance(decision, dict):
            raise InvalidPredictionOutputError("window decision is not an object")
        _validate_decision(decision, identity, timestamp.isoformat())
    if configuration_identity(
        {"window_id": window_id, "decisions": decisions}
    ) != manifest.get("window_result_id"):
        raise InvalidPredictionOutputError("window result identity is inconsistent")
