"""Bind consumed result evidence to the retained QF-40 request without evaluation."""

from datetime import date, datetime, timedelta

from quantforge.backtesting.config import EvaluationInterval
from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._aggregate_schema import record, records
from quantforge.experiments._holdout_membership_integrity import (
    validate_holdout_membership,
)
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments._prediction_counts_integrity import (
    prediction_window_observations,
    validate_prediction_summary_counts,
)
from quantforge.experiments._prediction_summary_integrity import (
    validate_prediction_summary,
)
from quantforge.experiments._prediction_trial_integrity import (
    prediction_components_match,
)
from quantforge.experiments._producer_integrity import validate_backtest_identity
from quantforge.experiments._window_integrity import validate_window_snapshot
from quantforge.oos.models import OOSSource
from quantforge.oos.prediction import PredictionMetricFields
from quantforge.prediction import PredictionDecisionSchedule
from quantforge.timeframes import resolve_exchange_session


def validate_holdout_consumption(
    source: OOSSource, consumption: PrimitiveMapping, reservation: PrimitiveMapping
) -> None:
    """Bind the envelope to the reservation already validated by ledger.state()."""
    expected: PrimitiveMapping = {
        "schema_version": "1",
        "state": "consumed",
        "lineage_id": source.lineage_id,
        "exposure_scope": mapping(reservation.get("exposure_scope")),
        "transition": "permanent_before_evaluation; interrupted_attempts_are_consumed",
    }
    if set(consumption) != set(expected) | {
        "request",
        "request_id",
        "consumption_run_id",
        "consumed_at",
    } or configuration_identity(
        {key: consumption.get(key) for key in expected}
    ) != configuration_identity(expected):
        raise ManifestError("holdout consumption envelope differs from its source")
    try:
        text(consumption["consumption_run_id"])
        consumed_at = datetime.fromisoformat(text(consumption["consumed_at"]))
        if consumed_at.utcoffset() != timedelta(0):
            raise ValueError("consumption timestamp must be UTC")
    except ValueError as error:
        raise ManifestError(
            "holdout consumption execution metadata is invalid"
        ) from error
    request = validate_holdout_request(source, consumption)
    if consumption["request_id"] != configuration_identity(request):
        raise ManifestError("holdout consumption request identity is inconsistent")


def validate_holdout_request(
    source: OOSSource, consumption: PrimitiveMapping
) -> PrimitiveMapping:
    """Bind permanent request metadata without preparing a holdout partition."""
    request = mapping(consumption.get("request"))
    frozen = mapping(request.get("frozen_selection"))
    expected: PrimitiveMapping = {
        "schema_version": "1",
        "operation": "evaluate_frozen_final_holdout",
        "lineage_id": source.lineage_id,
        "lineage": source.lineage.to_primitive(),
        "study_id": source.study_id,
        "plan_id": source.plan.plan_id,
        "study_definition": source.definition.to_primitive(),
        "holdout": source.plan.final_holdout.to_primitive(),
        "tail_policy": "outcome_horizon_remains_inside_holdout; no_tail_decisions",
    }
    if (
        set(request) != set(expected) | {"frozen_selection", "evaluation_membership"}
        or configuration_identity({key: request.get(key) for key in expected})
        != configuration_identity(expected)
        or not any(
            fold.selection is not None
            and configuration_identity(fold.selection.to_primitive())
            == configuration_identity(frozen)
            for fold in source.folds
        )
    ):
        raise ManifestError("holdout request differs from its frozen source selection")
    validate_holdout_membership(source, request["evaluation_membership"])
    return request


def validate_holdout_artifact(
    source: OOSSource, consumption: PrimitiveMapping, result: PrimitiveMapping
) -> None:
    if (
        result.get("schema_version") != "1"
        or result.get("kind") != "final_holdout_result"
        or result.get("state") != "consumed"
    ):
        raise ManifestError("unsupported holdout result envelope")
    request = validate_holdout_request(source, consumption)
    frozen = mapping(request["frozen_selection"])
    artifact = mapping(result.get("artifact"))
    if artifact.get("selection_id") != frozen.get("selection_id"):
        raise ManifestError("holdout artifact differs from frozen selection")
    payload = mapping(artifact.get("result"))
    manifest = mapping(payload.get("manifest"))
    definition = mapping(mapping(frozen.get("candidate")).get("definition"))
    adapter = mapping(source.definition.to_primitive().get("adapter"))
    if (
        artifact.get("kind") == "prediction"
        and adapter.get("adapter") == "qf39_prediction"
    ):
        validate_window_snapshot(payload)
        if artifact.get("result_id") != manifest.get("window_result_id"):
            raise ManifestError("holdout prediction result identity is inconsistent")
        configuration = mapping(manifest.get("configuration"))
        if not prediction_components_match(
            configuration, definition, definition.get("contract_version", "1")
        ):
            raise ManifestError("holdout prediction differs from frozen candidate")
        if any(
            configuration.get(key) != definition.get(reference)
            for key, reference in {
                "prediction_context_requirements": "prediction_context",
                "feature_configuration": "feature_configuration",
                "result_schema_version": "result_schema_version",
            }.items()
        ) or manifest.get("indicator_backend_environment") != definition.get(
            "indicator_backend_environment"
        ):
            raise ManifestError("holdout prediction differs from frozen candidate")
        membership = mapping(request.get("evaluation_membership"))
        labels = membership.get("evaluation_sessions")
        if not isinstance(labels, list) or not labels:
            raise ManifestError("holdout request evaluation sessions are invalid")
        universe = mapping(adapter.get("universe"))
        primary_configuration = mapping(universe.get("grid_definition")).get(
            "primary_timeframe"
        )
        primary = next(
            (
                timeframe
                for timeframe in source.plan.environment.timeframes
                if timeframe.to_primitive() == primary_configuration
            ),
            None,
        )
        if primary is None:
            raise ManifestError(
                "holdout prediction primary timeframe is not in its plan"
            )
        try:
            # QF-40 derives the complete closed schedule from the requested
            # partition and frozen primary timeframe, without generating research.
            expected_schedule = PredictionDecisionSchedule(
                primary,
                resolve_exchange_session(
                    date.fromisoformat(text(labels[0])), primary.session_policy
                ).open_timestamp,
                resolve_exchange_session(
                    date.fromisoformat(text(labels[-1])), primary.session_policy
                ).close_timestamp,
            )
        except ValueError as error:
            raise ManifestError(
                "holdout request evaluation sessions are invalid"
            ) from error
        if configuration_identity(
            mapping(manifest.get("schedule"))
        ) != configuration_identity(expected_schedule.to_primitive()):
            raise ManifestError("holdout prediction differs from requested schedule")
        context = mapping(
            mapping(manifest.get("context_environment")).get("configuration")
        )
        market = mapping(manifest.get("market_data"))
        if (
            context.get("partition") != membership
            or context.get("plan_id") != source.plan.plan_id
            or market.get("dataset_id") != membership.get("bounded_dataset_id")
            or market.get("bars_fingerprint") != membership.get("bounded_data_sha256")
        ):
            raise ManifestError("holdout prediction differs from requested partition")
        captured = artifact.get("holdout_summary")
        if captured is None:
            raise ManifestError("holdout prediction summary evidence is unavailable")
        captured = mapping(captured)
        try:
            record(
                captured,
                {"schema_version", "window_result_id", "summary"},
                "holdout summary",
            )
            summary = validate_prediction_summary(captured.get("summary"))
            validate_prediction_summary_counts(
                summary,
                prediction_window_observations(
                    payload,
                    "final_holdout",
                    text(artifact["selection_id"]),
                    text(artifact["result_id"]),
                ),
                len(records(payload["decisions"])),
                PredictionMetricFields().to_primitive(),
            )
        except ManifestError as error:
            raise ManifestError(
                "holdout prediction summary evidence is invalid"
            ) from error
        if (
            captured.get("schema_version") != "1"
            or captured.get("window_result_id") != manifest.get("window_result_id")
            or configuration_identity({"summary": result.get("summary")})
            != configuration_identity({"summary": mapping(captured.get("summary"))})
        ):
            raise ManifestError(
                "holdout prediction summary differs from captured evidence"
            )
    elif (
        artifact.get("kind") == "backtest" and adapter.get("adapter") == "qf39_backtest"
    ):
        validate_backtest_identity(manifest)
        if artifact.get("result_id") != manifest.get("run_id"):
            raise ManifestError("holdout backtest result identity is inconsistent")
        membership = mapping(request.get("evaluation_membership"))
        labels = membership.get("evaluation_sessions")
        if not isinstance(labels, list) or not labels:
            raise ManifestError("holdout request evaluation sessions are invalid")
        try:
            # Serialize the producer's pure boundary contract; no partition or
            # backtest is built, and the frozen execution settings stay intact.
            interval = EvaluationInterval(
                date.fromisoformat(text(labels[0])),
                date.fromisoformat(text(labels[-1])),
            ).to_primitive()
        except ValueError as error:
            raise ManifestError(
                "holdout request evaluation sessions are invalid"
            ) from error
        execution = source.plan.environment.execution
        if execution is None:
            raise ManifestError("holdout request has no frozen backtest configuration")
        expected_configuration = {
            **execution.configuration.configuration_snapshot.to_primitive(),
            "evaluation_interval": interval,
        }
        if (
            mapping(manifest.get("strategy")).get("configuration") != definition
            or manifest.get("backtest_configuration") != expected_configuration
            or any(
                manifest.get(key) != expected_configuration.get(key)
                for key in ("engine_version", "result_schema_version")
            )
        ):
            raise ManifestError("holdout backtest differs from frozen configuration")
        market = mapping(manifest.get("market_data"))
        if (
            market.get("dataset_id") != membership.get("bounded_dataset_id")
            # QF-5's independent bars_fingerprint uses a different serialization
            # from the QF-3 data_sha256 retained by the partition request.
            or market.get("data_sha256") != membership.get("bounded_data_sha256")
        ):
            raise ManifestError("holdout backtest differs from requested dataset")
        if configuration_identity(
            {"summary": result.get("summary")}
        ) != configuration_identity({"summary": mapping(manifest.get("performance"))}):
            raise ManifestError(
                "holdout backtest summary differs from captured performance"
            )
    else:
        raise ManifestError("holdout artifact kind differs from its request")
