"""Bind consumed result evidence to the retained QF-40 request without evaluation."""

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments._producer_integrity import validate_backtest_identity
from quantforge.experiments._window_integrity import validate_window_snapshot
from quantforge.experiments._window_sessions import scheduled_sessions
from quantforge.oos.models import OOSSource


def validate_holdout_artifact(
    source: OOSSource, consumption: PrimitiveMapping, result: PrimitiveMapping
) -> None:
    request = mapping(consumption.get("request"))
    frozen = mapping(request.get("frozen_selection"))
    if not any(
        fold.selection is not None and fold.selection.to_primitive() == frozen
        for fold in source.folds
    ) or any(
        request.get(key) != expected
        for key, expected in {
            "lineage_id": source.lineage_id,
            "study_id": source.study_id,
            "plan_id": source.plan.plan_id,
            "study_definition": source.definition.to_primitive(),
        }.items()
    ):
        raise ManifestError("holdout request differs from its frozen source selection")
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
        for name in ("prediction_rule", "outcome_labeler", "evaluator"):
            actual = mapping(configuration.get(name))
            expected = mapping(definition.get(name))
            if configuration_identity(
                {key: actual.get(key) for key in expected}
            ) != configuration_identity(expected):
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
        if not isinstance(labels, list):
            raise ManifestError("holdout request evaluation sessions are invalid")
        permitted_sessions = {text(label) for label in labels}
        context = mapping(
            mapping(manifest.get("context_environment")).get("configuration")
        )
        market = mapping(manifest.get("market_data"))
        if (
            context.get("partition") != membership
            or context.get("plan_id") != source.plan.plan_id
            or market.get("dataset_id") != membership.get("bounded_dataset_id")
            or market.get("bars_fingerprint") != membership.get("bounded_data_sha256")
            or not set(
                scheduled_sessions(mapping(manifest.get("schedule"))).values()
            ).issubset(permitted_sessions)
        ):
            raise ManifestError("holdout prediction differs from requested partition")
    elif (
        artifact.get("kind") == "backtest" and adapter.get("adapter") == "qf39_backtest"
    ):
        validate_backtest_identity(manifest)
        if artifact.get("result_id") != manifest.get("run_id"):
            raise ManifestError("holdout backtest result identity is inconsistent")
    else:
        raise ManifestError("holdout artifact kind differs from its request")
