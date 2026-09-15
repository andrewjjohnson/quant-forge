"""Bind persisted QF-32 results to their recorded candidate definition."""

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._json import ManifestError, mapping


def validate_prediction_trial_result(
    trial: PrimitiveMapping, artifact: PrimitiveMapping, study: PrimitiveMapping
) -> None:
    """Compare stored component/context/data inputs; never construct a candidate."""
    window = study.get("decision_schedule") is not None
    result = mapping(
        artifact.get("prediction_window" if window else "prediction_study")
    )
    manifest = mapping(result.get("manifest"))
    definition = mapping(trial.get("trial_definition"))
    configuration = mapping(manifest.get("configuration"))
    for name in ("prediction_rule", "outcome_labeler", "evaluator"):
        component = mapping(configuration.get(name))
        expected = mapping(definition.get(name))
        if any(component.get(key) != value for key, value in expected.items()):
            raise ManifestError("prediction result differs from its trial definition")
    if (
        manifest.get("market_data") != study.get("dataset")
        or configuration.get("prediction_context_requirements")
        != definition.get("prediction_context")
        or configuration.get("feature_configuration")
        != definition.get("feature_configuration")
        or configuration.get("result_schema_version")
        != definition.get("result_schema_version")
        or definition.get("indicator_backend_environment")
        != study.get("indicator_backend")
        or artifact.get("prediction_window_id" if window else "prediction_study_id")
        != manifest.get("window_result_id" if window else "study_id")
    ):
        raise ManifestError("prediction result differs from its trial definition")
    if window:
        if any(
            manifest.get(key) != study.get(key)
            for key in ("dataset_family_fingerprint", "context_environment")
        ) or (
            manifest.get("schedule") != study.get("decision_schedule")
            or manifest.get("engine_version")
            != study.get("prediction_window_engine_version")
            or manifest.get("indicator_backend_environment")
            != study.get("indicator_backend")
        ):
            raise ManifestError("prediction window differs from its trial definition")
    elif mapping(manifest.get("prediction_context")).get(
        "requirements"
    ) != definition.get("prediction_context"):
        raise ManifestError("prediction context differs from its trial definition")
