"""Bind persisted QF-32 results to their recorded candidate definition."""

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping


def frozen_prediction_components(
    definition: PrimitiveMapping, study: PrimitiveMapping
) -> PrimitiveMapping:
    """Read complete v2 wrappers or explicit declarations from legacy records."""
    from quantforge.experiments._producer_integrity import (
        validate_outcome_contract,
        validate_prediction_warm_up,
    )

    version = definition.get("contract_version", "1")
    if version not in ("1", "2") or version != study.get(
        "trial_definition_version", "1"
    ):
        raise ManifestError("unsupported or inconsistent trial definition version")
    components: PrimitiveMapping = {}
    for name, required in (
        ("prediction_rule", ("warm_up_observations",)),
        (
            "outcome_labeler",
            (
                "required_future_sessions",
                "required_market_fields",
                "result_schema_version",
            ),
        ),
        ("evaluator", ("result_schema_version",)),
    ):
        component = dict(mapping(definition.get(name)))
        captured = mapping(component.get("configuration"))
        for field in required:
            if field not in component:
                if version == "1" and field in captured:
                    component[field] = captured[field]
                elif (
                    version == "1"
                    and field == "required_future_sessions"
                    and (
                        isinstance(parameters := captured.get("parameters"), dict)
                        and "future_sessions" in parameters
                    )
                ):
                    component[field] = parameters["future_sessions"]
                else:
                    raise ManifestError(
                        "trial definition contract metadata is unavailable: "
                        f"{name}.{field}"
                    )
        components[name] = component
    validate_prediction_warm_up(components)
    validate_outcome_contract(components)
    return components


def prediction_components_match(
    configuration: PrimitiveMapping,
    components: PrimitiveMapping,
    contract_version: Primitive,
) -> bool:
    """Compare complete v2 wrappers; only legacy v1 projects captured fields."""
    if contract_version not in ("1", "2"):
        raise ManifestError("unsupported trial definition version")
    for name in ("prediction_rule", "outcome_labeler", "evaluator"):
        actual = mapping(configuration.get(name))
        expected = mapping(components.get(name))
        captured = (
            actual
            if contract_version == "2"
            else {key: actual.get(key) for key in expected}
        )
        if configuration_identity(captured) != configuration_identity(expected):
            return False
    return True


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
    components = frozen_prediction_components(definition, study)
    configuration = mapping(manifest.get("configuration"))
    if not prediction_components_match(
        configuration, components, definition.get("contract_version", "1")
    ):
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
