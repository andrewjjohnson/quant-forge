"""Verify QF-11 row provenance and identities without labeling or evaluating."""

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments._prediction_sessions import (
    validate_outcome_session,
    validate_signal_session,
)
from quantforge.experiments._prediction_temporal_integrity import (
    elapsed_temporal_configuration,
    primary_context_observations,
    validate_elapsed_resolution,
)


def _require_fields(record: PrimitiveMapping, expected: PrimitiveMapping) -> None:
    if configuration_identity(
        {key: record.get(key) for key in expected}
    ) != configuration_identity(expected):
        raise ManifestError("prediction row provenance differs from its manifest")


def validate_prediction_signal(
    manifest: PrimitiveMapping, prediction: PrimitiveMapping, indexes: dict[str, int]
) -> str:
    """Check all predictions, including end-of-data signals without labeled rows."""
    market_data = mapping(manifest.get("market_data"))
    rule = mapping(mapping(manifest.get("configuration")).get("prediction_rule"))
    _require_fields(
        prediction,
        {
            "strategy_id": rule.get("name"),
            "strategy_implementation_version": rule.get("implementation_version"),
            "strategy_configuration_id": rule.get("configuration_id"),
            "symbol": market_data.get("symbol"),
        },
    )
    mapping(prediction.get("values"))
    mapping(prediction.get("strategy_parameters"))
    rule_configuration = mapping(rule.get("configuration"))
    if "parameters" in rule_configuration:
        _require_fields(
            prediction, {"strategy_parameters": rule_configuration["parameters"]}
        )
    context = manifest.get("prediction_context")
    decision_session = None
    if context is not None and mapping(context).get("status") == "available":
        decision_session = text(mapping(context).get("decision_session"))
    return validate_signal_session(
        prediction,
        rule,
        indexes,
        decision_session=decision_session,
        context_warm_up_observations=(
            primary_context_observations(manifest)
            if elapsed_temporal_configuration(manifest) is not None
            else None
        ),
    )


def validate_prediction_row(
    manifest: PrimitiveMapping, row: PrimitiveMapping, indexes: dict[str, int]
) -> str:
    """Check existing dataset/component references and the producer's three hashes."""
    market_data = mapping(manifest.get("market_data"))
    configuration = mapping(manifest.get("configuration"))
    dataset: PrimitiveMapping = {
        "dataset_id": market_data.get("dataset_id"),
        "dataset_fingerprint": market_data.get("bars_fingerprint"),
    }
    _require_fields(row, {**dataset, "study_id": manifest.get("study_id")})
    prediction = mapping(row.get("prediction"))
    signal = validate_prediction_signal(manifest, prediction, indexes)
    outcome = mapping(row.get("outcome"))
    labeler = mapping(configuration.get("outcome_labeler"))
    if elapsed_temporal_configuration(manifest) is not None:
        validate_elapsed_resolution(manifest, prediction, outcome)
    else:
        validate_outcome_session(
            signal,
            outcome.get("outcome_session"),
            labeler.get("required_future_sessions"),
            indexes,
        )
    _require_fields(
        outcome,
        {
            **dataset,
            "signal_session": prediction.get("signal_session"),
            "outcome_name": labeler.get("name"),
            "outcome_implementation_version": labeler.get("implementation_version"),
            "outcome_configuration_id": labeler.get("configuration_id"),
            "outcome_result_schema_version": labeler.get("result_schema_version"),
        },
    )
    outcome_id = configuration_identity(
        {
            "record_type": "prediction_outcome",
            **{key: value for key, value in outcome.items() if key != "outcome_id"},
        }
    )
    if outcome.get("outcome_id") != outcome_id:
        raise ManifestError("prediction outcome identity is inconsistent")
    evaluation = mapping(row.get("evaluation"))
    evaluator = mapping(configuration.get("evaluator"))
    _require_fields(
        evaluation,
        {
            "outcome_id": outcome_id,
            "evaluator_name": evaluator.get("name"),
            "evaluator_implementation_version": evaluator.get("implementation_version"),
            "evaluator_configuration_id": evaluator.get("configuration_id"),
            "evaluation_result_schema_version": evaluator.get("result_schema_version"),
        },
    )
    evaluation_id = configuration_identity(
        {
            "record_type": "prediction_evaluation",
            "prediction": mapping(prediction.get("values")),
            **{
                key: value
                for key, value in evaluation.items()
                if key != "evaluation_id"
            },
        }
    )
    if evaluation.get("evaluation_id") != evaluation_id:
        raise ManifestError("prediction evaluation identity is inconsistent")
    row_id = configuration_identity(
        {
            "record_type": "prediction_study_row",
            "study_id": manifest.get("study_id"),
            "outcome_id": outcome_id,
            "evaluation_id": evaluation_id,
            "signal": {
                "features": mapping(row.get("features")),
                "prediction": prediction,
            },
        }
    )
    if row.get("row_id") != row_id:
        raise ManifestError("prediction row identity is inconsistent")
    return text(row.get("row_id"))
