"""Verify QF-11 row provenance and identities without labeling or evaluating."""

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text


def _require_fields(record: PrimitiveMapping, expected: PrimitiveMapping) -> None:
    if any(record.get(key) != value for key, value in expected.items()):
        raise ManifestError("prediction row provenance differs from its manifest")


def validate_prediction_row(manifest: PrimitiveMapping, row: PrimitiveMapping) -> str:
    """Check existing dataset/component references and the producer's three hashes."""
    market_data = mapping(manifest.get("market_data"))
    configuration = mapping(manifest.get("configuration"))
    dataset: PrimitiveMapping = {
        "dataset_id": market_data.get("dataset_id"),
        "dataset_fingerprint": market_data.get("bars_fingerprint"),
    }
    _require_fields(row, {**dataset, "study_id": manifest.get("study_id")})
    prediction = mapping(row.get("prediction"))
    rule = mapping(configuration.get("prediction_rule"))
    _require_fields(
        prediction,
        {
            "strategy_id": rule.get("name"),
            "strategy_implementation_version": rule.get("implementation_version"),
            "strategy_configuration_id": rule.get("configuration_id"),
            "symbol": market_data.get("symbol"),
        },
    )
    rule_configuration = mapping(rule.get("configuration"))
    if "parameters" in rule_configuration:
        _require_fields(
            prediction, {"strategy_parameters": rule_configuration["parameters"]}
        )
    outcome = mapping(row.get("outcome"))
    labeler = mapping(configuration.get("outcome_labeler"))
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
