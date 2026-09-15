"""Validate the stored QF-32 stability schema without computing statistics."""

from dataclasses import fields
from decimal import Decimal, InvalidOperation
from typing import cast

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._json import ManifestError
from quantforge.optimization.models import StabilityClassification
from quantforge.prediction.grid import PredictionStabilitySummary


def _decimal(value: object) -> Decimal:
    if not isinstance(value, str):
        raise ManifestError("prediction stability decimals must be strings")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ManifestError("prediction stability decimal is invalid") from error
    if not result.is_finite():
        raise ManifestError("prediction stability decimals must be finite")
    return result


def validate_prediction_stability(record: PrimitiveMapping) -> None:
    """Check producer fields, primitive types and intrinsic value domains only."""
    if set(record) != {field.name for field in fields(PredictionStabilitySummary)}:
        raise ManifestError("prediction stability fields differ from producer schema")
    for field in ("trial_id", "combination_id"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise ManifestError("prediction stability references must be strings")
    for field in (
        "objective_rank",
        "valid_neighbor_count",
        "excluded_neighbor_count",
        "eligible_neighbor_count",
    ):
        count = record[field]
        minimum = 1 if field == "objective_rank" else 0
        if type(count) is not int or count < minimum:
            raise ManifestError(
                "prediction stability rank or neighbor count is invalid"
            )
    _decimal(record["objective_value"])
    if not 0 <= _decimal(record["constraint_pass_fraction"]) <= 1:
        raise ManifestError(
            "prediction stability constraint fraction is outside [0, 1]"
        )
    for field in (
        "median_neighbor_objective",
        "relative_dispersion",
        "center_to_neighbor_difference",
        "relative_center_to_neighbor_difference",
    ):
        if record[field] is not None:
            value = _decimal(record[field])
            if field == "relative_dispersion" and value < 0:
                raise ManifestError(
                    "prediction stability dispersion must be nonnegative"
                )
    values = record["neighbor_objective_values"]
    if not isinstance(values, list):
        raise ManifestError("prediction stability neighbor objectives must be an array")
    for value in values:
        _decimal(value)
    if record["eligible_neighbor_count"] != len(values) or cast(
        int, record["eligible_neighbor_count"]
    ) > cast(int, record["valid_neighbor_count"]):
        raise ManifestError("prediction stability neighbor counts are inconsistent")
    for field in ("is_boundary", "is_isolated_peak"):
        if type(record[field]) is not bool:
            raise ManifestError("prediction stability flags must be booleans")
    if record["classification"] not in tuple(
        item.value for item in StabilityClassification
    ):
        raise ManifestError("prediction stability classification is invalid")
    reason = record["isolation_reason"]
    if (reason is not None and (not isinstance(reason, str) or not reason.strip())) or (
        record["is_isolated_peak"] and reason is None
    ):
        raise ManifestError("prediction stability isolation reason is invalid")
