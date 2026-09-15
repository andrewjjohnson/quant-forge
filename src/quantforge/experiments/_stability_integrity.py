"""Validate QF-6/QF-32 stability schemas without computing statistics."""

from dataclasses import fields
from decimal import Decimal, InvalidOperation
from typing import cast

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._json import ManifestError
from quantforge.optimization.models import StabilityClassification, StabilitySummary
from quantforge.prediction.grid import PredictionStabilitySummary


def _decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, str):
        raise ManifestError(f"{label} decimals must be strings")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ManifestError(f"{label} decimal is invalid") from error
    if not result.is_finite():
        raise ManifestError(f"{label} decimals must be finite")
    return result


def validate_prediction_stability(record: PrimitiveMapping) -> None:
    """Check QF-32 producer fields, primitive types and intrinsic domains."""
    _validate_stability(record, optimization=False)


def validate_optimization_stability(record: PrimitiveMapping) -> None:
    """Check the completed QF-6 stability record, including its assigned rank."""
    _validate_stability(record, optimization=True)


def _validate_stability(record: PrimitiveMapping, *, optimization: bool) -> None:
    label = "optimization stability" if optimization else "prediction stability"
    model = StabilitySummary if optimization else PredictionStabilitySummary
    eligible_count = (
        "successful_eligible_neighbor_count"
        if optimization
        else "eligible_neighbor_count"
    )
    dispersion = (
        "objective_standard_deviation" if optimization else "relative_dispersion"
    )
    if set(record) != {field.name for field in fields(model)}:
        raise ManifestError(f"{label} fields differ from producer schema")
    for field in ("trial_id", "combination_id"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise ManifestError(f"{label} references must be strings")
    for field in (
        "objective_rank",
        "valid_neighbor_count",
        "excluded_neighbor_count",
        eligible_count,
        *(("stability_rank",) if optimization else ()),
    ):
        count = record[field]
        minimum = 1 if field.endswith("rank") else 0
        if type(count) is not int or count < minimum:
            raise ManifestError(f"{label} rank or neighbor count is invalid")
    _decimal(record["objective_value"], label)
    if not 0 <= _decimal(record["constraint_pass_fraction"], label) <= 1:
        raise ManifestError(f"{label} constraint fraction is outside [0, 1]")
    if optimization and not 0 <= _decimal(record["stability_score"], label) <= 1:
        raise ManifestError(f"{label} score is outside [0, 1]")
    for field in (
        "median_neighbor_objective",
        dispersion,
        "center_to_neighbor_difference",
        "relative_center_to_neighbor_difference",
        *(
            ("mean_neighbor_objective", "worst_neighbor_objective")
            if optimization
            else ()
        ),
    ):
        if record[field] is not None:
            value = _decimal(record[field], label)
            if field == dispersion and value < 0:
                raise ManifestError(f"{label} dispersion must be nonnegative")
    values = record["neighbor_objective_values"]
    if not isinstance(values, list):
        raise ManifestError(f"{label} neighbor objectives must be an array")
    for value in values:
        _decimal(value, label)
    if record[eligible_count] != len(values) or cast(
        int, record[eligible_count]
    ) > cast(int, record["valid_neighbor_count"]):
        raise ManifestError(f"{label} neighbor counts are inconsistent")
    for field in ("is_boundary", "is_isolated_peak"):
        if type(record[field]) is not bool:
            raise ManifestError(f"{label} flags must be booleans")
    if record["classification"] not in tuple(
        item.value for item in StabilityClassification
    ):
        raise ManifestError(f"{label} classification is invalid")
    reason = record["isolation_reason"]
    if (reason is not None and (not isinstance(reason, str) or not reason.strip())) or (
        record["is_isolated_peak"] and reason is None
    ):
        raise ManifestError(f"{label} isolation reason is invalid")
