"""Exploratory descriptive analysis for completed QF-7 feature datasets."""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, DecimalException
from enum import StrEnum
from pathlib import Path
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.prediction._arithmetic import arithmetic
from quantforge.prediction.errors import SignalFeatureDatasetError
from quantforge.prediction.signal_feature_models import (
    SchemaFieldCategory,
    SignalFeatureDatasetResult,
)

FEATURE_ANALYSIS_ENGINE_VERSION = "11"

_NUMERIC_SCHEMA_TYPES = frozenset(("decimal", "integer"))
_SCALAR_SCHEMA_TYPES = frozenset(("boolean", "date", "decimal", "integer", "string"))


class WinnerDefinition(StrEnum):
    """Configurable rule for splitting eligible rows into two outcome groups."""

    DECIMAL_GREATER_THAN_ZERO = "decimal_greater_than_zero"
    VALUE_EQUALS = "value_equals"


@dataclass(frozen=True, slots=True)
class FeatureAnalysisBin:
    """One deterministic left-inclusive, right-exclusive numeric interval."""

    label: str
    lower_inclusive: Decimal | None
    upper_exclusive: Decimal | None

    def __post_init__(self) -> None:
        if not self.label:
            raise SignalFeatureDatasetError("feature analysis bin labels are required")
        if (
            self.lower_inclusive is not None
            and self.upper_exclusive is not None
            and self.lower_inclusive >= self.upper_exclusive
        ):
            raise SignalFeatureDatasetError(
                "feature analysis bin lower bound must be below its upper bound"
            )

    def contains(self, value: Decimal) -> bool:
        return (self.lower_inclusive is None or value >= self.lower_inclusive) and (
            self.upper_exclusive is None or value < self.upper_exclusive
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "label": self.label,
            "lower_inclusive": _optional_decimal(self.lower_inclusive),
            "upper_exclusive": _optional_decimal(self.upper_exclusive),
        }


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Descriptive statistics retaining sample count and dispersion."""

    sample_count: int
    mean: Decimal | None
    median: Decimal | None
    population_standard_deviation: Decimal | None
    first_quartile: Decimal | None
    third_quartile: Decimal | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "first_quartile": _optional_decimal(self.first_quartile),
            "mean": _optional_decimal(self.mean),
            "median": _optional_decimal(self.median),
            "population_standard_deviation": _optional_decimal(
                self.population_standard_deviation
            ),
            "sample_count": self.sample_count,
            "third_quartile": _optional_decimal(self.third_quartile),
        }


@dataclass(frozen=True, slots=True)
class FeatureGroupAnalysis:
    """One feature's distribution within winners or losers."""

    feature_name: str
    outcome_group: str
    statistics: DistributionSummary

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "feature_name": self.feature_name,
            "outcome_group": self.outcome_group,
            "statistics": self.statistics.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class FeatureBinAnalysis:
    """Outcome rate and honest count for one configured feature interval."""

    feature_name: str
    bin_label: str
    lower_inclusive: Decimal | None
    upper_exclusive: Decimal | None
    sample_count: int
    winner_count: int
    loser_count: int
    winner_rate: Decimal | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "bin_label": self.bin_label,
            "feature_name": self.feature_name,
            "loser_count": self.loser_count,
            "lower_inclusive": _optional_decimal(self.lower_inclusive),
            "sample_count": self.sample_count,
            "upper_exclusive": _optional_decimal(self.upper_exclusive),
            "winner_count": self.winner_count,
            "winner_rate": _optional_decimal(self.winner_rate),
        }


@dataclass(frozen=True, slots=True)
class SignalFeatureAnalysisResult:
    """Exploratory winner/loser comparison with no automatic filter selection."""

    analysis_id: str
    feature_dataset_id: str
    configuration_snapshot: PrimitiveMappingSnapshot
    eligible_row_count: int
    winner_count: int
    loser_count: int
    group_summaries: tuple[FeatureGroupAnalysis, ...]
    bin_summaries: tuple[FeatureBinAnalysis, ...]

    @property
    def configuration(self) -> PrimitiveMapping:
        return self.configuration_snapshot.to_primitive()

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "analysis_id": self.analysis_id,
            "component": "quantforge_signal_feature_analysis",
            "configuration": self.configuration,
            "feature_dataset_id": self.feature_dataset_id,
            "group_summaries": [
                summary.to_primitive() for summary in self.group_summaries
            ],
            "bin_summaries": [summary.to_primitive() for summary in self.bin_summaries],
            "record_counts": {
                "eligible_rows": self.eligible_row_count,
                "losers": self.loser_count,
                "winners": self.winner_count,
            },
            "warning": (
                "exploratory descriptive analysis only; observed associations are "
                "candidate hypotheses, not causal or validated trading filters"
            ),
        }


def analyze_signal_features(
    result: SignalFeatureDatasetResult,
    *,
    feature_names: tuple[str, ...],
    outcome_name: str,
    winner_definition: WinnerDefinition,
    bins: dict[str, tuple[FeatureAnalysisBin, ...]],
    winner_value: str | None = None,
) -> SignalFeatureAnalysisResult:
    """Compare configured features across an explicit outcome split and bins."""
    if (
        not feature_names
        or feature_names != tuple(sorted(feature_names))
        or len(feature_names) != len(set(feature_names))
    ):
        raise SignalFeatureDatasetError(
            "analysis feature names must be a sorted unique tuple"
        )
    schema_fields = {field.name: field for field in result.schema.fields}
    if outcome_name not in schema_fields or any(
        feature_name not in schema_fields for feature_name in feature_names
    ):
        raise SignalFeatureDatasetError(
            "analysis fields must exist in the signal-feature schema"
        )
    if any(
        schema_fields[feature_name].category
        is not SchemaFieldCategory.CONTEMPORANEOUS_FEATURE
        for feature_name in feature_names
    ):
        raise SignalFeatureDatasetError(
            "analysis features must be contemporaneous-feature schema fields"
        )
    if schema_fields[outcome_name].category is not SchemaFieldCategory.FUTURE_OUTCOME:
        raise SignalFeatureDatasetError(
            "analysis outcome must be a future-outcome schema field"
        )
    if any(
        schema_fields[feature_name].data_type not in _NUMERIC_SCHEMA_TYPES
        for feature_name in feature_names
    ):
        raise SignalFeatureDatasetError(
            "analysis features must use numeric decimal or integer schema types"
        )
    outcome_data_type = schema_fields[outcome_name].data_type
    if (
        winner_definition is WinnerDefinition.DECIMAL_GREATER_THAN_ZERO
        and outcome_data_type not in _NUMERIC_SCHEMA_TYPES
    ):
        raise SignalFeatureDatasetError(
            "decimal-greater-than-zero requires a numeric outcome schema type"
        )
    if (
        winner_definition is WinnerDefinition.VALUE_EQUALS
        and outcome_data_type not in _SCALAR_SCHEMA_TYPES
    ):
        raise SignalFeatureDatasetError(
            "value-equals requires a scalar outcome schema type"
        )
    if set(bins) != set(feature_names) or any(not bins[name] for name in feature_names):
        raise SignalFeatureDatasetError(
            "every analyzed feature requires explicit nonempty bins"
        )
    if winner_definition is WinnerDefinition.VALUE_EQUALS and winner_value is None:
        raise SignalFeatureDatasetError(
            "value-equals winner definition requires winner_value"
        )
    value_equals_winner_value = cast(str, winner_value)
    (
        outcome_availability_name,
        outcome_unavailable_value,
        availability_unavailable_value,
    ) = _outcome_eligibility(result, outcome_name)
    if outcome_availability_name == outcome_name:
        raise SignalFeatureDatasetError(
            "outcome availability fields cannot be analysis outcomes"
        )
    if outcome_availability_name is not None and (
        outcome_availability_name not in schema_fields
        or schema_fields[outcome_availability_name].category
        is not SchemaFieldCategory.FUTURE_OUTCOME
        or schema_fields[outcome_availability_name].data_type != "boolean"
        or schema_fields[outcome_availability_name].nullable
        or availability_unavailable_value is not False
    ):
        raise SignalFeatureDatasetError(
            "analysis outcome availability field must be a non-nullable boolean "
            "future outcome with an unavailable default of false"
        )
    if outcome_availability_name is None and outcome_unavailable_value is not None:
        raise SignalFeatureDatasetError(
            "analysis outcomes with non-null unavailable defaults require an "
            "availability flag"
        )
    normalized_winner_value = winner_value
    if (
        winner_definition is WinnerDefinition.VALUE_EQUALS
        and outcome_data_type == "decimal"
    ):
        parsed_winner_value = _decimal_value(value_equals_winner_value)
        if parsed_winner_value is None:
            raise SignalFeatureDatasetError(
                "value-equals requires a finite decimal winner_value for a decimal "
                "outcome"
            )
        normalized_winner_value = decimal_to_primitive(parsed_winner_value)
    if (
        winner_definition is WinnerDefinition.VALUE_EQUALS
        and outcome_data_type == "boolean"
        and value_equals_winner_value not in ("false", "true")
    ):
        raise SignalFeatureDatasetError(
            "value-equals requires winner_value true or false for a boolean outcome"
        )
    if (
        winner_definition is WinnerDefinition.VALUE_EQUALS
        and outcome_data_type == "integer"
    ):
        try:
            normalized_winner_value = str(int(value_equals_winner_value))
        except ValueError as error:
            raise SignalFeatureDatasetError(
                "value-equals requires an integer winner_value for an integer outcome"
            ) from error
    if (
        winner_definition is WinnerDefinition.VALUE_EQUALS
        and outcome_data_type == "date"
    ):
        try:
            parsed_winner_date = date.fromisoformat(value_equals_winner_value)
        except ValueError as error:
            raise SignalFeatureDatasetError(
                "value-equals requires a canonical ISO date winner_value for a date "
                "outcome"
            ) from error
        if parsed_winner_date.isoformat() != value_equals_winner_value:
            raise SignalFeatureDatasetError(
                "value-equals requires a canonical ISO date winner_value for a date "
                "outcome"
            )
        normalized_winner_value = parsed_winner_date.isoformat()
    configuration: PrimitiveMapping = {
        "bins": {
            name: [item.to_primitive() for item in bins[name]] for name in feature_names
        },
        "engine_version": FEATURE_ANALYSIS_ENGINE_VERSION,
        "feature_names": list(feature_names),
        "outcome_availability_name": outcome_availability_name,
        "outcome_name": outcome_name,
        "winner_definition": winner_definition.value,
        "winner_value": normalized_winner_value,
    }
    configuration_snapshot = PrimitiveMappingSnapshot.capture(configuration)
    analysis_id = configuration_identity(
        {
            "component": "quantforge_signal_feature_analysis",
            "configuration": configuration,
            "feature_dataset_id": result.dataset_id,
        }
    )
    classified: list[tuple[PrimitiveMapping, bool]] = []
    for row in result.rows:
        primitive = row.to_primitive()
        if (
            outcome_availability_name is not None
            and primitive.get(outcome_availability_name) is not True
        ):
            continue
        classification = _winner_classification(
            primitive.get(outcome_name),
            winner_definition,
            normalized_winner_value,
            outcome_data_type,
        )
        if classification is not None:
            classified.append((primitive, classification))

    group_summaries: list[FeatureGroupAnalysis] = []
    bin_summaries: list[FeatureBinAnalysis] = []
    for feature_name in feature_names:
        for outcome_group, is_winner in (("winner", True), ("loser", False)):
            values = tuple(
                parsed
                for primitive, classification in classified
                if classification is is_winner
                and (parsed := _decimal_value(primitive.get(feature_name))) is not None
            )
            group_summaries.append(
                FeatureGroupAnalysis(feature_name, outcome_group, _distribution(values))
            )
        for feature_bin in bins[feature_name]:
            classifications = tuple(
                classification
                for primitive, classification in classified
                if (value := _decimal_value(primitive.get(feature_name))) is not None
                and feature_bin.contains(value)
            )
            winner_count = sum(classifications)
            sample_count = len(classifications)
            with arithmetic():
                winner_rate = (
                    None
                    if sample_count == 0
                    else Decimal(winner_count) / Decimal(sample_count)
                )
            bin_summaries.append(
                FeatureBinAnalysis(
                    feature_name,
                    feature_bin.label,
                    feature_bin.lower_inclusive,
                    feature_bin.upper_exclusive,
                    sample_count,
                    winner_count,
                    sample_count - winner_count,
                    winner_rate,
                )
            )
    winner_count = sum(classification for _, classification in classified)
    return SignalFeatureAnalysisResult(
        analysis_id,
        result.dataset_id,
        configuration_snapshot,
        len(classified),
        winner_count,
        len(classified) - winner_count,
        tuple(group_summaries),
        tuple(bin_summaries),
    )


def export_signal_feature_analysis(
    result: SignalFeatureAnalysisResult, output_root: Path
) -> Path:
    """Atomically write one deterministic exploratory analysis JSON artifact."""
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"{result.analysis_id}.json"
    expected = (
        json.dumps(
            result.to_primitive(),
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if destination.exists():
        try:
            if destination.read_text(encoding="utf-8") != expected:
                raise SignalFeatureDatasetError(
                    "existing feature analysis conflicts with deterministic output"
                )
        except OSError as error:
            raise SignalFeatureDatasetError(
                "failed to validate existing feature analysis"
            ) from error
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{result.analysis_id}.", suffix=".tmp", dir=output_root
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except OSError as error:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise SignalFeatureDatasetError(
            "failed to export signal-feature analysis"
        ) from error
    return destination


def default_overnight_gap_feature_bins() -> dict[str, tuple[FeatureAnalysisBin, ...]]:
    """Return disclosed baseline bins for ATR, volume, and trend context."""
    return {
        "feature_atr_percentage_of_close": (
            FeatureAnalysisBin("below_1pct", None, Decimal("0.01")),
            FeatureAnalysisBin("1pct_to_2pct", Decimal("0.01"), Decimal("0.02")),
            FeatureAnalysisBin("2pct_and_above", Decimal("0.02"), None),
        ),
        "feature_trend_distance_percentage": (
            FeatureAnalysisBin("below_minus_1pct", None, Decimal("-0.01")),
            FeatureAnalysisBin(
                "minus_1pct_to_plus_1pct", Decimal("-0.01"), Decimal("0.01")
            ),
            FeatureAnalysisBin("plus_1pct_and_above", Decimal("0.01"), None),
        ),
        "feature_volume_ratio": (
            FeatureAnalysisBin("below_0_75", None, Decimal("0.75")),
            FeatureAnalysisBin("0_75_to_1_25", Decimal("0.75"), Decimal("1.25")),
            FeatureAnalysisBin("1_25_and_above", Decimal("1.25"), None),
        ),
    }


def _winner_classification(
    value: Primitive | None,
    winner_definition: WinnerDefinition,
    winner_value: str | None,
    outcome_data_type: str,
) -> bool | None:
    if value is None:
        return None
    if winner_definition is WinnerDefinition.DECIMAL_GREATER_THAN_ZERO:
        decimal_value = _decimal_value(value)
        return None if decimal_value is None else decimal_value > 0
    if outcome_data_type == "decimal":
        decimal_value = _decimal_value(value)
        decimal_winner_value = _decimal_value(winner_value)
        return (
            None
            if decimal_value is None or decimal_winner_value is None
            else decimal_value == decimal_winner_value
        )
    if outcome_data_type == "boolean":
        if not isinstance(value, bool) or winner_value not in ("false", "true"):
            return None
        return value is (winner_value == "true")
    if outcome_data_type == "integer":
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not isinstance(winner_value, str)
        ):
            return None
        try:
            return value == int(winner_value)
        except ValueError:
            return None
    if outcome_data_type == "date":
        if not isinstance(value, str) or not isinstance(winner_value, str):
            return None
        try:
            parsed_value = date.fromisoformat(value)
        except ValueError:
            return None
        return parsed_value.isoformat() == value == winner_value
    if not isinstance(value, (str, int, float, bool)):
        return None
    return str(value) == winner_value


def _outcome_eligibility(
    result: SignalFeatureDatasetResult, outcome_name: str
) -> tuple[str | None, Primitive | None, Primitive | None]:
    configured_outcomes = result.configuration.get("outcomes")
    if not isinstance(configured_outcomes, list):
        return None, None, None
    for raw_outcome in configured_outcomes:
        if not isinstance(raw_outcome, dict):
            continue
        outcome = cast(PrimitiveMapping, raw_outcome)
        namespace = outcome.get("namespace")
        raw_fields = outcome.get("fields")
        if not isinstance(namespace, str) or not isinstance(raw_fields, list):
            continue
        field_names = tuple(
            field_name
            for raw_field in raw_fields
            if isinstance(raw_field, dict)
            and isinstance(
                field_name := cast(PrimitiveMapping, raw_field).get("field_name"), str
            )
        )
        prefix = f"outcome_{namespace}_"
        if not outcome_name.startswith(prefix):
            continue
        field_name = outcome_name.removeprefix(prefix)
        if field_name not in field_names:
            continue
        availability_name = (
            f"outcome_{namespace}_available" if "available" in field_names else None
        )
        raw_unavailable_values = outcome.get("unavailable_values")
        unavailable_value = (
            raw_unavailable_values.get(field_name)
            if availability_name is None and isinstance(raw_unavailable_values, dict)
            else None
        )
        availability_unavailable_value = (
            raw_unavailable_values.get("available")
            if availability_name is not None
            and isinstance(raw_unavailable_values, dict)
            else None
        )
        return (
            availability_name,
            cast(Primitive | None, unavailable_value),
            cast(Primitive | None, availability_unavailable_value),
        )
    return None, None, None


def _decimal_value(value: Primitive | None) -> Decimal | None:
    if value is None or isinstance(value, (dict, list, bool)):
        return None
    try:
        parsed = Decimal(str(value))
    except (DecimalException, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _distribution(values: tuple[Decimal, ...]) -> DistributionSummary:
    if not values:
        return DistributionSummary(0, None, None, None, None, None)
    ordered = tuple(sorted(values))
    try:
        with arithmetic():
            mean = sum(ordered, Decimal(0)) / Decimal(len(ordered))
            variance = sum(
                ((value - mean) ** 2 for value in ordered), Decimal(0)
            ) / Decimal(len(ordered))
            standard_deviation = variance.sqrt()
            first_quartile = _quantile(ordered, Decimal("0.25"))
            median = _quantile(ordered, Decimal("0.5"))
            third_quartile = _quantile(ordered, Decimal("0.75"))
    except DecimalException as error:
        raise SignalFeatureDatasetError(
            "feature distribution arithmetic failed"
        ) from error
    return DistributionSummary(
        len(ordered),
        mean,
        median,
        standard_deviation,
        first_quartile,
        third_quartile,
    )


def _quantile(values: tuple[Decimal, ...], quantile: Decimal) -> Decimal:
    position = Decimal(len(values) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(values) - 1)
    fraction = position - Decimal(lower_index)
    return values[lower_index] + (values[upper_index] - values[lower_index]) * fraction


def _optional_decimal(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_primitive(value)
