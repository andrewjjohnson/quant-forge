"""Focused QF-7/QF-29 compositions for elapsed intraday path outcomes."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

from quantforge.configuration import PrimitiveMapping, decimal_to_primitive
from quantforge.data import TimeframeBarSeries
from quantforge.prediction._intraday_reference import REFERENCE_PRICE_CONVENTION
from quantforge.prediction.feature_dataset import (
    PredictionStudyOutcome,
    outcome_resolution_fields,
)
from quantforge.prediction.intraday_path import (
    IntradayPathOutcomeLabeler,
    IntradayPathValues,
)
from quantforge.prediction.intraday_path_evaluation import (
    IntradayExcursionEvaluationValues,
    IntradayExcursionEvaluator,
    IntradayTargetStopEvaluationValues,
    IntradayTargetStopEvaluator,
    SameBarConflictPolicy,
)
from quantforge.prediction.outcome_temporal import OutcomeTemporalConfiguration
from quantforge.prediction.signal_feature_models import SchemaField, SchemaFieldCategory


def _path_schema(
    duration: timedelta,
    source: TimeframeBarSeries,
    evaluator_id: str,
    extra_definitions: tuple[tuple[str, str, str], ...],
    extra_defaults: PrimitiveMapping,
) -> tuple[IntradayPathOutcomeLabeler, tuple[SchemaField, ...], PrimitiveMapping]:
    temporal = OutcomeTemporalConfiguration.elapsed_duration(duration, source.timeframe)
    labeler = IntradayPathOutcomeLabeler(temporal)
    defaults: PrimitiveMapping = {
        "available": False,
        "anchor_kind": temporal.anchor_kind.value,
        "horizon_kind": "elapsed_duration",
        "elapsed_duration_microseconds": duration // timedelta(microseconds=1),
        "outcome_configuration_id": labeler.configuration_id,
        "temporal_configuration_id": temporal.configuration_id,
        "evaluation_configuration_id": evaluator_id,
        "reference_price_convention": REFERENCE_PRICE_CONVENTION,
        "source_reference": source.dataset_reference.to_primitive(
            include_feed_scope=True
        ),
        **extra_defaults,
    }
    definitions = (
        ("reference_observation_id", "string", "sha256"),
        ("reference_price", "decimal", "price_per_share"),
        ("reference_price_convention", "string", "policy"),
        ("source_reference", "object", "provenance"),
        ("endpoint_status", "string", "availability"),
        ("endpoint_available", "boolean", "availability"),
        ("path_start_timestamp", "string", "UTC_timestamp"),
        ("path_end_timestamp", "string", "UTC_timestamp"),
        ("missing_observation_timestamp", "string", "UTC_timestamp"),
        ("unavailable_reason", "string", "reason_code"),
        ("direction", "string", "prediction_direction"),
        ("evaluation_configuration_id", "string", "sha256"),
        *extra_definitions,
    )
    timing = "future label; requires every completed path bar through QF-46 endpoint"
    fields = (
        *(
            replace(
                field,
                nullable=field.nullable or field.name not in defaults,
                calculation_or_source=(
                    "complete path availability; endpoint status is recorded separately"
                    if field.name in ("status", "available")
                    else field.calculation_or_source
                ),
            )
            for field in outcome_resolution_fields()
        ),
        *(
            SchemaField(
                name,
                SchemaFieldCategory.FUTURE_OUTCOME,
                data_type,
                unit,
                name not in defaults,
                "QF-47 canonical intraday path outcome",
                timing,
            )
            for name, data_type, unit in definitions
        ),
    )
    return labeler, tuple(sorted(fields, key=lambda field: field.name)), defaults


def intraday_excursion_outcome(
    duration: timedelta,
    source: TimeframeBarSeries,
    *,
    namespace: str | None = None,
) -> PredictionStudyOutcome[IntradayPathValues, IntradayExcursionEvaluationValues]:
    """Export direction-aware MFE/MAE ratios with complete path provenance."""
    evaluator = IntradayExcursionEvaluator()
    labeler, fields, defaults = _path_schema(
        duration,
        source,
        evaluator.configuration_id,
        (
            ("mfe_percentage", "decimal", "ratio"),
            ("mae_percentage", "decimal", "ratio"),
            ("mfe_timestamp", "string", "UTC_timestamp"),
            ("mae_timestamp", "string", "UTC_timestamp"),
            ("mfe_observation_id", "string", "sha256"),
            ("mae_observation_id", "string", "sha256"),
        ),
        {},
    )
    return PredictionStudyOutcome[
        IntradayPathValues, IntradayExcursionEvaluationValues
    ].create(
        namespace
        if namespace is not None
        else f"intraday_mfe_mae_{duration // timedelta(microseconds=1)}us",
        labeler,
        evaluator,
        fields,
        unavailable_values=defaults,
        outcome_source=source,
    )


def intraday_target_stop_outcome(
    duration: timedelta,
    source: TimeframeBarSeries,
    target_percentage: Decimal,
    stop_percentage: Decimal,
    *,
    same_bar_conflict_policy: SameBarConflictPolicy = SameBarConflictPolicy.AMBIGUOUS,
    namespace: str | None = None,
) -> PredictionStudyOutcome[IntradayPathValues, IntradayTargetStopEvaluationValues]:
    """Export inclusive target/stop ratios; collisions remain explicitly ambiguous.

    Use explicit namespaces when comparing thresholds at the same duration.
    Thresholds/policy bind the evaluator and composed outcome identity, while
    the direction-neutral path labeler can be shared across those comparisons.
    """
    evaluator = IntradayTargetStopEvaluator(
        target_percentage, stop_percentage, same_bar_conflict_policy
    )
    labeler, fields, defaults = _path_schema(
        duration,
        source,
        evaluator.configuration_id,
        (
            ("target_percentage", "decimal", "ratio"),
            ("stop_percentage", "decimal", "ratio"),
            ("same_bar_conflict_policy", "string", "policy"),
            ("label", "string", "target_stop_label"),
            ("target_level", "decimal", "price_per_share"),
            ("stop_level", "decimal", "price_per_share"),
            ("event_timestamp", "string", "UTC_timestamp"),
            ("event_observation_id", "string", "sha256"),
            ("ambiguous_timestamp", "string", "UTC_timestamp"),
            ("ambiguous_observation_id", "string", "sha256"),
            ("ambiguous_high", "decimal", "price_per_share"),
            ("ambiguous_low", "decimal", "price_per_share"),
        ),
        {
            "target_percentage": decimal_to_primitive(evaluator.target_percentage),
            "stop_percentage": decimal_to_primitive(evaluator.stop_percentage),
            "same_bar_conflict_policy": evaluator.same_bar_conflict_policy.value,
            "label": "unavailable",
        },
    )
    return PredictionStudyOutcome[
        IntradayPathValues, IntradayTargetStopEvaluationValues
    ].create(
        namespace
        if namespace is not None
        else f"intraday_target_stop_{duration // timedelta(microseconds=1)}us",
        labeler,
        evaluator,
        fields,
        unavailable_values=defaults,
        outcome_source=source,
    )
