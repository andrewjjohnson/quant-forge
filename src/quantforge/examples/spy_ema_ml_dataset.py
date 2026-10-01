"""QF-67 plumbing example: the QF-45 EMA population as an event ML dataset.

This only declares the explicit causal feature schema and the 30-minute binary
target used to assemble QF-45 trigger observations. It changes nothing in the
frozen QF-45 rule, grid, dates, outcomes or selection. The tiny QF-45 event
sample is not evidence of ML adequacy, of an edge, or of profitability.
"""

from quantforge.examples.spy_ema import DAILY, TWO_MINUTES, EmaSmokeRule
from quantforge.ml import (
    EventFeatureDefinition,
    EventFeatureSchema,
    FeatureValueType,
    ForwardReturnBinaryTarget,
    MissingValuePolicy,
)
from quantforge.prediction import SchemaFieldCategory
from quantforge.timeframes import Timeframe

# Model column, persisted QF-45 strategy feature, source timeframe.
EMA_EVENT_FEATURES: tuple[tuple[str, str, Timeframe], ...] = (
    ("ema_fast_previous", "previous_fast", TWO_MINUTES),
    ("ema_slow_previous", "previous_slow", TWO_MINUTES),
    ("ema_fast", "current_fast", TWO_MINUTES),
    ("ema_slow", "current_slow", TWO_MINUTES),
    ("daily_close", "daily_close", DAILY),
    ("daily_ema50", "daily_ema50", DAILY),
)


def ema_event_feature_schema() -> EventFeatureSchema:
    """The six causal QF-45 inputs, derived from the rule's declared fields."""
    declared = {
        field.name: field
        for field in EmaSmokeRule().strategy_feature_definitions
        if field.category is SchemaFieldCategory.CONTEMPORANEOUS_FEATURE
    }
    return EventFeatureSchema(
        "qf45_ema_trigger_features",
        "1",
        tuple(
            EventFeatureDefinition(
                name=name,
                source_field=source,
                value_type=FeatureValueType.DECIMAL,
                missing_values=MissingValuePolicy.REJECT,
                unit=declared[source].unit,
                description=(
                    f"{declared[source].calculation_or_source}; "
                    f"{declared[source].temporal_availability}"
                ),
                timeframe=timeframe,
            )
            for name, source, timeframe in EMA_EVENT_FEATURES
        ),
    )


def ema_forward_return_target() -> ForwardReturnBinaryTarget:
    """QF-45's existing 30-minute raw forward return > 0 (zero is negative)."""
    return ForwardReturnBinaryTarget()


__all__ = [
    "EMA_EVENT_FEATURES",
    "ema_event_feature_schema",
    "ema_forward_return_target",
]
