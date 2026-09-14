"""OOS-only summaries and explicit permanent final-holdout consumption (QF-40)."""

from quantforge.oos._records import OOSIntegrityError
from quantforge.oos.backtest import aggregate_backtest
from quantforge.oos.common import (
    configuration_stability,
    export_oos_aggregate,
    load_oos_aggregate,
)
from quantforge.oos.holdout import HoldoutConsumptionRecord, HoldoutLedger, HoldoutState
from quantforge.oos.holdout_evaluation import HoldoutEvaluation
from quantforge.oos.models import (
    BacktestOOSAggregate,
    ConfigurationStabilitySummary,
    MetricSummary,
    OOSSource,
    PredictionOOSAggregate,
)
from quantforge.oos.prediction import PredictionMetricFields, aggregate_prediction
from quantforge.oos.source import load_oos_source

__all__ = [
    "BacktestOOSAggregate",
    "ConfigurationStabilitySummary",
    "HoldoutConsumptionRecord",
    "HoldoutEvaluation",
    "HoldoutLedger",
    "HoldoutState",
    "MetricSummary",
    "OOSIntegrityError",
    "OOSSource",
    "PredictionMetricFields",
    "PredictionOOSAggregate",
    "aggregate_backtest",
    "aggregate_prediction",
    "configuration_stability",
    "export_oos_aggregate",
    "load_oos_aggregate",
    "load_oos_source",
]
