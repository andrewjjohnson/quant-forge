"""Reusable engine-neutral strategy and sizing contracts."""

from quantforge.strategies.base import Strategy
from quantforge.strategies.exceptions import (
    DuplicateStrategyDecisionError,
    InvalidStrategyOutputError,
    InvalidStrategyParametersError,
    InvalidTargetWeightError,
    MissingRequiredMarketFieldError,
    StrategyError,
    UnorderedStrategyInputError,
    UnsupportedTimingConventionError,
)
from quantforge.strategies.models import (
    STRATEGY_OUTPUT_CONTRACT_VERSION,
    ExecutionSessionStatus,
    ExecutionTiming,
    IndicatorObservation,
    MarketDataReference,
    ParameterValue,
    PositionIntent,
    StrategyDecision,
    StrategyOutput,
)
from quantforge.strategies.moving_average import MovingAverageCrossoverStrategy
from quantforge.strategies.parameters import (
    MovingAverageCrossoverParameters,
    StrategyParameters,
)
from quantforge.strategies.runner import run_strategy
from quantforge.strategies.sizing import (
    PositionSizingPolicy,
    SizingContext,
    SizingContextField,
    TargetWeightIntent,
    TargetWeightSizingPolicy,
)

__all__ = [
    "STRATEGY_OUTPUT_CONTRACT_VERSION",
    "DuplicateStrategyDecisionError",
    "ExecutionSessionStatus",
    "ExecutionTiming",
    "IndicatorObservation",
    "InvalidStrategyOutputError",
    "InvalidStrategyParametersError",
    "InvalidTargetWeightError",
    "MarketDataReference",
    "MissingRequiredMarketFieldError",
    "MovingAverageCrossoverParameters",
    "MovingAverageCrossoverStrategy",
    "ParameterValue",
    "PositionIntent",
    "PositionSizingPolicy",
    "SizingContext",
    "SizingContextField",
    "Strategy",
    "StrategyDecision",
    "StrategyError",
    "StrategyOutput",
    "StrategyParameters",
    "TargetWeightIntent",
    "TargetWeightSizingPolicy",
    "UnorderedStrategyInputError",
    "UnsupportedTimingConventionError",
    "run_strategy",
]
