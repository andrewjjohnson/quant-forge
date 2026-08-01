"""Reusable, aligned, and causal indicator contracts."""

from quantforge.indicators.base import Indicator, IndicatorParameters
from quantforge.indicators.exceptions import (
    IndicatorCalculationError,
    IndicatorError,
    InvalidIndicatorParametersError,
    MisalignedIndicatorOutputError,
    MissingMarketFieldError,
    UnorderedMarketDataError,
)
from quantforge.indicators.models import (
    IndicatorFieldOutput,
    IndicatorOutput,
    IndicatorValue,
    MarketField,
)
from quantforge.indicators.moving_average import (
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
)

__all__ = [
    "SIMPLE_MOVING_AVERAGE_OUTPUT",
    "Indicator",
    "IndicatorCalculationError",
    "IndicatorError",
    "IndicatorFieldOutput",
    "IndicatorOutput",
    "IndicatorParameters",
    "IndicatorValue",
    "InvalidIndicatorParametersError",
    "MarketField",
    "MisalignedIndicatorOutputError",
    "MissingMarketFieldError",
    "SimpleMovingAverage",
    "SimpleMovingAverageParameters",
    "UnorderedMarketDataError",
]
