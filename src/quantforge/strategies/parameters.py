"""Immutable strategy parameter contracts."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol

from quantforge.configuration import PrimitiveMapping, decimal_to_primitive
from quantforge.indicators.models import MarketField
from quantforge.strategies.exceptions import InvalidStrategyParametersError


class StrategyParameters(Protocol):
    """Typed parameters that serialize into a stable primitive mapping."""

    def to_primitive(self) -> PrimitiveMapping: ...


@dataclass(frozen=True, slots=True)
class MovingAverageCrossoverParameters:
    """Configuration for a long-only fast/slow moving-average crossover."""

    fast_window: int
    slow_window: int
    source_field: MarketField = MarketField.CLOSE
    target_long_weight: Decimal = Decimal(1)

    def __post_init__(self) -> None:
        _validate_window_types(self.fast_window, self.slow_window)
        if self.fast_window < 1 or self.slow_window < 1:
            raise InvalidStrategyParametersError(
                "fast_window and slow_window must be positive"
            )
        if self.fast_window >= self.slow_window:
            raise InvalidStrategyParametersError(
                "fast_window must be strictly less than slow_window"
            )
        _validate_source_field(self.source_field)
        try:
            weight = Decimal(str(self.target_long_weight))
        except InvalidOperation as error:
            raise InvalidStrategyParametersError(
                "target_long_weight must be numeric"
            ) from error
        if not weight.is_finite() or not Decimal(0) < weight <= Decimal(1):
            raise InvalidStrategyParametersError(
                "target_long_weight must be greater than 0 and at most 1"
            )
        object.__setattr__(self, "target_long_weight", weight.normalize())

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "source_field": self.source_field.value,
            "target_long_weight": decimal_to_primitive(self.target_long_weight),
        }


def _validate_window_types(fast_window: object, slow_window: object) -> None:
    if any(
        isinstance(window, bool) or not isinstance(window, int)
        for window in (fast_window, slow_window)
    ):
        raise InvalidStrategyParametersError(
            "fast_window and slow_window must be integers"
        )


def _validate_source_field(value: object) -> None:
    if not isinstance(value, MarketField):
        raise InvalidStrategyParametersError(
            "source_field must be a normalized market field"
        )
