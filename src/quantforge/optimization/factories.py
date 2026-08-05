"""Serializable strategy-construction boundary for optimization trials."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol, cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    decimal_to_primitive,
)
from quantforge.indicators import MarketField
from quantforge.optimization.errors import InvalidStudyConfigurationError
from quantforge.strategies import (
    InvalidStrategyParametersError,
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
    Strategy,
)


class StrategyFactory(Protocol):
    """Generic extension point that constructs a real QF-4 strategy contract."""

    @property
    def strategy_name(self) -> str: ...

    @property
    def strategy_version(self) -> str: ...

    @property
    def parameter_order(self) -> tuple[str, ...]: ...

    @property
    def required_parameter_names(self) -> frozenset[str]: ...

    def configuration(self) -> PrimitiveMapping: ...

    def build(self, parameters: PrimitiveMapping) -> Strategy: ...


@dataclass(frozen=True, slots=True)
class MovingAverageCrossoverFactory:
    """QF-4 moving-average constructor exposed through the generic factory API."""

    default_source_field: MarketField = MarketField.CLOSE
    default_target_long_weight: Decimal = Decimal("1")

    strategy_name = MovingAverageCrossoverStrategy.name
    strategy_version = MovingAverageCrossoverStrategy.implementation_version
    parameter_order = (
        "fast_window",
        "slow_window",
        "source_field",
        "target_long_weight",
    )
    required_parameter_names = frozenset(("fast_window", "slow_window"))

    def __post_init__(self) -> None:
        try:
            defaults = MovingAverageCrossoverParameters(
                fast_window=1,
                slow_window=2,
                source_field=self.default_source_field,
                target_long_weight=self.default_target_long_weight,
            )
        except InvalidStrategyParametersError as error:
            raise InvalidStudyConfigurationError(
                "moving-average factory defaults are invalid"
            ) from error
        object.__setattr__(self, "default_source_field", defaults.source_field)
        object.__setattr__(
            self,
            "default_target_long_weight",
            defaults.target_long_weight,
        )

    def configuration(self) -> PrimitiveMapping:
        return {
            "component": "quantforge_strategy_factory",
            "factory_schema_version": "1",
            "factory_type": (
                "quantforge.optimization.factories.MovingAverageCrossoverFactory"
            ),
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "parameter_contract": (
                "quantforge.strategies.MovingAverageCrossoverParameters"
            ),
            "parameter_order": list(self.parameter_order),
            "required_parameters": cast(
                list[Primitive], sorted(self.required_parameter_names)
            ),
            "default_parameters": {
                "source_field": self.default_source_field.value,
                "target_long_weight": decimal_to_primitive(
                    self.default_target_long_weight
                ),
            },
        }

    def build(self, parameters: PrimitiveMapping) -> Strategy:
        unknown = set(parameters).difference(self.parameter_order)
        if unknown:
            rendered = ", ".join(sorted(unknown))
            raise InvalidStudyConfigurationError(
                f"moving-average parameter contract does not contain: {rendered}"
            )
        missing = self.required_parameter_names.difference(parameters)
        if missing:
            rendered = ", ".join(sorted(missing))
            raise InvalidStudyConfigurationError(
                f"moving-average parameters are required: {rendered}"
            )
        fast_window = parameters["fast_window"]
        slow_window = parameters["slow_window"]
        source_value = parameters.get("source_field", self.default_source_field.value)
        weight_value = parameters.get(
            "target_long_weight",
            decimal_to_primitive(self.default_target_long_weight),
        )
        if isinstance(fast_window, bool) or not isinstance(fast_window, int):
            raise InvalidStrategyParametersError("fast_window must be an integer")
        if isinstance(slow_window, bool) or not isinstance(slow_window, int):
            raise InvalidStrategyParametersError("slow_window must be an integer")
        if not isinstance(source_value, str):
            raise InvalidStrategyParametersError(
                "source_field must be a string categorical value"
            )
        if not isinstance(weight_value, (str, int, float)) or isinstance(
            weight_value, bool
        ):
            raise InvalidStrategyParametersError(
                "target_long_weight must be a floating candidate"
            )
        try:
            source_field = MarketField(source_value)
        except ValueError as error:
            raise InvalidStrategyParametersError(
                "source_field must be a supported market field"
            ) from error
        try:
            target_long_weight = Decimal(str(weight_value))
        except (InvalidOperation, ValueError) as error:
            raise InvalidStrategyParametersError(
                "target_long_weight must be numeric"
            ) from error
        return MovingAverageCrossoverStrategy(
            MovingAverageCrossoverParameters(
                fast_window=fast_window,
                slow_window=slow_window,
                source_field=source_field,
                target_long_weight=target_long_weight,
            )
        )
