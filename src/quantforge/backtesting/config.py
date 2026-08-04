"""Immutable typed backtest configuration."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import cast

from quantforge.backtesting._arithmetic import (
    arithmetic,
    arithmetic_configuration,
    decimal_from,
)
from quantforge.backtesting.costs import CommissionModel, FeeModel, SlippageModel
from quantforge.backtesting.errors import InvalidBacktestConfigurationError
from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)

ENGINE_VERSION = "1"
RESULT_SCHEMA_VERSION = "1"


def _cost_model_configuration(
    model: object,
    model_label: str,
    *,
    require_non_decreasing_buy_cost: bool = False,
) -> PrimitiveMapping:
    implementation_version = cast(
        object, getattr(model, "implementation_version", None)
    )
    if (
        not isinstance(implementation_version, str)
        or not implementation_version.strip()
    ):
        raise InvalidBacktestConfigurationError(
            f"{model_label} model implementation version must be explicit"
        )
    if (
        require_non_decreasing_buy_cost
        and getattr(model, "buy_cost_is_non_decreasing_by_quantity", None) is not True
    ):
        raise InvalidBacktestConfigurationError(
            f"{model_label} model must guarantee nondecreasing buy cost by quantity"
        )
    try:
        with arithmetic():
            configuration = getattr(model, "configuration")()
        if not isinstance(configuration, dict):
            raise TypeError("cost model configuration must be an object")
        primitive_configuration = cast(PrimitiveMapping, configuration)
        if (
            primitive_configuration.get("implementation_version")
            != implementation_version
        ):
            raise ValueError(
                "cost model implementation version must match its configuration"
            )
        if (
            require_non_decreasing_buy_cost
            and primitive_configuration.get("buy_cost_is_non_decreasing_by_quantity")
            is not True
        ):
            raise ValueError(
                "cost model configuration must record its nondecreasing buy-cost "
                "guarantee"
            )
        configuration_identity(primitive_configuration)
        return PrimitiveMappingSnapshot.capture(primitive_configuration).to_primitive()
    except (ArithmeticError, AttributeError, TypeError, ValueError) as error:
        raise InvalidBacktestConfigurationError(
            f"{model_label} model must expose stable versioned primitive configuration"
        ) from error


@dataclass(frozen=True, slots=True)
class NextSessionOpenExecution:
    """Execute a close-derived decision at the next calendar session's open."""

    timing: str = "next_session_after_close"
    price_field: str = "open"
    order_type: str = "market"

    def __post_init__(self) -> None:
        if (
            self.timing != "next_session_after_close"
            or self.price_field != "open"
            or self.order_type != "market"
        ):
            raise InvalidBacktestConfigurationError(
                "only next-session-open market execution is supported"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "timing": self.timing,
            "price_field": self.price_field,
            "order_type": self.order_type,
        }


@dataclass(frozen=True, slots=True)
class DiscreteTargetWeightSizing:
    """Size once on flat-to-long transition and fully exit on long-to-flat."""

    whole_shares_only: bool = True
    rebalance_existing_position: bool = False

    def __post_init__(self) -> None:
        if not self.whole_shares_only or self.rebalance_existing_position:
            raise InvalidBacktestConfigurationError(
                "only discrete whole-share transition sizing is supported"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "model": "discrete_target_weight",
            "whole_shares_only": self.whole_shares_only,
            "rebalance_existing_position": self.rebalance_existing_position,
        }


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """All assumptions required for a reportable deterministic MVP backtest."""

    initial_capital: Decimal
    commission: CommissionModel
    fees: FeeModel
    slippage: SlippageModel
    execution: NextSessionOpenExecution = NextSessionOpenExecution()
    sizing: DiscreteTargetWeightSizing = DiscreteTargetWeightSizing()
    annual_risk_free_rate: Decimal = Decimal(0)
    annualization_factor: int = 252
    long_only: bool = True
    forced_liquidation: bool = False
    engine_version: str = field(default=ENGINE_VERSION, init=False)
    result_schema_version: str = field(default=RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        try:
            initial_capital = decimal_from(self.initial_capital, "initial capital")
            risk_free_rate = decimal_from(
                self.annual_risk_free_rate, "annual risk-free rate"
            )
        except ValueError as error:
            raise InvalidBacktestConfigurationError(str(error)) from error
        if initial_capital <= 0:
            raise InvalidBacktestConfigurationError("initial capital must be positive")
        if risk_free_rate <= Decimal(-1):
            raise InvalidBacktestConfigurationError(
                "annual risk-free rate must be greater than -1"
            )
        annualization_factor = cast(object, self.annualization_factor)
        if (
            isinstance(annualization_factor, bool)
            or not isinstance(annualization_factor, int)
            or annualization_factor <= 0
        ):
            raise InvalidBacktestConfigurationError(
                "annualization factor must be a positive integer"
            )
        commission = cast(object, self.commission)
        fees = cast(object, self.fees)
        slippage = cast(object, self.slippage)
        execution = cast(object, self.execution)
        sizing = cast(object, self.sizing)
        long_only = cast(object, self.long_only)
        forced_liquidation = cast(object, self.forced_liquidation)
        if commission is None or not callable(getattr(commission, "calculate", None)):
            raise InvalidBacktestConfigurationError(
                "an explicit commission model is required"
            )
        if fees is None or not callable(getattr(fees, "calculate", None)):
            raise InvalidBacktestConfigurationError(
                "an explicit transaction-fee model is required"
            )
        if slippage is None or not callable(getattr(slippage, "apply", None)):
            raise InvalidBacktestConfigurationError(
                "an explicit slippage model is required"
            )
        if not isinstance(execution, NextSessionOpenExecution):
            raise InvalidBacktestConfigurationError(
                "unsupported execution configuration"
            )
        if not isinstance(sizing, DiscreteTargetWeightSizing):
            raise InvalidBacktestConfigurationError("unsupported sizing configuration")
        if not isinstance(long_only, bool) or not long_only:
            raise InvalidBacktestConfigurationError(
                "only long-only behavior is supported"
            )
        if not isinstance(forced_liquidation, bool) or forced_liquidation:
            raise InvalidBacktestConfigurationError(
                "forced liquidation is not supported in the MVP"
            )
        object.__setattr__(self, "initial_capital", initial_capital)
        object.__setattr__(self, "annual_risk_free_rate", risk_free_rate)
        _cost_model_configuration(
            self.commission,
            "commission",
            require_non_decreasing_buy_cost=True,
        )
        _cost_model_configuration(
            self.fees,
            "transaction-fee",
            require_non_decreasing_buy_cost=True,
        )
        _cost_model_configuration(self.slippage, "slippage")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "initial_capital": decimal_to_primitive(self.initial_capital),
            "execution": self.execution.to_primitive(),
            "commission": _cost_model_configuration(
                self.commission,
                "commission",
                require_non_decreasing_buy_cost=True,
            ),
            "fees": _cost_model_configuration(
                self.fees,
                "transaction-fee",
                require_non_decreasing_buy_cost=True,
            ),
            "slippage": _cost_model_configuration(self.slippage, "slippage"),
            "sizing": self.sizing.to_primitive(),
            "annual_risk_free_rate": decimal_to_primitive(self.annual_risk_free_rate),
            "annualization_factor": self.annualization_factor,
            "long_only": self.long_only,
            "forced_liquidation": self.forced_liquidation,
            "engine_version": self.engine_version,
            "result_schema_version": self.result_schema_version,
            "arithmetic": arithmetic_configuration(),
        }
