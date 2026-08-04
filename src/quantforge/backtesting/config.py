"""Immutable typed backtest configuration."""

from dataclasses import dataclass
from decimal import Decimal
from typing import cast

from quantforge.backtesting._arithmetic import arithmetic_configuration, decimal_from
from quantforge.backtesting.costs import CommissionModel, SlippageModel
from quantforge.backtesting.errors import InvalidBacktestConfigurationError
from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)

ENGINE_VERSION = "1"
RESULT_SCHEMA_VERSION = "1"


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
    slippage: SlippageModel
    execution: NextSessionOpenExecution = NextSessionOpenExecution()
    sizing: DiscreteTargetWeightSizing = DiscreteTargetWeightSizing()
    annual_risk_free_rate: Decimal = Decimal(0)
    annualization_factor: int = 252
    long_only: bool = True
    forced_liquidation: bool = False
    engine_version: str = ENGINE_VERSION
    result_schema_version: str = RESULT_SCHEMA_VERSION

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
        slippage = cast(object, self.slippage)
        execution = cast(object, self.execution)
        sizing = cast(object, self.sizing)
        long_only = cast(object, self.long_only)
        forced_liquidation = cast(object, self.forced_liquidation)
        if commission is None or not callable(getattr(commission, "calculate", None)):
            raise InvalidBacktestConfigurationError(
                "an explicit commission model is required"
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
        if not self.engine_version or not self.result_schema_version:
            raise InvalidBacktestConfigurationError("schema versions must not be empty")
        object.__setattr__(self, "initial_capital", initial_capital)
        object.__setattr__(self, "annual_risk_free_rate", risk_free_rate)
        try:
            configuration_identity(
                {
                    "commission": self.commission.configuration(),
                    "slippage": self.slippage.configuration(),
                }
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise InvalidBacktestConfigurationError(
                "cost models must expose stable primitive configuration"
            ) from error

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "initial_capital": decimal_to_primitive(self.initial_capital),
            "execution": self.execution.to_primitive(),
            "commission": self.commission.configuration(),
            "slippage": self.slippage.configuration(),
            "sizing": self.sizing.to_primitive(),
            "annual_risk_free_rate": decimal_to_primitive(self.annual_risk_free_rate),
            "annualization_factor": self.annualization_factor,
            "long_only": self.long_only,
            "forced_liquidation": self.forced_liquidation,
            "engine_version": self.engine_version,
            "result_schema_version": self.result_schema_version,
            "arithmetic": arithmetic_configuration(),
        }
