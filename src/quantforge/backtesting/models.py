"""Immutable orders, fills, portfolio records, trades, and complete results."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import cast

from quantforge.backtesting.costs import OrderSide
from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    decimal_to_primitive,
)
from quantforge.data.models import DatasetMetadata
from quantforge.strategies.models import StrategyDecision


def _decimal(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_primitive(value)


class OrderStatus(StrEnum):
    """Final deterministic status of an MVP order."""

    FILLED = "filled"
    REJECTED = "rejected"
    UNEXECUTED_END_OF_DATA = "unexecuted_end_of_data"


@dataclass(frozen=True, slots=True)
class MarketDataMetadata:
    """Complete immutable QF-3 provenance required by a backtest result."""

    dataset_id: str
    bars_fingerprint: str
    schema_version: str
    canonical_symbol: str
    provider_name: str
    provider_symbol: str
    retrieved_at: datetime
    provider_timezone: str | None
    requested_start: date
    requested_end: date
    actual_first_session: date
    actual_last_session: date
    calendar: str
    adjustment_mode: str
    raw_location: str
    normalized_location: str
    bar_count: int
    missing_sessions: tuple[date, ...]
    split_sessions: tuple[date, ...]
    dividend_sessions: tuple[date, ...]
    adapter_version: str

    @classmethod
    def from_qf3(
        cls, metadata: DatasetMetadata, *, bars_fingerprint: str
    ) -> "MarketDataMetadata":
        return cls(
            dataset_id=metadata.dataset_id,
            bars_fingerprint=bars_fingerprint,
            schema_version=metadata.schema_version,
            canonical_symbol=metadata.canonical_symbol,
            provider_name=metadata.provider_name,
            provider_symbol=metadata.provider_symbol,
            retrieved_at=metadata.retrieved_at,
            provider_timezone=metadata.provider_timezone,
            requested_start=metadata.requested_start,
            requested_end=metadata.requested_end,
            actual_first_session=metadata.actual_first_session,
            actual_last_session=metadata.actual_last_session,
            calendar=metadata.calendar,
            adjustment_mode=metadata.adjustment_mode.value,
            raw_location=metadata.raw_location,
            normalized_location=metadata.normalized_location,
            bar_count=metadata.bar_count,
            missing_sessions=metadata.missing_sessions,
            split_sessions=metadata.split_sessions,
            dividend_sessions=metadata.dividend_sessions,
            adapter_version=metadata.adapter_version,
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "dataset_id": self.dataset_id,
            "bars_fingerprint": self.bars_fingerprint,
            "schema_version": self.schema_version,
            "canonical_symbol": self.canonical_symbol,
            "provider_name": self.provider_name,
            "provider_symbol": self.provider_symbol,
            "retrieved_at": self.retrieved_at.isoformat(),
            "provider_timezone": self.provider_timezone,
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "actual_first_session": self.actual_first_session.isoformat(),
            "actual_last_session": self.actual_last_session.isoformat(),
            "calendar": self.calendar,
            "adjustment_mode": self.adjustment_mode,
            "raw_location": self.raw_location,
            "normalized_location": self.normalized_location,
            "bar_count": self.bar_count,
            "missing_sessions": [item.isoformat() for item in self.missing_sessions],
            "split_sessions": [item.isoformat() for item in self.split_sessions],
            "dividend_sessions": [item.isoformat() for item in self.dividend_sessions],
            "adapter_version": self.adapter_version,
        }


@dataclass(frozen=True, slots=True)
class SignalRecord:
    """A QF-4 decision with a stable run-local signal identifier."""

    signal_id: str
    decision: StrategyDecision

    def to_primitive(self) -> PrimitiveMapping:
        return {"signal_id": self.signal_id, **self.decision.to_primitive()}


@dataclass(frozen=True, slots=True)
class OrderRecord:
    """One immutable market-order outcome derived from a generic signal."""

    order_id: str
    run_id: str
    symbol: str
    side: OrderSide
    requested_quantity: int | None
    originating_signal_id: str
    signal_session: date
    earliest_permitted_execution_session: date | None
    decision_session: date
    target_position: str
    target_weight: Decimal
    strategy_id: str
    strategy_configuration_id: str
    status: OrderStatus
    reason: str | None
    order_type: str = "market"

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "order_id": self.order_id,
            "run_id": self.run_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "requested_quantity": self.requested_quantity,
            "originating_signal_id": self.originating_signal_id,
            "signal_session": self.signal_session.isoformat(),
            "earliest_permitted_execution_session": (
                None
                if self.earliest_permitted_execution_session is None
                else self.earliest_permitted_execution_session.isoformat()
            ),
            "decision_session": self.decision_session.isoformat(),
            "order_type": self.order_type,
            "target_position": self.target_position,
            "target_weight": decimal_to_primitive(self.target_weight),
            "strategy_id": self.strategy_id,
            "strategy_configuration_id": self.strategy_configuration_id,
            "status": self.status.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class FillRecord:
    """One full immutable fill with separated price impact, commission, and fees."""

    fill_id: str
    order_id: str
    originating_signal_id: str
    symbol: str
    side: OrderSide
    quantity: int
    execution_session: date
    reference_price: Decimal
    fill_price: Decimal
    slippage_per_share: Decimal
    slippage_basis_points: Decimal
    gross_notional: Decimal
    commission: Decimal
    fees: Decimal
    net_cash_effect: Decimal
    strategy_id: str
    strategy_configuration_id: str

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "originating_signal_id": self.originating_signal_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "execution_session": self.execution_session.isoformat(),
            "reference_price": decimal_to_primitive(self.reference_price),
            "fill_price": decimal_to_primitive(self.fill_price),
            "slippage_per_share": decimal_to_primitive(self.slippage_per_share),
            "slippage_basis_points": decimal_to_primitive(self.slippage_basis_points),
            "gross_notional": decimal_to_primitive(self.gross_notional),
            "commission": decimal_to_primitive(self.commission),
            "fees": decimal_to_primitive(self.fees),
            "net_cash_effect": decimal_to_primitive(self.net_cash_effect),
            "strategy_id": self.strategy_id,
            "strategy_configuration_id": self.strategy_configuration_id,
        }


@dataclass(frozen=True, slots=True)
class PositionRecord:
    """End-of-session single-symbol holding and cost-basis record."""

    session: date
    symbol: str
    shares: int
    average_entry_cost: Decimal | None
    cost_basis: Decimal
    market_value: Decimal
    realized_profit_loss: Decimal
    unrealized_profit_loss: Decimal

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "session": self.session.isoformat(),
            "symbol": self.symbol,
            "shares": self.shares,
            "average_entry_cost": _decimal(self.average_entry_cost),
            "cost_basis": decimal_to_primitive(self.cost_basis),
            "market_value": decimal_to_primitive(self.market_value),
            "realized_profit_loss": decimal_to_primitive(self.realized_profit_loss),
            "unrealized_profit_loss": decimal_to_primitive(self.unrealized_profit_loss),
        }


@dataclass(frozen=True, slots=True)
class DailyPortfolioRecord:
    """End-of-session cash, equity, return, drawdown, and exposure snapshot."""

    session: date
    cash: Decimal
    shares: int
    closing_mark_price: Decimal
    market_value: Decimal
    total_equity: Decimal
    daily_return: Decimal | None
    running_equity_peak: Decimal
    drawdown: Decimal
    exposed: bool
    exposure_weight: Decimal
    order_ids: tuple[str, ...]
    fill_ids: tuple[str, ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "session": self.session.isoformat(),
            "cash": decimal_to_primitive(self.cash),
            "shares": self.shares,
            "closing_mark_price": decimal_to_primitive(self.closing_mark_price),
            "market_value": decimal_to_primitive(self.market_value),
            "total_equity": decimal_to_primitive(self.total_equity),
            "daily_return": _decimal(self.daily_return),
            "running_equity_peak": decimal_to_primitive(self.running_equity_peak),
            "drawdown": decimal_to_primitive(self.drawdown),
            "exposed": self.exposed,
            "exposure_weight": decimal_to_primitive(self.exposure_weight),
            "order_ids": list(self.order_ids),
            "fill_ids": list(self.fill_ids),
        }


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """Completed round trip or explicitly open end-of-data position."""

    trade_id: str
    symbol: str
    entry_signal_id: str
    entry_order_id: str
    entry_fill_id: str
    entry_session: date
    entry_price: Decimal
    entry_quantity: int
    entry_commission: Decimal
    entry_fees: Decimal
    exit_signal_id: str | None
    exit_order_id: str | None
    exit_fill_id: str | None
    exit_session: date | None
    exit_price: Decimal | None
    exit_commission: Decimal | None
    exit_fees: Decimal | None
    gross_profit_loss: Decimal | None
    net_profit_loss: Decimal | None
    return_percentage: Decimal | None
    holding_period_sessions: int | None
    strategy_id: str
    strategy_implementation_version: str
    strategy_configuration_id: str
    is_open: bool

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "entry_signal_id": self.entry_signal_id,
            "entry_order_id": self.entry_order_id,
            "entry_fill_id": self.entry_fill_id,
            "entry_session": self.entry_session.isoformat(),
            "entry_price": decimal_to_primitive(self.entry_price),
            "entry_quantity": self.entry_quantity,
            "entry_commission": decimal_to_primitive(self.entry_commission),
            "entry_fees": decimal_to_primitive(self.entry_fees),
            "exit_signal_id": self.exit_signal_id,
            "exit_order_id": self.exit_order_id,
            "exit_fill_id": self.exit_fill_id,
            "exit_session": (
                None if self.exit_session is None else self.exit_session.isoformat()
            ),
            "exit_price": _decimal(self.exit_price),
            "exit_commission": _decimal(self.exit_commission),
            "exit_fees": _decimal(self.exit_fees),
            "gross_profit_loss": _decimal(self.gross_profit_loss),
            "net_profit_loss": _decimal(self.net_profit_loss),
            "return_percentage": _decimal(self.return_percentage),
            "holding_period_sessions": self.holding_period_sessions,
            "strategy_id": self.strategy_id,
            "strategy_implementation_version": self.strategy_implementation_version,
            "strategy_configuration_id": self.strategy_configuration_id,
            "is_open": self.is_open,
        }


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """Typed metrics; decimals are ratios and undefined values are ``None``."""

    starting_equity: Decimal
    ending_equity: Decimal
    total_return: Decimal
    cagr: Decimal | None
    annualized_volatility: Decimal | None
    sharpe_ratio: Decimal | None
    sortino_ratio: Decimal | None
    maximum_drawdown: Decimal
    profit_factor: Decimal | None
    exposure: Decimal
    trade_count: int
    open_trade_count: int
    win_rate: Decimal | None
    gross_profit: Decimal
    gross_loss: Decimal
    winning_trades: int
    losing_trades: int
    average_trade_return: Decimal | None
    benchmark_total_return: Decimal | None
    annual_risk_free_rate: Decimal
    annualization_factor: int
    volatility_standard_deviation: str = "sample"
    maximum_drawdown_convention: str = "negative_decimal"

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "starting_equity": decimal_to_primitive(self.starting_equity),
            "ending_equity": decimal_to_primitive(self.ending_equity),
            "total_return": decimal_to_primitive(self.total_return),
            "cagr": _decimal(self.cagr),
            "annualized_volatility": _decimal(self.annualized_volatility),
            "sharpe_ratio": _decimal(self.sharpe_ratio),
            "sortino_ratio": _decimal(self.sortino_ratio),
            "maximum_drawdown": decimal_to_primitive(self.maximum_drawdown),
            "profit_factor": _decimal(self.profit_factor),
            "exposure": decimal_to_primitive(self.exposure),
            "trade_count": self.trade_count,
            "open_trade_count": self.open_trade_count,
            "win_rate": _decimal(self.win_rate),
            "gross_profit": decimal_to_primitive(self.gross_profit),
            "gross_loss": decimal_to_primitive(self.gross_loss),
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "average_trade_return": _decimal(self.average_trade_return),
            "benchmark_total_return": _decimal(self.benchmark_total_return),
            "annual_risk_free_rate": decimal_to_primitive(self.annual_risk_free_rate),
            "annualization_factor": self.annualization_factor,
            "volatility_standard_deviation": self.volatility_standard_deviation,
            "maximum_drawdown_convention": self.maximum_drawdown_convention,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Comparable full-period buy-and-hold output with no fabricated exit."""

    benchmark_id: str
    configuration_snapshot: PrimitiveMappingSnapshot
    order: OrderRecord
    fill: FillRecord | None
    daily_equity: tuple[DailyPortfolioRecord, ...]
    performance: PerformanceSummary

    @property
    def configuration(self) -> PrimitiveMapping:
        """Return a detached representation of the frozen benchmark assumptions."""
        return self.configuration_snapshot.to_primitive()

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "benchmark_id": self.benchmark_id,
            "configuration": self.configuration,
            "order": self.order.to_primitive(),
            "fill": None if self.fill is None else self.fill.to_primitive(),
            "daily_equity": cast(
                list[Primitive], [row.to_primitive() for row in self.daily_equity]
            ),
            "performance": self.performance.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete deterministic, traceable, tabular-friendly QF-5 result."""

    run_id: str
    engine_version: str
    result_schema_version: str
    market_data: MarketDataMetadata
    strategy_id: str
    strategy_implementation_version: str
    strategy_configuration_id: str
    strategy_configuration_snapshot: PrimitiveMappingSnapshot
    strategy_warm_up_observations: int
    backtest_configuration_snapshot: PrimitiveMappingSnapshot
    signals: tuple[SignalRecord, ...]
    orders: tuple[OrderRecord, ...]
    fills: tuple[FillRecord, ...]
    positions: tuple[PositionRecord, ...]
    completed_trades: tuple[TradeRecord, ...]
    open_trades: tuple[TradeRecord, ...]
    daily_equity: tuple[DailyPortfolioRecord, ...]
    performance: PerformanceSummary
    benchmark: BenchmarkResult
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    initiated_at: datetime | None = None

    @property
    def strategy_configuration(self) -> PrimitiveMapping:
        """Return a detached representation of frozen strategy provenance."""
        return self.strategy_configuration_snapshot.to_primitive()

    @property
    def backtest_configuration(self) -> PrimitiveMapping:
        """Return a detached representation of frozen execution assumptions."""
        return self.backtest_configuration_snapshot.to_primitive()

    def manifest_primitive(self) -> PrimitiveMapping:
        return {
            "run_id": self.run_id,
            "engine_version": self.engine_version,
            "result_schema_version": self.result_schema_version,
            "initiated_at": (
                None if self.initiated_at is None else self.initiated_at.isoformat()
            ),
            "market_data": self.market_data.to_primitive(),
            "strategy": {
                "strategy_id": self.strategy_id,
                "strategy_implementation_version": (
                    self.strategy_implementation_version
                ),
                "strategy_configuration_id": self.strategy_configuration_id,
                "configuration": self.strategy_configuration,
                "warm_up_observations": self.strategy_warm_up_observations,
            },
            "backtest_configuration": self.backtest_configuration,
            "performance": self.performance.to_primitive(),
            "benchmark": {
                "benchmark_id": self.benchmark.benchmark_id,
                "configuration": self.benchmark.configuration,
                "order": self.benchmark.order.to_primitive(),
                "fill": (
                    None
                    if self.benchmark.fill is None
                    else self.benchmark.fill.to_primitive()
                ),
                "performance": self.benchmark.performance.to_primitive(),
            },
            "record_counts": {
                "signals": len(self.signals),
                "orders": len(self.orders),
                "fills": len(self.fills),
                "positions": len(self.positions),
                "completed_trades": len(self.completed_trades),
                "open_trades": len(self.open_trades),
                "daily_equity": len(self.daily_equity),
                "benchmark_daily_equity": len(self.benchmark.daily_equity),
            },
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "manifest": self.manifest_primitive(),
            "signals": cast(
                list[Primitive], [item.to_primitive() for item in self.signals]
            ),
            "orders": cast(
                list[Primitive], [item.to_primitive() for item in self.orders]
            ),
            "fills": cast(
                list[Primitive], [item.to_primitive() for item in self.fills]
            ),
            "positions": cast(
                list[Primitive], [item.to_primitive() for item in self.positions]
            ),
            "completed_trades": cast(
                list[Primitive],
                [item.to_primitive() for item in self.completed_trades],
            ),
            "open_trades": cast(
                list[Primitive], [item.to_primitive() for item in self.open_trades]
            ),
            "daily_equity": cast(
                list[Primitive], [item.to_primitive() for item in self.daily_equity]
            ),
            "benchmark_daily_equity": cast(
                list[Primitive],
                [item.to_primitive() for item in self.benchmark.daily_equity],
            ),
        }
