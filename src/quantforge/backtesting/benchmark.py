"""Comparable deterministic full-period buy-and-hold benchmark."""

from decimal import Decimal

from quantforge.backtesting._arithmetic import arithmetic
from quantforge.backtesting._execution_costs import (
    affordable_quantity,
    commission_amount,
    fee_amount,
    slipped_price,
)
from quantforge.backtesting.config import BacktestConfig
from quantforge.backtesting.costs import OrderSide
from quantforge.backtesting.errors import ExecutionError
from quantforge.backtesting.metrics import calculate_performance
from quantforge.backtesting.models import (
    BenchmarkResult,
    DailyPortfolioRecord,
    FillRecord,
    OrderRecord,
    OrderStatus,
    TradeRecord,
)
from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.models import MarketDataset


def run_buy_and_hold_benchmark(
    dataset: MarketDataset,
    config: BacktestConfig,
    run_id: str,
    backtest_configuration: PrimitiveMapping,
) -> BenchmarkResult:
    """Buy whole shares at the first open and hold through the final close."""
    configuration: PrimitiveMapping = {
        "model": "buy_and_hold",
        "implementation_version": "2",
        "start": "first_dataset_session_open",
        "return_series_start": "initial_capital_to_first_session_close",
        "forced_liquidation": False,
        "initial_capital": decimal_to_primitive(config.initial_capital),
        "commission": backtest_configuration["commission"],
        "fees": backtest_configuration["fees"],
        "slippage": backtest_configuration["slippage"],
    }
    benchmark_id = configuration_identity(
        {
            "run_id": run_id,
            "record_type": "benchmark",
            "configuration": configuration,
        }
    )
    signal_id = configuration_identity(
        {"benchmark_id": benchmark_id, "record_type": "signal"}
    )
    order_id = configuration_identity(
        {"benchmark_id": benchmark_id, "record_type": "order"}
    )
    first_bar = dataset.bars[0]
    fill_price = slipped_price(
        config.slippage,
        first_bar.open,
        OrderSide.BUY,
        context="benchmark",
    )
    quantity = affordable_quantity(
        config.initial_capital,
        fill_price,
        config.commission,
        config.fees,
        context="benchmark",
    )
    fill: FillRecord | None = None
    if quantity == 0:
        order = OrderRecord(
            order_id=order_id,
            run_id=run_id,
            symbol=first_bar.symbol,
            side=OrderSide.BUY,
            requested_quantity=0,
            originating_signal_id=signal_id,
            signal_session=first_bar.session_date,
            earliest_permitted_execution_session=first_bar.session_date,
            decision_session=first_bar.session_date,
            target_position="long",
            target_weight=Decimal(1),
            strategy_id="buy_and_hold_benchmark",
            strategy_configuration_id=benchmark_id,
            status=OrderStatus.REJECTED,
            reason="insufficient_cash_for_one_share",
        )
        cash = config.initial_capital
        shares = 0
    else:
        commission = commission_amount(
            config.commission,
            quantity,
            fill_price,
            context="benchmark",
        )
        fees = fee_amount(
            config.fees,
            OrderSide.BUY,
            quantity,
            fill_price,
            context="benchmark",
        )
        with arithmetic():
            gross_notional = Decimal(quantity) * fill_price
            cash_effect = -(gross_notional + commission + fees)
            cash = config.initial_capital + cash_effect
            slippage_per_share = fill_price - first_bar.open
            effective_bps = abs(slippage_per_share) / first_bar.open * Decimal(10_000)
        if cash < 0:
            raise ExecutionError("benchmark purchase would make cash negative")
        shares = quantity
        fill_id = configuration_identity(
            {"benchmark_id": benchmark_id, "record_type": "fill"}
        )
        fill = FillRecord(
            fill_id=fill_id,
            order_id=order_id,
            originating_signal_id=signal_id,
            symbol=first_bar.symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            execution_session=first_bar.session_date,
            reference_price=first_bar.open,
            fill_price=fill_price,
            slippage_per_share=slippage_per_share,
            slippage_basis_points=effective_bps,
            gross_notional=gross_notional,
            commission=commission,
            fees=fees,
            net_cash_effect=cash_effect,
            strategy_id="buy_and_hold_benchmark",
            strategy_configuration_id=benchmark_id,
        )
        order = OrderRecord(
            order_id=order_id,
            run_id=run_id,
            symbol=first_bar.symbol,
            side=OrderSide.BUY,
            requested_quantity=quantity,
            originating_signal_id=signal_id,
            signal_session=first_bar.session_date,
            earliest_permitted_execution_session=first_bar.session_date,
            decision_session=first_bar.session_date,
            target_position="long",
            target_weight=Decimal(1),
            strategy_id="buy_and_hold_benchmark",
            strategy_configuration_id=benchmark_id,
            status=OrderStatus.FILLED,
            reason=None,
        )

    daily: list[DailyPortfolioRecord] = []
    peak = config.initial_capital
    previous_equity = config.initial_capital
    for index, bar in enumerate(dataset.bars):
        with arithmetic():
            market_value = Decimal(shares) * bar.close
            equity = cash + market_value
            if previous_equity == 0:
                daily_return = None
            else:
                daily_return = equity / previous_equity - Decimal(1)
            peak = max(peak, equity)
            drawdown = equity / peak - Decimal(1)
            exposure_weight = Decimal(0) if shares == 0 else market_value / equity
        daily.append(
            DailyPortfolioRecord(
                session=bar.session_date,
                cash=cash,
                shares=shares,
                closing_mark_price=bar.close,
                market_value=market_value,
                total_equity=equity,
                daily_return=daily_return,
                running_equity_peak=peak,
                drawdown=drawdown,
                exposed=shares > 0,
                exposure_weight=exposure_weight,
                order_ids=(order_id,) if index == 0 else (),
                fill_ids=(fill.fill_id,) if index == 0 and fill is not None else (),
            )
        )
        previous_equity = equity

    open_trades: tuple[TradeRecord, ...] = ()
    if fill is not None:
        trade_id = configuration_identity(
            {"benchmark_id": benchmark_id, "entry_fill_id": fill.fill_id}
        )
        open_trades = (
            TradeRecord(
                trade_id=trade_id,
                symbol=fill.symbol,
                entry_signal_id=signal_id,
                entry_order_id=order_id,
                entry_fill_id=fill.fill_id,
                entry_session=fill.execution_session,
                entry_price=fill.fill_price,
                entry_quantity=fill.quantity,
                entry_commission=fill.commission,
                entry_fees=fill.fees,
                exit_signal_id=None,
                exit_order_id=None,
                exit_fill_id=None,
                exit_session=None,
                exit_price=None,
                exit_commission=None,
                exit_fees=None,
                gross_profit_loss=None,
                net_profit_loss=None,
                return_percentage=None,
                holding_period_sessions=None,
                strategy_id=fill.strategy_id,
                strategy_implementation_version="2",
                strategy_configuration_id=fill.strategy_configuration_id,
                is_open=True,
            ),
        )
    performance = calculate_performance(
        tuple(daily),
        (),
        open_trades,
        initial_capital=config.initial_capital,
        annual_risk_free_rate=config.annual_risk_free_rate,
        annualization_factor=config.annualization_factor,
        include_first_daily_return=True,
    )
    return BenchmarkResult(
        benchmark_id=benchmark_id,
        configuration_snapshot=PrimitiveMappingSnapshot.capture(configuration),
        order=order,
        fill=fill,
        daily_equity=tuple(daily),
        performance=performance,
    )
