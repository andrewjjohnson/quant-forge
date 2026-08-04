"""Generic deterministic chronological single-symbol backtest runner."""

from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from typing import cast

from quantforge.backtesting._arithmetic import arithmetic
from quantforge.backtesting._execution_costs import (
    affordable_quantity,
    commission_amount,
    fee_amount,
    slipped_price,
)
from quantforge.backtesting.benchmark import run_buy_and_hold_benchmark
from quantforge.backtesting.config import BacktestConfig
from quantforge.backtesting.costs import OrderSide
from quantforge.backtesting.errors import (
    ExecutionError,
    InvalidMarketDataError,
    InvalidSignalError,
    PortfolioAccountingError,
)
from quantforge.backtesting.metrics import calculate_performance
from quantforge.backtesting.models import (
    BacktestResult,
    DailyPortfolioRecord,
    FillRecord,
    MarketDataMetadata,
    OrderRecord,
    OrderStatus,
    PositionRecord,
    SignalRecord,
    TradeRecord,
)
from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.exceptions import ValidationError as MarketDataValidationError
from quantforge.data.models import (
    SCHEMA_VERSION as MARKET_DATA_SCHEMA_VERSION,
)
from quantforge.data.models import (
    AdjustmentMode,
    DailyBar,
    MarketDataset,
)
from quantforge.data.validate import validate_market_dataset
from quantforge.strategies import (
    ExecutionTiming,
    MarketDataReference,
    PositionIntent,
    Strategy,
    StrategyDecision,
    run_strategy,
)

LIMITATIONS = (
    "single instrument per backtest",
    "long-only with no leverage, margin, borrowing, or short selling",
    "market orders with full whole-share fills only",
    "next-session open execution only",
    "no volume, liquidity, partial-fill, or intraday sequencing model",
    "no split- or dividend-bearing ranges or modeled corporate-action cash flows",
    "no forced end-of-data liquidation",
    "target weights are applied only on flat-to-long transitions",
)


@dataclass(frozen=True, slots=True)
class _OpenTrade:
    signal: SignalRecord
    order: OrderRecord
    fill: FillRecord
    total_entry_cost: Decimal


def _stable_id(values: PrimitiveMapping) -> str:
    try:
        return configuration_identity(values)
    except (TypeError, ValueError) as error:
        raise InvalidSignalError(
            "run inputs must have stable JSON-compatible serialization"
        ) from error


def _bars_fingerprint(bars: tuple[DailyBar, ...]) -> str:
    """Fingerprint the validated bar values independently of caller metadata."""
    serialized_bars: list[Primitive] = [
        {
            "symbol": bar.symbol,
            "session_date": bar.session_date.isoformat(),
            "open": decimal_to_primitive(bar.open),
            "high": decimal_to_primitive(bar.high),
            "low": decimal_to_primitive(bar.low),
            "close": decimal_to_primitive(bar.close),
            "volume": decimal_to_primitive(bar.volume),
        }
        for bar in bars
    ]
    return _stable_id(
        {
            "component": "quantforge_market_bars",
            "fingerprint_version": "1",
            "bars": serialized_bars,
        }
    )


def _validate_dataset(dataset: MarketDataset) -> None:
    dataset_value = cast(object, dataset)
    if not isinstance(dataset_value, MarketDataset) or not dataset_value.bars:
        raise InvalidMarketDataError("a nonempty QF-3 MarketDataset is required")
    metadata = dataset_value.metadata
    if not metadata.dataset_id or not metadata.schema_version:
        raise InvalidMarketDataError("dataset identity and schema version are required")
    if metadata.schema_version != MARKET_DATA_SCHEMA_VERSION:
        raise InvalidMarketDataError(
            f"market data schema {MARKET_DATA_SCHEMA_VERSION} is required for "
            "verified corporate-action provenance"
        )
    try:
        recomputed_missing_sessions = validate_market_dataset(dataset_value)
    except MarketDataValidationError as error:
        raise InvalidMarketDataError(str(error)) from error
    if metadata.adjustment_mode is not AdjustmentMode.UNADJUSTED:
        raise InvalidMarketDataError(
            "adjusted market data requires point-in-time corporate-action data "
            "that QF-3/QF-5 do not provide"
        )
    internal_missing_sessions = tuple(
        missing_session
        for missing_session in recomputed_missing_sessions
        if metadata.actual_first_session
        <= missing_session
        <= metadata.actual_last_session
    )
    if internal_missing_sessions:
        rendered = ", ".join(
            missing_session.isoformat() for missing_session in internal_missing_sessions
        )
        raise InvalidMarketDataError(
            "dataset has missing expected sessions within its observed range: "
            f"{rendered}"
        )
    observed_split_sessions = tuple(
        split_session
        for split_session in metadata.split_sessions
        if metadata.actual_first_session
        <= split_session
        <= metadata.actual_last_session
    )
    if observed_split_sessions:
        rendered = ", ".join(
            split_session.isoformat() for split_session in observed_split_sessions
        )
        raise InvalidMarketDataError(
            "unadjusted market data contains stock splits within its observed range: "
            f"{rendered}"
        )
    observed_dividend_sessions = tuple(
        dividend_session
        for dividend_session in metadata.dividend_sessions
        if metadata.actual_first_session
        <= dividend_session
        <= metadata.actual_last_session
    )
    if observed_dividend_sessions:
        rendered = ", ".join(
            dividend_session.isoformat()
            for dividend_session in observed_dividend_sessions
        )
        raise InvalidMarketDataError(
            "unadjusted market data contains cash dividends within its observed "
            f"range: {rendered}"
        )


def _order_id(run_id: str, signal_id: str) -> str:
    return _stable_id(
        {"run_id": run_id, "record_type": "order", "signal_id": signal_id}
    )


def _base_order(
    run_id: str,
    signal: SignalRecord,
    *,
    quantity: int | None,
    status: OrderStatus,
    reason: str | None,
) -> OrderRecord:
    decision = signal.decision
    side = (
        OrderSide.BUY
        if decision.target_position is PositionIntent.LONG
        else OrderSide.SELL
    )
    return OrderRecord(
        order_id=_order_id(run_id, signal.signal_id),
        run_id=run_id,
        symbol=decision.canonical_symbol,
        side=side,
        requested_quantity=quantity,
        originating_signal_id=signal.signal_id,
        signal_session=decision.signal_session,
        earliest_permitted_execution_session=decision.earliest_executable_session,
        decision_session=decision.signal_session,
        target_position=decision.target_position.value,
        target_weight=decision.target_weight,
        strategy_id=decision.strategy_id,
        strategy_configuration_id=decision.strategy_configuration_id,
        status=status,
        reason=reason,
    )


def _fill(
    run_id: str,
    order: OrderRecord,
    execution_session: date,
    reference_price: Decimal,
    fill_price: Decimal,
    commission: Decimal,
    fees: Decimal,
) -> FillRecord:
    quantity = order.requested_quantity
    if quantity is None or quantity <= 0:
        raise ExecutionError("a fill requires a positive integer quantity")
    with arithmetic():
        gross_notional = Decimal(quantity) * fill_price
        slippage_per_share = fill_price - reference_price
        effective_bps = abs(slippage_per_share) / reference_price * Decimal(10_000)
        net_cash_effect = (
            -(gross_notional + commission + fees)
            if order.side is OrderSide.BUY
            else gross_notional - commission - fees
        )
    fill_id = _stable_id(
        {
            "run_id": run_id,
            "record_type": "fill",
            "order_id": order.order_id,
        }
    )
    return FillRecord(
        fill_id=fill_id,
        order_id=order.order_id,
        originating_signal_id=order.originating_signal_id,
        symbol=order.symbol,
        side=order.side,
        quantity=quantity,
        execution_session=execution_session,
        reference_price=reference_price,
        fill_price=fill_price,
        slippage_per_share=slippage_per_share,
        slippage_basis_points=effective_bps,
        gross_notional=gross_notional,
        commission=commission,
        fees=fees,
        net_cash_effect=net_cash_effect,
        strategy_id=order.strategy_id,
        strategy_configuration_id=order.strategy_configuration_id,
    )


def _signal_records(
    run_id: str, decisions: tuple[StrategyDecision, ...]
) -> tuple[SignalRecord, ...]:
    records: list[SignalRecord] = []
    for ordinal, decision in enumerate(decisions):
        if decision.execution_timing is not ExecutionTiming.NEXT_SESSION_AFTER_CLOSE:
            raise InvalidSignalError("unsupported strategy execution timing")
        signal_id = _stable_id(
            {
                "run_id": run_id,
                "record_type": "signal",
                "ordinal": ordinal,
                "decision": decision.to_primitive(),
            }
        )
        records.append(SignalRecord(signal_id, decision))
    if len({record.signal_id for record in records}) != len(records):
        raise InvalidSignalError("duplicate signal identifiers")
    return tuple(records)


def _open_trade_record(
    run_id: str, open_trade: _OpenTrade, strategy_implementation_version: str
) -> TradeRecord:
    fill = open_trade.fill
    trade_id = _stable_id(
        {"run_id": run_id, "record_type": "trade", "entry_fill_id": fill.fill_id}
    )
    return TradeRecord(
        trade_id=trade_id,
        symbol=fill.symbol,
        entry_signal_id=open_trade.signal.signal_id,
        entry_order_id=open_trade.order.order_id,
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
        strategy_implementation_version=strategy_implementation_version,
        strategy_configuration_id=fill.strategy_configuration_id,
        is_open=True,
    )


def run_backtest(
    dataset: MarketDataset,
    strategy: Strategy,
    config: BacktestConfig,
    *,
    initiated_at: datetime | None = None,
) -> BacktestResult:
    """Run QF-4 decisions through deterministic next-open execution and accounting."""
    _validate_dataset(dataset)
    bars_fingerprint = _bars_fingerprint(dataset.bars)
    market_data_reference = MarketDataReference.from_dataset(dataset)
    if initiated_at is not None and initiated_at.utcoffset() is None:
        raise InvalidSignalError("initiated_at must include a defined UTC offset")
    strategy_configuration_snapshot = PrimitiveMappingSnapshot.capture(
        strategy.configuration()
    )
    strategy_configuration = strategy_configuration_snapshot.to_primitive()
    backtest_configuration_snapshot = PrimitiveMappingSnapshot.capture(
        config.to_primitive()
    )
    backtest_configuration = backtest_configuration_snapshot.to_primitive()
    strategy_implementation_version = strategy.implementation_version
    strategy_configuration_id = strategy.configuration_id
    run_id = _stable_id(
        {
            "component": "quantforge_backtest",
            "engine_version": config.engine_version,
            "result_schema_version": config.result_schema_version,
            "market_data": {
                **market_data_reference.to_primitive(),
                "bars_fingerprint": bars_fingerprint,
            },
            "strategy": {
                "strategy_id": strategy.name,
                "strategy_implementation_version": strategy_implementation_version,
                "strategy_configuration_id": strategy_configuration_id,
                "configuration": strategy_configuration,
            },
            "backtest_configuration": backtest_configuration,
        }
    )
    strategy_output = run_strategy(strategy, dataset)
    if strategy_output.strategy_configuration_id != strategy_configuration_id:
        raise InvalidSignalError(
            "strategy configuration changed during backtest initialization"
        )
    signals = _signal_records(run_id, strategy_output.decisions)
    signals_by_execution: dict[date, list[SignalRecord]] = {}
    order_ids_by_signal_session: dict[date, list[str]] = {}
    for signal in signals:
        decision = signal.decision
        order_ids_by_signal_session.setdefault(decision.signal_session, []).append(
            _order_id(run_id, signal.signal_id)
        )
        if decision.earliest_executable_session is not None:
            signals_by_execution.setdefault(
                decision.earliest_executable_session, []
            ).append(signal)

    cash = config.initial_capital
    shares = 0
    total_entry_cost = Decimal(0)
    realized_profit_loss = Decimal(0)
    previous_equity = config.initial_capital
    running_peak = config.initial_capital
    open_trade: _OpenTrade | None = None
    order_outcomes: dict[str, OrderRecord] = {}
    fills: list[FillRecord] = []
    completed_trades: list[TradeRecord] = []
    positions: list[PositionRecord] = []
    daily_equity: list[DailyPortfolioRecord] = []
    session_index = {bar.session_date: index for index, bar in enumerate(dataset.bars)}

    for index, bar in enumerate(dataset.bars):
        session_fill_ids: list[str] = []
        for signal in signals_by_execution.get(bar.session_date, []):
            decision = signal.decision
            if decision.signal_session >= bar.session_date:
                raise InvalidSignalError(
                    "a signal cannot execute during or before its decision session"
                )
            if decision.target_position is PositionIntent.LONG:
                if shares != 0:
                    order_outcomes[signal.signal_id] = _base_order(
                        run_id,
                        signal,
                        quantity=0,
                        status=OrderStatus.REJECTED,
                        reason="target_already_satisfied_no_rebalance",
                    )
                    continue
                fill_price = slipped_price(config.slippage, bar.open, OrderSide.BUY)
                with arithmetic():
                    cash_budget = min(cash, cash * decision.target_weight)
                quantity = affordable_quantity(
                    cash_budget,
                    fill_price,
                    config.commission,
                    config.fees,
                )
                if quantity == 0:
                    order_outcomes[signal.signal_id] = _base_order(
                        run_id,
                        signal,
                        quantity=0,
                        status=OrderStatus.REJECTED,
                        reason="insufficient_cash_for_one_share",
                    )
                    continue
                commission = commission_amount(config.commission, quantity, fill_price)
                fees = fee_amount(config.fees, OrderSide.BUY, quantity, fill_price)
                order = _base_order(
                    run_id,
                    signal,
                    quantity=quantity,
                    status=OrderStatus.FILLED,
                    reason=None,
                )
                fill = _fill(
                    run_id,
                    order,
                    bar.session_date,
                    bar.open,
                    fill_price,
                    commission,
                    fees,
                )
                with arithmetic():
                    next_cash = cash + fill.net_cash_effect
                    total_entry_cost = fill.gross_notional + fill.commission + fill.fees
                if next_cash < 0:
                    raise PortfolioAccountingError("purchase would make cash negative")
                cash = next_cash
                shares = quantity
                open_trade = _OpenTrade(signal, order, fill, total_entry_cost)
            else:
                if shares == 0 or open_trade is None:
                    order_outcomes[signal.signal_id] = _base_order(
                        run_id,
                        signal,
                        quantity=0,
                        status=OrderStatus.REJECTED,
                        reason="target_already_flat",
                    )
                    continue
                quantity = shares
                fill_price = slipped_price(config.slippage, bar.open, OrderSide.SELL)
                commission = commission_amount(config.commission, quantity, fill_price)
                fees = fee_amount(config.fees, OrderSide.SELL, quantity, fill_price)
                order = _base_order(
                    run_id,
                    signal,
                    quantity=quantity,
                    status=OrderStatus.FILLED,
                    reason=None,
                )
                fill = _fill(
                    run_id,
                    order,
                    bar.session_date,
                    bar.open,
                    fill_price,
                    commission,
                    fees,
                )
                with arithmetic():
                    next_cash = cash + fill.net_cash_effect
                    gross_profit_loss = (
                        fill.fill_price - open_trade.fill.fill_price
                    ) * Decimal(quantity)
                    net_profit_loss = (
                        gross_profit_loss
                        - open_trade.fill.commission
                        - open_trade.fill.fees
                        - fill.commission
                        - fill.fees
                    )
                    trade_return = net_profit_loss / open_trade.total_entry_cost
                if next_cash < 0:
                    order_outcomes[signal.signal_id] = replace(
                        order,
                        status=OrderStatus.REJECTED,
                        reason="insufficient_cash_for_exit_costs",
                    )
                    continue
                cash = next_cash
                with arithmetic():
                    realized_profit_loss += net_profit_loss
                trade_id = _stable_id(
                    {
                        "run_id": run_id,
                        "record_type": "trade",
                        "entry_fill_id": open_trade.fill.fill_id,
                        "exit_fill_id": fill.fill_id,
                    }
                )
                completed_trades.append(
                    TradeRecord(
                        trade_id=trade_id,
                        symbol=bar.symbol,
                        entry_signal_id=open_trade.signal.signal_id,
                        entry_order_id=open_trade.order.order_id,
                        entry_fill_id=open_trade.fill.fill_id,
                        entry_session=open_trade.fill.execution_session,
                        entry_price=open_trade.fill.fill_price,
                        entry_quantity=open_trade.fill.quantity,
                        entry_commission=open_trade.fill.commission,
                        entry_fees=open_trade.fill.fees,
                        exit_signal_id=signal.signal_id,
                        exit_order_id=order.order_id,
                        exit_fill_id=fill.fill_id,
                        exit_session=fill.execution_session,
                        exit_price=fill.fill_price,
                        exit_commission=fill.commission,
                        exit_fees=fill.fees,
                        gross_profit_loss=gross_profit_loss,
                        net_profit_loss=net_profit_loss,
                        return_percentage=trade_return,
                        holding_period_sessions=(
                            session_index[fill.execution_session]
                            - session_index[open_trade.fill.execution_session]
                        ),
                        strategy_id=fill.strategy_id,
                        strategy_implementation_version=(
                            strategy_implementation_version
                        ),
                        strategy_configuration_id=fill.strategy_configuration_id,
                        is_open=False,
                    )
                )
                shares = 0
                total_entry_cost = Decimal(0)
                open_trade = None

            order_outcomes[signal.signal_id] = order
            fills.append(fill)
            session_fill_ids.append(fill.fill_id)

        with arithmetic():
            market_value = Decimal(shares) * bar.close
            equity = cash + market_value
            if equity != cash + market_value:
                raise PortfolioAccountingError("equity invariant failed")
            if index == 0:
                daily_return = Decimal(0)
            elif previous_equity == 0:
                daily_return = None
            else:
                daily_return = equity / previous_equity - Decimal(1)
            running_peak = max(running_peak, equity)
            drawdown = equity / running_peak - Decimal(1)
            exposure_weight = Decimal(0) if shares == 0 else market_value / equity
            unrealized_profit_loss = (
                Decimal(0) if shares == 0 else market_value - total_entry_cost
            )
            average_entry_cost = (
                None if shares == 0 else total_entry_cost / Decimal(shares)
            )
        if shares < 0 or cash < 0:
            raise PortfolioAccountingError("cash and shares must remain nonnegative")
        positions.append(
            PositionRecord(
                session=bar.session_date,
                symbol=bar.symbol,
                shares=shares,
                average_entry_cost=average_entry_cost,
                cost_basis=total_entry_cost,
                market_value=market_value,
                realized_profit_loss=realized_profit_loss,
                unrealized_profit_loss=unrealized_profit_loss,
            )
        )
        daily_equity.append(
            DailyPortfolioRecord(
                session=bar.session_date,
                cash=cash,
                shares=shares,
                closing_mark_price=bar.close,
                market_value=market_value,
                total_equity=equity,
                daily_return=daily_return,
                running_equity_peak=running_peak,
                drawdown=drawdown,
                exposed=shares > 0,
                exposure_weight=exposure_weight,
                order_ids=tuple(order_ids_by_signal_session.get(bar.session_date, ())),
                fill_ids=tuple(session_fill_ids),
            )
        )
        previous_equity = equity

    final_session = dataset.bars[-1].session_date
    available_sessions = set(session_index)
    for signal in signals:
        if signal.signal_id in order_outcomes:
            continue
        eligible = signal.decision.earliest_executable_session
        if eligible is None:
            status = OrderStatus.REJECTED
            reason = "execution_session_unresolved"
        elif eligible > final_session:
            status = OrderStatus.UNEXECUTED_END_OF_DATA
            reason = "no_later_execution_bar"
        elif eligible not in available_sessions:
            status = OrderStatus.REJECTED
            reason = "missing_execution_bar"
        else:
            raise InvalidSignalError("eligible signal was not processed")
        order_outcomes[signal.signal_id] = _base_order(
            run_id,
            signal,
            quantity=None,
            status=status,
            reason=reason,
        )

    orders = tuple(order_outcomes[signal.signal_id] for signal in signals)
    if len({order.order_id for order in orders}) != len(orders):
        raise InvalidSignalError("duplicate order identifiers")
    if len({fill.fill_id for fill in fills}) != len(fills):
        raise InvalidSignalError("duplicate fill identifiers")
    open_trades = (
        ()
        if open_trade is None
        else (_open_trade_record(run_id, open_trade, strategy_implementation_version),)
    )
    benchmark = run_buy_and_hold_benchmark(
        dataset,
        config,
        run_id,
        backtest_configuration,
    )
    performance = calculate_performance(
        tuple(daily_equity),
        tuple(completed_trades),
        open_trades,
        initial_capital=config.initial_capital,
        annual_risk_free_rate=config.annual_risk_free_rate,
        annualization_factor=config.annualization_factor,
        benchmark_total_return=benchmark.performance.total_return,
    )
    return BacktestResult(
        run_id=run_id,
        engine_version=config.engine_version,
        result_schema_version=config.result_schema_version,
        market_data=MarketDataMetadata.from_qf3(
            dataset.metadata, bars_fingerprint=bars_fingerprint
        ),
        strategy_id=strategy.name,
        strategy_implementation_version=strategy_implementation_version,
        strategy_configuration_id=strategy_configuration_id,
        strategy_configuration_snapshot=strategy_configuration_snapshot,
        strategy_warm_up_observations=strategy.warm_up_observations,
        backtest_configuration_snapshot=backtest_configuration_snapshot,
        signals=signals,
        orders=orders,
        fills=tuple(fills),
        positions=tuple(positions),
        completed_trades=tuple(completed_trades),
        open_trades=open_trades,
        daily_equity=tuple(daily_equity),
        performance=performance,
        benchmark=benchmark,
        warnings=(),
        limitations=LIMITATIONS,
        initiated_at=initiated_at,
    )
