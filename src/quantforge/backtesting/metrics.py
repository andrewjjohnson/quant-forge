"""Pure deterministic performance metric calculations."""

from decimal import Decimal

from quantforge.backtesting._arithmetic import (
    arithmetic,
    decimal_sqrt,
    fractional_power,
)
from quantforge.backtesting.models import (
    DailyPortfolioRecord,
    PerformanceSummary,
    TradeRecord,
)

CALENDAR_DAYS_PER_YEAR = Decimal("365.2425")


def calculate_performance(
    daily_equity: tuple[DailyPortfolioRecord, ...],
    completed_trades: tuple[TradeRecord, ...],
    open_trades: tuple[TradeRecord, ...],
    *,
    initial_capital: Decimal,
    annual_risk_free_rate: Decimal,
    annualization_factor: int,
    benchmark_total_return: Decimal | None = None,
    include_first_daily_return: bool = False,
    total_dividend_income: Decimal = Decimal(0),
    dividend_event_count: int = 0,
    split_event_count: int = 0,
) -> PerformanceSummary:
    """Calculate typed arithmetic-return metrics without NaN or infinity.

    Most daily series use the first record as their initial close and therefore
    start returns at index one. A first-open benchmark can opt into its invested
    inception-to-first-close return explicitly.
    """
    if not daily_equity:
        raise ValueError("daily equity records are required")
    ending_equity = daily_equity[-1].total_equity
    with arithmetic():
        total_return = ending_equity / initial_capital - Decimal(1)
        exposure = Decimal(sum(record.exposed for record in daily_equity)) / Decimal(
            len(daily_equity)
        )

    elapsed_days = (daily_equity[-1].session - daily_equity[0].session).days
    cagr: Decimal | None = None
    if elapsed_days > 0 and ending_equity > 0:
        with arithmetic():
            ratio = ending_equity / initial_capital
            cagr = fractional_power(
                ratio, CALENDAR_DAYS_PER_YEAR / Decimal(elapsed_days)
            ) - Decimal(1)

    return_start = 0 if include_first_daily_return else 1
    returns = tuple(
        record.daily_return
        for record in daily_equity[return_start:]
        if record.daily_return is not None
    )
    annualized_volatility: Decimal | None = None
    sharpe_ratio: Decimal | None = None
    sortino_ratio: Decimal | None = None
    if returns:
        with arithmetic():
            daily_risk_free = fractional_power(
                Decimal(1) + annual_risk_free_rate,
                Decimal(1) / Decimal(annualization_factor),
            ) - Decimal(1)
            excess_returns = tuple(item - daily_risk_free for item in returns)
            mean_excess = sum(excess_returns, start=Decimal(0)) / Decimal(
                len(excess_returns)
            )
            annualization_root = decimal_sqrt(Decimal(annualization_factor))
            if len(returns) >= 2:
                mean_return = sum(returns, start=Decimal(0)) / Decimal(len(returns))
                variance = sum(
                    ((item - mean_return) ** 2 for item in returns),
                    start=Decimal(0),
                ) / Decimal(len(returns) - 1)
                daily_volatility = decimal_sqrt(variance)
                annualized_volatility = daily_volatility * annualization_root

                mean_excess_for_sample = sum(
                    excess_returns, start=Decimal(0)
                ) / Decimal(len(excess_returns))
                excess_variance = sum(
                    ((item - mean_excess_for_sample) ** 2 for item in excess_returns),
                    start=Decimal(0),
                ) / Decimal(len(excess_returns) - 1)
                excess_deviation = decimal_sqrt(excess_variance)
                if excess_deviation > 0:
                    sharpe_ratio = mean_excess / excess_deviation * annualization_root

            squared_downside = tuple(
                min(item, Decimal(0)) ** 2 for item in excess_returns
            )
            downside_deviation = decimal_sqrt(
                sum(squared_downside, start=Decimal(0)) / Decimal(len(squared_downside))
            )
            if downside_deviation > 0:
                sortino_ratio = mean_excess / downside_deviation * annualization_root

    net_result_values: list[Decimal] = []
    trade_return_values: list[Decimal] = []
    for trade in completed_trades:
        net_result = (
            trade.total_economic_profit_loss
            if trade.total_economic_profit_loss is not None
            else trade.net_profit_loss
        )
        trade_return = (
            trade.total_economic_return
            if trade.total_economic_return is not None
            else trade.return_percentage
        )
        if net_result is not None:
            net_result_values.append(net_result)
        if trade_return is not None:
            trade_return_values.append(trade_return)
    net_results = tuple(net_result_values)
    trade_returns = tuple(trade_return_values)
    winning_trades = sum(result > 0 for result in net_results)
    losing_trades = sum(result < 0 for result in net_results)
    with arithmetic():
        gross_profit = sum(
            (result for result in net_results if result > 0), start=Decimal(0)
        )
        gross_loss = sum(
            (result for result in net_results if result < 0), start=Decimal(0)
        )
        profit_factor = None if gross_loss == 0 else gross_profit / abs(gross_loss)
        win_rate = (
            None
            if not completed_trades
            else Decimal(winning_trades) / Decimal(len(completed_trades))
        )
        average_trade_return = (
            None
            if not trade_returns
            else sum(trade_returns, start=Decimal(0)) / Decimal(len(trade_returns))
        )

    return PerformanceSummary(
        starting_equity=initial_capital,
        ending_equity=ending_equity,
        total_return=total_return,
        cagr=cagr,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        maximum_drawdown=min(record.drawdown for record in daily_equity),
        profit_factor=profit_factor,
        exposure=exposure,
        trade_count=len(completed_trades),
        open_trade_count=len(open_trades),
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        average_trade_return=average_trade_return,
        benchmark_total_return=benchmark_total_return,
        annual_risk_free_rate=annual_risk_free_rate,
        annualization_factor=annualization_factor,
        total_dividend_income=total_dividend_income,
        dividend_event_count=dividend_event_count,
        split_event_count=split_event_count,
    )
