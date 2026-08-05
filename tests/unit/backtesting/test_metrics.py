from datetime import date
from decimal import Decimal

from quantforge.backtesting import DailyPortfolioRecord
from quantforge.backtesting.metrics import calculate_performance


def equity_record(
    session: date,
    equity: str,
    daily_return: str,
    peak: str,
    drawdown: str,
    *,
    exposed: bool,
) -> DailyPortfolioRecord:
    total_equity = Decimal(equity)
    shares = 1 if exposed else 0
    market_value = Decimal(shares)
    return DailyPortfolioRecord(
        session=session,
        cash=total_equity - market_value,
        shares=shares,
        closing_mark_price=Decimal(1),
        market_value=market_value,
        total_equity=total_equity,
        daily_return=Decimal(daily_return),
        running_equity_peak=Decimal(peak),
        drawdown=Decimal(drawdown),
        exposed=exposed,
        exposure_weight=(market_value / total_equity if exposed else Decimal(0)),
        order_ids=(),
        fill_ids=(),
    )


def test_hand_calculated_return_risk_drawdown_and_exposure_metrics() -> None:
    daily = (
        equity_record(date(2024, 1, 2), "100", "0", "100", "0", exposed=False),
        equity_record(date(2024, 1, 3), "110", "0.1", "110", "0", exposed=True),
        equity_record(date(2024, 1, 4), "99", "-0.1", "110", "-0.1", exposed=True),
    )

    summary = calculate_performance(
        daily,
        (),
        (),
        initial_capital=Decimal(100),
        annual_risk_free_rate=Decimal(0),
        annualization_factor=1,
    )

    assert summary.total_return == Decimal("-0.01")
    assert summary.cagr is not None
    assert str(summary.annualized_volatility).startswith(
        "0.1414213562373095048801688724"
    )
    assert summary.sharpe_ratio == Decimal(0)
    assert summary.sortino_ratio == Decimal(0)
    assert summary.maximum_drawdown == Decimal("-0.1")
    assert summary.exposure == Decimal("0.6666666666666666666666666666666667")
    assert summary.trade_count == 0
    assert summary.win_rate is None
    assert summary.profit_factor is None


def test_single_session_metrics_leave_calendar_and_dispersion_values_undefined() -> (
    None
):
    summary = calculate_performance(
        (equity_record(date(2024, 1, 2), "100", "0", "100", "0", exposed=False),),
        (),
        (),
        initial_capital=Decimal(100),
        annual_risk_free_rate=Decimal(0),
        annualization_factor=252,
    )

    assert summary.total_return == Decimal(0)
    assert summary.cagr is None
    assert summary.annualized_volatility is None
    assert summary.sharpe_ratio is None
    assert summary.sortino_ratio is None
