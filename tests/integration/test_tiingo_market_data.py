import os
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from quantforge.backtesting import (
    BacktestConfig,
    BasisPointSlippage,
    DividendPolicy,
    ExplicitZeroFees,
    FixedCommission,
    export_backtest_result,
    run_backtest,
)
from quantforge.data import AdjustmentMode, MarketDataCache, MarketDataService
from quantforge.data.providers import TiingoProvider
from quantforge.strategies import (
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
)


@pytest.mark.integration
def test_live_tiingo_spy_dividend_cache_backtest_and_export(tmp_path: Path) -> None:
    if os.environ.get("QUANTFORGE_RUN_LIVE_TIINGO") != "1":
        pytest.skip("set QUANTFORGE_RUN_LIVE_TIINGO=1 to opt in")
    api_key = os.environ.get("TIINGO_API_KEY")
    if not api_key:
        pytest.skip("TIINGO_API_KEY is not configured")
    cache = MarketDataCache(tmp_path / "cache")
    service = MarketDataService(TiingoProvider(api_key), cache)
    dataset = service.get_daily_bars(
        "SPY",
        date(2024, 6, 17),
        date(2024, 6, 28),
        AdjustmentMode.UNADJUSTED,
    )

    assert dataset.metadata.dividend_count >= 1
    assert cache.load(dataset.metadata.dataset_id) == dataset
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))
    cash_result = run_backtest(
        dataset,
        strategy,
        BacktestConfig(
            Decimal(100_000),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(1)),
            dividend_policy=DividendPolicy.CASH_DIVIDENDS,
        ),
    )
    price_result = run_backtest(
        dataset,
        strategy,
        BacktestConfig(
            Decimal(100_000),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(1)),
            dividend_policy=DividendPolicy.PRICE_RETURN_ONLY,
        ),
    )

    assert cash_result.benchmark.performance.total_dividend_income > 0
    assert price_result.performance.total_dividend_income == 0
    assert price_result.dividend_accounting.dividend_events_ignored >= 1
    assert cash_result.run_id != price_result.run_id
    assert export_backtest_result(cash_result, tmp_path / "reports").is_dir()
    assert export_backtest_result(price_result, tmp_path / "reports").is_dir()
