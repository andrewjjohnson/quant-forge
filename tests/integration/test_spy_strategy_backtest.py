from decimal import Decimal

from quantforge.backtesting import (
    BacktestConfig,
    BasisPointSlippage,
    PerShareCommission,
    run_backtest,
)
from quantforge.strategies import (
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
)

from ..unit.helpers import make_dataset


def test_local_spy_dataset_runs_qf4_strategy_end_to_end() -> None:
    dataset = make_dataset(
        ("100", "99", "98", "99", "101", "103", "102", "99", "97"),
        dataset_id="immutable-local-spy-fixture",
    )
    strategy = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverParameters(2, 3, target_long_weight=Decimal("0.75"))
    )

    result = run_backtest(
        dataset,
        strategy,
        BacktestConfig(
            Decimal("100000"),
            PerShareCommission(Decimal("0.005"), minimum=Decimal("1")),
            BasisPointSlippage(Decimal("5")),
        ),
    )

    assert result.market_data.dataset_id == "immutable-local-spy-fixture"
    assert result.strategy_id == "moving_average_crossover"
    assert len(result.signals) == len(result.orders) == 2
    assert len(result.fills) == 2
    assert result.performance.trade_count == 1
    assert result.benchmark.fill is not None
