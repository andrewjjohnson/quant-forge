"""Trial execution helpers, including process-worker initialization."""

from collections.abc import Callable

from quantforge.backtesting import BacktestConfig, BacktestResult, run_backtest
from quantforge.configuration import PrimitiveMapping
from quantforge.data import MarketDataset
from quantforge.optimization.factories import StrategyFactory
from quantforge.strategies import Strategy

type BacktestRunner = Callable[
    [MarketDataset, Strategy, BacktestConfig], BacktestResult
]

_worker_dataset: MarketDataset | None = None
_worker_factory: StrategyFactory | None = None
_worker_backtest_config: BacktestConfig | None = None


def initialize_process_worker(
    dataset: MarketDataset,
    strategy_factory: StrategyFactory,
    backtest_config: BacktestConfig,
) -> None:
    """Load immutable trial inputs once in each local process."""
    global _worker_backtest_config, _worker_dataset, _worker_factory
    _worker_dataset = dataset
    _worker_factory = strategy_factory
    _worker_backtest_config = backtest_config


def run_process_trial(parameters: PrimitiveMapping) -> BacktestResult:
    """Build a QF-4 strategy and invoke the unmodified QF-5 runner."""
    if (
        _worker_dataset is None
        or _worker_factory is None
        or _worker_backtest_config is None
    ):
        raise RuntimeError("optimization process worker was not initialized")
    strategy = _worker_factory.build(parameters)
    return run_backtest(_worker_dataset, strategy, _worker_backtest_config)
