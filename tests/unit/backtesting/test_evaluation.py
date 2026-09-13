"""QF-43 boundary regressions against independent evaluation-only accounting."""

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import (
    BacktestConfig,
    BasisPointFees,
    BasisPointSlippage,
    DividendPolicy,
    EvaluationInterval,
    FixedCommission,
    InvalidBacktestConfigurationError,
    InvalidMarketDataError,
    OrderStatus,
    export_backtest_result,
    run_backtest,
    validate_backtest_result_export,
)
from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import MarketDataset
from quantforge.strategies import (
    ExecutionSessionStatus,
    MarketDataReference,
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
    PositionIntent,
    StrategyDecision,
    StrategyOutput,
)
from quantforge.strategies.base import next_exchange_session

from ..helpers import SESSIONS, make_dataset
from .test_runner import ManualTransitionStrategy, configured_result, zero_cost_config


class ScheduledStrategy(ManualTransitionStrategy):
    """Explicit causal targets, independent of dataset-relative row positions."""

    def __init__(self, schedule: tuple[tuple[date, PositionIntent], ...]) -> None:
        self.schedule = schedule
        self.seen_dataset: MarketDataset | None = None

    def configuration(self) -> PrimitiveMapping:
        return {
            **super().configuration(),
            "schedule": [
                {"session": session.isoformat(), "target": target.value}
                for session, target in self.schedule
            ],
        }

    def generate(self, dataset: MarketDataset) -> StrategyOutput:
        self.seen_dataset = dataset
        available = {bar.session_date for bar in dataset.bars}
        return StrategyOutput(
            self.name,
            self.configuration_id,
            MarketDataReference.from_dataset(dataset),
            tuple(
                StrategyDecision(
                    canonical_symbol="SPY",
                    signal_session=session,
                    earliest_executable_session=next_exchange_session(
                        session, dataset.metadata.calendar
                    ),
                    execution_timing=self.timing,
                    execution_session_status=ExecutionSessionStatus.PENDING,
                    target_position=target,
                    target_weight=Decimal("0.5")
                    if target is PositionIntent.LONG
                    else Decimal(0),
                    strategy_id=self.name,
                    strategy_configuration_id=self.configuration_id,
                    strategy_parameters=(),
                    reason="scheduled causal target",
                    indicator_values=(),
                )
                for session, target in self.schedule
                if session in available
            ),
        )


def test_default_preserves_pre_qf43_complete_result_identity() -> None:
    # Captured on main e4cdbdd before QF-43: covers every serialized record/metric.
    result = configured_result()
    assert (
        result.run_id
        == "3d0ae92e6de009e972cb212dd26f92407d8141e97239b5acb3f2ea6ba1f5fcd8"
    )
    assert configuration_identity(result.to_primitive()) == (
        "0792ee5072d9714efc8eea14f1bbc98f1e3e0e1d9280da1d56a82ff1494e529b"
    )
    assert "evaluation_interval" not in result.backtest_configuration


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (SESSIONS[3], SESSIONS[2]),
        (datetime(2024, 7, 1, tzinfo=UTC), SESSIONS[3]),
        (SESSIONS[0], datetime(2024, 7, 5, tzinfo=UTC)),
        ("2024-07-01", SESSIONS[3]),
        (None, SESSIONS[3]),
    ],
)
def test_invalid_boundary_types_and_order_fail(start: object, end: object) -> None:
    with pytest.raises(InvalidBacktestConfigurationError):
        EvaluationInterval(cast(date, start), cast(date, end))


def test_unsupported_reset_and_untyped_interval_fail() -> None:
    with pytest.raises(InvalidBacktestConfigurationError, match="no carried positions"):
        EvaluationInterval(SESSIONS[0], SESSIONS[1], "carry_training_positions")
    with pytest.raises(InvalidBacktestConfigurationError, match="EvaluationInterval"):
        replace(zero_cost_config(), evaluation_interval=cast(EvaluationInterval, {}))


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (date(2024, 6, 28), SESSIONS[2]),
        (date(2024, 7, 4), SESSIONS[3]),
        (SESSIONS[0], date(2024, 7, 4)),
        (SESSIONS[0], SESSIONS[5]),
    ],
)
def test_endpoints_must_be_observed_sessions_before_strategy_runs(
    start: date, end: date
) -> None:
    strategy = ScheduledStrategy(())
    with pytest.raises(InvalidMarketDataError, match="both evaluation endpoints"):
        run_backtest(
            make_dataset(("10",) * 4),
            strategy,
            replace(
                zero_cost_config(), evaluation_interval=EvaluationInterval(start, end)
            ),
        )
    assert strategy.seen_dataset is None


def test_long_warmup_supplies_first_evaluation_decision_and_next_open_fill() -> None:
    prices = ("10",) * 9 + ("9", "12", "13", "14", "13", "12")
    dataset = make_dataset(prices)
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 10))
    config = replace(
        zero_cost_config(),
        evaluation_interval=EvaluationInterval(SESSIONS[10], SESSIONS[14]),
    )
    result = run_backtest(dataset, strategy, config)
    first = result.signals[0].decision
    assert first.signal_session == SESSIONS[10]
    assert {item.name: item.value for item in first.indicator_values} == {
        "fast_sma": Decimal("10.5"),
        "previous_fast_sma": Decimal("9.5"),
        "previous_slow_sma": Decimal("9.9"),
        "slow_sma": Decimal("10.1"),
    }
    assert result.strategy_warm_up_observations == 10
    assert result.daily_equity[0].cash == 1000
    assert result.daily_equity[0].shares == 0
    assert result.daily_equity[0].daily_return == 0
    assert result.fills[0].execution_session == SESSIONS[11]
    assert result.fills[0].reference_price == 13

    # Removing legitimate history makes the same slow indicator unavailable.
    short_history = make_dataset(prices[9:], sessions=SESSIONS[9:])
    insufficient = run_backtest(short_history, strategy, config)
    assert insufficient.signals == insufficient.fills == ()
    assert insufficient.performance.total_return == 0
    assert insufficient.performance.exposure == 0
    assert all(row.total_equity == 1000 for row in insufficient.daily_equity)


def test_context_target_state_is_preserved_without_carrying_a_position() -> None:
    dataset = make_dataset(("3", "2", "1", "2", "3", "4", "3", "2", "1"))
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))
    result = run_backtest(
        dataset,
        strategy,
        replace(
            zero_cost_config(),
            evaluation_interval=EvaluationInterval(SESSIONS[5], SESSIONS[8]),
        ),
    )
    # July 8's LONG would ordinarily fill at the first evaluation open, July 9.
    assert [signal.decision.target_position for signal in result.signals] == [
        PositionIntent.FLAT
    ]
    assert result.orders[0].reason == "target_already_flat"
    assert result.fills == result.completed_trades == result.open_trades == ()
    assert result.performance.exposure == result.performance.total_return == 0
    assert all(row.cash == 1000 and row.shares == 0 for row in result.daily_equity)
    assert all(
        row.realized_profit_loss == row.unrealized_profit_loss == 0
        for row in result.positions
    )


@pytest.mark.parametrize("with_costs", [False, True])
def test_context_cannot_affect_any_accounting_benchmark_or_metric(
    with_costs: bool,
) -> None:
    prices = ("100", "200", "50", "50", "10", "11", "9", "12", "13", "900")
    schedule = (
        (SESSIONS[0], PositionIntent.LONG),
        (SESSIONS[2], PositionIntent.FLAT),
        (SESSIONS[3], PositionIntent.LONG),  # would fill at evaluation start
        (SESSIONS[4], PositionIntent.LONG),
        (SESSIONS[6], PositionIntent.FLAT),
        (SESSIONS[8], PositionIntent.LONG),  # must not fill on source's future bar
    )
    costs = zero_cost_config()
    if with_costs:
        costs = replace(
            costs,
            commission=FixedCommission(Decimal(1)),
            fees=BasisPointFees(Decimal(10)),
            slippage=BasisPointSlippage(Decimal(100)),
        )
    bounded = run_backtest(
        make_dataset(prices),
        ScheduledStrategy(schedule),
        replace(
            costs, evaluation_interval=EvaluationInterval(SESSIONS[4], SESSIONS[8])
        ),
    )
    isolated = run_backtest(
        make_dataset(prices[4:9], sessions=SESSIONS[4:9]),
        ScheduledStrategy(schedule),
        costs,
    )
    # The independent short dataset has no warm-up trades or returns to slice out.
    assert bounded.performance == isolated.performance
    assert bounded.benchmark.performance == isolated.benchmark.performance
    for actual, expected in zip(
        bounded.daily_equity, isolated.daily_equity, strict=True
    ):
        assert (
            actual.cash,
            actual.shares,
            actual.total_equity,
            actual.daily_return,
            actual.drawdown,
            actual.exposure_weight,
        ) == (
            expected.cash,
            expected.shares,
            expected.total_equity,
            expected.daily_return,
            expected.drawdown,
            expected.exposure_weight,
        )
    assert bounded.positions == isolated.positions
    assert [row.session for row in bounded.daily_equity] == list(SESSIONS[4:9])
    assert bounded.performance.trade_count == 1
    assert bounded.performance.open_trade_count == 0
    assert bounded.performance.exposure == Decimal("0.4")
    assert bounded.daily_equity[0].cash == 1000
    assert bounded.orders[-1].status is OrderStatus.UNEXECUTED_END_OF_DATA
    assert bounded.orders[-1].reason == "no_later_execution_bar"
    assert [fill.execution_session for fill in bounded.fills] == [
        SESSIONS[5],
        SESSIONS[7],
    ]
    assert bounded.benchmark.fill is not None
    assert bounded.benchmark.fill.execution_session == SESSIONS[4]
    assert bounded.benchmark.fill.reference_price == 10
    assert bounded.benchmark.configuration["start"] == "first_evaluation_session_open"
    if with_costs:
        buy, sell = bounded.fills
        assert buy.fill_price == Decimal("11.11")
        assert sell.fill_price == Decimal("11.88")
        assert buy.commission == sell.commission == 1
        assert buy.fees == buy.gross_notional * Decimal("0.001")
        assert sell.fees == sell.gross_notional * Decimal("0.001")
    else:
        assert bounded.benchmark.performance.total_return == Decimal("0.3")


@pytest.mark.parametrize(
    "policy", [DividendPolicy.CASH_DIVIDENDS, DividendPolicy.PRICE_RETURN_ONLY]
)
def test_corporate_actions_before_at_and_inside_boundary(
    policy: DividendPolicy,
) -> None:
    prices = ("200", "100", "50", "50", "25", "25", "99")
    dataset = make_dataset(
        prices,
        splits=((SESSIONS[1], "2"), (SESSIONS[2], "2"), (SESSIONS[4], "2")),
        dividends=(
            (SESSIONS[1], "7"),
            (SESSIONS[2], "3"),
            (SESSIONS[4], "1"),
            (SESSIONS[6], "8"),
        ),
    )
    strategy = ScheduledStrategy(
        ((SESSIONS[0], PositionIntent.LONG), (SESSIONS[2], PositionIntent.LONG))
    )
    result = run_backtest(
        dataset,
        strategy,
        replace(
            zero_cost_config(policy),
            evaluation_interval=EvaluationInterval(SESSIONS[2], SESSIONS[5]),
        ),
    )
    assert strategy.seen_dataset is not None
    assert [bar.close for bar in strategy.seen_dataset.bars] == [Decimal(200)] * 6
    assert len(strategy.seen_dataset.corporate_actions) == 6
    assert result.daily_equity[0].cash == 1000
    assert result.daily_equity[0].shares == 0
    assert [row.effective_session for row in result.split_adjustments] == [
        SESSIONS[2],
        SESSIONS[4],
    ]
    assert (
        result.split_adjustments[0].shares_before
        == result.split_adjustments[0].shares_after
        == 0
    )
    assert result.split_adjustments[1].shares_before == 10
    assert result.split_adjustments[1].shares_after == 20
    assert result.fills[0].reference_price == 50  # fills use raw, not feature prices
    assert result.dividend_accounting.dividend_events_present == 2
    assert result.benchmark.dividend_accounting.dividend_events_present == 2
    assert result.performance.split_event_count == 2
    assert result.benchmark.performance.split_event_count == 2
    assert result.benchmark.fill is not None
    assert result.benchmark.fill.quantity == 20
    if policy is DividendPolicy.CASH_DIVIDENDS:
        assert result.performance.total_dividend_income == 10
        assert result.benchmark.performance.total_dividend_income == 20
        assert result.performance.dividend_event_count == 1
        assert result.daily_equity[-1].total_equity == 1010
    else:
        assert result.performance.total_dividend_income == 0
        assert result.benchmark.performance.total_dividend_income == 0
        assert result.daily_equity[-1].total_equity == 1000


def test_single_session_interval_keeps_signal_but_cannot_fill() -> None:
    result = run_backtest(
        make_dataset(
            ("100", "90", "100"),
            opens=("100", "90", "90"),
            lows=("100", "90", "90"),
        ),
        ScheduledStrategy(((SESSIONS[2], PositionIntent.LONG),)),
        replace(
            zero_cost_config(),
            evaluation_interval=EvaluationInterval(SESSIONS[2], SESSIONS[2]),
        ),
    )
    assert len(result.daily_equity) == len(result.signals) == 1
    assert result.fills == ()
    assert result.orders[0].status is OrderStatus.UNEXECUTED_END_OF_DATA
    assert result.performance.cagr is None
    assert result.performance.annualized_volatility is None
    assert result.performance.total_return == 0
    assert result.benchmark.daily_equity[0].daily_return == Decimal("0.11")
    assert result.benchmark.performance.total_return == Decimal("0.11")


def test_future_append_cannot_change_historical_economics_or_reuse_source_identity(
    tmp_path: Path,
) -> None:
    prices = ("3", "2", "1", "2", "3", "4", "3", "2", "1")
    source = make_dataset(prices)
    extended = make_dataset((*prices, "9000", "1"), splits=((SESSIONS[10], "7"),))
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))
    config = replace(
        zero_cost_config(),
        evaluation_interval=EvaluationInterval(SESSIONS[4], SESSIONS[8]),
    )
    historical = run_backtest(source, strategy, config)
    future = run_backtest(extended, strategy, config)
    assert historical.performance == future.performance
    assert historical.benchmark.performance == future.benchmark.performance
    assert historical.positions == future.positions
    assert [record.decision for record in historical.signals] == [
        record.decision for record in future.signals
    ]
    assert [
        (row.execution_session, row.quantity, row.fill_price, row.net_cash_effect)
        for row in historical.fills
    ] == [
        (row.execution_session, row.quantity, row.fill_price, row.net_cash_effect)
        for row in future.fills
    ]
    assert historical.run_id != future.run_id  # QF-3's immutable source changed
    assert historical.market_data.dataset_id != future.market_data.dataset_id
    assert (
        run_backtest(source, strategy, config).to_primitive()
        == historical.to_primitive()
    )
    artifact = export_backtest_result(historical, tmp_path)
    assert validate_backtest_result_export(historical, artifact) == artifact
    with pytest.raises(InvalidMarketDataError):
        run_backtest(replace(source, bars=extended.bars), strategy, config)


def test_interval_and_source_context_are_identity_bearing() -> None:
    dataset = make_dataset(("10",) * 9)
    baseline = zero_cost_config()
    variants: tuple[BacktestConfig, ...] = (
        baseline,
        replace(
            baseline, evaluation_interval=EvaluationInterval(SESSIONS[0], SESSIONS[8])
        ),
        replace(
            baseline, evaluation_interval=EvaluationInterval(SESSIONS[3], SESSIONS[8])
        ),
        replace(
            baseline, evaluation_interval=EvaluationInterval(SESSIONS[3], SESSIONS[7])
        ),
        replace(
            baseline,
            initial_capital=Decimal(2000),
            evaluation_interval=EvaluationInterval(SESSIONS[3], SESSIONS[7]),
        ),
    )
    results = [
        run_backtest(dataset, ScheduledStrategy(()), config) for config in variants
    ]
    assert len({result.run_id for result in results}) == len(variants)
    assert len({result.benchmark.benchmark_id for result in results}) == len(variants)
    shorter_context = make_dataset(("10",) * 8, sessions=SESSIONS[1:9])
    assert (
        run_backtest(shorter_context, ScheduledStrategy(()), variants[-1]).run_id
        != results[-1].run_id
    )
