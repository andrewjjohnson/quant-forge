import json
from dataclasses import dataclass, replace
from datetime import date
from decimal import ROUND_DOWN, ROUND_UP, Decimal, localcontext
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import (
    BacktestConfig,
    BacktestResult,
    BasisPointFees,
    BasisPointSlippage,
    ExecutionError,
    ExplicitZeroFees,
    FixedCommission,
    InvalidMarketDataError,
    OrderSide,
    OrderStatus,
    ResultExportError,
    export_backtest_result,
    load_backtest_manifest,
    run_backtest,
)
from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data.models import MarketDataset
from quantforge.indicators import Indicator, MarketField
from quantforge.strategies import (
    ExecutionSessionStatus,
    ExecutionTiming,
    MarketDataReference,
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
    PositionIntent,
    PositionSizingPolicy,
    StrategyDecision,
    StrategyOutput,
    StrategyParameters,
    TargetWeightSizingPolicy,
)

from ..helpers import make_dataset

PRICES = ("3", "2", "1", "2", "3", "4", "3", "2", "1")
GOLDEN_PATH = (
    Path(__file__).parents[2]
    / "fixtures"
    / "backtesting"
    / "deterministic_moving_average.json"
)


def configured_result(
    prices: tuple[str, ...] = PRICES, *, dataset_id: str = "synthetic-dataset"
) -> BacktestResult:
    return run_backtest(
        make_dataset(prices, dataset_id=dataset_id),
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(100)),
        ),
    )


def selected(row: PrimitiveMapping, keys: tuple[str, ...]) -> PrimitiveMapping:
    return {key: row[key] for key in keys}


def test_human_auditable_golden_backtest() -> None:
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    result = configured_result()

    assert [
        selected(item.to_primitive(), ("signal_session", "target_position"))
        for item in result.signals
    ] == expected["signals"]
    assert [
        selected(item.to_primitive(), ("side", "requested_quantity", "status"))
        for item in result.orders
    ] == expected["orders"]
    fill_keys = (
        "side",
        "execution_session",
        "reference_price",
        "fill_price",
        "gross_notional",
        "commission",
        "fees",
        "net_cash_effect",
    )
    assert [
        selected(item.to_primitive(), fill_keys) for item in result.fills
    ] == expected["fills"]
    trade_keys = (
        "entry_session",
        "exit_session",
        "entry_quantity",
        "entry_fees",
        "exit_fees",
        "gross_profit_loss",
        "net_profit_loss",
        "holding_period_sessions",
    )
    assert [
        selected(item.to_primitive(), trade_keys) for item in result.completed_trades
    ] == expected["trades"]
    equity_keys = ("session", "cash", "shares", "total_equity", "drawdown")
    assert [
        selected(item.to_primitive(), equity_keys) for item in result.daily_equity
    ] == expected["equity"]
    assert (
        selected(result.performance.to_primitive(), tuple(expected["metrics"]))
        == (expected["metrics"])
    )


def test_next_session_timing_costs_accounting_and_traceability() -> None:
    result = configured_result()
    entry_order, exit_order = result.orders
    entry_fill, exit_fill = result.fills
    trade = result.completed_trades[0]

    assert entry_order.signal_session == date(2024, 7, 8)
    assert entry_fill.execution_session == date(2024, 7, 9)
    assert entry_fill.execution_session > entry_order.signal_session
    assert entry_fill.fill_price > entry_fill.reference_price
    assert exit_fill.fill_price < exit_fill.reference_price
    assert entry_fill.commission == exit_fill.commission == Decimal(1)
    assert entry_fill.fees == exit_fill.fees == Decimal(0)
    assert all(
        record.cash >= 0 and record.shares >= 0 for record in result.daily_equity
    )
    assert all(
        record.total_equity == record.cash + record.market_value
        for record in result.daily_equity
    )
    assert trade.entry_fill_id == entry_fill.fill_id
    assert trade.exit_fill_id == exit_fill.fill_id
    assert entry_fill.order_id == entry_order.order_id
    assert exit_fill.order_id == exit_order.order_id
    assert entry_order.originating_signal_id == result.signals[0].signal_id
    assert trade.strategy_configuration_id == result.strategy_configuration_id
    assert trade.strategy_implementation_version == "1"
    assert result.strategy_implementation_version == "1"


def test_repeated_equivalent_inputs_replay_identically() -> None:
    first = configured_result()
    second = configured_result()

    assert first == second
    assert first.run_id == second.run_id
    assert first.to_primitive() == second.to_primitive()
    json.dumps(first.to_primitive(), allow_nan=False, sort_keys=True)


def test_backtest_ignores_the_callers_ambient_decimal_context() -> None:
    with localcontext() as low_precision:
        low_precision.prec = 8
        low_precision.rounding = ROUND_DOWN
        low_result = configured_result()
    with localcontext() as high_precision:
        high_precision.prec = 50
        high_precision.rounding = ROUND_UP
        high_result = configured_result()

    assert low_result == high_result


def test_different_explicit_costs_change_results_deterministically() -> None:
    dataset = make_dataset(PRICES)
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))
    zero_cost = run_backtest(
        dataset,
        strategy,
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(0)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(0)),
        ),
    )
    realistic_cost = configured_result()

    assert zero_cost.run_id != realistic_cost.run_id
    assert (
        zero_cost.performance.ending_equity > realistic_cost.performance.ending_equity
    )


def test_additional_fees_are_separate_and_reduce_cash_and_trade_results() -> None:
    result = run_backtest(
        make_dataset(PRICES),
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(1)),
            BasisPointFees(Decimal(10)),
            BasisPointSlippage(Decimal(100)),
        ),
    )
    entry, exit_fill = result.fills
    trade = result.completed_trades[0]

    assert entry.fees == Decimal("0.09696")
    assert exit_fill.fees == Decimal("0.02376")
    assert entry.net_cash_effect == -(
        entry.gross_notional + entry.commission + entry.fees
    )
    assert exit_fill.net_cash_effect == (
        exit_fill.gross_notional - exit_fill.commission - exit_fill.fees
    )
    assert trade.gross_profit_loss is not None
    assert trade.exit_commission is not None
    assert trade.exit_fees is not None
    assert trade.net_profit_loss == (
        trade.gross_profit_loss
        - trade.entry_commission
        - trade.entry_fees
        - trade.exit_commission
        - trade.exit_fees
    )
    assert result.benchmark.configuration["fees"] == {
        "model": "basis_point_fees",
        "implementation_version": "1",
        "parameters": {"basis_points": "10"},
    }
    assert result.benchmark.fill is not None
    assert result.benchmark.fill.fees > 0


def test_final_session_signal_is_preserved_as_unexecuted_order() -> None:
    result = configured_result(("3", "2", "1", "2", "3"))

    assert len(result.signals) == 1
    assert result.orders[0].status is OrderStatus.UNEXECUTED_END_OF_DATA
    assert result.orders[0].requested_quantity is None
    assert result.fills == ()
    assert result.completed_trades == ()


def test_unaffordable_entry_is_explicit_and_never_creates_fractional_shares() -> None:
    result = run_backtest(
        make_dataset(("3", "2", "1", "2", "3", "4")),
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        BacktestConfig(
            Decimal(1),
            FixedCommission(Decimal("0.5")),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(100)),
        ),
    )

    assert result.orders[0].status is OrderStatus.REJECTED
    assert result.orders[0].reason == "insufficient_cash_for_one_share"
    assert result.fills == ()
    assert all(
        record.shares == 0 and record.cash == Decimal(1)
        for record in result.daily_equity
    )


def test_open_position_is_not_forced_closed_at_end_of_data() -> None:
    result = configured_result(("3", "2", "1", "2", "3", "4"))

    assert len(result.fills) == 1
    assert result.completed_trades == ()
    assert len(result.open_trades) == 1
    assert result.open_trades[0].exit_fill_id is None
    assert result.daily_equity[-1].shares > 0


def test_post_depletion_daily_return_is_explicitly_undefined() -> None:
    sessions = (
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
        date(2024, 7, 8),
        date(2024, 7, 9),
        date(2024, 7, 10),
        date(2024, 7, 11),
        date(2024, 7, 12),
        date(2024, 7, 15),
    )
    result = run_backtest(
        make_dataset(
            ("3", "2", "1", "2", "3", "4", "3", "2", "1", "1"),
            sessions=sessions,
            dataset_id="complete-loss",
        ),
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        BacktestConfig(
            Decimal(10),
            FixedCommission(Decimal(2)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(0)),
        ),
    )

    assert result.daily_equity[-2].total_equity == Decimal(0)
    assert result.daily_equity[-2].daily_return == Decimal(-1)
    assert result.daily_equity[-1].total_equity == Decimal(0)
    assert result.daily_equity[-1].daily_return is None
    assert result.daily_equity[-1].to_primitive()["daily_return"] is None
    assert result.performance.total_return == Decimal(-1)
    assert result.performance.cagr is None


def test_future_bars_do_not_change_prior_order_or_fill_economics() -> None:
    through_entry = configured_result(
        ("3", "2", "1", "2", "3", "4"), dataset_id="through-entry"
    )
    with_future = configured_result(PRICES, dataset_id="with-future")

    first_order = through_entry.orders[0]
    future_order = with_future.orders[0]
    assert (
        first_order.side,
        first_order.requested_quantity,
        first_order.signal_session,
        first_order.earliest_permitted_execution_session,
        first_order.status,
    ) == (
        future_order.side,
        future_order.requested_quantity,
        future_order.signal_session,
        future_order.earliest_permitted_execution_session,
        future_order.status,
    )
    first_fill = through_entry.fills[0]
    future_fill = with_future.fills[0]
    assert (
        first_fill.execution_session,
        first_fill.reference_price,
        first_fill.fill_price,
        first_fill.quantity,
        first_fill.commission,
    ) == (
        future_fill.execution_session,
        future_fill.reference_price,
        future_fill.fill_price,
        future_fill.quantity,
        future_fill.commission,
    )


def test_invalid_execution_price_fails_with_market_data_domain_error() -> None:
    dataset = make_dataset(("3", "2", "1", "2", "3", "4"))
    invalid_bar = replace(dataset.bars[-1], open=Decimal("NaN"))
    invalid = replace(dataset, bars=(*dataset.bars[:-1], invalid_bar))

    with pytest.raises(InvalidMarketDataError, match="positive and finite"):
        run_backtest(
            invalid,
            MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
            BacktestConfig(
                Decimal(100),
                FixedCommission(Decimal(1)),
                ExplicitZeroFees(),
                BasisPointSlippage(Decimal(1)),
            ),
        )


class FavorableSlippage:
    name = "invalid_favorable_slippage"
    implementation_version = "1"

    def apply(self, reference_price: Decimal, side: OrderSide) -> Decimal:
        del side
        return reference_price - Decimal(1)

    def configuration(self) -> PrimitiveMapping:
        return {
            "model": self.name,
            "implementation_version": self.implementation_version,
            "parameters": {},
        }


def test_custom_slippage_cannot_improve_a_buy_fill() -> None:
    with pytest.raises(ExecutionError, match="adverse"):
        run_backtest(
            make_dataset(("3", "2", "1", "2", "3", "4")),
            MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
            BacktestConfig(
                Decimal(100),
                FixedCommission(Decimal(1)),
                ExplicitZeroFees(),
                FavorableSlippage(),
            ),
        )


def test_metrics_and_benchmark_edge_cases_are_explicit() -> None:
    result = configured_result(("6", "5", "4", "3", "2", "1"))

    assert result.performance.trade_count == 0
    assert result.performance.win_rate is None
    assert result.performance.profit_factor is None
    assert result.performance.sharpe_ratio is None
    assert result.performance.sortino_ratio is None
    assert result.benchmark.fill is not None
    assert result.benchmark.fill.commission == Decimal(1)
    assert result.benchmark.daily_equity[-1].shares == result.benchmark.fill.quantity
    assert result.benchmark.performance.trade_count == 0
    assert result.benchmark.performance.open_trade_count == 1


def test_structured_export_is_stable_reloadable_and_never_overwrites(
    tmp_path: Path,
) -> None:
    result = configured_result()
    exported = export_backtest_result(result, tmp_path / "reports")

    assert sorted(path.name for path in exported.iterdir()) == [
        "benchmark_equity.csv",
        "equity.csv",
        "fills.csv",
        "manifest.json",
        "orders.csv",
        "positions.csv",
        "signals.csv",
        "trades.csv",
    ]
    assert load_backtest_manifest(exported / "manifest.json") == (
        result.manifest_primitive()
    )
    with pytest.raises(ResultExportError, match="already exists"):
        export_backtest_result(result, tmp_path / "reports")


@dataclass(frozen=True, slots=True)
class ManualParameters:
    def to_primitive(self) -> PrimitiveMapping:
        return {}


class ManualTransitionStrategy:
    name = "manual_transition"
    implementation_version = "1"
    timing = ExecutionTiming.NEXT_SESSION_AFTER_CLOSE
    asset_assumptions = ("single symbol", "long-only")
    parameters: StrategyParameters = ManualParameters()
    required_fields = frozenset((MarketField.CLOSE,))
    required_indicators: tuple[Indicator, ...] = ()
    warm_up_observations = 1
    sizing_policy: PositionSizingPolicy = TargetWeightSizingPolicy(Decimal("0.5"))

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_type": "strategy",
            "component_name": self.name,
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": {},
            "required_fields": ["close"],
            "required_indicators": [],
            "warm_up_observations": 1,
            "timing_convention": self.timing.value,
            "sizing": self.sizing_policy.configuration(),
            "asset_assumptions": list(self.asset_assumptions),
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def generate(self, dataset: MarketDataset) -> StrategyOutput:
        decisions = (
            StrategyDecision(
                canonical_symbol=dataset.metadata.canonical_symbol,
                signal_session=dataset.bars[0].session_date,
                earliest_executable_session=dataset.bars[1].session_date,
                execution_timing=self.timing,
                execution_session_status=ExecutionSessionStatus.PENDING,
                target_position=PositionIntent.LONG,
                target_weight=Decimal("0.5"),
                strategy_id=self.name,
                strategy_configuration_id=self.configuration_id,
                strategy_parameters=(),
                reason="manual entry",
                indicator_values=(),
            ),
            StrategyDecision(
                canonical_symbol=dataset.metadata.canonical_symbol,
                signal_session=dataset.bars[2].session_date,
                earliest_executable_session=dataset.bars[3].session_date,
                execution_timing=self.timing,
                execution_session_status=ExecutionSessionStatus.PENDING,
                target_position=PositionIntent.FLAT,
                target_weight=Decimal(0),
                strategy_id=self.name,
                strategy_configuration_id=self.configuration_id,
                strategy_parameters=(),
                reason="manual exit",
                indicator_values=(),
            ),
        )
        return StrategyOutput(
            self.name,
            self.configuration_id,
            MarketDataReference.from_dataset(dataset),
            decisions,
        )


class RevisedManualTransitionStrategy(ManualTransitionStrategy):
    implementation_version = "2"


class MutableConfigurationStrategy(ManualTransitionStrategy):
    def __init__(self) -> None:
        self.mutable_configuration = super().configuration()

    def configuration(self) -> PrimitiveMapping:
        return self.mutable_configuration


class RevisedFixedCommission(FixedCommission):
    implementation_version = "2"


def test_another_generic_strategy_runs_without_backtester_changes() -> None:
    result = run_backtest(
        make_dataset(("10", "11", "12", "13")),
        ManualTransitionStrategy(),
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(10)),
        ),
    )

    assert result.strategy_id == "manual_transition"
    assert [order.status for order in result.orders] == [
        OrderStatus.FILLED,
        OrderStatus.FILLED,
    ]
    assert len(result.completed_trades) == 1
    assert result.performance.win_rate == Decimal(1)
    assert result.performance.profit_factor is None


def test_strategy_implementation_version_changes_run_and_trade_provenance() -> None:
    dataset = make_dataset(("10", "11", "12", "13"))
    config = BacktestConfig(
        Decimal(100),
        FixedCommission(Decimal(1)),
        ExplicitZeroFees(),
        BasisPointSlippage(Decimal(10)),
    )

    original = run_backtest(dataset, ManualTransitionStrategy(), config)
    revised = run_backtest(dataset, RevisedManualTransitionStrategy(), config)

    assert original.run_id != revised.run_id
    assert original.strategy_implementation_version == "1"
    assert revised.strategy_implementation_version == "2"
    assert original.completed_trades[0].strategy_implementation_version == "1"
    assert revised.completed_trades[0].strategy_implementation_version == "2"


def test_cost_model_implementation_version_changes_run_identity() -> None:
    dataset = make_dataset(PRICES)
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))
    original = run_backtest(
        dataset,
        strategy,
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(100)),
        ),
    )
    revised = run_backtest(
        dataset,
        strategy,
        BacktestConfig(
            Decimal(100),
            RevisedFixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(100)),
        ),
    )

    assert original.run_id != revised.run_id
    assert [
        (
            fill.side,
            fill.quantity,
            fill.execution_session,
            fill.fill_price,
            fill.commission,
            fill.fees,
            fill.net_cash_effect,
        )
        for fill in original.fills
    ] == [
        (
            fill.side,
            fill.quantity,
            fill.execution_session,
            fill.fill_price,
            fill.commission,
            fill.fees,
            fill.net_cash_effect,
        )
        for fill in revised.fills
    ]
    assert (
        cast(PrimitiveMapping, original.backtest_configuration["commission"])[
            "implementation_version"
        ]
        == "1"
    )
    assert (
        cast(PrimitiveMapping, revised.backtest_configuration["commission"])[
            "implementation_version"
        ]
        == "2"
    )


def test_result_provenance_is_deeply_snapshotted_from_strategy_configuration() -> None:
    strategy = MutableConfigurationStrategy()
    result = run_backtest(
        make_dataset(("10", "11", "12", "13")),
        strategy,
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(10)),
        ),
    )
    expected_manifest = result.manifest_primitive()

    mutable_parameters = cast(
        PrimitiveMapping, strategy.mutable_configuration["parameters"]
    )
    mutable_parameters["changed_after_run"] = True
    detached_result_parameters = cast(
        PrimitiveMapping, result.strategy_configuration["parameters"]
    )
    detached_result_parameters["changed_through_result"] = True

    assert result.manifest_primitive() == expected_manifest
    assert result.strategy_configuration["parameters"] == {}
