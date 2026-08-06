import csv
import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, tzinfo
from decimal import ROUND_DOWN, ROUND_UP, Decimal, getcontext, localcontext
from pathlib import Path
from typing import ClassVar, Literal, cast

import pytest

from quantforge.backtesting import (
    BACKTEST_ARTIFACT_FILENAMES,
    BacktestConfig,
    BacktestResult,
    BasisPointFees,
    BasisPointSlippage,
    DividendPolicy,
    ExecutionError,
    ExplicitZeroFees,
    FixedCommission,
    InvalidMarketDataError,
    InvalidSignalError,
    OrderSide,
    OrderStatus,
    PortfolioAccountingError,
    ResultExportError,
    ReturnBasis,
    export_backtest_result,
    load_backtest_manifest,
    run_backtest,
    validate_backtest_result_artifact,
    validate_backtest_result_export,
)
from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import dataset_identity_matches
from quantforge.data.models import AdjustmentMode, MarketDataset
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


class UndefinedOffsetTimezone(tzinfo):
    def utcoffset(self, date_time: datetime | None) -> None:
        del date_time
        return None

    def dst(self, date_time: datetime | None) -> None:
        del date_time
        return None

    def tzname(self, date_time: datetime | None) -> str:
        del date_time
        return "undefined-offset"


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
    assert result.market_data.schema_version == "4"
    assert result.market_data.split_sessions == ()
    assert result.market_data.dividend_sessions == ()
    assert result.warnings == ()


def test_repeated_equivalent_inputs_replay_identically() -> None:
    first = configured_result()
    second = configured_result()

    assert first == second
    assert first.run_id == second.run_id
    assert first.to_primitive() == second.to_primitive()
    json.dumps(first.to_primitive(), allow_nan=False, sort_keys=True)


def test_changed_bars_are_rejected_when_dataset_id_is_reused() -> None:
    original_dataset = make_dataset(PRICES, dataset_id="reused-dataset-id")
    revised_bars = list(original_dataset.bars)
    revised_bars[-1] = replace(revised_bars[-1], open=Decimal("1.5"))
    revised_dataset = replace(original_dataset, bars=tuple(revised_bars))

    with pytest.raises(InvalidMarketDataError, match=r"do not match.*dataset identity"):
        run_backtest(
            revised_dataset,
            MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
            BacktestConfig(
                Decimal(100),
                FixedCommission(Decimal(1)),
                ExplicitZeroFees(),
                BasisPointSlippage(Decimal(100)),
            ),
        )


def test_execution_calendar_is_verified_and_changes_valid_run_identity() -> None:
    sessions = (
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
    )
    original_dataset = make_dataset(
        ("3", "2", "4"),
        sessions=sessions,
        dataset_id="reused-dataset-id",
    )
    copied_dataset = replace(
        original_dataset,
        metadata=replace(original_dataset.metadata, calendar="24/7"),
    )
    revised_dataset = make_dataset(
        ("3", "2", "4"),
        sessions=sessions,
        dataset_id="reused-dataset-id",
        calendar="24/7",
    )
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(1, 2))
    config = BacktestConfig(
        Decimal(100),
        FixedCommission(Decimal(1)),
        ExplicitZeroFees(),
        BasisPointSlippage(Decimal(100)),
    )

    with pytest.raises(InvalidMarketDataError, match=r"do not match.*dataset identity"):
        run_backtest(copied_dataset, strategy, config)

    original = run_backtest(original_dataset, strategy, config)
    revised = run_backtest(revised_dataset, strategy, config)

    assert original.market_data.dataset_id != revised.market_data.dataset_id
    assert original.market_data.bars_fingerprint == (
        revised.market_data.bars_fingerprint
    )
    assert original.signals[0].decision.earliest_executable_session == date(2024, 7, 5)
    assert revised.signals[0].decision.earliest_executable_session == date(2024, 7, 4)
    assert original.run_id != revised.run_id


@pytest.mark.parametrize(
    "initiated_at",
    [
        datetime(2024, 7, 15, 12),
        datetime(2024, 7, 15, 12, tzinfo=UndefinedOffsetTimezone()),
    ],
)
def test_initiation_timestamp_requires_a_defined_utc_offset(
    initiated_at: datetime,
) -> None:
    with pytest.raises(InvalidSignalError, match="defined UTC offset"):
        run_backtest(
            make_dataset(PRICES),
            MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
            BacktestConfig(
                Decimal(100),
                FixedCommission(Decimal(1)),
                ExplicitZeroFees(),
                BasisPointSlippage(Decimal(100)),
            ),
            initiated_at=initiated_at,
        )


def test_initiation_timestamp_preserves_its_defined_utc_offset() -> None:
    initiated_at = datetime(2024, 7, 15, 12, tzinfo=UTC)
    result = run_backtest(
        make_dataset(PRICES),
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(100)),
        ),
        initiated_at=initiated_at,
    )

    assert result.manifest_primitive()["initiated_at"] == "2024-07-15T12:00:00+00:00"


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
        "buy_cost_is_non_decreasing_by_quantity": True,
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


def test_internal_missing_market_session_is_rejected_before_daily_metrics() -> None:
    sessions = (
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 5),
        date(2024, 7, 8),
        date(2024, 7, 9),
        date(2024, 7, 10),
    )
    dataset = make_dataset(
        ("3", "2", "1", "2", "3", "4"),
        sessions=sessions,
        missing_sessions=(date(2024, 7, 3),),
    )

    with pytest.raises(
        InvalidMarketDataError,
        match="missing expected sessions within its observed range: 2024-07-03",
    ):
        run_backtest(
            dataset,
            MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
            BacktestConfig(
                Decimal(100),
                FixedCommission(Decimal(1)),
                ExplicitZeroFees(),
                BasisPointSlippage(Decimal(1)),
            ),
        )


def test_internal_session_gaps_are_recomputed_instead_of_trusting_metadata() -> None:
    dataset = make_dataset(
        ("1", "2", "3"),
        sessions=(
            date(2024, 7, 1),
            date(2024, 7, 3),
            date(2024, 7, 5),
        ),
        missing_sessions=(),
    )
    assert dataset_identity_matches(dataset)

    with pytest.raises(
        InvalidMarketDataError,
        match=(
            "missing-session provenance does not match the calendar; "
            "computed: 2024-07-02; declared: none"
        ),
    ):
        run_backtest(
            dataset,
            MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(1, 2)),
            BacktestConfig(
                Decimal(100),
                FixedCommission(Decimal(0)),
                ExplicitZeroFees(),
                BasisPointSlippage(Decimal(0)),
            ),
        )


def test_requested_range_gaps_outside_observed_bars_remain_provenance() -> None:
    sessions = (
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
        date(2024, 7, 8),
        date(2024, 7, 9),
        date(2024, 7, 10),
    )
    outside_gaps = (date(2024, 7, 1), date(2024, 7, 11))
    dataset = make_dataset(
        ("3", "2", "1", "2", "3", "4"),
        sessions=sessions,
        requested_start=outside_gaps[0],
        requested_end=outside_gaps[1],
        missing_sessions=outside_gaps,
    )

    result = run_backtest(
        dataset,
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(1)),
        ),
    )

    assert result.market_data.missing_sessions == outside_gaps
    assert tuple(record.session for record in result.daily_equity) == sessions


@pytest.mark.parametrize(
    "adjustment_mode",
    [
        AdjustmentMode.SPLIT_ADJUSTED,
        AdjustmentMode.SPLIT_AND_DIVIDEND_ADJUSTED,
    ],
)
def test_adjusted_data_is_rejected_by_raw_price_action_accounting(
    adjustment_mode: AdjustmentMode,
) -> None:
    dataset = make_dataset(
        PRICES,
        adjustment_mode=adjustment_mode,
    )

    with pytest.raises(InvalidMarketDataError, match="raw-price explicit"):
        run_backtest(
            dataset,
            MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
            BacktestConfig(
                Decimal(100),
                FixedCommission(Decimal(1)),
                ExplicitZeroFees(),
                BasisPointSlippage(Decimal(100)),
            ),
        )


def test_market_data_schema_without_dividend_provenance_is_rejected() -> None:
    dataset = make_dataset(PRICES)
    legacy = replace(
        dataset,
        metadata=replace(dataset.metadata, schema_version="2"),
    )

    with pytest.raises(InvalidMarketDataError, match="corporate-action provenance"):
        run_backtest(
            legacy,
            MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
            BacktestConfig(
                Decimal(100),
                FixedCommission(Decimal(1)),
                ExplicitZeroFees(),
                BasisPointSlippage(Decimal(100)),
            ),
        )


def test_unadjusted_data_with_stock_split_is_accounted_for() -> None:
    split_session = date(2024, 7, 9)
    split_bearing = make_dataset(
        PRICES,
        split_sessions=(split_session,),
    )

    result = run_backtest(
        split_bearing,
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(100)),
        ),
    )

    assert len(result.split_adjustments) == 1
    assert result.split_adjustments[0].effective_session == split_session


def test_split_normalized_strategy_view_prevents_false_moving_average_exit() -> None:
    continuous_dataset = make_dataset(("100", "100", "99", "101", "102", "103", "105"))
    split_dataset = make_dataset(
        ("100", "100", "99", "101", "102", "51.5", "52.5"),
        splits=((date(2024, 7, 9), "2"),),
    )
    strategy_parameters = MovingAverageCrossoverParameters(2, 3)
    continuous_result = run_backtest(
        continuous_dataset,
        MovingAverageCrossoverStrategy(strategy_parameters),
        zero_cost_config(),
    )
    split_result = run_backtest(
        split_dataset,
        MovingAverageCrossoverStrategy(strategy_parameters),
        zero_cost_config(),
    )

    continuous_decisions = tuple(
        (
            signal.decision.signal_session,
            signal.decision.target_position,
            signal.decision.indicator_values,
        )
        for signal in continuous_result.signals
    )
    split_decisions = tuple(
        (
            signal.decision.signal_session,
            signal.decision.target_position,
            signal.decision.indicator_values,
        )
        for signal in split_result.signals
    )
    assert split_decisions == continuous_decisions
    assert tuple(item[1] for item in split_decisions) == (PositionIntent.LONG,)
    assert split_result.fills[0].reference_price == Decimal("51.5")


def test_future_split_does_not_revise_prior_strategy_decisions() -> None:
    strategy_parameters = MovingAverageCrossoverParameters(2, 3)
    prefix_result = run_backtest(
        make_dataset(("100", "100", "99", "101", "102", "103")),
        MovingAverageCrossoverStrategy(strategy_parameters),
        zero_cost_config(),
    )
    future_split_result = run_backtest(
        make_dataset(
            ("100", "100", "99", "101", "102", "103", "52.5"),
            splits=((date(2024, 7, 10), "2"),),
        ),
        MovingAverageCrossoverStrategy(strategy_parameters),
        zero_cost_config(),
    )

    prefix_decisions = tuple(
        (
            signal.decision.signal_session,
            signal.decision.target_position,
            signal.decision.indicator_values,
        )
        for signal in prefix_result.signals
    )
    matching_full_decisions = tuple(
        (
            signal.decision.signal_session,
            signal.decision.target_position,
            signal.decision.indicator_values,
        )
        for signal in future_split_result.signals
        if signal.decision.signal_session
        <= prefix_result.market_data.actual_last_session
    )
    assert matching_full_decisions == prefix_decisions


def test_unadjusted_data_with_cash_dividend_is_accounted_for() -> None:
    dividend_session = date(2024, 7, 9)
    dividend_bearing = make_dataset(
        PRICES,
        dividend_sessions=(dividend_session,),
    )

    result = run_backtest(
        dividend_bearing,
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(100)),
            dividend_policy=DividendPolicy.CASH_DIVIDENDS,
        ),
    )

    assert len(result.dividend_cashflows) == 1
    assert result.dividend_cashflows[0].ex_dividend_session == dividend_session


def test_removed_dividend_provenance_is_rejected_by_dataset_identity() -> None:
    dividend_bearing = make_dataset(
        PRICES,
        dividend_sessions=(date(2024, 7, 9),),
    )
    stripped = replace(
        dividend_bearing,
        metadata=replace(dividend_bearing.metadata, dividend_sessions=()),
    )

    with pytest.raises(
        InvalidMarketDataError,
        match="corporate-action counts or sessions do not match records",
    ):
        run_backtest(
            stripped,
            MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
            BacktestConfig(
                Decimal(100),
                FixedCommission(Decimal(1)),
                ExplicitZeroFees(),
                BasisPointSlippage(Decimal(100)),
            ),
        )


class FavorableSlippage:
    cost_category: ClassVar[Literal["slippage"]] = "slippage"
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


class AmbientContextCommission:
    cost_category: ClassVar[Literal["commission"]] = "commission"
    name = "ambient_context_commission"
    implementation_version = "1"
    buy_cost_is_non_decreasing_by_quantity: Literal[True] = True

    def __init__(self) -> None:
        self.observed_precisions: set[int] = set()
        self.observed_configuration_precisions: set[int] = set()

    def calculate(self, quantity: int, fill_price: Decimal) -> Decimal:
        del quantity
        self.observed_precisions.add(getcontext().prec)
        return fill_price / Decimal(7)

    def configuration(self) -> PrimitiveMapping:
        self.observed_configuration_precisions.add(getcontext().prec)
        return {
            "model": self.name,
            "implementation_version": self.implementation_version,
            "buy_cost_is_non_decreasing_by_quantity": True,
            "parameters": {},
        }


class AmbientContextFees:
    cost_category: ClassVar[Literal["transaction_fee"]] = "transaction_fee"
    name = "ambient_context_fees"
    implementation_version = "1"
    buy_cost_is_non_decreasing_by_quantity: Literal[True] = True

    def __init__(self) -> None:
        self.observed_precisions: set[int] = set()
        self.observed_configuration_precisions: set[int] = set()

    def calculate(
        self,
        side: OrderSide,
        quantity: int,
        fill_price: Decimal,
    ) -> Decimal:
        del side, quantity
        self.observed_precisions.add(getcontext().prec)
        return fill_price / Decimal(13)

    def configuration(self) -> PrimitiveMapping:
        self.observed_configuration_precisions.add(getcontext().prec)
        return {
            "model": self.name,
            "implementation_version": self.implementation_version,
            "buy_cost_is_non_decreasing_by_quantity": True,
            "parameters": {},
        }


class AmbientContextSlippage:
    cost_category: ClassVar[Literal["slippage"]] = "slippage"
    name = "ambient_context_slippage"
    implementation_version = "1"

    def __init__(self) -> None:
        self.observed_precisions: set[int] = set()
        self.observed_configuration_precisions: set[int] = set()

    def apply(self, reference_price: Decimal, side: OrderSide) -> Decimal:
        self.observed_precisions.add(getcontext().prec)
        rate = Decimal(1) / Decimal(97_000)
        multiplier = Decimal(1) + rate if side is OrderSide.BUY else Decimal(1) - rate
        return reference_price * multiplier

    def configuration(self) -> PrimitiveMapping:
        self.observed_configuration_precisions.add(getcontext().prec)
        return {
            "model": self.name,
            "implementation_version": self.implementation_version,
            "parameters": {},
        }


def test_custom_cost_callbacks_use_the_serialized_decimal_policy() -> None:
    commission = AmbientContextCommission()
    fees = AmbientContextFees()
    slippage = AmbientContextSlippage()
    dataset = make_dataset(PRICES)
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))
    config = BacktestConfig(Decimal(100), commission, fees, slippage)

    with localcontext() as low_precision:
        low_precision.prec = 8
        low_precision.rounding = ROUND_DOWN
        low_result = run_backtest(dataset, strategy, config)
    with localcontext() as high_precision:
        high_precision.prec = 50
        high_precision.rounding = ROUND_UP
        high_result = run_backtest(dataset, strategy, config)

    assert low_result == high_result
    assert commission.observed_precisions == {34}
    assert fees.observed_precisions == {34}
    assert slippage.observed_precisions == {34}
    assert commission.observed_configuration_precisions == {34}
    assert fees.observed_configuration_precisions == {34}
    assert slippage.observed_configuration_precisions == {34}


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


def test_benchmark_includes_first_invested_return_in_risk_metrics() -> None:
    result = run_backtest(
        make_dataset(("10", "11")),
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(1, 2)),
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(10)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(0)),
            annualization_factor=1,
        ),
    )

    assert result.daily_equity[0].daily_return == Decimal(0)
    assert tuple(record.daily_return for record in result.benchmark.daily_equity) == (
        Decimal("-0.1"),
        Decimal("0.1"),
    )
    assert str(result.benchmark.performance.annualized_volatility).startswith(
        "0.1414213562373095048801688724"
    )
    assert result.benchmark.performance.sharpe_ratio == Decimal(0)
    assert result.benchmark.performance.sortino_ratio == Decimal(0)
    assert result.benchmark.configuration["implementation_version"] == "4"
    assert result.benchmark.configuration["return_series_start"] == (
        "initial_capital_to_first_session_close"
    )


def test_structured_export_is_stable_reloadable_and_never_overwrites(
    tmp_path: Path,
) -> None:
    result = configured_result()
    exported = export_backtest_result(result, tmp_path / "reports")

    assert sorted(path.name for path in exported.iterdir()) == [
        "benchmark_dividend_cashflows.csv",
        "benchmark_equity.csv",
        "benchmark_split_adjustments.csv",
        "dividend_cashflows.csv",
        "equity.csv",
        "fills.csv",
        "integrity.json",
        "manifest.json",
        "orders.csv",
        "positions.csv",
        "signals.csv",
        "split_adjustments.csv",
        "trades.csv",
    ]
    manifest = load_backtest_manifest(exported / "manifest.json")
    integrity = cast(
        PrimitiveMapping,
        json.loads((exported / "integrity.json").read_text(encoding="utf-8")),
    )
    assert manifest == result.manifest_primitive()
    assert integrity["schema_version"] == "1"
    assert integrity["algorithm"] == "sha256"
    assert set(cast(PrimitiveMapping, integrity["files"])) == set(
        BACKTEST_ARTIFACT_FILENAMES
    ) - {"integrity.json"}
    assert validate_backtest_result_artifact(exported) == exported
    assert validate_backtest_result_export(result, exported) == exported
    with pytest.raises(ResultExportError, match="already exists"):
        export_backtest_result(result, tmp_path / "reports")


@pytest.mark.parametrize("artifact_name", BACKTEST_ARTIFACT_FILENAMES)
@pytest.mark.parametrize("mutation", ["delete", "truncate"])
def test_export_validation_rejects_incomplete_or_modified_artifacts(
    tmp_path: Path,
    artifact_name: str,
    mutation: str,
) -> None:
    result = configured_result()
    exported = export_backtest_result(result, tmp_path / "reports")
    artifact = exported / artifact_name
    if mutation == "delete":
        artifact.unlink()
    else:
        artifact.write_bytes(b"")

    with pytest.raises(ResultExportError, match="expected immutable result"):
        validate_backtest_result_export(result, exported)


def test_export_validation_rejects_unexpected_artifacts(tmp_path: Path) -> None:
    result = configured_result()
    exported = export_backtest_result(result, tmp_path / "reports")
    (exported / "unexpected.csv").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(ResultExportError, match="expected immutable result"):
        validate_backtest_result_export(result, exported)


@pytest.mark.parametrize(
    "manifest_field",
    ["corporate_action_accounting", "benchmark", "record_counts", "warnings"],
)
def test_artifact_validation_binds_manifest_contents(
    tmp_path: Path, manifest_field: str
) -> None:
    result = configured_result()
    exported = export_backtest_result(result, tmp_path / "reports")
    manifest_path = exported / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[manifest_field] = "corrupt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ResultExportError, match="integrity validation failed"):
        validate_backtest_result_artifact(exported)


def test_empty_fill_export_preserves_the_full_fill_schema(tmp_path: Path) -> None:
    result = run_backtest(
        make_dataset(("1000", "1000")),
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(1, 2)),
        BacktestConfig(
            Decimal(1),
            FixedCommission(Decimal(0)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(0)),
        ),
    )

    assert result.fills == ()
    assert result.benchmark.fill is None
    exported = export_backtest_result(result, tmp_path / "reports")
    with (exported / "fills.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert rows == [
        [
            "fill_id",
            "order_id",
            "originating_signal_id",
            "symbol",
            "side",
            "quantity",
            "execution_session",
            "reference_price",
            "fill_price",
            "slippage_per_share",
            "slippage_basis_points",
            "gross_notional",
            "commission",
            "fees",
            "net_cash_effect",
            "strategy_id",
            "strategy_configuration_id",
        ]
    ]


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


class ConfigurationChangingDuringInitializationStrategy(ManualTransitionStrategy):
    def __init__(self) -> None:
        self._configuration_generation = 0

    def configuration(self) -> PrimitiveMapping:
        self._configuration_generation += 1
        configuration = super().configuration()
        configuration["configuration_generation"] = self._configuration_generation
        return configuration


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


def test_backtest_rejects_identity_that_does_not_match_captured_configuration() -> None:
    with pytest.raises(
        InvalidSignalError,
        match="configuration identity does not match the captured configuration",
    ):
        run_backtest(
            make_dataset(("10", "11", "12", "13")),
            ConfigurationChangingDuringInitializationStrategy(),
            BacktestConfig(
                Decimal(100),
                FixedCommission(Decimal(1)),
                ExplicitZeroFees(),
                BasisPointSlippage(Decimal(10)),
            ),
        )


def zero_cost_config(
    dividend_policy: DividendPolicy = DividendPolicy.CASH_DIVIDENDS,
) -> BacktestConfig:
    return BacktestConfig(
        Decimal(1000),
        FixedCommission(Decimal(0)),
        ExplicitZeroFees(),
        BasisPointSlippage(Decimal(0)),
        dividend_policy=dividend_policy,
    )


def test_dividend_uses_previous_close_entitlement_and_economic_trade_return() -> None:
    dataset = make_dataset(
        ("100", "100", "100", "100"),
        dividends=((date(2024, 7, 3), "1"),),
    )

    result = run_backtest(dataset, ManualTransitionStrategy(), zero_cost_config())

    strategy_cashflow = result.dividend_cashflows[0]
    benchmark_cashflow = result.benchmark.dividend_cashflows[0]
    trade = result.completed_trades[0]
    assert strategy_cashflow.entitled_share_quantity == 5
    assert strategy_cashflow.total_dividend_cash == Decimal(5)
    assert len(result.dividend_cashflows) == 1
    assert benchmark_cashflow.entitled_share_quantity == 10
    assert benchmark_cashflow.total_dividend_cash == Decimal(10)
    assert result.performance.total_dividend_income == Decimal(5)
    assert result.benchmark.performance.total_dividend_income == Decimal(10)
    assert result.performance.ending_equity == Decimal(1005)
    assert result.daily_equity[2].daily_return == Decimal("0.005")
    assert trade.net_profit_loss == Decimal(0)
    assert trade.dividend_income == Decimal(5)
    assert trade.total_economic_profit_loss == Decimal(5)
    assert trade.total_economic_return == Decimal("0.01")
    assert result.performance.win_rate == Decimal(1)
    assert result.daily_equity[2].dividend_cashflow_ids == (
        strategy_cashflow.dividend_cashflow_id,
    )


def test_buying_at_open_on_ex_date_does_not_receive_dividend() -> None:
    dataset = make_dataset(
        ("100", "100", "100", "100"),
        dividends=((date(2024, 7, 2), "1"),),
    )

    result = run_backtest(dataset, ManualTransitionStrategy(), zero_cost_config())

    assert result.dividend_cashflows[0].entitled_share_quantity == 0
    assert result.dividend_cashflows[0].total_dividend_cash == Decimal(0)
    assert result.completed_trades[0].dividend_income == Decimal(0)
    assert result.benchmark.dividend_cashflows[0].entitled_share_quantity == 10


def test_selling_at_open_on_ex_date_retains_dividend_entitlement() -> None:
    dataset = make_dataset(
        ("100", "100", "100", "100"),
        dividends=((date(2024, 7, 5), "1"),),
    )

    result = run_backtest(dataset, ManualTransitionStrategy(), zero_cost_config())

    cashflow = result.dividend_cashflows[0]
    assert result.fills[-1].execution_session == date(2024, 7, 5)
    assert cashflow.entitled_share_quantity == 5
    assert cashflow.resulting_cash_balance == Decimal(1005)
    assert result.completed_trades[0].dividend_income == Decimal(5)


def test_first_session_ex_date_does_not_entitle_first_open_buys() -> None:
    dataset = make_dataset(
        ("100", "100", "100", "100"),
        dividends=((date(2024, 7, 1), "1"),),
    )

    result = run_backtest(dataset, ManualTransitionStrategy(), zero_cost_config())

    assert result.dividend_cashflows[0].entitled_share_quantity == 0
    assert result.benchmark.dividend_cashflows[0].entitled_share_quantity == 0
    assert result.performance.total_dividend_income == Decimal(0)
    assert result.benchmark.performance.total_dividend_income == Decimal(0)


def test_price_return_only_ignores_cash_with_complete_disclosure() -> None:
    dataset = make_dataset(
        ("100", "100", "100", "100"),
        dividends=((date(2024, 7, 3), "1"),),
    )

    price_only = run_backtest(
        dataset,
        ManualTransitionStrategy(),
        zero_cost_config(DividendPolicy.PRICE_RETURN_ONLY),
    )
    cash_dividends = run_backtest(
        dataset,
        ManualTransitionStrategy(),
        zero_cost_config(DividendPolicy.CASH_DIVIDENDS),
    )

    strategy_summary = price_only.dividend_accounting
    benchmark_summary = price_only.benchmark.dividend_accounting
    assert price_only.run_id != cash_dividends.run_id
    assert price_only.dividend_cashflows == ()
    assert price_only.benchmark.dividend_cashflows == ()
    assert price_only.performance.total_dividend_income == Decimal(0)
    assert price_only.performance.ending_equity == Decimal(1000)
    assert cash_dividends.performance.ending_equity == Decimal(1005)
    assert strategy_summary.dividend_policy is DividendPolicy.PRICE_RETURN_ONLY
    assert strategy_summary.return_basis is ReturnBasis.PRICE_RETURN
    assert (
        strategy_summary.corporate_action_snapshot_id
        == dataset.metadata.corporate_action_snapshot_id
    )
    assert strategy_summary.dividend_events_present == 1
    assert strategy_summary.dividend_events_credited == 0
    assert strategy_summary.dividend_events_ignored == 1
    assert strategy_summary.total_dividend_cash_credited == Decimal(0)
    assert strategy_summary.estimated_ignored_dividend_cash == Decimal(5)
    assert strategy_summary.warning is not None
    assert price_only.warnings == (strategy_summary.warning,)
    assert benchmark_summary.dividend_policy is DividendPolicy.PRICE_RETURN_ONLY
    assert benchmark_summary.return_basis is ReturnBasis.PRICE_RETURN
    assert (
        benchmark_summary.corporate_action_snapshot_id
        == dataset.metadata.corporate_action_snapshot_id
    )
    assert benchmark_summary.dividend_events_ignored == 1
    assert benchmark_summary.estimated_ignored_dividend_cash == Decimal(10)
    assert price_only.completed_trades[0].dividend_income == Decimal(0)
    assert price_only.completed_trades[0].total_economic_profit_loss == Decimal(0)
    assert (
        cash_dividends.dividend_accounting.return_basis
        is ReturnBasis.TOTAL_RETURN_WITH_CASH_DIVIDENDS
    )
    assert cash_dividends.dividend_accounting.dividend_events_credited == 1
    assert cash_dividends.dividend_accounting.dividend_events_ignored == 0
    assert cash_dividends.benchmark.dividend_accounting.dividend_events_credited == 1


def test_price_return_only_does_not_sum_dividends_across_split_share_units() -> None:
    result = run_backtest(
        make_dataset(
            ("100", "100", "50", "50"),
            dividends=((date(2024, 7, 2), "1"), (date(2024, 7, 5), "0.60")),
            splits=((date(2024, 7, 3), "2"),),
        ),
        ManualTransitionStrategy(),
        zero_cost_config(DividendPolicy.PRICE_RETURN_ONLY),
    )

    strategy_summary = result.dividend_accounting
    benchmark_summary = result.benchmark.dividend_accounting
    assert strategy_summary.dividend_events_ignored == 2
    assert strategy_summary.estimated_ignored_dividend_cash == Decimal(6)
    assert benchmark_summary.estimated_ignored_dividend_cash == Decimal(22)
    assert (
        "total_ignored_dividend_amount_per_share" not in strategy_summary.to_primitive()
    )
    assert (
        "total_ignored_dividend_amount_per_share"
        not in benchmark_summary.to_primitive()
    )


def test_reject_if_dividends_requires_an_explicit_alternative() -> None:
    dataset = make_dataset(
        ("100", "100", "100", "100"),
        dividends=((date(2024, 7, 3), "1"),),
    )

    with pytest.raises(
        InvalidMarketDataError,
        match=r"select DividendPolicy\.PRICE_RETURN_ONLY or "
        r"DividendPolicy\.CASH_DIVIDENDS",
    ):
        run_backtest(
            dataset,
            ManualTransitionStrategy(),
            zero_cost_config(DividendPolicy.REJECT_IF_DIVIDENDS),
        )


@pytest.mark.parametrize("dividend_policy", list(DividendPolicy))
def test_dividend_free_raw_data_runs_under_every_policy(
    dividend_policy: DividendPolicy,
) -> None:
    result = run_backtest(
        make_dataset(("100", "100", "100", "100")),
        ManualTransitionStrategy(),
        zero_cost_config(dividend_policy),
    )

    assert result.dividend_accounting.dividend_policy is dividend_policy
    assert result.dividend_accounting.dividend_events_present == 0
    assert result.dividend_accounting.warning is None


@pytest.mark.parametrize("dividend_policy", list(DividendPolicy))
def test_inconsistent_raw_price_metadata_is_rejected_under_every_policy(
    dividend_policy: DividendPolicy,
) -> None:
    dataset = make_dataset(("100", "100", "100", "100"))
    inconsistent = replace(
        dataset,
        metadata=replace(dataset.metadata, ohlc_basis="provider_adjusted"),
    )

    with pytest.raises(InvalidMarketDataError, match="price or volume basis"):
        run_backtest(
            inconsistent,
            ManualTransitionStrategy(),
            zero_cost_config(dividend_policy),
        )


@pytest.mark.parametrize("dividend_policy", list(DividendPolicy))
def test_split_accounting_remains_active_under_every_dividend_policy(
    dividend_policy: DividendPolicy,
) -> None:
    result = run_backtest(
        make_dataset(
            ("100", "100", "50", "50"),
            splits=((date(2024, 7, 3), "2"),),
        ),
        ManualTransitionStrategy(),
        zero_cost_config(dividend_policy),
    )

    assert result.split_adjustments[0].shares_before == 5
    assert result.split_adjustments[0].shares_after == 10
    assert result.benchmark.split_adjustments[0].shares_before == 10
    assert result.benchmark.split_adjustments[0].shares_after == 20


def test_export_preserves_dividend_policy_return_basis_and_disclosures(
    tmp_path: Path,
) -> None:
    result = run_backtest(
        make_dataset(
            ("100", "100", "100", "100"),
            dividends=((date(2024, 7, 3), "1"),),
        ),
        ManualTransitionStrategy(),
        zero_cost_config(DividendPolicy.PRICE_RETURN_ONLY),
    )

    exported = export_backtest_result(result, tmp_path)
    manifest = load_backtest_manifest(exported / "manifest.json")
    action_accounting = cast(PrimitiveMapping, manifest["corporate_action_accounting"])
    strategy_dividends = cast(PrimitiveMapping, action_accounting["dividends"])
    benchmark = cast(PrimitiveMapping, manifest["benchmark"])
    benchmark_dividends = cast(PrimitiveMapping, benchmark["dividend_accounting"])
    assert strategy_dividends["dividend_policy"] == "price_return_only"
    assert strategy_dividends["return_basis"] == "price_return"
    assert (
        strategy_dividends["corporate_action_snapshot_id"]
        == result.market_data.corporate_action_snapshot_id
    )
    assert strategy_dividends["dividend_events_ignored"] == 1
    assert strategy_dividends["estimated_ignored_dividend_cash"] == "5"
    assert benchmark_dividends["dividend_policy"] == "price_return_only"
    assert (
        benchmark_dividends["corporate_action_snapshot_id"]
        == result.market_data.corporate_action_snapshot_id
    )
    assert benchmark_dividends["estimated_ignored_dividend_cash"] == "10"
    assert result.backtest_configuration["dividend_policy"] == "price_return_only"
    json.dumps(result.to_primitive(), allow_nan=False, sort_keys=True)


def test_split_preserves_cash_equity_and_aggregate_cost_basis() -> None:
    dataset = make_dataset(
        ("100", "100", "50", "50"),
        splits=((date(2024, 7, 3), "2"),),
    )

    result = run_backtest(dataset, ManualTransitionStrategy(), zero_cost_config())

    adjustment = result.split_adjustments[0]
    benchmark_adjustment = result.benchmark.split_adjustments[0]
    trade = result.completed_trades[0]
    assert adjustment.shares_before == 5
    assert adjustment.shares_after == 10
    assert adjustment.average_entry_cost_before == Decimal(100)
    assert adjustment.average_entry_cost_after == Decimal(50)
    assert adjustment.total_cost_basis_before == Decimal(500)
    assert adjustment.total_cost_basis_after == Decimal(500)
    assert adjustment.resulting_cash_balance == Decimal(500)
    assert adjustment.corporate_action_id == dataset.corporate_actions[0].action_id
    assert adjustment.source_dataset_id == dataset.metadata.dataset_id
    assert result.daily_equity[1].total_equity == Decimal(1000)
    assert result.daily_equity[2].total_equity == Decimal(1000)
    assert trade.entry_quantity == 5
    assert trade.exit_quantity == 10
    assert trade.net_profit_loss == Decimal(0)
    assert benchmark_adjustment.shares_before == 10
    assert benchmark_adjustment.shares_after == 20


def test_same_session_dividend_entitlement_precedes_split_adjustment() -> None:
    dataset = make_dataset(
        ("100", "100", "50", "50"),
        dividends=((date(2024, 7, 3), "1"),),
        splits=((date(2024, 7, 3), "2"),),
    )

    result = run_backtest(dataset, ManualTransitionStrategy(), zero_cost_config())

    assert result.dividend_cashflows[0].entitled_share_quantity == 5
    assert result.dividend_cashflows[0].total_dividend_cash == Decimal(5)
    assert result.split_adjustments[0].shares_after == 10


def test_fractional_split_shares_are_rejected_without_cash_in_lieu() -> None:
    dataset = make_dataset(
        ("100", "100", "66.6666666667", "66.6666666667"),
        splits=((date(2024, 7, 3), "1.5"),),
    )

    with pytest.raises(PortfolioAccountingError, match="fractional shares"):
        run_backtest(dataset, ManualTransitionStrategy(), zero_cost_config())


def test_reverse_split_float_factor_recovers_integral_share_ratio() -> None:
    dataset = make_dataset(
        ("100", "100", "300", "300"),
        splits=((date(2024, 7, 3), "0.3333333333333333"),),
    )
    config = BacktestConfig(
        Decimal(600),
        FixedCommission(Decimal(0)),
        ExplicitZeroFees(),
        BasisPointSlippage(Decimal(0)),
        dividend_policy=DividendPolicy.CASH_DIVIDENDS,
    )

    result = run_backtest(dataset, ManualTransitionStrategy(), config)

    strategy_adjustment = result.split_adjustments[0]
    benchmark_adjustment = result.benchmark.split_adjustments[0]
    assert strategy_adjustment.split_factor == Decimal("0.3333333333333333")
    assert strategy_adjustment.split_ratio_numerator == 1
    assert strategy_adjustment.split_ratio_denominator == 3
    assert strategy_adjustment.shares_before == 3
    assert strategy_adjustment.shares_after == 1
    assert benchmark_adjustment.shares_before == 6
    assert benchmark_adjustment.shares_after == 2
    assert result.daily_equity[2].total_equity == Decimal(600)


def test_non_roundtripping_reverse_split_factor_remains_fractional() -> None:
    dataset = make_dataset(
        ("100", "100", "299.9400119976", "299.9400119976"),
        splits=((date(2024, 7, 3), "0.3334"),),
    )
    config = BacktestConfig(
        Decimal(600),
        FixedCommission(Decimal(0)),
        ExplicitZeroFees(),
        BasisPointSlippage(Decimal(0)),
        dividend_policy=DividendPolicy.CASH_DIVIDENDS,
    )

    with pytest.raises(PortfolioAccountingError, match="fractional shares"):
        run_backtest(dataset, ManualTransitionStrategy(), config)


def test_adjusted_or_incomplete_action_data_is_rejected() -> None:
    adjusted = make_dataset(
        ("100", "100", "100", "100"),
        adjustment_mode=AdjustmentMode.SPLIT_ADJUSTED,
        dividends=((date(2024, 7, 3), "1"),),
    )
    incomplete = make_dataset(
        ("100", "100", "100", "100"),
        corporate_actions_complete=False,
    )

    with pytest.raises(InvalidMarketDataError, match="adjusted market data"):
        run_backtest(adjusted, ManualTransitionStrategy(), zero_cost_config())
    with pytest.raises(
        InvalidMarketDataError,
        match="requires complete explicit corporate actions",
    ):
        run_backtest(incomplete, ManualTransitionStrategy(), zero_cost_config())


def test_corporate_action_snapshot_changes_dataset_and_run_identity(
    tmp_path: Path,
) -> None:
    first_dataset = make_dataset(
        ("100", "100", "100", "100"),
        dividends=((date(2024, 7, 3), "1"),),
    )
    equivalent_dataset = make_dataset(
        ("100", "100", "100", "100"),
        dividends=((date(2024, 7, 3), "1"),),
    )
    revised_dataset = make_dataset(
        ("100", "100", "100", "100"),
        dividends=((date(2024, 7, 3), "1.01"),),
    )

    first = run_backtest(first_dataset, ManualTransitionStrategy(), zero_cost_config())
    equivalent = run_backtest(
        equivalent_dataset, ManualTransitionStrategy(), zero_cost_config()
    )
    revised = run_backtest(
        revised_dataset, ManualTransitionStrategy(), zero_cost_config()
    )

    assert first == equivalent
    assert first_dataset.metadata.dataset_id == equivalent_dataset.metadata.dataset_id
    assert (
        first_dataset.metadata.corporate_action_snapshot_id
        == equivalent_dataset.metadata.corporate_action_snapshot_id
    )
    assert revised_dataset.metadata.dataset_id != first_dataset.metadata.dataset_id
    assert (
        revised_dataset.metadata.corporate_action_snapshot_id
        != first_dataset.metadata.corporate_action_snapshot_id
    )
    assert revised.run_id != first.run_id
    first_export = export_backtest_result(first, tmp_path / "first")
    equivalent_export = export_backtest_result(equivalent, tmp_path / "equivalent")
    assert [path.name for path in first_export.iterdir()] == [
        path.name for path in equivalent_export.iterdir()
    ]
    assert {path.name: path.read_bytes() for path in first_export.iterdir()} == {
        path.name: path.read_bytes() for path in equivalent_export.iterdir()
    }


def test_split_factor_changes_snapshot_and_run_identity() -> None:
    first_dataset = make_dataset(
        ("100", "100", "50", "50"),
        splits=((date(2024, 7, 3), "2"),),
    )
    revised_dataset = make_dataset(
        (
            "100",
            "100",
            "33.33333333333333333333333333",
            "33.33333333333333333333333333",
        ),
        splits=((date(2024, 7, 3), "3"),),
    )

    first = run_backtest(first_dataset, ManualTransitionStrategy(), zero_cost_config())
    revised = run_backtest(
        revised_dataset, ManualTransitionStrategy(), zero_cost_config()
    )

    assert (
        first_dataset.metadata.corporate_action_snapshot_id
        != revised_dataset.metadata.corporate_action_snapshot_id
    )
    assert first.run_id != revised.run_id
