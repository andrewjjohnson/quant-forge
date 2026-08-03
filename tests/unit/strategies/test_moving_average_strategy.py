import json
from datetime import date
from decimal import ROUND_DOWN, ROUND_UP, Decimal, localcontext

from quantforge.strategies import (
    ExecutionSessionStatus,
    ExecutionTiming,
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
    PositionIntent,
    Strategy,
    StrategyOutput,
    run_strategy,
)

from ..helpers import make_dataset

PRICES = ("3", "2", "1", "2", "3", "4", "3", "2", "1")


def run_through_contract(strategy: Strategy, prices: tuple[str, ...]) -> StrategyOutput:
    return run_strategy(strategy, make_dataset(prices))


def test_upward_and_downward_crossovers_emit_single_ordered_state_changes() -> None:
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))

    output = run_through_contract(strategy, PRICES)

    assert [decision.signal_session for decision in output] == [
        date(2024, 7, 8),
        date(2024, 7, 11),
    ]
    assert [decision.target_position for decision in output] == [
        PositionIntent.LONG,
        PositionIntent.FLAT,
    ]
    assert [decision.target_weight for decision in output] == [Decimal(1), Decimal(0)]
    assert output.decisions[0].reason == (
        "fast moving average crossed above slow moving average"
    )
    assert {item.name for item in output.decisions[0].indicator_values} == {
        "fast_sma",
        "previous_fast_sma",
        "previous_slow_sma",
        "slow_sma",
    }


def test_baseline_begins_flat_and_does_not_emit_downward_or_repeated_targets() -> None:
    always_falling = ("6", "5", "4", "3", "2", "1")
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))

    assert run_through_contract(strategy, always_falling).decisions == ()
    assert len(run_through_contract(strategy, PRICES).decisions) == 2


def test_no_signal_on_first_valid_pair_or_during_warmup() -> None:
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))

    assert strategy.warm_up_observations == 3
    assert run_through_contract(strategy, ("1", "2")).decisions == ()
    assert run_through_contract(strategy, ("1", "2", "3")).decisions == ()


def test_equality_row_does_not_emit_but_leaving_equality_can_confirm_cross() -> None:
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))

    through_equality = run_through_contract(strategy, ("3", "2", "1", "3"))
    after_cross = run_through_contract(strategy, ("3", "2", "1", "3", "4"))

    assert through_equality.decisions == ()
    assert len(after_cross.decisions) == 1
    assert after_cross.decisions[0].signal_session == date(2024, 7, 8)


def test_signal_timing_is_next_exchange_session_and_never_same_close() -> None:
    sessions = (
        date(2024, 6, 28),
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
    )
    dataset = make_dataset(("4", "3", "1", "6"), sessions=sessions)
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))

    decision = run_strategy(strategy, dataset).decisions[0]

    assert decision.signal_session == date(2024, 7, 3)
    assert decision.earliest_executable_session == date(2024, 7, 5)
    assert decision.execution_timing is ExecutionTiming.NEXT_SESSION_AFTER_CLOSE
    assert decision.execution_session_status is ExecutionSessionStatus.PENDING
    assert decision.earliest_executable_session is not None
    assert decision.earliest_executable_session > decision.signal_session


def test_final_session_signal_is_calendar_resolved_but_remains_pending() -> None:
    dataset = make_dataset(("3", "2", "1", "2", "3"))
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))

    decision = run_strategy(strategy, dataset).decisions[0]

    assert decision.signal_session == dataset.bars[-1].session_date
    assert decision.earliest_executable_session == date(2024, 7, 9)
    assert decision.execution_session_status is ExecutionSessionStatus.PENDING
    assert "price" not in decision.to_primitive()


def test_next_session_timing_skips_a_weekend() -> None:
    sessions = (
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
    )
    dataset = make_dataset(("4", "3", "1", "6"), sessions=sessions)
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))

    decision = run_strategy(strategy, dataset).decisions[0]

    assert decision.signal_session == date(2024, 7, 5)
    assert decision.earliest_executable_session == date(2024, 7, 8)


def test_output_supports_tabular_and_event_consumers_and_json_serialization() -> None:
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))
    output = run_through_contract(strategy, PRICES)

    rows = output.to_rows()
    chronological = tuple(output)

    assert rows[0]["target_position"] == "long"
    assert chronological == output.decisions
    assert json.loads(json.dumps(output.to_primitive()))["contract_version"] == "1"
    assert output.market_data.adjustment_mode == "split_adjusted"


def test_future_bars_do_not_change_historical_signals() -> None:
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))
    through_cutoff = run_strategy(
        strategy,
        make_dataset(("3", "2", "1", "2", "3"), dataset_id="cutoff"),
    )
    with_future = run_strategy(
        strategy,
        make_dataset(PRICES, dataset_id="with-future"),
    )

    historical = tuple(
        decision
        for decision in with_future
        if decision.signal_session <= date(2024, 7, 8)
    )
    assert historical == through_cutoff.decisions


def test_strategy_decisions_ignore_ambient_decimal_context() -> None:
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))

    with localcontext() as low_precision:
        low_precision.prec = 8
        low_precision.rounding = ROUND_DOWN
        low_decisions = run_through_contract(strategy, PRICES).decisions
    with localcontext() as high_precision:
        high_precision.prec = 50
        high_precision.rounding = ROUND_UP
        high_decisions = run_through_contract(strategy, PRICES).decisions

    assert low_decisions == high_decisions
