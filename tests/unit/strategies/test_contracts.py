from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data.models import DailyBar, MarketDataset
from quantforge.indicators import Indicator, MarketField
from quantforge.strategies import (
    DuplicateStrategyDecisionError,
    ExecutionSessionStatus,
    ExecutionTiming,
    InvalidStrategyOutputError,
    MissingRequiredMarketFieldError,
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
    PositionSizingPolicy,
    StrategyOutput,
    StrategyParameters,
    UnorderedStrategyInputError,
    UnsupportedTimingConventionError,
    run_strategy,
)
from quantforge.strategies.base import next_exchange_session

from ..helpers import make_dataset


@dataclass(frozen=True, slots=True)
class SessionOnlyBar:
    session_date: date


class OutputTransformingStrategy:
    """Contract implementation that transforms a delegate's output for validation."""

    def __init__(
        self,
        delegate: MovingAverageCrossoverStrategy,
        transform: Callable[[StrategyOutput], StrategyOutput],
    ) -> None:
        self._delegate = delegate
        self._transform = transform

    @property
    def name(self) -> str:
        return self._delegate.name

    @property
    def parameters(self) -> StrategyParameters:
        return self._delegate.parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return self._delegate.required_fields

    @property
    def required_indicators(self) -> tuple[Indicator, ...]:
        return self._delegate.required_indicators

    @property
    def warm_up_observations(self) -> int:
        return self._delegate.warm_up_observations

    @property
    def timing(self) -> ExecutionTiming:
        return self._delegate.timing

    @property
    def sizing_policy(self) -> PositionSizingPolicy:
        return self._delegate.sizing_policy

    @property
    def asset_assumptions(self) -> tuple[str, ...]:
        return self._delegate.asset_assumptions

    @property
    def configuration_id(self) -> str:
        return self._delegate.configuration_id

    def configuration(self) -> PrimitiveMapping:
        return self._delegate.configuration()

    def generate(self, dataset: MarketDataset) -> StrategyOutput:
        return self._transform(self._delegate.generate(dataset))


def duplicate_first_decision(output: StrategyOutput) -> StrategyOutput:
    decision = output.decisions[0]
    return replace(output, decisions=(decision, decision))


def mark_first_decision_unresolved(output: StrategyOutput) -> StrategyOutput:
    decision = replace(
        output.decisions[0],
        earliest_executable_session=None,
        execution_session_status=ExecutionSessionStatus.UNRESOLVED,
    )
    return replace(output, decisions=(decision,))


def clear_first_decision_parameters(output: StrategyOutput) -> StrategyOutput:
    decision = replace(output.decisions[0], strategy_parameters=())
    return replace(output, decisions=(decision,))


def test_generic_runner_enforces_required_market_fields() -> None:
    dataset = make_dataset(("1",))
    incomplete = replace(
        dataset,
        bars=cast(tuple[DailyBar, ...], (SessionOnlyBar(date(2024, 7, 1)),)),
    )
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))

    with pytest.raises(MissingRequiredMarketFieldError, match="close"):
        run_strategy(strategy, incomplete)


def test_generic_runner_rejects_unordered_input() -> None:
    dataset = make_dataset(("1", "2"))
    unordered = replace(dataset, bars=tuple(reversed(dataset.bars)))
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))

    with pytest.raises(UnorderedStrategyInputError, match="chronological"):
        run_strategy(strategy, unordered)


def test_generic_runner_rejects_duplicate_decisions() -> None:
    strategy = OutputTransformingStrategy(
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        duplicate_first_decision,
    )

    with pytest.raises(DuplicateStrategyDecisionError, match="duplicate"):
        run_strategy(strategy, make_dataset(("3", "2", "1", "2", "3")))


def test_generic_runner_rejects_falsely_unresolved_execution_session() -> None:
    strategy = OutputTransformingStrategy(
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        mark_first_decision_unresolved,
    )

    with pytest.raises(InvalidStrategyOutputError, match="next exchange-session"):
        run_strategy(strategy, make_dataset(("3", "2", "1", "2", "3")))


def test_generic_runner_accepts_unresolved_session_when_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = OutputTransformingStrategy(
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        mark_first_decision_unresolved,
    )

    def fail_to_resolve_next_session(signal_session: date, calendar: str) -> date:
        raise UnsupportedTimingConventionError(
            f"cannot resolve {calendar} after {signal_session}"
        )

    monkeypatch.setattr(
        "quantforge.strategies.runner.next_exchange_session",
        fail_to_resolve_next_session,
    )

    decision = run_strategy(
        strategy, make_dataset(("3", "2", "1", "2", "3"))
    ).decisions[0]

    assert decision.earliest_executable_session is None
    assert decision.execution_session_status is ExecutionSessionStatus.UNRESOLVED


def test_generic_runner_rejects_incorrect_parameter_snapshot() -> None:
    strategy = OutputTransformingStrategy(
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        clear_first_decision_parameters,
    )

    with pytest.raises(InvalidStrategyOutputError, match="parameter snapshot"):
        run_strategy(strategy, make_dataset(("3", "2", "1", "2", "3")))


def test_decision_constructor_rejects_same_session_execution() -> None:
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3))
    decision = run_strategy(
        strategy, make_dataset(("3", "2", "1", "2", "3"))
    ).decisions[0]

    with pytest.raises(InvalidStrategyOutputError, match="later"):
        replace(
            decision,
            earliest_executable_session=decision.signal_session,
            execution_session_status=ExecutionSessionStatus.PENDING,
        )


def test_strategy_contract_exposes_owned_components_and_configuration() -> None:
    strategy = MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 5))

    assert strategy.required_fields == frozenset((MarketField.CLOSE,))
    assert len(strategy.required_indicators) == 2
    assert strategy.warm_up_observations == 5
    assert strategy.timing is ExecutionTiming.NEXT_SESSION_AFTER_CLOSE
    assert strategy.configuration()["sizing"] == strategy.sizing_policy.configuration()


def test_unknown_calendar_raises_timing_domain_error() -> None:
    with pytest.raises(UnsupportedTimingConventionError, match="calendar"):
        next_exchange_session(date(2024, 7, 3), "NOT_A_CALENDAR")
