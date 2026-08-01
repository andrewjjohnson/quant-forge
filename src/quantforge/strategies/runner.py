"""Minimal generic strategy invocation and output validation."""

from itertools import pairwise

from quantforge.data.models import MarketDataset
from quantforge.strategies.base import Strategy, next_exchange_session
from quantforge.strategies.exceptions import (
    DuplicateStrategyDecisionError,
    InvalidStrategyOutputError,
    MissingRequiredMarketFieldError,
    UnorderedStrategyInputError,
)
from quantforge.strategies.models import (
    ExecutionSessionStatus,
    MarketDataReference,
    StrategyOutput,
)


def run_strategy(strategy: Strategy, dataset: MarketDataset) -> StrategyOutput:
    """Run any strategy contract without indicator- or strategy-specific branches."""
    _validate_input(strategy, dataset)
    output = strategy.generate(dataset)
    _validate_output(strategy, dataset, output)
    return output


def _validate_input(strategy: Strategy, dataset: MarketDataset) -> None:
    sessions = tuple(bar.session_date for bar in dataset.bars)
    if any(current >= following for current, following in pairwise(sessions)):
        raise UnorderedStrategyInputError(
            "market sessions must be unique and strictly chronological"
        )
    for field in sorted(strategy.required_fields, key=str):
        if any(not hasattr(bar, field.value) for bar in dataset.bars):
            raise MissingRequiredMarketFieldError(
                f"market data is missing required field: {field.value}"
            )


def _validate_output(
    strategy: Strategy, dataset: MarketDataset, output: StrategyOutput
) -> None:
    if (
        output.contract_version != "1"
        or output.strategy_id != strategy.name
        or output.strategy_configuration_id != strategy.configuration_id
    ):
        raise InvalidStrategyOutputError(
            "strategy output identity does not match the invoked strategy"
        )
    if output.market_data != MarketDataReference.from_dataset(dataset):
        raise InvalidStrategyOutputError(
            "strategy output does not reference the input market dataset"
        )
    expected_order = tuple(
        sorted(
            output.decisions,
            key=lambda decision: (
                decision.signal_session,
                decision.canonical_symbol,
                decision.strategy_id,
            ),
        )
    )
    if output.decisions != expected_order:
        raise InvalidStrategyOutputError(
            "strategy decisions must be deterministically ordered"
        )
    keys = tuple(
        (
            decision.strategy_id,
            decision.canonical_symbol,
            decision.signal_session,
        )
        for decision in output.decisions
    )
    if len(keys) != len(set(keys)):
        raise DuplicateStrategyDecisionError(
            "duplicate strategy, symbol, and session decision"
        )
    available_sessions = {bar.session_date for bar in dataset.bars}
    for decision in output.decisions:
        if (
            decision.strategy_id != strategy.name
            or decision.strategy_configuration_id != strategy.configuration_id
            or decision.canonical_symbol != dataset.metadata.canonical_symbol
            or decision.signal_session not in available_sessions
            or decision.execution_timing is not strategy.timing
        ):
            raise InvalidStrategyOutputError(
                "decision identity or signal session does not match strategy input"
            )
        if decision.earliest_executable_session is not None:
            expected_session = next_exchange_session(
                decision.signal_session, dataset.metadata.calendar
            )
            if (
                decision.earliest_executable_session != expected_session
                or decision.execution_session_status
                is not ExecutionSessionStatus.PENDING
            ):
                raise InvalidStrategyOutputError(
                    "decision does not use the next exchange-session convention"
                )
