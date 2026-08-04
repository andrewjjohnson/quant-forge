"""Minimal generic strategy invocation and output validation."""

from itertools import pairwise
from typing import cast

from quantforge.configuration import PrimitiveMapping, PrimitiveScalar
from quantforge.data.models import MarketDataset
from quantforge.strategies.base import Strategy, next_exchange_session
from quantforge.strategies.exceptions import (
    DuplicateStrategyDecisionError,
    InvalidStrategyOutputError,
    MissingRequiredMarketFieldError,
    UnorderedStrategyInputError,
    UnsupportedTimingConventionError,
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
    implementation_version = cast(object, strategy.implementation_version)
    if (
        not isinstance(implementation_version, str)
        or not implementation_version.strip()
        or strategy.configuration().get("implementation_version")
        != implementation_version
    ):
        raise InvalidStrategyOutputError(
            "strategy implementation version must be explicit in configuration"
        )
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
    expected_parameters = strategy.parameters.to_primitive()
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
        parameter_snapshot = {
            parameter.name: parameter.value
            for parameter in decision.strategy_parameters
        }
        if not _parameter_snapshots_match(parameter_snapshot, expected_parameters):
            raise InvalidStrategyOutputError(
                "decision parameter snapshot does not match the invoked strategy"
            )
        try:
            expected_session = next_exchange_session(
                decision.signal_session, dataset.metadata.calendar
            )
        except UnsupportedTimingConventionError:
            if (
                decision.earliest_executable_session is not None
                or decision.execution_session_status
                is not ExecutionSessionStatus.UNRESOLVED
            ):
                raise InvalidStrategyOutputError(
                    "decision must be unresolved when its next exchange session "
                    "cannot be resolved"
                ) from None
        else:
            if (
                decision.earliest_executable_session != expected_session
                or decision.execution_session_status
                is not ExecutionSessionStatus.PENDING
            ):
                raise InvalidStrategyOutputError(
                    "decision does not use the next exchange-session convention"
                )


def _parameter_snapshots_match(
    actual: dict[str, PrimitiveScalar], expected: PrimitiveMapping
) -> bool:
    """Compare parameter names, primitive types, and values exactly."""
    return actual.keys() == expected.keys() and all(
        type(actual[name]) is type(expected[name]) and actual[name] == expected[name]
        for name in expected
    )
