"""Deterministic causal prediction generation and forward-label alignment."""

from datetime import date
from decimal import Decimal, DecimalException
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import dataset_identity_matches, validate_market_dataset
from quantforge.data.calendar import next_session_after
from quantforge.data.exceptions import ValidationError as MarketDataValidationError
from quantforge.data.models import SCHEMA_VERSION, AdjustmentMode, MarketDataset
from quantforge.prediction._arithmetic import (
    DECIMAL_CAPITALS,
    DECIMAL_CLAMP,
    DECIMAL_EMAX,
    DECIMAL_EMIN,
    DECIMAL_PRECISION,
    DECIMAL_ROUNDING,
    DECIMAL_TRAPS,
    arithmetic,
)
from quantforge.prediction.base import PredictionStrategy
from quantforge.prediction.errors import (
    InvalidPredictionDataError,
    InvalidPredictionOutputError,
)
from quantforge.prediction.models import (
    PredictionAnalysisResult,
    PredictionDirection,
    PredictionMarketData,
    PredictionMetrics,
    PredictionRow,
    PredictionStrategyOutput,
)

ENGINE_VERSION = "1"
RESULT_SCHEMA_VERSION = "1"
LIMITATIONS = (
    "direction accuracy is descriptive and is not an executable trade backtest",
    "the signal close is a prediction anchor, not a claimed fill price",
    "option prices, spreads, Greeks, and intraday exits are not modeled",
    "observed gaps include ex-dividend price effects and are not total returns",
    "raw datasets containing stock splits are rejected to avoid mechanical gaps",
    "flat zero-size gaps count as incorrect predictions",
)


def run_prediction_analysis(
    dataset: MarketDataset,
    strategy: PredictionStrategy,
) -> PredictionAnalysisResult:
    """Generate causal signals, then label them from the next session's open."""
    _validate_dataset(dataset)
    strategy_configuration_snapshot = PrimitiveMappingSnapshot.capture(
        strategy.configuration()
    )
    strategy_configuration = strategy_configuration_snapshot.to_primitive()
    strategy_configuration_id = _validate_strategy_identity(
        strategy, strategy_configuration
    )
    market_data = PredictionMarketData.from_qf3(dataset.metadata)
    analysis_configuration_snapshot = PrimitiveMappingSnapshot.capture(
        _analysis_configuration()
    )
    analysis_configuration = analysis_configuration_snapshot.to_primitive()
    analysis_id = _stable_id(
        {
            "component": "quantforge_prediction_analysis",
            "engine_version": ENGINE_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "market_data": market_data.to_primitive(),
            "strategy": {
                "strategy_id": strategy.name,
                "strategy_implementation_version": strategy.implementation_version,
                "strategy_configuration_id": strategy_configuration_id,
                "configuration": strategy_configuration,
            },
            "analysis_configuration": analysis_configuration,
        }
    )

    output = strategy.generate(dataset)
    current_configuration = strategy.configuration()
    if configuration_identity(current_configuration) != strategy_configuration_id:
        raise InvalidPredictionOutputError(
            "prediction strategy configuration changed during generation"
        )
    _validate_output(dataset, strategy, output, strategy_configuration_id)

    rows: list[PredictionRow] = []
    unlabeled_end_of_data_count = 0
    bar_indexes = {bar.session_date: index for index, bar in enumerate(dataset.bars)}
    for signal in output.signals:
        signal_index = bar_indexes[signal.signal_session]
        if signal_index == len(dataset.bars) - 1:
            unlabeled_end_of_data_count += 1
            continue
        signal_bar = dataset.bars[signal_index]
        outcome_bar = dataset.bars[signal_index + 1]
        expected_outcome_session = _next_session(
            signal.signal_session, dataset.metadata.calendar
        )
        if outcome_bar.session_date != expected_outcome_session:
            raise InvalidPredictionDataError(
                "a prediction outcome requires the immediate next exchange session"
            )
        try:
            with arithmetic():
                overnight_gap = outcome_bar.open / signal_bar.close - Decimal(1)
                gap_size = abs(overnight_gap)
                signed_return = (
                    overnight_gap
                    if signal.direction is PredictionDirection.UP
                    else -overnight_gap
                )
        except DecimalException as error:
            raise InvalidPredictionOutputError(
                "prediction label arithmetic failed under its configured policy"
            ) from error
        prediction_id = _stable_id(
            {
                "analysis_id": analysis_id,
                "record_type": "prediction",
                "symbol": signal.symbol,
                "signal_session": signal.signal_session.isoformat(),
                "outcome_session": outcome_bar.session_date.isoformat(),
            }
        )
        rows.append(
            PredictionRow(
                prediction_id=prediction_id,
                dataset_id=market_data.dataset_id,
                dataset_fingerprint=market_data.bars_fingerprint,
                symbol=signal.symbol,
                signal_session=signal.signal_session,
                outcome_session=outcome_bar.session_date,
                direction=signal.direction,
                strategy_id=signal.strategy_id,
                strategy_implementation_version=(
                    signal.strategy_implementation_version
                ),
                strategy_configuration_id=signal.strategy_configuration_id,
                strategy_parameters=signal.strategy_parameters,
                reason=signal.reason,
                feature_values=signal.feature_values,
                signal_close=signal_bar.close,
                next_open=outcome_bar.open,
                overnight_gap_percentage=overnight_gap,
                gap_size_percentage=gap_size,
                signed_prediction_return=signed_return,
                correct=signed_return > 0,
            )
        )

    metrics = _calculate_metrics(tuple(rows))
    return PredictionAnalysisResult(
        analysis_id=analysis_id,
        engine_version=ENGINE_VERSION,
        result_schema_version=RESULT_SCHEMA_VERSION,
        market_data=market_data,
        strategy_id=strategy.name,
        strategy_implementation_version=strategy.implementation_version,
        strategy_configuration_id=strategy_configuration_id,
        strategy_configuration_snapshot=strategy_configuration_snapshot,
        strategy_warm_up_observations=strategy.warm_up_observations,
        analysis_configuration_snapshot=analysis_configuration_snapshot,
        generated_signal_count=len(output.signals),
        unlabeled_end_of_data_count=unlabeled_end_of_data_count,
        rows=tuple(rows),
        metrics=metrics,
        limitations=LIMITATIONS,
    )


def _validate_dataset(dataset: MarketDataset) -> None:
    dataset_value = cast(object, dataset)
    if not isinstance(dataset_value, MarketDataset) or not dataset_value.bars:
        raise InvalidPredictionDataError("a nonempty QF-3 MarketDataset is required")
    if dataset_value.metadata.schema_version != SCHEMA_VERSION:
        raise InvalidPredictionDataError(
            f"market data schema {SCHEMA_VERSION} is required"
        )
    try:
        missing_sessions = validate_market_dataset(dataset_value)
    except MarketDataValidationError as error:
        raise InvalidPredictionDataError(str(error)) from error
    if not dataset_identity_matches(dataset_value):
        raise InvalidPredictionDataError(
            "market bars and provenance do not reproduce the dataset identity"
        )
    metadata = dataset_value.metadata
    if (
        metadata.adjustment_mode is AdjustmentMode.UNADJUSTED
        and metadata.split_count > 0
    ):
        raise InvalidPredictionDataError(
            "raw unadjusted datasets containing stock splits are unsupported "
            "because a mechanical split must not be labeled as an overnight gap"
        )
    internal_missing = tuple(
        session
        for session in missing_sessions
        if metadata.actual_first_session < session < metadata.actual_last_session
    )
    if internal_missing:
        raise InvalidPredictionDataError(
            "prediction analysis does not permit missing sessions inside the "
            "observed range"
        )


def _validate_strategy_identity(
    strategy: PredictionStrategy, configuration: PrimitiveMapping
) -> str:
    expected_id = configuration_identity(configuration)
    declared_id = cast(object, strategy.configuration_id)
    implementation_version = cast(object, strategy.implementation_version)
    if (
        not isinstance(declared_id, str)
        or not declared_id
        or declared_id != expected_id
    ):
        raise InvalidPredictionOutputError(
            "prediction strategy configuration identity is invalid"
        )
    if (
        not isinstance(implementation_version, str)
        or not implementation_version
        or configuration.get("implementation_version") != implementation_version
    ):
        raise InvalidPredictionOutputError(
            "prediction strategy implementation version is invalid"
        )
    return declared_id


def _validate_output(
    dataset: MarketDataset,
    strategy: PredictionStrategy,
    output: PredictionStrategyOutput,
    expected_configuration_id: str,
) -> None:
    if (
        output.contract_version != "1"
        or output.strategy_id != strategy.name
        or output.strategy_configuration_id != expected_configuration_id
        or output.dataset_id != dataset.metadata.dataset_id
    ):
        raise InvalidPredictionOutputError(
            "prediction output identity does not match its strategy and dataset"
        )
    expected_order = tuple(
        sorted(
            output.signals, key=lambda signal: (signal.signal_session, signal.symbol)
        )
    )
    if output.signals != expected_order:
        raise InvalidPredictionOutputError(
            "prediction signals must be deterministically ordered"
        )
    sessions = tuple(signal.signal_session for signal in output.signals)
    if len(sessions) != len(set(sessions)):
        raise InvalidPredictionOutputError(
            "a strategy may emit at most one prediction per session"
        )
    available_sessions = {bar.session_date for bar in dataset.bars}
    expected_parameters = strategy.parameters.to_primitive()
    for signal in output.signals:
        if (
            signal.symbol != dataset.metadata.canonical_symbol
            or signal.signal_session not in available_sessions
            or signal.strategy_id != strategy.name
            or signal.strategy_implementation_version != strategy.implementation_version
            or signal.strategy_configuration_id != expected_configuration_id
            or not _parameter_snapshots_match(
                signal.parameters_primitive(), expected_parameters
            )
        ):
            raise InvalidPredictionOutputError(
                "prediction signal identity or parameter snapshot is invalid"
            )


def _calculate_metrics(rows: tuple[PredictionRow, ...]) -> PredictionMetrics:
    correct = tuple(row for row in rows if row.correct)
    incorrect = tuple(row for row in rows if not row.correct)
    try:
        with arithmetic():
            accuracy = None if not rows else Decimal(len(correct)) / Decimal(len(rows))
            average_gap_size_correct = _average(
                tuple(row.gap_size_percentage for row in correct)
            )
            average_gap_size_incorrect = _average(
                tuple(row.gap_size_percentage for row in incorrect)
            )
            average_signed_return_correct = _average(
                tuple(row.signed_prediction_return for row in correct)
            )
            average_signed_return_incorrect = _average(
                tuple(row.signed_prediction_return for row in incorrect)
            )
    except DecimalException as error:
        raise InvalidPredictionOutputError(
            "prediction metric arithmetic failed under its configured policy"
        ) from error
    return PredictionMetrics(
        prediction_count=len(rows),
        correct_count=len(correct),
        incorrect_count=len(incorrect),
        accuracy=accuracy,
        average_gap_size_correct=average_gap_size_correct,
        average_gap_size_incorrect=average_gap_size_incorrect,
        average_signed_return_correct=average_signed_return_correct,
        average_signed_return_incorrect=average_signed_return_incorrect,
    )


def _average(values: tuple[Decimal, ...]) -> Decimal | None:
    return None if not values else sum(values, Decimal(0)) / Decimal(len(values))


def _next_session(signal_session: date, calendar: str) -> date:
    try:
        return next_session_after(signal_session, calendar)
    except Exception as error:
        raise InvalidPredictionDataError(
            f"cannot resolve next session for calendar: {calendar}"
        ) from error


def _stable_id(values: PrimitiveMapping) -> str:
    try:
        return configuration_identity(values)
    except (TypeError, ValueError) as error:
        raise InvalidPredictionOutputError(
            "prediction inputs must have stable JSON-compatible serialization"
        ) from error


def _parameter_snapshots_match(
    actual: PrimitiveMapping, expected: PrimitiveMapping
) -> bool:
    return actual.keys() == expected.keys() and all(
        type(actual[name]) is type(expected[name]) and actual[name] == expected[name]
        for name in expected
    )


def _analysis_configuration() -> PrimitiveMapping:
    return {
        "outcome_timing": "immediate_next_exchange_session_open",
        "reference_price": "completed_signal_session_close",
        "flat_gap_policy": "incorrect",
        "cash_dividend_label_policy": (
            "observed_underlying_price_gap_includes_ex_dividend_effect"
        ),
        "stock_split_label_policy": "reject_raw_unadjusted_split_datasets",
        "label_arithmetic": {
            "decimal_precision": DECIMAL_PRECISION,
            "rounding": DECIMAL_ROUNDING,
            "decimal_emin": DECIMAL_EMIN,
            "decimal_emax": DECIMAL_EMAX,
            "capitals": DECIMAL_CAPITALS,
            "clamp": DECIMAL_CLAMP,
            "initial_flags": [],
            "traps": [signal.__name__ for signal in DECIMAL_TRAPS],
        },
    }
