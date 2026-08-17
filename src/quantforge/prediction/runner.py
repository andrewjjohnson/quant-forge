"""Backward-compatible QF-11 overnight-gap analysis adapter."""

from decimal import Decimal, DecimalException

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.models import MarketDataset
from quantforge.prediction._arithmetic import arithmetic
from quantforge.prediction.base import PredictionStrategy
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.models import (
    PredictionAnalysisResult,
    PredictionMetrics,
    PredictionRow,
)
from quantforge.prediction.outcomes.overnight_gap import (
    NextSessionOpenGapOutcomeLabeler,
    create_overnight_gap_prediction_study,
)
from quantforge.prediction.study import run_prediction_study

ENGINE_VERSION = "5"
RESULT_SCHEMA_VERSION = "4"
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
    """Run the concrete gap study while preserving the public v1 result schema."""
    study_result = run_prediction_study(
        dataset, create_overnight_gap_prediction_study(strategy)
    )
    study_configuration = study_result.configuration
    strategy_configuration_snapshot = (
        study_configuration.strategy_configuration_snapshot
    )
    strategy_configuration = strategy_configuration_snapshot.to_primitive()
    strategy_configuration_id = study_configuration.strategy_configuration_id
    market_data = study_result.market_data
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

    rows: list[PredictionRow] = []
    for study_row in study_result.rows:
        signal = study_row.signal
        study_row_snapshot = study_row.to_primitive()
        fixed_signal_snapshot: PrimitiveMapping = {
            "features": study_row_snapshot["features"],
            "prediction": study_row_snapshot["prediction"],
        }
        outcome = study_row.outcome
        outcome_values = outcome.values
        evaluation_values = study_row.evaluation.values
        prediction_id = _stable_id(
            {
                "analysis_id": analysis_id,
                "record_type": "prediction",
                "signal": fixed_signal_snapshot,
                "symbol": signal.symbol,
                "signal_session": signal.signal_session.isoformat(),
                "outcome_session": outcome.outcome_session.isoformat(),
            }
        )
        rows.append(
            PredictionRow(
                prediction_id=prediction_id,
                dataset_id=market_data.dataset_id,
                dataset_fingerprint=market_data.bars_fingerprint,
                symbol=signal.symbol,
                signal_session=signal.signal_session,
                outcome_session=outcome.outcome_session,
                direction=signal.direction,
                strategy_id=signal.strategy_id,
                strategy_implementation_version=(
                    signal.strategy_implementation_version
                ),
                strategy_configuration_id=signal.strategy_configuration_id,
                strategy_parameters=signal.strategy_parameters,
                reason=signal.reason,
                feature_values=signal.feature_values,
                signal_close=outcome_values.signal_close,
                next_open=outcome_values.next_open,
                overnight_gap_percentage=(outcome_values.overnight_gap_percentage),
                gap_size_percentage=outcome_values.gap_size_percentage,
                signed_prediction_return=(evaluation_values.signed_prediction_return),
                correct=evaluation_values.direction_correct,
            )
        )

    result_rows = tuple(rows)
    metrics = _calculate_metrics(result_rows)
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
        generated_signal_count=study_result.generated_prediction_count,
        unlabeled_end_of_data_count=study_result.unavailable_outcome_count,
        generated_signals=study_result.signals,
        analysis_records_snapshot=PredictionAnalysisResult.capture_records(
            study_result.signals, result_rows
        ),
        rows=result_rows,
        metrics=metrics,
        limitations=LIMITATIONS,
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


def _stable_id(values: PrimitiveMapping) -> str:
    try:
        return configuration_identity(values)
    except (TypeError, ValueError) as error:
        raise InvalidPredictionOutputError(
            "prediction inputs must have stable JSON-compatible serialization"
        ) from error


def _analysis_configuration() -> PrimitiveMapping:
    return NextSessionOpenGapOutcomeLabeler().legacy_analysis_configuration()
