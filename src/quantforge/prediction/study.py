"""Generic orchestration for causal prediction, outcome, and evaluation stages."""

from dataclasses import dataclass
from datetime import date
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import dataset_identity_matches, validate_market_dataset
from quantforge.data.exceptions import ValidationError as MarketDataValidationError
from quantforge.data.models import SCHEMA_VERSION, MarketDataset
from quantforge.prediction.contracts import (
    EvaluationValuesT,
    OutcomeValuesT,
    PredictionEvaluation,
    PredictionOutcome,
    PredictionRecord,
    PredictionRecordT,
    PredictionRule,
    PredictionRuleOutput,
    PredictionStudy,
    PredictionValues,
)
from quantforge.prediction.errors import (
    InvalidPredictionConfigurationError,
    InvalidPredictionDataError,
    InvalidPredictionOutputError,
)
from quantforge.prediction.models import PredictionMarketData

STUDY_ENGINE_VERSION = "1"
STUDY_CONTRACT_VERSION = "1"


@dataclass(frozen=True, slots=True)
class PredictionStudyConfiguration:
    """Immutable identity inputs for a generic prediction study."""

    strategy_id: str
    strategy_implementation_version: str
    strategy_configuration_id: str
    strategy_configuration_snapshot: PrimitiveMappingSnapshot
    strategy_warm_up_observations: int
    outcome_name: str
    outcome_implementation_version: str
    outcome_configuration_id: str
    outcome_configuration_snapshot: PrimitiveMappingSnapshot
    outcome_result_schema_version: str
    required_future_sessions: int
    required_market_fields: tuple[str, ...]
    evaluator_name: str
    evaluator_implementation_version: str
    evaluator_configuration_id: str
    evaluator_configuration_snapshot: PrimitiveMappingSnapshot
    evaluation_result_schema_version: str
    feature_configuration_snapshot: PrimitiveMappingSnapshot
    result_schema_version: str

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "contract_version": STUDY_CONTRACT_VERSION,
            "evaluator": {
                "configuration": (self.evaluator_configuration_snapshot.to_primitive()),
                "configuration_id": self.evaluator_configuration_id,
                "implementation_version": self.evaluator_implementation_version,
                "name": self.evaluator_name,
                "result_schema_version": self.evaluation_result_schema_version,
            },
            "feature_configuration": (
                self.feature_configuration_snapshot.to_primitive()
            ),
            "outcome_labeler": {
                "configuration": self.outcome_configuration_snapshot.to_primitive(),
                "configuration_id": self.outcome_configuration_id,
                "implementation_version": self.outcome_implementation_version,
                "name": self.outcome_name,
                "required_future_sessions": self.required_future_sessions,
                "required_market_fields": list(self.required_market_fields),
                "result_schema_version": self.outcome_result_schema_version,
            },
            "prediction_rule": {
                "configuration": self.strategy_configuration_snapshot.to_primitive(),
                "configuration_id": self.strategy_configuration_id,
                "implementation_version": self.strategy_implementation_version,
                "name": self.strategy_id,
                "warm_up_observations": self.strategy_warm_up_observations,
            },
            "result_schema_version": self.result_schema_version,
        }


@dataclass(frozen=True, slots=True)
class PredictionStudyRow[
    PredictionRecordT: PredictionRecord,
    OutcomeValuesT: PredictionValues,
    EvaluationValuesT: PredictionValues,
]:
    """One fixed causal signal paired with a later outcome and evaluation."""

    row_id: str
    study_id: str
    dataset_id: str
    dataset_fingerprint: str
    signal: PredictionRecordT
    outcome: PredictionOutcome[OutcomeValuesT]
    evaluation: PredictionEvaluation[EvaluationValuesT]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_id": self.dataset_id,
            "evaluation": self.evaluation.to_primitive(),
            "features": self.signal.features_primitive(),
            "outcome": self.outcome.to_primitive(),
            "prediction": {
                "signal_session": self.signal.signal_session.isoformat(),
                "strategy_configuration_id": (self.signal.strategy_configuration_id),
                "strategy_id": self.signal.strategy_id,
                "strategy_implementation_version": (
                    self.signal.strategy_implementation_version
                ),
                "strategy_parameters": self.signal.parameters_primitive(),
                "symbol": self.signal.symbol,
                "values": self.signal.prediction_primitive(),
            },
            "row_id": self.row_id,
            "study_id": self.study_id,
        }


@dataclass(frozen=True, slots=True)
class PredictionStudyResult[
    PredictionRecordT: PredictionRecord,
    OutcomeValuesT: PredictionValues,
    EvaluationValuesT: PredictionValues,
]:
    """Generic study result with no required classification or gap fields."""

    study_id: str
    engine_version: str
    market_data: PredictionMarketData
    configuration: PredictionStudyConfiguration
    generated_prediction_count: int
    unavailable_outcome_count: int
    rows: tuple[
        PredictionStudyRow[PredictionRecordT, OutcomeValuesT, EvaluationValuesT], ...
    ]

    def manifest_primitive(self) -> PrimitiveMapping:
        return {
            "component": "quantforge_prediction_study",
            "configuration": self.configuration.to_primitive(),
            "engine_version": self.engine_version,
            "feature_outcome_boundary": (
                "prediction signals are fixed before outcome labeling; evaluators "
                "receive only a fixed signal and an already-generated outcome"
            ),
            "market_data": self.market_data.to_primitive(),
            "record_counts": {
                "generated_predictions": self.generated_prediction_count,
                "labeled_rows": len(self.rows),
                "unavailable_outcomes": self.unavailable_outcome_count,
            },
            "study_id": self.study_id,
        }

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "manifest": self.manifest_primitive(),
            "rows": cast(list[Primitive], [row.to_primitive() for row in self.rows]),
        }


def run_prediction_study(
    dataset: MarketDataset,
    study: PredictionStudy[PredictionRecordT, OutcomeValuesT, EvaluationValuesT],
) -> PredictionStudyResult[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]:
    """Run prediction, labeling, then evaluation as three ordered stages."""
    _validate_dataset(dataset)
    configuration = _capture_study_configuration(study)
    market_data = PredictionMarketData.from_qf3(dataset.metadata)
    study_id = _stable_id(
        {
            "component": "quantforge_prediction_study",
            "engine_version": STUDY_ENGINE_VERSION,
            "market_data": market_data.to_primitive(),
            "study_configuration": configuration.to_primitive(),
        }
    )

    output = study.strategy.generate(dataset)
    _validate_unchanged_component(
        "prediction strategy",
        study.strategy.configuration(),
        configuration.strategy_configuration_id,
    )
    _validate_strategy_output(
        dataset,
        study.strategy,
        output,
        configuration.strategy_configuration_id,
    )

    # This is intentionally after signal generation. Dataset-specific future-label
    # checks must never run early enough to influence prediction generation.
    study.outcome_labeler.validate_dataset(dataset)
    rows: list[
        PredictionStudyRow[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]
    ] = []
    unavailable_outcome_count = 0
    available_sessions = {bar.session_date for bar in dataset.bars}
    for signal in output.signals:
        label = study.outcome_labeler.label(dataset, signal.signal_session)
        if label is None:
            unavailable_outcome_count += 1
            continue
        if (
            label.signal_session != signal.signal_session
            or label.outcome_session <= signal.signal_session
            or label.outcome_session not in available_sessions
        ):
            raise InvalidPredictionOutputError(
                "outcome labels require matching signal and later dataset sessions"
            )
        outcome = _prediction_outcome(
            market_data,
            configuration,
            label.signal_session,
            label.outcome_session,
            label.values,
        )
        evaluation_values = study.evaluator.evaluate(signal, outcome)
        evaluation = _prediction_evaluation(
            configuration, signal, outcome, evaluation_values
        )
        row_id = _stable_id(
            {
                "evaluation_id": evaluation.evaluation_id,
                "outcome_id": outcome.outcome_id,
                "record_type": "prediction_study_row",
                "signal_session": signal.signal_session.isoformat(),
                "study_id": study_id,
                "symbol": signal.symbol,
            }
        )
        rows.append(
            PredictionStudyRow(
                row_id,
                study_id,
                market_data.dataset_id,
                market_data.bars_fingerprint,
                signal,
                outcome,
                evaluation,
            )
        )

    _validate_unchanged_component(
        "outcome labeler",
        study.outcome_labeler.configuration(),
        configuration.outcome_configuration_id,
    )
    _validate_unchanged_component(
        "prediction evaluator",
        study.evaluator.configuration(),
        configuration.evaluator_configuration_id,
    )
    return PredictionStudyResult(
        study_id,
        STUDY_ENGINE_VERSION,
        market_data,
        configuration,
        len(output.signals),
        unavailable_outcome_count,
        tuple(rows),
    )


def _capture_study_configuration(
    study: PredictionStudy[PredictionRecordT, OutcomeValuesT, EvaluationValuesT],
) -> PredictionStudyConfiguration:
    if not study.result_schema_version:
        raise InvalidPredictionConfigurationError(
            "prediction study result schema version is required"
        )
    strategy_snapshot, strategy_id = _component_snapshot(
        "prediction strategy",
        study.strategy.name,
        study.strategy.implementation_version,
        study.strategy.configuration_id,
        study.strategy.configuration(),
    )
    outcome_snapshot, outcome_id = _component_snapshot(
        "outcome labeler",
        study.outcome_labeler.name,
        study.outcome_labeler.implementation_version,
        study.outcome_labeler.configuration_id,
        study.outcome_labeler.configuration(),
    )
    evaluator_snapshot, evaluator_id = _component_snapshot(
        "prediction evaluator",
        study.evaluator.name,
        study.evaluator.implementation_version,
        study.evaluator.configuration_id,
        study.evaluator.configuration(),
    )
    future_sessions_value = cast(object, study.outcome_labeler.required_future_sessions)
    fields_value = cast(object, study.outcome_labeler.required_market_fields)
    if (
        isinstance(future_sessions_value, bool)
        or not isinstance(future_sessions_value, int)
        or future_sessions_value < 1
    ):
        raise InvalidPredictionConfigurationError(
            "outcome labelers require a positive future-session horizon"
        )
    if not isinstance(fields_value, tuple) or not fields_value:
        raise InvalidPredictionConfigurationError(
            "outcome required market fields must be sorted unique names"
        )
    raw_fields = cast(tuple[object, ...], fields_value)
    if any(not isinstance(field, str) or not field for field in raw_fields):
        raise InvalidPredictionConfigurationError(
            "outcome required market fields must be sorted unique names"
        )
    fields = cast(tuple[str, ...], raw_fields)
    if fields != tuple(sorted(fields)) or len(fields) != len(set(fields)):
        raise InvalidPredictionConfigurationError(
            "outcome required market fields must be sorted unique names"
        )
    if (
        not study.outcome_labeler.result_schema_version
        or not study.evaluator.result_schema_version
    ):
        raise InvalidPredictionConfigurationError(
            "outcome and evaluation result schema versions are required"
        )
    return PredictionStudyConfiguration(
        strategy_id=study.strategy.name,
        strategy_implementation_version=study.strategy.implementation_version,
        strategy_configuration_id=strategy_id,
        strategy_configuration_snapshot=strategy_snapshot,
        strategy_warm_up_observations=study.strategy.warm_up_observations,
        outcome_name=study.outcome_labeler.name,
        outcome_implementation_version=(study.outcome_labeler.implementation_version),
        outcome_configuration_id=outcome_id,
        outcome_configuration_snapshot=outcome_snapshot,
        outcome_result_schema_version=(study.outcome_labeler.result_schema_version),
        required_future_sessions=future_sessions_value,
        required_market_fields=fields,
        evaluator_name=study.evaluator.name,
        evaluator_implementation_version=study.evaluator.implementation_version,
        evaluator_configuration_id=evaluator_id,
        evaluator_configuration_snapshot=evaluator_snapshot,
        evaluation_result_schema_version=study.evaluator.result_schema_version,
        feature_configuration_snapshot=PrimitiveMappingSnapshot.capture(
            study.feature_configuration
        ),
        result_schema_version=study.result_schema_version,
    )


def _component_snapshot(
    label: str,
    name: object,
    implementation_version: object,
    declared_configuration_id: object,
    configuration: PrimitiveMapping,
) -> tuple[PrimitiveMappingSnapshot, str]:
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(implementation_version, str)
        or not implementation_version
    ):
        raise InvalidPredictionConfigurationError(
            f"{label} requires stable name and implementation version"
        )
    expected_id = configuration_identity(configuration)
    if (
        not isinstance(declared_configuration_id, str)
        or not declared_configuration_id
        or declared_configuration_id != expected_id
        or configuration.get("component_name") != name
        or configuration.get("implementation_version") != implementation_version
    ):
        raise InvalidPredictionConfigurationError(
            f"{label} configuration identity is invalid"
        )
    return PrimitiveMappingSnapshot.capture(configuration), expected_id


def _prediction_outcome(
    market_data: PredictionMarketData,
    configuration: PredictionStudyConfiguration,
    signal_session: date,
    outcome_session: date,
    values: OutcomeValuesT,
) -> PredictionOutcome[OutcomeValuesT]:
    outcome_id = _stable_id(
        {
            "dataset_fingerprint": market_data.bars_fingerprint,
            "dataset_id": market_data.dataset_id,
            "outcome_configuration_id": configuration.outcome_configuration_id,
            "outcome_implementation_version": (
                configuration.outcome_implementation_version
            ),
            "outcome_name": configuration.outcome_name,
            "outcome_result_schema_version": (
                configuration.outcome_result_schema_version
            ),
            "outcome_session": outcome_session.isoformat(),
            "record_type": "prediction_outcome",
            "signal_session": signal_session.isoformat(),
            "values": values.to_primitive(),
        }
    )
    return PredictionOutcome(
        outcome_id,
        configuration.outcome_name,
        configuration.outcome_implementation_version,
        configuration.outcome_configuration_id,
        configuration.outcome_result_schema_version,
        market_data.dataset_id,
        market_data.bars_fingerprint,
        signal_session,
        outcome_session,
        values,
    )


def _prediction_evaluation(
    configuration: PredictionStudyConfiguration,
    signal: PredictionRecord,
    outcome: PredictionOutcome[OutcomeValuesT],
    values: EvaluationValuesT,
) -> PredictionEvaluation[EvaluationValuesT]:
    evaluation_id = _stable_id(
        {
            "evaluation_result_schema_version": (
                configuration.evaluation_result_schema_version
            ),
            "evaluator_configuration_id": configuration.evaluator_configuration_id,
            "evaluator_implementation_version": (
                configuration.evaluator_implementation_version
            ),
            "evaluator_name": configuration.evaluator_name,
            "outcome_id": outcome.outcome_id,
            "prediction": signal.prediction_primitive(),
            "record_type": "prediction_evaluation",
            "values": values.to_primitive(),
        }
    )
    return PredictionEvaluation(
        evaluation_id,
        configuration.evaluator_name,
        configuration.evaluator_implementation_version,
        configuration.evaluator_configuration_id,
        configuration.evaluation_result_schema_version,
        outcome.outcome_id,
        values,
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


def _validate_strategy_output(
    dataset: MarketDataset,
    strategy: PredictionRule[PredictionRecordT],
    output: PredictionRuleOutput[PredictionRecordT],
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


def _validate_unchanged_component(
    label: str, configuration: PrimitiveMapping, expected_id: str
) -> None:
    if configuration_identity(configuration) != expected_id:
        raise InvalidPredictionOutputError(f"{label} configuration changed during run")


def _parameter_snapshots_match(
    actual: PrimitiveMapping, expected: PrimitiveMapping
) -> bool:
    return actual.keys() == expected.keys() and all(
        type(actual[name]) is type(expected[name]) and actual[name] == expected[name]
        for name in expected
    )


def _stable_id(values: PrimitiveMapping) -> str:
    try:
        return configuration_identity(values)
    except (TypeError, ValueError) as error:
        raise InvalidPredictionOutputError(
            "prediction study values require stable JSON-compatible serialization"
        ) from error
