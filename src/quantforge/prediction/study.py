"""Generic orchestration for causal prediction, outcome, and evaluation stages."""

from copy import deepcopy
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
from quantforge.data.lineage import AdjustmentBasis
from quantforge.data.models import SCHEMA_VERSION, MarketDataset
from quantforge.data.multi_timeframe import MultiTimeframeContextError
from quantforge.prediction.context import (
    PredictionContextError,
    PredictionContextFailurePolicy,
    PredictionContextProvider,
    PredictionContextRequirements,
    PredictionRuleContext,
    available_prediction_context_manifest,
    build_prediction_rule_context,
    skipped_prediction_context_manifest,
)
from quantforge.prediction.contracts import (
    EvaluationValuesT,
    MultiTimeframePredictionRule,
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

STUDY_ENGINE_VERSION = "12"
STUDY_CONTRACT_VERSION = "1"


@dataclass(frozen=True, slots=True)
class PredictionStudyDatasetSession:
    """One validated, detached dataset session reusable across related studies."""

    source_dataset: MarketDataset
    dataset_snapshot: MarketDataset
    component_dataset: MarketDataset
    market_data: PredictionMarketData
    available_sessions: frozenset[date]
    bar_indexes: dict[date, int]
    validated_labelers: set[tuple[int, str]]


@dataclass(frozen=True, slots=True)
class _SkippedPredictionRuleOutput[PredictionRecordT: PredictionRecord]:
    """Empty internal output when declared context policy skips execution."""

    strategy_id: str
    strategy_configuration_id: str
    dataset_id: str
    signals: tuple[PredictionRecordT, ...] = ()
    contract_version: str = "1"


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
    context_requirements_snapshot: PrimitiveMappingSnapshot | None = None

    def to_primitive(self) -> PrimitiveMapping:
        primitive: PrimitiveMapping = {
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
        if self.context_requirements_snapshot is not None:
            primitive["prediction_context_requirements"] = (
                self.context_requirements_snapshot.to_primitive()
            )
        return primitive


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
    primitive_snapshot: PrimitiveMappingSnapshot | None = None

    def __post_init__(self) -> None:
        expected_snapshot = _capture_values_snapshot(
            "prediction study row",
            _prediction_study_row_primitive(
                self.row_id,
                self.study_id,
                self.dataset_id,
                self.dataset_fingerprint,
                _prediction_record_primitive(self.signal),
                self.outcome.to_primitive(),
                self.evaluation.to_primitive(),
            ),
        )
        if self.primitive_snapshot is None:
            object.__setattr__(self, "primitive_snapshot", expected_snapshot)
        elif self.primitive_snapshot != expected_snapshot:
            raise InvalidPredictionOutputError(
                "prediction study row does not match its immutable snapshot"
            )

    def to_primitive(self) -> PrimitiveMapping:
        if self.primitive_snapshot is None:
            raise InvalidPredictionOutputError(
                "prediction study row requires an immutable snapshot"
            )
        return self.primitive_snapshot.to_primitive()


@dataclass(frozen=True, slots=True)
class _ComponentValueState[
    PredictionRecordT: PredictionRecord,
    OutcomeValuesT: PredictionValues,
    EvaluationValuesT: PredictionValues,
]:
    """Original component-owned values retained only for run-time validation."""

    signal: PredictionRecordT
    signal_snapshot: PrimitiveMappingSnapshot
    outcome: PredictionOutcome[OutcomeValuesT]
    outcome_snapshot: PrimitiveMappingSnapshot
    evaluation_values: EvaluationValuesT
    evaluation_values_snapshot: PrimitiveMappingSnapshot


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
    signals: tuple[PredictionRecordT, ...]
    rows: tuple[
        PredictionStudyRow[PredictionRecordT, OutcomeValuesT, EvaluationValuesT], ...
    ]
    prediction_context_snapshot: PrimitiveMappingSnapshot | None = None

    def manifest_primitive(self) -> PrimitiveMapping:
        manifest: PrimitiveMapping = {
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
        if self.prediction_context_snapshot is not None:
            manifest["prediction_context"] = (
                self.prediction_context_snapshot.to_primitive()
            )
        return manifest

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "manifest": self.manifest_primitive(),
            "rows": cast(list[Primitive], [row.to_primitive() for row in self.rows]),
        }


def run_prediction_study(
    dataset: MarketDataset,
    study: PredictionStudy[PredictionRecordT, OutcomeValuesT, EvaluationValuesT],
    *,
    context_provider: PredictionContextProvider | None = None,
) -> PredictionStudyResult[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]:
    """Run prediction, labeling, then evaluation as three ordered stages."""
    return run_prediction_study_in_session(
        prepare_prediction_study_dataset(dataset),
        study,
        context_provider=context_provider,
    )


def prepare_prediction_study_dataset(
    dataset: MarketDataset,
) -> PredictionStudyDatasetSession:
    """Validate and detach one dataset for related sequential study runs."""
    _validate_dataset(dataset)
    dataset_snapshot = _detached_copy("prediction market dataset snapshot", dataset)
    component_dataset = _detached_copy(
        "prediction component market dataset", dataset_snapshot
    )
    return PredictionStudyDatasetSession(
        dataset,
        dataset_snapshot,
        component_dataset,
        PredictionMarketData.from_qf3(dataset_snapshot.metadata),
        frozenset(bar.session_date for bar in component_dataset.bars),
        {bar.session_date: index for index, bar in enumerate(component_dataset.bars)},
        set(),
    )


def run_prediction_study_in_session(
    prepared: PredictionStudyDatasetSession,
    study: PredictionStudy[PredictionRecordT, OutcomeValuesT, EvaluationValuesT],
    *,
    context_provider: PredictionContextProvider | None = None,
) -> PredictionStudyResult[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]:
    """Run one study while preserving the session's validation boundaries."""
    dataset_snapshot = prepared.dataset_snapshot
    component_dataset = prepared.component_dataset
    _validate_unchanged_dataset(prepared.source_dataset, dataset_snapshot)
    _validate_unchanged_dataset(component_dataset, dataset_snapshot)
    configuration = _capture_study_configuration(study)
    market_data = prepared.market_data
    rule_context, prediction_context_snapshot = _prepare_prediction_context(
        study.strategy, context_provider, component_dataset
    )
    study_identity: PrimitiveMapping = {
        "component": "quantforge_prediction_study",
        "engine_version": STUDY_ENGINE_VERSION,
        "market_data": market_data.to_primitive(),
        "study_configuration": configuration.to_primitive(),
    }
    if prediction_context_snapshot is not None:
        study_identity["prediction_context"] = (
            prediction_context_snapshot.to_primitive()
        )
    study_id = _stable_id(study_identity)

    if configuration.context_requirements_snapshot is None:
        output = cast(PredictionRule[PredictionRecordT], study.strategy).generate(
            component_dataset
        )
    elif rule_context is None:
        output = _SkippedPredictionRuleOutput[PredictionRecordT](
            study.strategy.name,
            configuration.strategy_configuration_id,
            component_dataset.metadata.dataset_id,
        )
    else:
        rule_context_snapshot = _capture_values_snapshot(
            "prediction rule context", rule_context.values_primitive()
        )
        output = cast(
            MultiTimeframePredictionRule[PredictionRecordT], study.strategy
        ).generate_with_context(rule_context)
        _validate_unchanged_values(
            "prediction rule context",
            rule_context.values_primitive(),
            rule_context_snapshot,
        )
    generated_signals = tuple(output.signals)
    _validate_unchanged_dataset(component_dataset, dataset_snapshot)
    _validate_unchanged_component(
        "prediction strategy",
        study.strategy.configuration(),
        configuration.strategy_configuration_id,
    )
    _validate_unchanged_context_requirements(study.strategy, configuration)
    _validate_unchanged_dataset(component_dataset, dataset_snapshot)
    signal_snapshots = tuple(
        _capture_values_snapshot(
            "prediction signal", _prediction_record_primitive(signal)
        )
        for signal in generated_signals
    )
    _validate_strategy_output(
        component_dataset,
        study.strategy,
        output,
        generated_signals,
        signal_snapshots,
        configuration.strategy_configuration_id,
        configuration.strategy_warm_up_observations,
        context_decision_session=(
            None if rule_context is None else rule_context.decision_session
        ),
    )
    _validate_signal_snapshots(generated_signals, signal_snapshots)

    # This is intentionally after signal generation. Dataset-specific future-label
    # checks must never run early enough to influence prediction generation.
    labeler_session_key = (
        id(study.outcome_labeler),
        configuration.outcome_configuration_id,
    )
    if labeler_session_key not in prepared.validated_labelers:
        study.outcome_labeler.validate_dataset(component_dataset)
        prepared.validated_labelers.add(labeler_session_key)
    _validate_unchanged_component(
        "outcome labeler",
        study.outcome_labeler.configuration(),
        configuration.outcome_configuration_id,
    )
    _validate_unchanged_dataset(component_dataset, dataset_snapshot)
    _validate_signal_snapshots(generated_signals, signal_snapshots)
    rows: list[
        PredictionStudyRow[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]
    ] = []
    component_value_states: list[
        _ComponentValueState[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]
    ] = []
    unavailable_outcome_count = 0
    available_sessions = prepared.available_sessions
    bar_indexes = prepared.bar_indexes
    for signal, signal_snapshot in zip(
        generated_signals, signal_snapshots, strict=True
    ):
        _validate_unchanged_values(
            "prediction signal",
            _prediction_record_primitive(signal),
            signal_snapshot,
        )
        expected_outcome_index = (
            bar_indexes[signal.signal_session] + configuration.required_future_sessions
        )
        label = study.outcome_labeler.label(component_dataset, signal.signal_session)
        _validate_unchanged_component(
            "outcome labeler",
            study.outcome_labeler.configuration(),
            configuration.outcome_configuration_id,
        )
        _validate_unchanged_dataset(component_dataset, dataset_snapshot)
        _validate_unchanged_values(
            "prediction signal",
            _prediction_record_primitive(signal),
            signal_snapshot,
        )
        if label is None:
            if expected_outcome_index < len(component_dataset.bars):
                raise InvalidPredictionOutputError(
                    "outcome labeler returned no label although its declared future "
                    "session is available"
                )
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
        if (
            expected_outcome_index >= len(component_dataset.bars)
            or label.outcome_session
            != component_dataset.bars[expected_outcome_index].session_date
        ):
            raise InvalidPredictionOutputError(
                "outcome label session does not match its declared future-session "
                "horizon"
            )
        outcome_values_snapshot = _capture_values_snapshot(
            "prediction outcome values", label.values.to_primitive()
        )
        outcome = _prediction_outcome(
            market_data,
            configuration,
            label.signal_session,
            label.outcome_session,
            label.values,
            outcome_values_snapshot,
        )
        outcome_snapshot = _capture_values_snapshot(
            "prediction outcome",
            _prediction_outcome_primitive(
                outcome, outcome_values_snapshot.to_primitive()
            ),
        )
        _validate_unchanged_values(
            "prediction outcome values",
            label.values.to_primitive(),
            outcome_values_snapshot,
        )
        evaluation_values = study.evaluator.evaluate(signal, outcome)
        _validate_unchanged_component(
            "prediction evaluator",
            study.evaluator.configuration(),
            configuration.evaluator_configuration_id,
        )
        _validate_unchanged_dataset(component_dataset, dataset_snapshot)
        _validate_unchanged_values(
            "prediction signal",
            _prediction_record_primitive(signal),
            signal_snapshot,
        )
        _validate_unchanged_values(
            "prediction outcome", outcome.to_primitive(), outcome_snapshot
        )
        evaluation_values_snapshot = _capture_values_snapshot(
            "prediction evaluation values", evaluation_values.to_primitive()
        )
        evaluation = _prediction_evaluation(
            configuration,
            signal_snapshot,
            outcome,
            evaluation_values,
            evaluation_values_snapshot,
        )
        evaluation_snapshot = _capture_values_snapshot(
            "prediction evaluation", evaluation.to_primitive()
        )
        _validate_component_values(
            _prediction_record_primitive(signal),
            signal_snapshot,
            outcome.to_primitive(),
            outcome_snapshot,
            evaluation_values.to_primitive(),
            evaluation_values_snapshot,
        )
        row_id = _stable_id(
            {
                "evaluation_id": evaluation.evaluation_id,
                "outcome_id": outcome.outcome_id,
                "record_type": "prediction_study_row",
                "signal": signal_snapshot.to_primitive(),
                "study_id": study_id,
            }
        )
        detached_signal = _detached_copy("prediction signal", signal)
        detached_outcome = PredictionOutcome(
            outcome.outcome_id,
            outcome.outcome_name,
            outcome.outcome_implementation_version,
            outcome.outcome_configuration_id,
            outcome.outcome_result_schema_version,
            outcome.dataset_id,
            outcome.dataset_fingerprint,
            outcome.signal_session,
            outcome.outcome_session,
            _detached_copy("prediction outcome values", outcome.values),
        )
        detached_evaluation = PredictionEvaluation(
            evaluation.evaluation_id,
            evaluation.evaluator_name,
            evaluation.evaluator_implementation_version,
            evaluation.evaluator_configuration_id,
            evaluation.evaluation_result_schema_version,
            evaluation.outcome_id,
            _detached_copy("prediction evaluation values", evaluation.values),
        )
        detached_signal_primitive = _prediction_record_primitive(detached_signal)
        detached_outcome_primitive = detached_outcome.to_primitive()
        detached_evaluation_primitive = detached_evaluation.to_primitive()
        _validate_unchanged_values(
            "detached prediction signal",
            detached_signal_primitive,
            signal_snapshot,
        )
        _validate_unchanged_values(
            "detached prediction outcome",
            detached_outcome_primitive,
            outcome_snapshot,
        )
        _validate_unchanged_values(
            "detached prediction evaluation",
            detached_evaluation_primitive,
            evaluation_snapshot,
        )
        row_snapshot = _capture_values_snapshot(
            "prediction study row",
            _prediction_study_row_primitive(
                row_id,
                study_id,
                market_data.dataset_id,
                market_data.bars_fingerprint,
                detached_signal_primitive,
                detached_outcome_primitive,
                detached_evaluation_primitive,
            ),
        )
        row = PredictionStudyRow(
            row_id,
            study_id,
            market_data.dataset_id,
            market_data.bars_fingerprint,
            detached_signal,
            detached_outcome,
            detached_evaluation,
            row_snapshot,
        )
        rows.append(row)
        component_value_states.append(
            _ComponentValueState(
                signal,
                signal_snapshot,
                outcome,
                outcome_snapshot,
                evaluation_values,
                evaluation_values_snapshot,
            )
        )

    for state in component_value_states:
        _validate_component_values(
            _prediction_record_primitive(state.signal),
            state.signal_snapshot,
            state.outcome.to_primitive(),
            state.outcome_snapshot,
            state.evaluation_values.to_primitive(),
            state.evaluation_values_snapshot,
        )

    detached_signals = tuple(
        _detached_copy("prediction signal", signal) for signal in generated_signals
    )
    _validate_signal_snapshots(detached_signals, signal_snapshots)

    _validate_unchanged_component(
        "outcome labeler",
        study.outcome_labeler.configuration(),
        configuration.outcome_configuration_id,
    )
    _validate_unchanged_dataset(component_dataset, dataset_snapshot)
    _validate_unchanged_component(
        "prediction evaluator",
        study.evaluator.configuration(),
        configuration.evaluator_configuration_id,
    )
    _validate_unchanged_dataset(component_dataset, dataset_snapshot)
    _validate_unchanged_dataset(prepared.source_dataset, dataset_snapshot)
    return PredictionStudyResult(
        study_id,
        STUDY_ENGINE_VERSION,
        market_data,
        configuration,
        len(generated_signals),
        unavailable_outcome_count,
        detached_signals,
        tuple(rows),
        prediction_context_snapshot,
    )


def _capture_study_configuration(
    study: PredictionStudy[PredictionRecordT, OutcomeValuesT, EvaluationValuesT],
) -> PredictionStudyConfiguration:
    if not study.result_schema_version:
        raise InvalidPredictionConfigurationError(
            "prediction study result schema version is required"
        )
    warm_up_value = cast(object, study.strategy.warm_up_observations)
    if (
        isinstance(warm_up_value, bool)
        or not isinstance(warm_up_value, int)
        or warm_up_value < 1
    ):
        raise InvalidPredictionConfigurationError(
            "prediction strategies require a positive integer warm-up"
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
    context_requirements = _strategy_context_requirements(study.strategy)
    context_requirements_snapshot = (
        None
        if context_requirements is None
        else PrimitiveMappingSnapshot.capture(context_requirements.to_primitive())
    )
    return PredictionStudyConfiguration(
        strategy_id=study.strategy.name,
        strategy_implementation_version=study.strategy.implementation_version,
        strategy_configuration_id=strategy_id,
        strategy_configuration_snapshot=strategy_snapshot,
        strategy_warm_up_observations=warm_up_value,
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
        context_requirements_snapshot=context_requirements_snapshot,
    )


def _strategy_context_requirements(
    strategy: PredictionRule[PredictionRecordT]
    | MultiTimeframePredictionRule[PredictionRecordT],
) -> PredictionContextRequirements | None:
    raw_requirements = getattr(strategy, "context_requirements", None)
    if raw_requirements is None:
        if not callable(getattr(strategy, "generate", None)):
            raise InvalidPredictionConfigurationError(
                "single-timeframe prediction rule requires generate()"
            )
        return None
    if not isinstance(raw_requirements, PredictionContextRequirements) or not callable(
        getattr(strategy, "generate_with_context", None)
    ):
        raise InvalidPredictionConfigurationError(
            "multi-timeframe prediction rule requirements are invalid"
        )
    try:
        PrimitiveMappingSnapshot.capture(raw_requirements.to_primitive())
    except (TypeError, ValueError) as error:
        raise InvalidPredictionConfigurationError(
            "multi-timeframe prediction requirements require stable serialization"
        ) from error
    return raw_requirements


def _prepare_prediction_context(
    strategy: PredictionRule[PredictionRecordT]
    | MultiTimeframePredictionRule[PredictionRecordT],
    provider: PredictionContextProvider | None,
    dataset: MarketDataset,
) -> tuple[PredictionRuleContext | None, PrimitiveMappingSnapshot | None]:
    requirements = _strategy_context_requirements(strategy)
    if requirements is None:
        return None, None
    if provider is None or not callable(getattr(provider, "get_context", None)):
        raise InvalidPredictionConfigurationError(
            "multi-timeframe prediction study requires a context provider"
        )
    source_context = None
    try:
        source_context = provider.get_context(requirements)
        rule_context = build_prediction_rule_context(
            requirements,
            source_context,
            prediction_dataset_id=dataset.metadata.dataset_id,
            symbol=dataset.metadata.canonical_symbol,
            prediction_adjustment_basis=AdjustmentBasis(
                adjustment_mode=dataset.metadata.adjustment_mode,
                ohlc_basis=dataset.metadata.ohlc_basis,
                volume_basis=dataset.metadata.volume_basis,
                corporate_action_policy=dataset.metadata.corporate_action_policy,
                adjusted_fields_used=dataset.metadata.adjusted_fields_used,
            ),
        )
    except (PredictionContextError, MultiTimeframeContextError) as error:
        if requirements.failure_policy is PredictionContextFailurePolicy.FAIL:
            raise InvalidPredictionDataError(
                f"multi-timeframe prediction context failed: {error}"
            ) from error
        skipped = skipped_prediction_context_manifest(
            requirements,
            str(error),
            source_context=source_context,
        )
        return None, PrimitiveMappingSnapshot.capture(skipped)
    available = available_prediction_context_manifest(rule_context)
    return rule_context, PrimitiveMappingSnapshot.capture(available)


def _validate_unchanged_context_requirements(
    strategy: PredictionRule[PredictionRecordT]
    | MultiTimeframePredictionRule[PredictionRecordT],
    configuration: PredictionStudyConfiguration,
) -> None:
    expected = configuration.context_requirements_snapshot
    current = _strategy_context_requirements(strategy)
    if expected is None:
        if current is not None:
            raise InvalidPredictionOutputError(
                "prediction rule context requirements changed during run"
            )
        return
    if (
        current is None
        or PrimitiveMappingSnapshot.capture(current.to_primitive()) != expected
    ):
        raise InvalidPredictionOutputError(
            "prediction rule context requirements changed during run"
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
    values_snapshot: PrimitiveMappingSnapshot,
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
            "values": values_snapshot.to_primitive(),
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


def _prediction_outcome_primitive(
    outcome: PredictionOutcome[OutcomeValuesT],
    values: PrimitiveMapping,
) -> PrimitiveMapping:
    return {
        "dataset_fingerprint": outcome.dataset_fingerprint,
        "dataset_id": outcome.dataset_id,
        "outcome_configuration_id": outcome.outcome_configuration_id,
        "outcome_id": outcome.outcome_id,
        "outcome_implementation_version": outcome.outcome_implementation_version,
        "outcome_name": outcome.outcome_name,
        "outcome_result_schema_version": outcome.outcome_result_schema_version,
        "outcome_session": outcome.outcome_session.isoformat(),
        "signal_session": outcome.signal_session.isoformat(),
        "values": values,
    }


def _prediction_evaluation(
    configuration: PredictionStudyConfiguration,
    signal_snapshot: PrimitiveMappingSnapshot,
    outcome: PredictionOutcome[OutcomeValuesT],
    values: EvaluationValuesT,
    values_snapshot: PrimitiveMappingSnapshot,
) -> PredictionEvaluation[EvaluationValuesT]:
    prediction = cast(
        PrimitiveMapping,
        cast(PrimitiveMapping, signal_snapshot.to_primitive()["prediction"])["values"],
    )
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
            "prediction": prediction,
            "record_type": "prediction_evaluation",
            "values": values_snapshot.to_primitive(),
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


def _prediction_record_primitive(signal: PredictionRecord) -> PrimitiveMapping:
    return {
        "features": signal.features_primitive(),
        "prediction": {
            "signal_session": signal.signal_session.isoformat(),
            "strategy_configuration_id": signal.strategy_configuration_id,
            "strategy_id": signal.strategy_id,
            "strategy_implementation_version": (signal.strategy_implementation_version),
            "strategy_parameters": signal.parameters_primitive(),
            "symbol": signal.symbol,
            "values": signal.prediction_primitive(),
        },
    }


def _prediction_study_row_primitive(
    row_id: str,
    study_id: str,
    dataset_id: str,
    dataset_fingerprint: str,
    signal: PrimitiveMapping,
    outcome: PrimitiveMapping,
    evaluation: PrimitiveMapping,
) -> PrimitiveMapping:
    return {
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_id": dataset_id,
        "evaluation": evaluation,
        "features": signal["features"],
        "outcome": outcome,
        "prediction": signal["prediction"],
        "row_id": row_id,
        "study_id": study_id,
    }


def _detached_copy[ValueT](label: str, value: ValueT) -> ValueT:
    try:
        detached = deepcopy(value)
    except Exception as error:
        raise InvalidPredictionOutputError(
            f"{label} could not be detached from component-owned state"
        ) from error
    if detached is value:
        raise InvalidPredictionOutputError(
            f"{label} did not produce a detached component-independent copy"
        )
    return detached


def _validate_component_values(
    signal: PrimitiveMapping,
    signal_snapshot: PrimitiveMappingSnapshot,
    outcome: PrimitiveMapping,
    outcome_snapshot: PrimitiveMappingSnapshot,
    evaluation_values: PrimitiveMapping,
    evaluation_values_snapshot: PrimitiveMappingSnapshot,
) -> None:
    _validate_unchanged_values("prediction signal", signal, signal_snapshot)
    _validate_unchanged_values("prediction outcome", outcome, outcome_snapshot)
    _validate_unchanged_values(
        "prediction evaluation values",
        evaluation_values,
        evaluation_values_snapshot,
    )


def _validate_signal_snapshots(
    signals: tuple[PredictionRecordT, ...],
    snapshots: tuple[PrimitiveMappingSnapshot, ...],
) -> None:
    for signal, snapshot in zip(signals, snapshots, strict=True):
        _validate_unchanged_values(
            "prediction signal", _prediction_record_primitive(signal), snapshot
        )


def _capture_values_snapshot(
    label: str, values: PrimitiveMapping
) -> PrimitiveMappingSnapshot:
    try:
        return PrimitiveMappingSnapshot.capture(values)
    except (TypeError, ValueError) as error:
        raise InvalidPredictionOutputError(
            f"{label} requires stable JSON-compatible serialization"
        ) from error


def _validate_unchanged_values(
    label: str,
    values: PrimitiveMapping,
    expected_snapshot: PrimitiveMappingSnapshot,
) -> None:
    current_snapshot = _capture_values_snapshot(label, values)
    if current_snapshot != expected_snapshot:
        raise InvalidPredictionOutputError(
            f"{label} changed while the prediction evaluator was running"
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


def _validate_unchanged_dataset(
    dataset: MarketDataset, expected: MarketDataset
) -> None:
    if dataset != expected:
        raise InvalidPredictionOutputError(
            "prediction component mutated the market dataset"
        )


def _validate_strategy_output(
    dataset: MarketDataset,
    strategy: PredictionRule[PredictionRecordT]
    | MultiTimeframePredictionRule[PredictionRecordT],
    output: PredictionRuleOutput[PredictionRecordT],
    signals: tuple[PredictionRecordT, ...],
    signal_snapshots: tuple[PrimitiveMappingSnapshot, ...],
    expected_configuration_id: str,
    expected_warm_up_observations: int,
    context_decision_session: date | None,
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
        sorted(signals, key=lambda signal: (signal.signal_session, signal.symbol))
    )
    if signals != expected_order:
        raise InvalidPredictionOutputError(
            "prediction signals must be deterministically ordered"
        )
    sessions = tuple(signal.signal_session for signal in signals)
    if len(sessions) != len(set(sessions)):
        raise InvalidPredictionOutputError(
            "a strategy may emit at most one prediction per session"
        )
    bar_indexes = {bar.session_date: index for index, bar in enumerate(dataset.bars)}
    expected_parameters = strategy.parameters.to_primitive()
    for signal, signal_snapshot in zip(signals, signal_snapshots, strict=True):
        signal_index = bar_indexes.get(signal.signal_session)
        if (
            signal.symbol != dataset.metadata.canonical_symbol
            or signal_index is None
            or signal.strategy_id != strategy.name
            or signal.strategy_implementation_version != strategy.implementation_version
            or signal.strategy_configuration_id != expected_configuration_id
            or not _parameter_snapshots_match(
                _signal_parameters_from_snapshot(signal_snapshot),
                expected_parameters,
            )
        ):
            raise InvalidPredictionOutputError(
                "prediction signal identity or parameter snapshot is invalid"
            )
        if (
            context_decision_session is not None
            and signal.signal_session != context_decision_session
        ):
            raise InvalidPredictionOutputError(
                "multi-timeframe prediction signal must use the context decision "
                "session"
            )
        if signal_index + 1 < expected_warm_up_observations:
            raise InvalidPredictionOutputError(
                "prediction signal was emitted before the strategy's declared "
                "warm-up completed"
            )


def _signal_parameters_from_snapshot(
    signal_snapshot: PrimitiveMappingSnapshot,
) -> PrimitiveMapping:
    signal = signal_snapshot.to_primitive()
    prediction = signal.get("prediction")
    if not isinstance(prediction, dict):
        raise InvalidPredictionOutputError(
            "prediction signal requires a canonical prediction record"
        )
    parameters = prediction.get("strategy_parameters")
    if not isinstance(parameters, dict):
        raise InvalidPredictionOutputError(
            "prediction signal requires a canonical parameter snapshot"
        )
    return cast(PrimitiveMapping, parameters)


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
