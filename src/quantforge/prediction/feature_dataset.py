"""Deterministic, resumable QF-7 datasets built through QF-11 studies."""

import csv
import io
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Protocol, cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data import dataset_identity_matches, validate_market_dataset
from quantforge.data.exceptions import ValidationError as MarketDataValidationError
from quantforge.data.models import (
    SCHEMA_VERSION,
    CashDividend,
    CorporateAction,
    MarketDataset,
)
from quantforge.indicators import Indicator
from quantforge.prediction.contracts import (
    OutcomeLabel,
    OutcomeLabeler,
    PredictionEvaluator,
    PredictionOutcome,
    PredictionRule,
    PredictionRuleParameters,
    PredictionStudy,
    PredictionValues,
)
from quantforge.prediction.errors import (
    InvalidPredictionDataError,
    InvalidPredictionOutputError,
    SignalFeatureDatasetError,
    SignalFeaturePersistenceError,
)
from quantforge.prediction.feature_outcomes import (
    DEFAULT_FORWARD_RETURN_HORIZONS,
    DirectionalExcursionEvaluator,
    ExcursionEvaluationValues,
    ExcursionOutcomeLabeler,
    ExcursionPathValues,
    ForwardReturnEvaluator,
    ForwardReturnOutcomeLabeler,
    ForwardReturnValues,
    SameSessionConflictPolicy,
    TargetStopEvaluationValues,
    TargetStopEvaluator,
    TargetStopOutcomeLabeler,
    TargetStopPathValues,
)
from quantforge.prediction.models import PredictionMarketData
from quantforge.prediction.signal_feature_context import (
    AtrPercentageContext,
    ContextualFeature,
    TrendDistanceContext,
    VolumeRatioContext,
)
from quantforge.prediction.signal_feature_models import (
    FEATURE_DATASET_ENGINE_VERSION,
    FEATURE_SCHEMA_VERSION,
    OUTCOME_SCHEMA_VERSION,
    SchemaField,
    SchemaFieldCategory,
    SignalFeatureCandidate,
    SignalFeatureCandidateOutput,
    SignalFeatureDatasetResult,
    SignalFeatureRow,
    SignalFeatureSchema,
    SignalFeatureValue,
    summarize_dispositions,
)
from quantforge.prediction.study import (
    PredictionStudyDatasetSession,
    prepare_prediction_study_dataset,
    run_prediction_study,
    run_prediction_study_in_session,
)

_LIMITATIONS = (
    "all feature relationships are exploratory hypotheses, not validated filters",
    "MFE and MAE are research labels and do not imply executable extreme prices",
    "daily bars cannot order target and stop touches within the same session",
    "overlapping status is present in the schema but must not be fabricated when "
    "a prediction rule has no overlap concept",
    "CSV is the supported analytics artifact; Parquet is not emitted because the "
    "repository has no existing Parquet dependency",
)
_CAUSAL_PROVENANCE_SENTINEL = "causal-prefix-not-persisted"

_TRUSTED_ALIGNED_CONTEXT_TYPES = (
    AtrPercentageContext,
    TrendDistanceContext,
    VolumeRatioContext,
)


@dataclass(frozen=True, slots=True)
class OutcomeRun:
    """Normalized output of one generic QF-11 study composition."""

    study_id: str
    values_by_session: dict[date, PrimitiveMapping]


@dataclass(frozen=True, slots=True)
class _CandidatePopulationValues:
    """Unreachable values type for the candidate-only QF-11 boundary."""

    def to_primitive(self) -> PrimitiveMapping:
        return {}


class _CandidatePopulationLabeler:
    """Keep QF-11 signal guards without evaluating a configured future outcome."""

    name = "qf7_candidate_population_boundary"
    implementation_version = "1"
    result_schema_version = "1"
    required_market_fields = ("close",)

    def __init__(self, bar_count: int) -> None:
        self.required_future_sessions = bar_count

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_outcome_labeler",
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": {
                "purpose": "fix_candidate_population_without_future_evaluation",
                "unavailable_horizon_sessions": self.required_future_sessions,
            },
            "required_market_fields": list(self.required_market_fields),
            "result_schema_version": self.result_schema_version,
        }

    def validate_dataset(self, dataset: MarketDataset) -> None:
        del dataset

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[_CandidatePopulationValues] | None:
        del dataset, signal_session
        return None


class _CandidatePopulationEvaluator:
    """Evaluator that cannot be reached because candidate-only labels are absent."""

    name = "qf7_candidate_population_boundary"
    implementation_version = "1"
    result_schema_version = "1"

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_evaluator",
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": {
                "purpose": "candidate_population_has_no_evaluation",
            },
            "result_schema_version": self.result_schema_version,
        }

    def evaluate(
        self,
        signal: SignalFeatureCandidate,
        outcome: PredictionOutcome[_CandidatePopulationValues],
    ) -> _CandidatePopulationValues:
        del signal, outcome
        raise InvalidPredictionOutputError(
            "candidate-only QF-11 boundary unexpectedly produced an outcome"
        )


class ConfiguredOutcome(Protocol):
    """Type-erased adapter around one typed QF-11 labeler/evaluator pair."""

    @property
    def namespace(self) -> str: ...

    @property
    def fields(self) -> tuple[SchemaField, ...]: ...

    @property
    def configuration_id(self) -> str: ...

    @property
    def labeler_configuration_id(self) -> str: ...

    @property
    def evaluator_configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...

    def unavailable_row(self) -> PrimitiveMapping: ...

    def run(
        self,
        dataset: MarketDataset,
        strategy: PredictionRule[SignalFeatureCandidate],
        feature_configuration: PrimitiveMapping,
    ) -> OutcomeRun: ...


@dataclass(frozen=True, slots=True)
class PredictionStudyOutcome[
    OutcomeT: PredictionValues,
    EvaluationT: PredictionValues,
]:
    """Expose one typed QF-11 study as flattened QF-7 outcome columns."""

    namespace: str
    labeler: OutcomeLabeler[OutcomeT]
    evaluator: PredictionEvaluator[SignalFeatureCandidate, OutcomeT, EvaluationT]
    fields: tuple[SchemaField, ...]
    unavailable_values_snapshot: PrimitiveMappingSnapshot

    @classmethod
    def create(
        cls,
        namespace: str,
        labeler: OutcomeLabeler[OutcomeT],
        evaluator: PredictionEvaluator[SignalFeatureCandidate, OutcomeT, EvaluationT],
        fields: tuple[SchemaField, ...],
        *,
        unavailable_values: PrimitiveMapping,
    ) -> "PredictionStudyOutcome[OutcomeT, EvaluationT]":
        if not namespace or not namespace.replace("_", "").isalnum():
            raise SignalFeatureDatasetError(
                "outcome namespaces must use nonempty alphanumeric snake case"
            )
        field_names = tuple(field.name for field in fields)
        if (
            not fields
            or field_names != tuple(sorted(field_names))
            or len(field_names) != len(set(field_names))
            or any(
                field.category is not SchemaFieldCategory.FUTURE_OUTCOME
                for field in fields
            )
        ):
            raise SignalFeatureDatasetError(
                "outcome fields must be sorted, unique future-outcome definitions"
            )
        unavailable_field_names = set(unavailable_values)
        if not unavailable_field_names.issubset(field_names):
            raise SignalFeatureDatasetError(
                "unavailable outcome defaults must name declared fields"
            )
        non_nullable_field_names = {
            field.name for field in fields if not field.nullable
        }
        if any(
            field_name not in unavailable_values
            or unavailable_values[field_name] is None
            for field_name in non_nullable_field_names
        ):
            raise SignalFeatureDatasetError(
                "non-nullable outcome fields require non-null unavailable defaults"
            )
        fields_by_name = {field.name: field for field in fields}
        invalid_default_names = tuple(
            sorted(
                field_name
                for field_name, field_value in unavailable_values.items()
                if not _schema_value_matches(fields_by_name[field_name], field_value)
            )
        )
        if invalid_default_names:
            raise SignalFeatureDatasetError(
                "unavailable outcome defaults do not match declared field types: "
                f"{invalid_default_names}"
            )
        availability_field = fields_by_name.get("available")
        if availability_field is not None and (
            availability_field.data_type != "boolean"
            or availability_field.nullable
            or unavailable_values.get("available") is not False
        ):
            raise SignalFeatureDatasetError(
                "outcome availability fields must be non-nullable booleans with "
                "an unavailable default of false"
            )
        return cls(
            namespace,
            labeler,
            evaluator,
            fields,
            PrimitiveMappingSnapshot.capture(unavailable_values),
        )

    @property
    def labeler_configuration_id(self) -> str:
        return self.labeler.configuration_id

    @property
    def evaluator_configuration_id(self) -> str:
        return self.evaluator.configuration_id

    def configuration(self) -> PrimitiveMapping:
        return {
            "component": "qf11_prediction_study_outcome",
            "evaluator": self.evaluator.configuration(),
            "fields": [field.to_primitive() for field in self.fields],
            "labeler": self.labeler.configuration(),
            "namespace": self.namespace,
            "unavailable_values": self.unavailable_values_snapshot.to_primitive(),
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def unavailable_row(self) -> PrimitiveMapping:
        values: PrimitiveMapping = {field.name: None for field in self.fields}
        values.update(self.unavailable_values_snapshot.to_primitive())
        return values

    def run(
        self,
        dataset: MarketDataset,
        strategy: PredictionRule[SignalFeatureCandidate],
        feature_configuration: PrimitiveMapping,
    ) -> OutcomeRun:
        return self.run_prepared(
            prepare_prediction_study_dataset(dataset),
            strategy,
            feature_configuration,
        )

    def run_prepared(
        self,
        prepared_dataset: PredictionStudyDatasetSession,
        strategy: PredictionRule[SignalFeatureCandidate],
        feature_configuration: PrimitiveMapping,
    ) -> OutcomeRun:
        study = PredictionStudy[SignalFeatureCandidate, OutcomeT, EvaluationT].create(
            strategy,
            self.labeler,
            self.evaluator,
            feature_configuration=feature_configuration,
            result_schema_version=OUTCOME_SCHEMA_VERSION,
        )
        result = run_prediction_study_in_session(prepared_dataset, study)
        field_names = {field.name for field in self.fields}
        non_nullable_field_names = {
            field.name for field in self.fields if not field.nullable
        }
        values_by_session: dict[date, PrimitiveMapping] = {}
        for row in result.rows:
            values = row.evaluation.values.to_primitive()
            if "outcome_session" in field_names:
                values["outcome_session"] = row.outcome.outcome_session.isoformat()
            if set(values) != field_names:
                raise InvalidPredictionOutputError(
                    f"outcome {self.namespace} values do not match their schema"
                )
            if any(
                values[field_name] is None for field_name in non_nullable_field_names
            ):
                raise InvalidPredictionOutputError(
                    f"outcome {self.namespace} produced null for a non-nullable field"
                )
            invalid_value_names = tuple(
                sorted(
                    field.name
                    for field in self.fields
                    if not _schema_value_matches(field, values[field.name])
                )
            )
            if invalid_value_names:
                raise InvalidPredictionOutputError(
                    f"outcome {self.namespace} values do not match declared field "
                    f"types: {invalid_value_names}"
                )
            values_by_session[row.signal.signal_session] = values
        return OutcomeRun(result.study_id, values_by_session)


def _run_configured_outcome(
    configured_outcome: ConfiguredOutcome,
    expected_configuration_id: str,
    dataset: MarketDataset,
    prepared_dataset: PredictionStudyDatasetSession,
    strategy: PredictionRule[SignalFeatureCandidate],
    feature_configuration: PrimitiveMapping,
) -> OutcomeRun:
    feature_configuration_snapshot = PrimitiveMappingSnapshot.capture(
        feature_configuration
    )
    outcome_feature_configuration = feature_configuration_snapshot.to_primitive()
    if isinstance(configured_outcome, PredictionStudyOutcome):
        outcome_run = configured_outcome.run_prepared(
            prepared_dataset, strategy, outcome_feature_configuration
        )
    else:
        outcome_run = configured_outcome.run(
            dataset, strategy, outcome_feature_configuration
        )
    if outcome_feature_configuration != feature_configuration_snapshot.to_primitive():
        raise InvalidPredictionOutputError(
            "configured outcome mutated feature configuration during execution"
        )
    if (
        configured_outcome.configuration_id != expected_configuration_id
        or configuration_identity(configured_outcome.configuration())
        != expected_configuration_id
    ):
        raise InvalidPredictionOutputError(
            "configured outcome configuration changed during execution"
        )
    return outcome_run


def _schema_value_matches(field: SchemaField, value: Primitive) -> bool:
    if value is None:
        return field.nullable
    if field.data_type == "boolean":
        return isinstance(value, bool)
    if field.data_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if field.data_type == "decimal":
        if not isinstance(value, str):
            return False
        try:
            return Decimal(value).is_finite()
        except DecimalException:
            return False
    if field.data_type == "date":
        if not isinstance(value, str):
            return False
        try:
            return date.fromisoformat(value).isoformat() == value
        except ValueError:
            return False
    if field.data_type == "string":
        return isinstance(value, str) and (not field.nullable or value != "")
    if field.data_type == "object":
        return isinstance(value, dict)
    if field.data_type == "array":
        return isinstance(value, list)
    return False


def _is_canonical_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_study_id(namespace: str, study_id: object) -> str:
    if not _is_canonical_sha256(study_id):
        raise InvalidPredictionOutputError(
            f"outcome {namespace} returned a non-canonical QF-11 study ID"
        )
    return cast(str, study_id)


def _bound_outcome_study_id(
    configured_outcome: ConfiguredOutcome,
    expected_configuration_id: str,
    reported_study_id: object,
    market_data: PredictionMarketData,
    strategy: PredictionRule[SignalFeatureCandidate],
    feature_configuration: PrimitiveMapping,
    fields: tuple[SchemaField, ...],
    unavailable_values: PrimitiveMapping,
) -> str:
    validated_reported_id = _validated_study_id(
        configured_outcome.namespace, reported_study_id
    )
    if isinstance(configured_outcome, PredictionStudyOutcome):
        return validated_reported_id
    return configuration_identity(
        {
            "binding_version": "1",
            "component": "quantforge_qf7_direct_outcome_study_binding",
            "feature_configuration": feature_configuration,
            "market_data": market_data.to_primitive(),
            "outcome": {
                "configuration_id": expected_configuration_id,
                "evaluator_configuration_id": (
                    configured_outcome.evaluator_configuration_id
                ),
                "fields": [field.to_primitive() for field in fields],
                "labeler_configuration_id": (
                    configured_outcome.labeler_configuration_id
                ),
                "namespace": configured_outcome.namespace,
                "reported_qf11_study_id": validated_reported_id,
                "unavailable_values": unavailable_values,
            },
            "prediction_rule_configuration_id": strategy.configuration_id,
        }
    )


def _validate_outcome_session_keys(
    configured_outcome: ConfiguredOutcome,
    expected_sessions: frozenset[date],
    values_by_session: dict[date, PrimitiveMapping],
) -> None:
    unexpected_sessions = tuple(
        sorted(set(values_by_session).difference(expected_sessions))
    )
    if unexpected_sessions:
        raise InvalidPredictionOutputError(
            f"outcome {configured_outcome.namespace} returned values for sessions "
            "outside the current "
            f"candidate chunk: {unexpected_sessions}"
        )
    if not isinstance(configured_outcome, PredictionStudyOutcome):
        missing_sessions = tuple(
            sorted(expected_sessions.difference(values_by_session))
        )
        if missing_sessions:
            raise InvalidPredictionOutputError(
                f"direct outcome {configured_outcome.namespace} omitted candidate "
                f"sessions instead of returning explicit availability: "
                f"{missing_sessions}"
            )


class _SignalFeatureRule(Protocol):
    """QF-11 candidate rule with documented strategy-input fields."""

    @property
    def name(self) -> str: ...

    @property
    def implementation_version(self) -> str: ...

    @property
    def parameters(self) -> PredictionRuleParameters: ...

    @property
    def required_indicators(self) -> tuple[Indicator, ...]: ...

    @property
    def warm_up_observations(self) -> int: ...

    @property
    def configuration_id(self) -> str: ...

    @property
    def strategy_feature_definitions(self) -> tuple[SchemaField, ...]: ...

    def configuration(self) -> PrimitiveMapping: ...

    def generate(self, dataset: MarketDataset) -> SignalFeatureCandidateOutput: ...


class _FixedCandidateRule:
    """Replay already-fixed causal candidates through additional QF-11 studies."""

    def __init__(
        self,
        source: _SignalFeatureRule,
        source_configuration_snapshot: PrimitiveMappingSnapshot,
        signals: tuple[SignalFeatureCandidate, ...],
        population_id: str,
        population_count: int,
    ) -> None:
        self._source = source
        self._source_configuration_snapshot = source_configuration_snapshot
        self._signals = signals
        self._population_id = population_id
        self._population_count = population_count

    @property
    def name(self) -> str:
        return self._source.name

    @property
    def implementation_version(self) -> str:
        return self._source.implementation_version

    @property
    def parameters(self) -> PredictionRuleParameters:
        return self._source.parameters

    @property
    def required_indicators(self) -> tuple[Indicator, ...]:
        return self._source.required_indicators

    @property
    def warm_up_observations(self) -> int:
        return self._source.warm_up_observations

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "qf7_fixed_candidate_replay",
            "fixed_candidate_population_count": self._population_count,
            "fixed_candidate_population_id": self._population_id,
            "implementation_version": self.implementation_version,
            "replay_semantics": "operational chunks are parts of one fixed population",
            "source_configuration": self._source_configuration_snapshot.to_primitive(),
            "source_configuration_id": configuration_identity(
                self._source_configuration_snapshot.to_primitive()
            ),
        }

    def generate(self, dataset: MarketDataset) -> SignalFeatureCandidateOutput:
        configuration_id = self.configuration_id
        return SignalFeatureCandidateOutput(
            self.name,
            configuration_id,
            dataset.metadata.dataset_id,
            tuple(
                replace(signal, strategy_configuration_id=configuration_id)
                for signal in self._signals
            ),
        )


def _fixed_candidate_population_id(
    signals: tuple[SignalFeatureCandidate, ...],
) -> str:
    return configuration_identity(
        {
            "component": "qf7_fixed_candidate_population",
            "signals": [
                {
                    "features": signal.features_primitive(),
                    "prediction": signal.prediction_primitive(),
                    "signal_session": signal.signal_session.isoformat(),
                    "strategy_configuration_id": signal.strategy_configuration_id,
                    "strategy_id": signal.strategy_id,
                    "strategy_implementation_version": (
                        signal.strategy_implementation_version
                    ),
                    "strategy_parameters": signal.parameters_primitive(),
                    "symbol": signal.symbol,
                }
                for signal in signals
            ],
        }
    )


def forward_return_outcome(
    horizon_sessions: int,
) -> PredictionStudyOutcome[ForwardReturnValues, ForwardReturnValues]:
    """Create one close-to-close return study with explicit unavailable fields."""
    timing = f"available only after exchange session t+{horizon_sessions} closes"
    fields = _sorted_fields(
        (
            _outcome_field("available", "boolean", "flag", False, timing),
            _outcome_field(
                "horizon_sessions", "integer", "exchange_sessions", False, timing
            ),
            _outcome_field("outcome_price", "decimal", "price_per_share", True, timing),
            _outcome_field("outcome_session", "date", "exchange_session", True, timing),
            _outcome_field(
                "raw_return",
                "decimal",
                "ratio",
                True,
                timing,
                "future close / signal close - 1",
            ),
            _outcome_field(
                "reference_price", "decimal", "price_per_share", True, timing
            ),
        )
    )
    return PredictionStudyOutcome[ForwardReturnValues, ForwardReturnValues].create(
        f"forward_return_{horizon_sessions}",
        ForwardReturnOutcomeLabeler(horizon_sessions),
        ForwardReturnEvaluator(),
        fields,
        unavailable_values={
            "available": False,
            "horizon_sessions": horizon_sessions,
        },
    )


def excursion_outcome(
    horizon_sessions: int,
) -> PredictionStudyOutcome[ExcursionPathValues, ExcursionEvaluationValues]:
    """Create one direction-aware MFE/MAE QF-11 study composition."""
    timing = f"available only after exchange session t+{horizon_sessions} closes"
    fields = _sorted_fields(
        (
            _outcome_field("available", "boolean", "flag", False, timing),
            _outcome_field(
                "horizon_sessions", "integer", "exchange_sessions", False, timing
            ),
            _outcome_field("mae_percentage", "decimal", "ratio", True, timing),
            _outcome_field("mae_session", "date", "exchange_session", True, timing),
            _outcome_field("mfe_percentage", "decimal", "ratio", True, timing),
            _outcome_field("mfe_session", "date", "exchange_session", True, timing),
            _outcome_field("outcome_session", "date", "exchange_session", True, timing),
            _outcome_field(
                "reference_price", "decimal", "price_per_share", True, timing
            ),
            _outcome_field("unavailable_reason", "string", "reason_code", True, timing),
        )
    )
    return PredictionStudyOutcome[
        ExcursionPathValues, ExcursionEvaluationValues
    ].create(
        f"mfe_mae_{horizon_sessions}",
        ExcursionOutcomeLabeler(horizon_sessions),
        DirectionalExcursionEvaluator(),
        fields,
        unavailable_values={
            "available": False,
            "horizon_sessions": horizon_sessions,
            "unavailable_reason": "insufficient_future_sessions",
        },
    )


def target_stop_outcome(
    horizon_sessions: int,
    target_percentage: Decimal,
    stop_percentage: Decimal,
    same_session_conflict_policy: SameSessionConflictPolicy = (
        SameSessionConflictPolicy.AMBIGUOUS
    ),
) -> PredictionStudyOutcome[TargetStopPathValues, TargetStopEvaluationValues]:
    """Create one target/stop study with explicit daily-bar conflict policy."""
    timing = f"available only after exchange session t+{horizon_sessions} closes"
    fields = _sorted_fields(
        tuple(
            _outcome_field(name, data_type, unit, nullable, timing)
            for name, data_type, unit, nullable in (
                ("ambiguous_high", "decimal", "price_per_share", True),
                ("ambiguous_low", "decimal", "price_per_share", True),
                ("ambiguous_session", "date", "exchange_session", True),
                ("available", "boolean", "flag", False),
                ("event_session", "date", "exchange_session", True),
                ("horizon_sessions", "integer", "exchange_sessions", False),
                ("label", "string", "target_stop_label", False),
                ("outcome_session", "date", "exchange_session", True),
                ("reference_price", "decimal", "price_per_share", True),
                (
                    "same_session_conflict_policy",
                    "string",
                    "policy_name",
                    False,
                ),
                ("stop_level", "decimal", "price_per_share", True),
                ("stop_percentage", "decimal", "ratio", False),
                ("target_level", "decimal", "price_per_share", True),
                ("target_percentage", "decimal", "ratio", False),
                ("unavailable_reason", "string", "reason_code", True),
            )
        )
    )
    labeler = TargetStopOutcomeLabeler(
        horizon_sessions,
        target_percentage,
        stop_percentage,
        same_session_conflict_policy,
    )
    evaluator = TargetStopEvaluator(
        target_percentage, stop_percentage, same_session_conflict_policy
    )
    return PredictionStudyOutcome[
        TargetStopPathValues, TargetStopEvaluationValues
    ].create(
        f"target_stop_{horizon_sessions}",
        labeler,
        evaluator,
        fields,
        unavailable_values={
            "available": False,
            "horizon_sessions": horizon_sessions,
            "label": "unavailable",
            "same_session_conflict_policy": same_session_conflict_policy.value,
            "stop_percentage": decimal_to_primitive(evaluator.stop_percentage),
            "target_percentage": decimal_to_primitive(evaluator.target_percentage),
            "unavailable_reason": "insufficient_future_sessions",
        },
    )


def default_signal_feature_outcomes() -> tuple[ConfiguredOutcome, ...]:
    """Return documented forward horizons plus five-session path labels."""
    return (
        *(
            forward_return_outcome(horizon)
            for horizon in DEFAULT_FORWARD_RETURN_HORIZONS
        ),
        excursion_outcome(5),
        target_stop_outcome(5, Decimal("0.01"), Decimal("0.005")),
    )


def build_signal_feature_dataset[
    InitialOutcomeT: PredictionValues,
    InitialEvaluationT: PredictionValues,
](
    *,
    dataset: MarketDataset,
    prediction_study: PredictionStudy[
        SignalFeatureCandidate, InitialOutcomeT, InitialEvaluationT
    ],
    contextual_features: Sequence[ContextualFeature],
    outcomes: Sequence[ConfiguredOutcome],
    output_root: Path,
    chunk_size: int = 100,
) -> SignalFeatureDatasetResult:
    """Build or resume one deterministic analytics dataset through QF-11 studies."""
    if isinstance(chunk_size, bool) or chunk_size < 1:
        raise SignalFeatureDatasetError("chunk_size must be a positive integer")
    market_data = _validated_market_data(dataset)
    strategy = cast(_SignalFeatureRule, prediction_study.strategy)
    strategy_configuration_snapshot = PrimitiveMappingSnapshot.capture(
        strategy.configuration()
    )
    strategy_configuration_id = configuration_identity(
        strategy_configuration_snapshot.to_primitive()
    )
    if strategy.configuration_id != strategy_configuration_id:
        raise SignalFeatureDatasetError(
            "prediction strategy configuration identity is invalid"
        )
    strategy_fields = _strategy_fields(strategy)
    sorted_features = tuple(sorted(contextual_features, key=lambda item: item.name))
    contextual_definitions = tuple(feature.definition for feature in sorted_features)
    contextual_configuration_snapshots = tuple(
        PrimitiveMappingSnapshot.capture(feature.configuration())
        for feature in sorted_features
    )
    sorted_outcomes = tuple(sorted(outcomes, key=lambda item: item.namespace))
    outcome_field_snapshots = tuple(outcome.fields for outcome in sorted_outcomes)
    outcome_configuration_snapshots = tuple(
        PrimitiveMappingSnapshot.capture(outcome.configuration())
        for outcome in sorted_outcomes
    )
    _validate_feature_configuration(
        strategy_fields,
        sorted_features,
        contextual_definitions,
        contextual_configuration_snapshots,
        sorted_outcomes,
        outcome_field_snapshots,
        outcome_configuration_snapshots,
    )
    if not any(
        item.labeler_configuration_id
        == prediction_study.outcome_labeler.configuration_id
        and item.evaluator_configuration_id
        == prediction_study.evaluator.configuration_id
        for item in sorted_outcomes
    ):
        raise SignalFeatureDatasetError(
            "configured outcomes must include the supplied PredictionStudy composition"
        )
    unavailable_outcome_values = {
        outcome.namespace: _validated_unavailable_outcome_values(
            outcome,
            fields,
            outcome.unavailable_row(),
        )
        for outcome, fields in zip(
            sorted_outcomes, outcome_field_snapshots, strict=True
        )
    }
    configuration = _dataset_configuration(
        market_data,
        prediction_study,
        strategy_configuration_snapshot,
        strategy_fields,
        contextual_definitions,
        contextual_configuration_snapshots,
        sorted_outcomes,
        outcome_field_snapshots,
        outcome_configuration_snapshots,
        unavailable_outcome_values,
    )
    configuration_snapshot = PrimitiveMappingSnapshot.capture(configuration)
    outcome_configuration_ids = {
        outcome.namespace: configuration_identity(snapshot.to_primitive())
        for outcome, snapshot in zip(
            sorted_outcomes, outcome_configuration_snapshots, strict=True
        )
    }
    feature_dataset_id = configuration_identity(configuration)
    schema = _schema(
        strategy_fields,
        contextual_definitions,
        sorted_outcomes,
        outcome_field_snapshots,
    )
    feature_configuration = _feature_configuration(
        strategy_fields,
        contextual_definitions,
        contextual_configuration_snapshots,
    )
    destination = output_root / feature_dataset_id

    if destination.exists() and _manifest_status(destination) == "complete":
        return _load_completed_result(
            destination,
            feature_dataset_id,
            market_data,
            configuration_snapshot,
            schema,
            dataset,
            strategy,
            strategy_configuration_snapshot,
            feature_configuration,
            sorted_outcomes,
            outcome_field_snapshots,
            unavailable_outcome_values,
            outcome_configuration_ids,
        )
    _initialize_or_validate_progress(
        destination,
        feature_dataset_id,
        market_data,
        configuration_snapshot,
        schema,
    )
    completed_rows = _load_progress_rows(destination, feature_dataset_id, schema)

    candidates = _generate_candidate_population(
        dataset,
        prediction_study,
        strategy_configuration_id,
    )
    candidate_ids = {
        _candidate_id(market_data, candidate): candidate for candidate in candidates
    }
    if len(candidate_ids) != len(candidates):
        raise InvalidPredictionOutputError(
            "candidate generation produced duplicate logical identities"
        )
    unknown_completed = set(completed_rows) - set(candidate_ids)
    if unknown_completed:
        raise SignalFeaturePersistenceError(
            "persisted rows do not belong to the regenerated candidate population"
        )
    bar_indexes = {bar.session_date: index for index, bar in enumerate(dataset.bars)}
    enriched_candidates = _enrich_candidates(
        dataset,
        candidates,
        bar_indexes,
        sorted_features,
        contextual_definitions,
        contextual_configuration_snapshots,
    )
    _validate_regenerated_completed_rows(
        feature_dataset_id,
        market_data,
        schema,
        enriched_candidates,
        completed_rows,
        sorted_outcomes,
        outcome_field_snapshots,
        unavailable_outcome_values,
    )
    missing_candidates = tuple(
        candidate
        for candidate in enriched_candidates
        if (candidate_id := _candidate_id(market_data, candidate))
        if candidate_id not in completed_rows
    )
    candidate_population_id = _fixed_candidate_population_id(enriched_candidates)
    study_ids_by_namespace = _study_ids_from_rows(completed_rows.values())
    prepared_outcome_dataset = (
        prepare_prediction_study_dataset(dataset)
        if missing_candidates or not enriched_candidates
        else None
    )

    for chunk_start in range(0, len(missing_candidates), chunk_size):
        chunk = missing_candidates[chunk_start : chunk_start + chunk_size]
        chunk_signal_sessions = frozenset(
            candidate.signal_session for candidate in chunk
        )
        fixed_rule = _FixedCandidateRule(
            strategy,
            strategy_configuration_snapshot,
            chunk,
            candidate_population_id,
            len(enriched_candidates),
        )
        outcome_values: dict[str, dict[date, PrimitiveMapping]] = {}
        chunk_study_ids: dict[str, str] = {}
        for configured_outcome, fields in zip(
            sorted_outcomes, outcome_field_snapshots, strict=True
        ):
            if prepared_outcome_dataset is None:
                raise InvalidPredictionOutputError(
                    "missing candidates require a prepared outcome dataset"
                )
            outcome_run = _run_configured_outcome(
                configured_outcome,
                outcome_configuration_ids[configured_outcome.namespace],
                dataset,
                prepared_outcome_dataset,
                fixed_rule,
                feature_configuration,
            )
            _validate_outcome_session_keys(
                configured_outcome,
                chunk_signal_sessions,
                outcome_run.values_by_session,
            )
            outcome_values[configured_outcome.namespace] = {
                signal_session: _validated_flattened_outcome_values(
                    configured_outcome, fields, values
                )
                for signal_session, values in outcome_run.values_by_session.items()
            }
            study_id = _bound_outcome_study_id(
                configured_outcome,
                outcome_configuration_ids[configured_outcome.namespace],
                outcome_run.study_id,
                market_data,
                fixed_rule,
                feature_configuration,
                fields,
                unavailable_outcome_values[configured_outcome.namespace],
            )
            chunk_study_ids[configured_outcome.namespace] = study_id
            previous_study_id = study_ids_by_namespace.get(configured_outcome.namespace)
            if previous_study_id not in (None, study_id):
                raise InvalidPredictionOutputError(
                    "QF-11 study identity changed across deterministic chunks"
                )
            study_ids_by_namespace[configured_outcome.namespace] = study_id
        for candidate in chunk:
            candidate_id = _candidate_id(market_data, candidate)
            row = _build_row(
                feature_dataset_id,
                market_data,
                schema,
                candidate,
                candidate_id,
                sorted_outcomes,
                outcome_values,
                chunk_study_ids,
                unavailable_outcome_values,
            )
            _persist_progress_row(destination, row)
            completed_rows[candidate_id] = row

    ordered_rows = tuple(
        completed_rows[_candidate_id(market_data, candidate)]
        for candidate in enriched_candidates
    )
    if len(ordered_rows) != len({row.candidate_id for row in ordered_rows}):
        raise SignalFeaturePersistenceError(
            "completed signal-feature rows contain duplicate candidates"
        )
    if not enriched_candidates:
        empty_rule = _FixedCandidateRule(
            strategy,
            strategy_configuration_snapshot,
            (),
            candidate_population_id,
            0,
        )
        for configured_outcome, fields in zip(
            sorted_outcomes, outcome_field_snapshots, strict=True
        ):
            if prepared_outcome_dataset is None:
                raise InvalidPredictionOutputError(
                    "empty candidates require a prepared outcome dataset"
                )
            outcome_run = _run_configured_outcome(
                configured_outcome,
                outcome_configuration_ids[configured_outcome.namespace],
                dataset,
                prepared_outcome_dataset,
                empty_rule,
                feature_configuration,
            )
            _validate_outcome_session_keys(
                configured_outcome,
                frozenset(),
                outcome_run.values_by_session,
            )
            study_ids_by_namespace[configured_outcome.namespace] = (
                _bound_outcome_study_id(
                    configured_outcome,
                    outcome_configuration_ids[configured_outcome.namespace],
                    outcome_run.study_id,
                    market_data,
                    empty_rule,
                    feature_configuration,
                    fields,
                    unavailable_outcome_values[configured_outcome.namespace],
                )
            )
    result = SignalFeatureDatasetResult(
        feature_dataset_id,
        FEATURE_DATASET_ENGINE_VERSION,
        market_data,
        configuration_snapshot,
        schema,
        ordered_rows,
        summarize_dispositions(ordered_rows),
        tuple(study_ids_by_namespace[name] for name in sorted(study_ids_by_namespace)),
        _LIMITATIONS,
    )
    _finalize_export(destination, result)
    return result


def _generate_candidate_population[
    InitialOutcomeT: PredictionValues,
    InitialEvaluationT: PredictionValues,
](
    dataset: MarketDataset,
    prediction_study: PredictionStudy[
        SignalFeatureCandidate, InitialOutcomeT, InitialEvaluationT
    ],
    expected_strategy_configuration_id: str,
) -> tuple[SignalFeatureCandidate, ...]:
    _validate_strategy_configuration(
        cast(PredictionRule[SignalFeatureCandidate], prediction_study.strategy),
        expected_strategy_configuration_id,
    )
    boundary_study = PredictionStudy[
        SignalFeatureCandidate,
        _CandidatePopulationValues,
        _CandidatePopulationValues,
    ].create(
        prediction_study.strategy,
        _CandidatePopulationLabeler(len(dataset.bars)),
        _CandidatePopulationEvaluator(),
        feature_configuration=prediction_study.feature_configuration,
        result_schema_version=prediction_study.result_schema_version,
    )
    signals = tuple(run_prediction_study(dataset, boundary_study).signals)
    _validate_strategy_configuration(
        cast(PredictionRule[SignalFeatureCandidate], prediction_study.strategy),
        expected_strategy_configuration_id,
    )
    return signals


def _validate_strategy_configuration(
    strategy: PredictionRule[SignalFeatureCandidate],
    expected_configuration_id: str,
) -> None:
    if (
        strategy.configuration_id != expected_configuration_id
        or configuration_identity(strategy.configuration()) != expected_configuration_id
    ):
        raise InvalidPredictionOutputError(
            "prediction strategy configuration changed during candidate generation"
        )


def _validate_regenerated_completed_rows(
    feature_dataset_id: str,
    market_data: PredictionMarketData,
    schema: SignalFeatureSchema,
    candidates: tuple[SignalFeatureCandidate, ...],
    completed_rows: dict[str, SignalFeatureRow],
    outcomes: tuple[ConfiguredOutcome, ...],
    outcome_field_snapshots: tuple[tuple[SchemaField, ...], ...],
    unavailable_outcome_values: dict[str, PrimitiveMapping],
) -> None:
    for candidate in candidates:
        candidate_id = _candidate_id(market_data, candidate)
        completed_row = completed_rows.get(candidate_id)
        if completed_row is None:
            continue
        completed_values = completed_row.to_primitive()
        outcome_values = {
            outcome.namespace: {
                candidate.signal_session: {
                    field.name: completed_values[
                        f"outcome_{outcome.namespace}_{field.name}"
                    ]
                    for field in fields
                }
            }
            for outcome, fields in zip(outcomes, outcome_field_snapshots, strict=True)
        }
        regenerated_row = _build_row(
            feature_dataset_id,
            market_data,
            schema,
            candidate,
            candidate_id,
            outcomes,
            outcome_values,
            _study_ids_from_rows((completed_row,)),
            unavailable_outcome_values,
        )
        if regenerated_row.to_primitive() != completed_values:
            raise SignalFeaturePersistenceError(
                "persisted row does not match its regenerated causal candidate"
            )


def _validated_market_data(dataset: MarketDataset) -> PredictionMarketData:
    if dataset.metadata.schema_version != SCHEMA_VERSION or not dataset.bars:
        raise InvalidPredictionDataError(
            f"a nonempty market dataset using schema {SCHEMA_VERSION} is required"
        )
    try:
        missing_sessions = validate_market_dataset(dataset)
    except MarketDataValidationError as error:
        raise InvalidPredictionDataError(str(error)) from error
    if not dataset_identity_matches(dataset):
        raise InvalidPredictionDataError(
            "market bars and provenance do not reproduce the dataset identity"
        )
    metadata = dataset.metadata
    if any(
        metadata.actual_first_session < session < metadata.actual_last_session
        for session in missing_sessions
    ):
        raise InvalidPredictionDataError(
            "signal-feature generation does not permit internal missing sessions"
        )
    return PredictionMarketData.from_qf3(metadata)


def _strategy_fields(strategy: _SignalFeatureRule) -> tuple[SchemaField, ...]:
    fields = strategy.strategy_feature_definitions
    if tuple(field.name for field in fields) != tuple(
        sorted(field.name for field in fields)
    ):
        raise SignalFeatureDatasetError(
            "strategy input feature definitions must be a sorted typed tuple"
        )
    if any(
        field.category is not SchemaFieldCategory.CONTEMPORANEOUS_FEATURE
        for field in fields
    ):
        raise SignalFeatureDatasetError(
            "strategy input feature definitions must be contemporaneous features"
        )
    return fields


def _validate_feature_configuration(
    strategy_fields: tuple[SchemaField, ...],
    contextual_features: tuple[ContextualFeature, ...],
    contextual_definitions: tuple[SchemaField, ...],
    contextual_configuration_snapshots: tuple[PrimitiveMappingSnapshot, ...],
    outcomes: tuple[ConfiguredOutcome, ...],
    outcome_field_snapshots: tuple[tuple[SchemaField, ...], ...],
    outcome_configuration_snapshots: tuple[PrimitiveMappingSnapshot, ...],
) -> None:
    if not outcomes:
        raise SignalFeatureDatasetError("at least one QF-11 outcome is required")
    contextual_names = tuple(item.name for item in contextual_features)
    outcome_names = tuple(item.namespace for item in outcomes)
    strategy_names = {field.name for field in strategy_fields}
    if (
        len(contextual_names) != len(set(contextual_names))
        or strategy_names & set(contextual_names)
        or len(outcome_names) != len(set(outcome_names))
    ):
        raise SignalFeatureDatasetError(
            "strategy, contextual, and outcome names must be unique"
        )
    for feature, definition, configuration_snapshot in zip(
        contextual_features,
        contextual_definitions,
        contextual_configuration_snapshots,
        strict=True,
    ):
        if definition.name != feature.name:
            raise SignalFeatureDatasetError(
                "contextual feature definition does not match its name"
            )
        if definition.category is not SchemaFieldCategory.CONTEMPORANEOUS_FEATURE:
            raise SignalFeatureDatasetError(
                "contextual feature definitions must be contemporaneous features"
            )
        if (
            configuration_identity(configuration_snapshot.to_primitive())
            != feature.configuration_id
        ):
            raise SignalFeatureDatasetError(
                "contextual feature configuration identity is invalid"
            )
    for outcome, fields, configuration_snapshot in zip(
        outcomes,
        outcome_field_snapshots,
        outcome_configuration_snapshots,
        strict=True,
    ):
        field_names = tuple(field.name for field in fields)
        if (
            not field_names
            or field_names != tuple(sorted(field_names))
            or len(field_names) != len(set(field_names))
            or any(
                field.category is not SchemaFieldCategory.FUTURE_OUTCOME
                for field in fields
            )
        ):
            raise SignalFeatureDatasetError(
                "configured outcome fields must be sorted, unique future-outcome "
                "definitions"
            )
        if (
            configuration_identity(configuration_snapshot.to_primitive())
            != outcome.configuration_id
        ):
            raise SignalFeatureDatasetError(
                "configured outcome configuration identity is invalid"
            )


def _dataset_configuration[
    InitialOutcomeT: PredictionValues,
    InitialEvaluationT: PredictionValues,
](
    market_data: PredictionMarketData,
    prediction_study: PredictionStudy[
        SignalFeatureCandidate, InitialOutcomeT, InitialEvaluationT
    ],
    strategy_configuration_snapshot: PrimitiveMappingSnapshot,
    strategy_fields: tuple[SchemaField, ...],
    contextual_definitions: tuple[SchemaField, ...],
    contextual_configuration_snapshots: tuple[PrimitiveMappingSnapshot, ...],
    outcomes: tuple[ConfiguredOutcome, ...],
    outcome_field_snapshots: tuple[tuple[SchemaField, ...], ...],
    outcome_configuration_snapshots: tuple[PrimitiveMappingSnapshot, ...],
    unavailable_outcome_values: dict[str, PrimitiveMapping],
) -> PrimitiveMapping:
    return {
        "component": "quantforge_signal_feature_dataset",
        "engine_version": FEATURE_DATASET_ENGINE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "prediction_study_template": {
            "evaluator": prediction_study.evaluator.configuration(),
            "feature_configuration": prediction_study.feature_configuration,
            "outcome_labeler": prediction_study.outcome_labeler.configuration(),
            "result_schema_version": prediction_study.result_schema_version,
            "strategy": strategy_configuration_snapshot.to_primitive(),
        },
        "source_data": market_data.to_primitive(),
        "strategy_feature_fields": [field.to_primitive() for field in strategy_fields],
        "contextual_features": [
            _contextual_feature_configuration(definition, configuration_snapshot)
            for definition, configuration_snapshot in zip(
                contextual_definitions,
                contextual_configuration_snapshots,
                strict=True,
            )
        ],
        "outcomes": [
            _normalized_outcome_configuration(
                outcome,
                fields,
                configuration_snapshot,
                unavailable_outcome_values[outcome.namespace],
            )
            for outcome, fields, configuration_snapshot in zip(
                outcomes,
                outcome_field_snapshots,
                outcome_configuration_snapshots,
                strict=True,
            )
        ],
    }


def _normalized_outcome_configuration(
    outcome: ConfiguredOutcome,
    fields: tuple[SchemaField, ...],
    configuration_snapshot: PrimitiveMappingSnapshot,
    unavailable_values: PrimitiveMapping,
) -> PrimitiveMapping:
    return {
        "component_configuration": configuration_snapshot.to_primitive(),
        "configuration_id": configuration_identity(
            configuration_snapshot.to_primitive()
        ),
        "fields": [field.to_primitive() for field in fields],
        "namespace": outcome.namespace,
        "unavailable_values": unavailable_values,
    }


def _feature_configuration(
    strategy_fields: tuple[SchemaField, ...],
    contextual_definitions: tuple[SchemaField, ...],
    contextual_configuration_snapshots: tuple[PrimitiveMappingSnapshot, ...],
) -> PrimitiveMapping:
    return {
        "candidate_context_features": [
            _contextual_feature_configuration(definition, configuration_snapshot)
            for definition, configuration_snapshot in zip(
                contextual_definitions,
                contextual_configuration_snapshots,
                strict=True,
            )
        ],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "required_strategy_features": [
            field.to_primitive() for field in strategy_fields
        ],
        "temporal_policy": "context_features_receive_signal_session_prefix_only",
    }


def _contextual_feature_configuration(
    definition: SchemaField,
    configuration_snapshot: PrimitiveMappingSnapshot,
) -> PrimitiveMapping:
    return {
        "configuration": configuration_snapshot.to_primitive(),
        "definition": definition.to_primitive(),
    }


def _schema(
    strategy_fields: tuple[SchemaField, ...],
    contextual_definitions: tuple[SchemaField, ...],
    outcomes: tuple[ConfiguredOutcome, ...],
    outcome_field_snapshots: tuple[tuple[SchemaField, ...], ...],
) -> SignalFeatureSchema:
    feature_fields = tuple(
        replace(field, name=f"feature_{field.name}")
        for field in sorted(
            (*strategy_fields, *contextual_definitions),
            key=lambda field: field.name,
        )
    )
    outcome_fields = tuple(
        replace(field, name=f"outcome_{outcome.namespace}_{field.name}")
        for outcome, fields in zip(outcomes, outcome_field_snapshots, strict=True)
        for field in fields
    )
    return SignalFeatureSchema(
        FEATURE_SCHEMA_VERSION,
        OUTCOME_SCHEMA_VERSION,
        (*_identity_fields(), *_disposition_fields(), *feature_fields, *outcome_fields),
    )


def _identity_fields() -> tuple[SchemaField, ...]:
    timing = "known before outcome labeling"
    definitions = (
        ("row_id", "string", "sha256", "QF-7 row identity"),
        ("candidate_id", "string", "sha256", "stable logical candidate identity"),
        ("study_id", "string", "sha256", "QF-7 feature-dataset identity"),
        (
            "prediction_study_ids",
            "object",
            "namespace_to_sha256",
            "QF-11 study identities used for outcome construction",
        ),
        ("source_dataset_id", "string", "sha256", "immutable QF-3 dataset ID"),
        (
            "dataset_fingerprint",
            "string",
            "sha256",
            "QF-3 normalized bar fingerprint",
        ),
        ("symbol", "string", "canonical_symbol", "QF-3 canonical symbol"),
        (
            "signal_session",
            "date",
            "exchange_session",
            "completed signal/session date",
        ),
        ("strategy_id", "string", "component_name", "source QF-11 rule ID"),
        (
            "strategy_implementation_version",
            "string",
            "version",
            "source QF-11 rule implementation version",
        ),
        (
            "strategy_configuration_id",
            "string",
            "sha256",
            "complete source QF-11 rule configuration identity",
        ),
        (
            "candidate_rule_id",
            "string",
            "component_name",
            "QF-7 candidate prediction-rule adapter ID",
        ),
        (
            "candidate_rule_configuration_id",
            "string",
            "sha256",
            "QF-7 candidate rule configuration identity",
        ),
        (
            "strategy_parameters_id",
            "string",
            "sha256",
            "complete parameter-set identity",
        ),
        (
            "strategy_parameters",
            "object",
            "parameter_mapping",
            "complete parameter-set snapshot",
        ),
        ("provider_name", "string", "provider_name", "QF-3 provider provenance"),
        (
            "adjustment_mode",
            "string",
            "adjustment_policy",
            "QF-3 price adjustment mode",
        ),
        ("ohlc_basis", "string", "price_basis", "QF-3 OHLC basis"),
        ("volume_basis", "string", "volume_basis", "QF-3 volume basis"),
        (
            "feature_schema_version",
            "string",
            "schema_version",
            "QF-7 feature schema version",
        ),
        (
            "outcome_schema_version",
            "string",
            "schema_version",
            "QF-7 outcome schema version",
        ),
    )
    return tuple(
        SchemaField(
            name,
            SchemaFieldCategory.IDENTITY,
            data_type,
            unit,
            False,
            source,
            timing,
        )
        for name, data_type, unit, source in definitions
    )


def _disposition_fields() -> tuple[SchemaField, ...]:
    timing = "fixed by the prediction rule before outcome labeling"
    definitions = (
        ("signal_disposition", "string", "enum", False),
        ("disposition_reason_codes", "array", "reason_codes", False),
        ("disposition_explanation", "string", "text", True),
        ("direction", "string", "up_or_down", True),
        ("selected_rule_reason", "string", "reason_code", True),
        ("matched_rule_reasons", "array", "reason_codes", False),
    )
    return tuple(
        SchemaField(
            name,
            SchemaFieldCategory.DISPOSITION,
            data_type,
            unit,
            nullable,
            "typed QF-7 SignalFeatureCandidate classification",
            timing,
        )
        for name, data_type, unit, nullable in definitions
    )


def _enrich_candidates(
    dataset: MarketDataset,
    candidates: tuple[SignalFeatureCandidate, ...],
    bar_indexes: dict[date, int],
    contextual_features: tuple[ContextualFeature, ...],
    contextual_definitions: tuple[SchemaField, ...],
    contextual_configuration_snapshots: tuple[PrimitiveMappingSnapshot, ...],
) -> tuple[SignalFeatureCandidate, ...]:
    if any(candidate.contextual_features for candidate in candidates):
        raise InvalidPredictionOutputError(
            "candidate rules must not pre-populate builder-owned contextual features"
        )
    if not candidates:
        return ()
    values_by_session: dict[date, list[SignalFeatureValue]] = {
        candidate.signal_session: [] for candidate in candidates
    }
    for feature, definition, configuration_snapshot in zip(
        contextual_features,
        contextual_definitions,
        contextual_configuration_snapshots,
        strict=True,
    ):
        expected_configuration_id = configuration_identity(
            configuration_snapshot.to_primitive()
        )
        if feature.__class__ not in _TRUSTED_ALIGNED_CONTEXT_TYPES:
            values: dict[date, Decimal | None] = {}
            for candidate in candidates:
                value = feature.value_from_history(
                    _causal_history(
                        dataset,
                        bar_indexes[candidate.signal_session],
                        candidate.signal_session,
                    )
                )
                if (
                    feature.configuration_id != expected_configuration_id
                    or configuration_identity(feature.configuration())
                    != expected_configuration_id
                ):
                    raise InvalidPredictionOutputError(
                        "contextual feature configuration changed during calculation"
                    )
                values[candidate.signal_session] = value
        else:
            aligned_callback = getattr(feature, "values_for_dataset", None)
            if not callable(aligned_callback):
                raise InvalidPredictionOutputError(
                    f"contextual feature {feature.name} has an invalid aligned "
                    "calculation callback"
                )
            calculate_aligned = cast(
                Callable[[MarketDataset], tuple[Decimal | None, ...]],
                aligned_callback,
            )
            raw_aligned_values = cast(object, calculate_aligned(dataset))
            if not isinstance(raw_aligned_values, tuple):
                raise InvalidPredictionOutputError(
                    f"contextual feature {feature.name} returned a misaligned series"
                )
            raw_aligned_tuple = cast(tuple[object, ...], raw_aligned_values)
            if len(raw_aligned_tuple) != len(dataset.bars):
                raise InvalidPredictionOutputError(
                    f"contextual feature {feature.name} returned a misaligned series"
                )
            aligned_values = cast(tuple[Decimal | None, ...], raw_aligned_tuple)
            values = {
                candidate.signal_session: aligned_values[
                    bar_indexes[candidate.signal_session]
                ]
                for candidate in candidates
            }
        if (
            feature.configuration_id != expected_configuration_id
            or configuration_identity(feature.configuration())
            != expected_configuration_id
        ):
            raise InvalidPredictionOutputError(
                "contextual feature configuration changed during calculation"
            )
        for signal_session, value in values.items():
            if value is None and not definition.nullable:
                raise InvalidPredictionOutputError(
                    f"contextual feature {feature.name} returned null for a "
                    "non-nullable schema field"
                )
            values_by_session[signal_session].append(
                SignalFeatureValue(feature.name, value)
            )
    return tuple(
        replace(
            candidate,
            contextual_features=tuple(
                sorted(
                    values_by_session[candidate.signal_session],
                    key=lambda item: item.name,
                )
            ),
        )
        for candidate in candidates
    )


def _causal_history(
    dataset: MarketDataset, final_index: int, final_session: date
) -> MarketDataset:
    actions = tuple(
        replace(
            action,
            action_id=f"causal-prefix-action-{index}",
            source_dataset_id=_CAUSAL_PROVENANCE_SENTINEL,
        )
        for index, action in enumerate(
            action
            for action in dataset.corporate_actions
            if _action_session(action) <= final_session
        )
    )
    dividends = tuple(action for action in actions if isinstance(action, CashDividend))
    splits = tuple(action for action in actions if not isinstance(action, CashDividend))
    metadata = replace(
        dataset.metadata,
        retrieved_at=datetime(1970, 1, 1, tzinfo=UTC),
        requested_end=min(dataset.metadata.requested_end, final_session),
        actual_last_session=final_session,
        bar_count=final_index + 1,
        missing_sessions=tuple(
            session
            for session in dataset.metadata.missing_sessions
            if session <= final_session
        ),
        split_sessions=tuple(
            session
            for session in dataset.metadata.split_sessions
            if session <= final_session
        ),
        dividend_sessions=tuple(
            session
            for session in dataset.metadata.dividend_sessions
            if session <= final_session
        ),
        corporate_action_count=len(actions),
        dividend_count=len(dividends),
        split_count=len(splits),
        corporate_actions_complete=False,
        corporate_action_snapshot_id=_CAUSAL_PROVENANCE_SENTINEL,
        raw_sha256=_CAUSAL_PROVENANCE_SENTINEL,
        data_sha256=_CAUSAL_PROVENANCE_SENTINEL,
        dataset_id=_CAUSAL_PROVENANCE_SENTINEL,
        raw_location=_CAUSAL_PROVENANCE_SENTINEL,
        normalized_location=_CAUSAL_PROVENANCE_SENTINEL,
        corporate_actions_location=_CAUSAL_PROVENANCE_SENTINEL,
        adapter_version=_CAUSAL_PROVENANCE_SENTINEL,
    )
    return MarketDataset(dataset.bars[: final_index + 1], metadata, actions)


def _action_session(action: CorporateAction) -> date:
    if isinstance(action, CashDividend):
        return action.ex_dividend_session
    return action.effective_session


def _candidate_id(
    market_data: PredictionMarketData, candidate: SignalFeatureCandidate
) -> str:
    if not _is_canonical_sha256(candidate.source_rule_configuration_id):
        raise InvalidPredictionOutputError(
            "candidate source rule configuration ID must be a canonical SHA-256"
        )
    return configuration_identity(
        {
            "candidate_rule_configuration_id": candidate.strategy_configuration_id,
            "dataset_fingerprint": market_data.bars_fingerprint,
            "parameters": candidate.parameters_primitive(),
            "record_type": "signal_feature_candidate",
            "signal_session": candidate.signal_session.isoformat(),
            "source_rule_configuration_id": candidate.source_rule_configuration_id,
            "source_rule_id": candidate.source_rule_id,
            "source_rule_implementation_version": (
                candidate.source_rule_implementation_version
            ),
            "symbol": candidate.symbol,
        }
    )


def _build_row(
    feature_dataset_id: str,
    market_data: PredictionMarketData,
    schema: SignalFeatureSchema,
    candidate: SignalFeatureCandidate,
    candidate_id: str,
    outcomes: tuple[ConfiguredOutcome, ...],
    outcome_values: dict[str, dict[date, PrimitiveMapping]],
    study_ids: dict[str, str],
    unavailable_outcome_values: dict[str, PrimitiveMapping],
) -> SignalFeatureRow:
    candidate_features = {
        f"feature_{feature_name}": feature_value
        for feature_name, feature_value in candidate.features_primitive().items()
    }
    declared_feature_fields = {
        field.name: field
        for field in schema.fields
        if field.category is SchemaFieldCategory.CONTEMPORANEOUS_FEATURE
    }
    declared_feature_names = set(declared_feature_fields)
    if set(candidate_features) != declared_feature_names:
        missing = sorted(declared_feature_names - set(candidate_features))
        unexpected = sorted(set(candidate_features) - declared_feature_names)
        raise InvalidPredictionOutputError(
            "candidate feature names do not match their declared schema: "
            f"missing={missing}, unexpected={unexpected}"
        )
    invalid_feature_names = tuple(
        sorted(
            feature_name
            for feature_name, feature_value in candidate_features.items()
            if not _schema_value_matches(
                declared_feature_fields[feature_name], feature_value
            )
        )
    )
    if invalid_feature_names:
        raise InvalidPredictionOutputError(
            "candidate feature values do not match their declared schema types or "
            f"nullability: {invalid_feature_names}"
        )
    parameters = candidate.parameters_primitive()
    values: PrimitiveMapping = {
        "adjustment_mode": market_data.adjustment_mode,
        "candidate_id": candidate_id,
        "candidate_rule_configuration_id": candidate.strategy_configuration_id,
        "candidate_rule_id": candidate.strategy_id,
        "dataset_fingerprint": market_data.bars_fingerprint,
        "direction": None if candidate.direction is None else candidate.direction.value,
        "disposition_explanation": candidate.explanation,
        "disposition_reason_codes": list(candidate.reason_codes),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "matched_rule_reasons": list(candidate.matched_rule_reasons),
        "ohlc_basis": market_data.ohlc_basis,
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "prediction_study_ids": dict(sorted(study_ids.items())),
        "provider_name": market_data.provider_name,
        "selected_rule_reason": candidate.selected_rule_reason,
        "signal_disposition": candidate.disposition.value,
        "signal_session": candidate.signal_session.isoformat(),
        "source_dataset_id": market_data.dataset_id,
        "strategy_configuration_id": candidate.source_rule_configuration_id,
        "strategy_id": candidate.source_rule_id,
        "strategy_implementation_version": (
            candidate.source_rule_implementation_version
        ),
        "strategy_parameters": parameters,
        "strategy_parameters_id": configuration_identity(parameters),
        "study_id": feature_dataset_id,
        "symbol": candidate.symbol,
        "volume_basis": market_data.volume_basis,
    }
    values.update(candidate_features)
    for configured_outcome in outcomes:
        unprefixed = outcome_values[configured_outcome.namespace].get(
            candidate.signal_session
        )
        if unprefixed is None:
            unprefixed = unavailable_outcome_values[configured_outcome.namespace]
        for field_name, field_value in unprefixed.items():
            values[f"outcome_{configured_outcome.namespace}_{field_name}"] = field_value
    values["row_id"] = _row_id(feature_dataset_id, values)
    if set(values) != set(schema.column_names):
        raise InvalidPredictionOutputError(
            "signal-feature row values do not match their complete schema"
        )
    fields_by_name = {field.name: field for field in schema.fields}
    invalid_value_names = tuple(
        sorted(
            field_name
            for field_name, field_value in values.items()
            if not _schema_value_matches(fields_by_name[field_name], field_value)
        )
    )
    if invalid_value_names:
        raise InvalidPredictionOutputError(
            "signal-feature row values do not match their declared schema types or "
            f"nullability: {invalid_value_names}"
        )
    values = {name: values[name] for name in schema.column_names}
    return SignalFeatureRow.capture(values)


def _validated_flattened_outcome_values(
    configured_outcome: ConfiguredOutcome,
    fields: tuple[SchemaField, ...],
    values: PrimitiveMapping,
) -> PrimitiveMapping:
    fields_by_name = {field.name: field for field in fields}
    if set(values) != set(fields_by_name):
        raise InvalidPredictionOutputError(
            f"outcome {configured_outcome.namespace} flattened values do not match "
            "their declared schema"
        )
    invalid_value_names = tuple(
        sorted(
            field_name
            for field_name, field_value in values.items()
            if not _schema_value_matches(fields_by_name[field_name], field_value)
        )
    )
    if invalid_value_names:
        raise InvalidPredictionOutputError(
            f"outcome {configured_outcome.namespace} flattened values do not match "
            "declared field types or nullability: "
            f"{invalid_value_names}"
        )
    return PrimitiveMappingSnapshot.capture(values).to_primitive()


def _validated_unavailable_outcome_values(
    configured_outcome: ConfiguredOutcome,
    fields: tuple[SchemaField, ...],
    values: PrimitiveMapping,
) -> PrimitiveMapping:
    validated = _validated_flattened_outcome_values(configured_outcome, fields, values)
    fields_by_name = {field.name: field for field in fields}
    availability_field = fields_by_name.get("available")
    if availability_field is not None and (
        availability_field.data_type != "boolean"
        or availability_field.nullable
        or validated.get("available") is not False
    ):
        raise SignalFeatureDatasetError(
            "outcome availability fields must be non-nullable booleans with "
            "an unavailable default of false"
        )
    return validated


def _row_id(feature_dataset_id: str, values: PrimitiveMapping) -> str:
    row_payload = dict(values)
    row_payload.pop("row_id", None)
    return configuration_identity(
        {
            "feature_dataset_id": feature_dataset_id,
            "record_type": "signal_feature_row",
            "row_payload": row_payload,
        }
    )


def _initialize_or_validate_progress(
    destination: Path,
    feature_dataset_id: str,
    market_data: PredictionMarketData,
    configuration: PrimitiveMappingSnapshot,
    schema: SignalFeatureSchema,
) -> None:
    manifest: PrimitiveMapping = {
        "component": "quantforge_signal_feature_dataset",
        "configuration": configuration.to_primitive(),
        "dataset_id": feature_dataset_id,
        "engine_version": FEATURE_DATASET_ENGINE_VERSION,
        "market_data": market_data.to_primitive(),
        "status": "in_progress",
    }
    if destination.exists():
        if not (destination / "manifest.json").exists():
            temporary_paths = _validate_manifestless_staging_destination(destination)
            existing_schema_path = destination / "schema.json"
            if existing_schema_path.exists():
                if _read_mapping(existing_schema_path) != schema.to_primitive():
                    raise SignalFeaturePersistenceError(
                        "manifest-less signal-feature schema is incompatible with "
                        "this run"
                    )
            else:
                _atomic_json(existing_schema_path, schema.to_primitive())
            (destination / "rows").mkdir(exist_ok=True)
            for temporary_path in temporary_paths:
                try:
                    temporary_path.unlink()
                except OSError as error:
                    raise SignalFeaturePersistenceError(
                        "failed to remove an incomplete startup artifact"
                    ) from error
            _atomic_json(destination / "manifest.json", manifest)
            return
        try:
            existing = _read_mapping(destination / "manifest.json")
            existing_schema = _read_mapping(destination / "schema.json")
        except SignalFeaturePersistenceError:
            raise
        if existing != manifest or existing_schema != schema.to_primitive():
            raise SignalFeaturePersistenceError(
                "persisted signal-feature state is incompatible with this run"
            )
    else:
        destination.mkdir(parents=True)
        (destination / "rows").mkdir()
        _atomic_json(destination / "schema.json", schema.to_primitive())
        _atomic_json(destination / "manifest.json", manifest)


def _manifest_status(destination: Path) -> str | None:
    manifest_path = destination / "manifest.json"
    if not manifest_path.exists():
        _validate_manifestless_staging_destination(destination)
        return None
    status = _read_mapping(manifest_path).get("status")
    return status if isinstance(status, str) else None


def _validate_manifestless_staging_destination(destination: Path) -> tuple[Path, ...]:
    if not destination.is_dir():
        raise SignalFeaturePersistenceError(
            "signal-feature destination without a manifest must be a directory"
        )
    rows_directory = destination / "rows"
    try:
        if rows_directory.exists() and (
            not rows_directory.is_dir() or any(rows_directory.iterdir())
        ):
            raise SignalFeaturePersistenceError(
                "manifest-less signal-feature state contains row checkpoints"
            )
        temporary_paths = tuple(
            child
            for child in destination.iterdir()
            if child.is_file()
            and (
                child.name.startswith(".schema.json.")
                or child.name.startswith(".manifest.json.")
            )
            and child.name.endswith(".tmp")
        )
        allowed_paths = {
            destination / "schema.json",
            rows_directory,
            *temporary_paths,
        }
        if any(child not in allowed_paths for child in destination.iterdir()):
            raise SignalFeaturePersistenceError(
                "signal-feature destination exists without a manifest and contains "
                "non-restartable state"
            )
    except OSError as error:
        raise SignalFeaturePersistenceError(
            "failed to inspect manifest-less signal-feature state"
        ) from error
    return temporary_paths


def _load_progress_rows(
    destination: Path,
    feature_dataset_id: str,
    schema: SignalFeatureSchema,
) -> dict[str, SignalFeatureRow]:
    rows_directory = destination / "rows"
    if not rows_directory.is_dir():
        raise SignalFeaturePersistenceError(
            "signal-feature progress rows directory is missing"
        )
    rows: dict[str, SignalFeatureRow] = {}
    for path in sorted(rows_directory.glob("*.json")):
        row = SignalFeatureRow.capture(_read_mapping(path))
        row_values = row.to_primitive()
        expected_row_id = _row_id(feature_dataset_id, row_values)
        if (
            path.stem != row.row_id
            or row.row_id != expected_row_id
            or set(row_values) != set(schema.column_names)
            or row.candidate_id in rows
        ):
            raise SignalFeaturePersistenceError(
                "persisted signal-feature row identity or schema is invalid"
            )
        rows[row.candidate_id] = row
    return rows


def _study_ids_from_rows(rows: Iterable[SignalFeatureRow]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        raw = row.to_primitive().get("prediction_study_ids")
        if not isinstance(raw, dict) or any(
            not _is_canonical_sha256(value) for value in raw.values()
        ):
            raise SignalFeaturePersistenceError(
                "persisted row has invalid QF-11 study identities"
            )
        current = cast(dict[str, str], raw)
        for namespace, study_id in current.items():
            if result.get(namespace, study_id) != study_id:
                raise SignalFeaturePersistenceError(
                    "persisted rows disagree on QF-11 study identity"
                )
            result[namespace] = study_id
    return result


def _persist_progress_row(destination: Path, row: SignalFeatureRow) -> None:
    path = destination / "rows" / f"{row.row_id}.json"
    if path.exists():
        if _read_mapping(path) != row.to_primitive():
            raise SignalFeaturePersistenceError(
                "existing signal-feature row conflicts with deterministic output"
            )
        return
    _atomic_json(path, row.to_primitive())


def _finalize_export(destination: Path, result: SignalFeatureDatasetResult) -> None:
    _atomic_text(destination / "features.csv", _render_csv(result))
    _atomic_json(destination / "summary.json", result.summary.to_primitive())
    _atomic_json(destination / "schema.json", result.schema.to_primitive())
    _atomic_json(destination / "manifest.json", result.manifest_primitive())


def _load_completed_result(
    destination: Path,
    feature_dataset_id: str,
    market_data: PredictionMarketData,
    configuration: PrimitiveMappingSnapshot,
    schema: SignalFeatureSchema,
    dataset: MarketDataset,
    strategy: _SignalFeatureRule,
    strategy_configuration_snapshot: PrimitiveMappingSnapshot,
    feature_configuration: PrimitiveMapping,
    outcomes: tuple[ConfiguredOutcome, ...],
    outcome_field_snapshots: tuple[tuple[SchemaField, ...], ...],
    unavailable_outcome_values: dict[str, PrimitiveMapping],
    outcome_configuration_ids: dict[str, str],
) -> SignalFeatureDatasetResult:
    manifest = _read_mapping(destination / "manifest.json")
    rows_by_candidate = _load_progress_rows(destination, feature_dataset_id, schema)
    rows = tuple(sorted(rows_by_candidate.values(), key=lambda row: row.signal_session))
    study_ids = _study_ids_from_rows(rows)
    if rows:
        prediction_study_ids = tuple(study_ids[name] for name in sorted(study_ids))
    else:
        prediction_study_ids = _empty_dataset_study_ids(
            dataset,
            strategy,
            strategy_configuration_snapshot,
            feature_configuration,
            outcomes,
            outcome_field_snapshots,
            unavailable_outcome_values,
            outcome_configuration_ids,
        )
    result = SignalFeatureDatasetResult(
        feature_dataset_id,
        FEATURE_DATASET_ENGINE_VERSION,
        market_data,
        configuration,
        schema,
        rows,
        summarize_dispositions(rows),
        prediction_study_ids,
        _LIMITATIONS,
    )
    if (
        manifest != result.manifest_primitive()
        or _read_mapping(destination / "schema.json") != schema.to_primitive()
        or _read_mapping(destination / "summary.json") != result.summary.to_primitive()
    ):
        raise SignalFeaturePersistenceError(
            "completed signal-feature metadata does not match deterministic state"
        )
    try:
        existing_csv = (destination / "features.csv").read_text(encoding="utf-8")
    except OSError as error:
        raise SignalFeaturePersistenceError(
            "completed signal-feature CSV cannot be read"
        ) from error
    if existing_csv != _render_csv(result):
        raise SignalFeaturePersistenceError(
            "completed signal-feature CSV does not match deterministic rows"
        )
    return result


def _empty_dataset_study_ids(
    dataset: MarketDataset,
    strategy: _SignalFeatureRule,
    strategy_configuration_snapshot: PrimitiveMappingSnapshot,
    feature_configuration: PrimitiveMapping,
    outcomes: tuple[ConfiguredOutcome, ...],
    outcome_field_snapshots: tuple[tuple[SchemaField, ...], ...],
    unavailable_outcome_values: dict[str, PrimitiveMapping],
    outcome_configuration_ids: dict[str, str],
) -> tuple[str, ...]:
    empty_rule = _FixedCandidateRule(
        strategy,
        strategy_configuration_snapshot,
        (),
        _fixed_candidate_population_id(()),
        0,
    )
    prepared_dataset = prepare_prediction_study_dataset(dataset)
    study_ids: list[str] = []
    market_data = PredictionMarketData.from_qf3(dataset.metadata)
    for outcome, fields in zip(outcomes, outcome_field_snapshots, strict=True):
        outcome_run = _run_configured_outcome(
            outcome,
            outcome_configuration_ids[outcome.namespace],
            dataset,
            prepared_dataset,
            empty_rule,
            feature_configuration,
        )
        _validate_outcome_session_keys(
            outcome,
            frozenset(),
            outcome_run.values_by_session,
        )
        study_ids.append(
            _bound_outcome_study_id(
                outcome,
                outcome_configuration_ids[outcome.namespace],
                outcome_run.study_id,
                market_data,
                empty_rule,
                feature_configuration,
                fields,
                unavailable_outcome_values[outcome.namespace],
            )
        )
    return tuple(study_ids)


def _render_csv(result: SignalFeatureDatasetResult) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=result.schema.column_names, lineterminator="\n"
    )
    writer.writeheader()
    for row in result.rows:
        primitive = row.to_primitive()
        writer.writerow(
            {
                name: _csv_value(primitive.get(name))
                for name in result.schema.column_names
            }
        )
    return stream.getvalue()


def _csv_value(value: Primitive) -> str | int | float | bool | None:
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return value


def _atomic_json(path: Path, values: PrimitiveMapping) -> None:
    _atomic_text(
        path,
        json.dumps(
            values,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _atomic_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except OSError as error:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise SignalFeaturePersistenceError(
            f"failed to atomically persist signal-feature artifact: {path.name}"
        ) from error


def _read_mapping(path: Path) -> PrimitiveMapping:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SignalFeaturePersistenceError(
            f"failed to load signal-feature artifact: {path.name}"
        ) from error
    if not isinstance(loaded, dict):
        raise SignalFeaturePersistenceError(
            f"signal-feature artifact must be a JSON object: {path.name}"
        )
    loaded_mapping = cast(dict[object, object], loaded)
    if any(not isinstance(key, str) for key in loaded_mapping):
        raise SignalFeaturePersistenceError(
            f"signal-feature artifact keys must be strings: {path.name}"
        )
    return cast(PrimitiveMapping, loaded_mapping)


def _sorted_fields(fields: tuple[SchemaField, ...]) -> tuple[SchemaField, ...]:
    return tuple(sorted(fields, key=lambda field: field.name))


def _outcome_field(
    name: str,
    data_type: str,
    unit: str,
    nullable: bool,
    timing: str,
    calculation: str = "typed QF-11 outcome labeler and evaluator",
) -> SchemaField:
    return SchemaField(
        name,
        SchemaFieldCategory.FUTURE_OUTCOME,
        data_type,
        unit,
        nullable,
        calculation,
        timing,
    )
