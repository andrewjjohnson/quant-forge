"""Generic typed contracts for causal prediction studies."""

from dataclasses import dataclass
from datetime import date
from typing import Protocol, TypeVar

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.data.models import MarketDataset
from quantforge.indicators import Indicator
from quantforge.prediction.context import (
    PredictionContextRequirements,
    PredictionRuleContext,
)


class PredictionRuleParameters(Protocol):
    """Typed prediction-rule parameters with stable serialization."""

    def to_primitive(self) -> PrimitiveMapping: ...


class PredictionRecord(Protocol):
    """One causal prediction without any required classification payload.

    Implementations must support a component-independent ``copy.deepcopy`` so a
    completed study row never retains the strategy's owned instance.
    """

    @property
    def symbol(self) -> str: ...

    @property
    def signal_session(self) -> date: ...

    @property
    def strategy_id(self) -> str: ...

    @property
    def strategy_implementation_version(self) -> str: ...

    @property
    def strategy_configuration_id(self) -> str: ...

    def parameters_primitive(self) -> PrimitiveMapping: ...

    def features_primitive(self) -> PrimitiveMapping: ...

    def prediction_primitive(self) -> PrimitiveMapping: ...


class PredictionValues(Protocol):
    """Typed study-specific values with deterministic primitive serialization.

    Implementations must support a component-independent ``copy.deepcopy`` for
    immutable study-row capture.
    """

    def to_primitive(self) -> PrimitiveMapping: ...


OutcomeValuesT = TypeVar("OutcomeValuesT", bound=PredictionValues)
EvaluationValuesT = TypeVar("EvaluationValuesT", bound=PredictionValues, covariant=True)
EvaluatorOutcomeValuesT = TypeVar("EvaluatorOutcomeValuesT", bound=PredictionValues)
PredictionRecordT = TypeVar("PredictionRecordT", bound=PredictionRecord, covariant=True)
EvaluatorPredictionRecordT = TypeVar(
    "EvaluatorPredictionRecordT", bound=PredictionRecord, contravariant=True
)


class PredictionRuleOutput(Protocol[PredictionRecordT]):
    """Ordered predictions emitted before any outcome labeling occurs."""

    @property
    def strategy_id(self) -> str: ...

    @property
    def strategy_configuration_id(self) -> str: ...

    @property
    def dataset_id(self) -> str: ...

    @property
    def signals(self) -> tuple[PredictionRecordT, ...]: ...

    @property
    def contract_version(self) -> str: ...


class PredictionRule(Protocol[PredictionRecordT]):
    """Generate typed causal predictions without calculating outcomes."""

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

    def configuration(self) -> PrimitiveMapping: ...

    def generate(
        self, dataset: MarketDataset
    ) -> PredictionRuleOutput[PredictionRecordT]: ...


class MultiTimeframePredictionRule(Protocol[PredictionRecordT]):
    """Generate predictions from only declared multi-timeframe inputs."""

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
    def context_requirements(self) -> PredictionContextRequirements: ...

    @property
    def configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...

    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> PredictionRuleOutput[PredictionRecordT]: ...


@dataclass(frozen=True, slots=True)
class OutcomeLabel[OutcomeValuesT: PredictionValues]:
    """Study-specific future values before generic provenance is attached."""

    signal_session: date
    outcome_session: date
    values: OutcomeValuesT


class OutcomeLabeler(Protocol[OutcomeValuesT]):
    """Generate future outcomes only after prediction signals are fixed."""

    @property
    def name(self) -> str: ...

    @property
    def implementation_version(self) -> str: ...

    @property
    def result_schema_version(self) -> str: ...

    @property
    def required_future_sessions(self) -> int: ...

    @property
    def required_market_fields(self) -> tuple[str, ...]: ...

    @property
    def configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...

    def validate_dataset(self, dataset: MarketDataset) -> None: ...

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[OutcomeValuesT] | None: ...


class PredictionEvaluator(
    Protocol[
        EvaluatorPredictionRecordT,
        EvaluatorOutcomeValuesT,
        EvaluationValuesT,
    ]
):
    """Evaluate an already-fixed prediction against an already-built outcome."""

    @property
    def name(self) -> str: ...

    @property
    def implementation_version(self) -> str: ...

    @property
    def result_schema_version(self) -> str: ...

    @property
    def configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...

    def evaluate(
        self,
        signal: EvaluatorPredictionRecordT,
        outcome: "PredictionOutcome[EvaluatorOutcomeValuesT]",
    ) -> EvaluationValuesT: ...


@dataclass(frozen=True, slots=True)
class PredictionOutcome[OutcomeValuesT: PredictionValues]:
    """A typed future outcome with stable labeler and dataset provenance."""

    outcome_id: str
    outcome_name: str
    outcome_implementation_version: str
    outcome_configuration_id: str
    outcome_result_schema_version: str
    dataset_id: str
    dataset_fingerprint: str
    signal_session: date
    outcome_session: date
    values: OutcomeValuesT

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_id": self.dataset_id,
            "outcome_configuration_id": self.outcome_configuration_id,
            "outcome_id": self.outcome_id,
            "outcome_implementation_version": self.outcome_implementation_version,
            "outcome_name": self.outcome_name,
            "outcome_result_schema_version": self.outcome_result_schema_version,
            "outcome_session": self.outcome_session.isoformat(),
            "signal_session": self.signal_session.isoformat(),
            "values": self.values.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class PredictionEvaluation[EvaluationValuesT: PredictionValues]:
    """Typed evaluator output that does not impose classification semantics."""

    evaluation_id: str
    evaluator_name: str
    evaluator_implementation_version: str
    evaluator_configuration_id: str
    evaluation_result_schema_version: str
    outcome_id: str
    values: EvaluationValuesT

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "evaluation_id": self.evaluation_id,
            "evaluation_result_schema_version": (self.evaluation_result_schema_version),
            "evaluator_configuration_id": self.evaluator_configuration_id,
            "evaluator_implementation_version": (self.evaluator_implementation_version),
            "evaluator_name": self.evaluator_name,
            "outcome_id": self.outcome_id,
            "values": self.values.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class PredictionStudy[
    PredictionRecordT: PredictionRecord,
    OutcomeValuesT: PredictionValues,
    EvaluationValuesT: PredictionValues,
]:
    """Runtime composition of one rule, outcome labeler, and evaluator."""

    strategy: (
        PredictionRule[PredictionRecordT]
        | MultiTimeframePredictionRule[PredictionRecordT]
    )
    outcome_labeler: OutcomeLabeler[OutcomeValuesT]
    evaluator: PredictionEvaluator[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]
    feature_configuration_snapshot: PrimitiveMappingSnapshot
    result_schema_version: str = "1"

    @classmethod
    def create(
        cls,
        strategy: PredictionRule[PredictionRecordT]
        | MultiTimeframePredictionRule[PredictionRecordT],
        outcome_labeler: OutcomeLabeler[OutcomeValuesT],
        evaluator: PredictionEvaluator[
            PredictionRecordT, OutcomeValuesT, EvaluationValuesT
        ],
        *,
        feature_configuration: PrimitiveMapping | None = None,
        result_schema_version: str = "1",
    ) -> "PredictionStudy[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]":
        configured_features: PrimitiveMapping = (
            {
                "feature_schema_version": "1",
                "source": "prediction_signal_feature_values",
            }
            if feature_configuration is None
            else feature_configuration
        )
        return cls(
            strategy,
            outcome_labeler,
            evaluator,
            PrimitiveMappingSnapshot.capture(configured_features),
            result_schema_version,
        )

    @property
    def feature_configuration(self) -> PrimitiveMapping:
        return self.feature_configuration_snapshot.to_primitive()
