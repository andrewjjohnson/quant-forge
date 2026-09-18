"""Generic typed contracts for causal prediction studies."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol, TypeVar, cast

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.data.models import MarketDataset
from quantforge.data.multi_timeframe import TimeframeBarSeries
from quantforge.indicators import Indicator
from quantforge.prediction.context import (
    PredictionContextRequirements,
    PredictionRuleContext,
)
from quantforge.prediction.outcome_resolution import (
    OutcomeEvaluationRequest,
    OutcomeResolution,
)
from quantforge.prediction.outcome_temporal import (
    OutcomeAnchorKind,
    OutcomeTemporalError,
    outcome_temporal_configuration,
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


class StudyOutcomeComponent(Protocol):
    """Metadata and dataset validation shared by session and request consumers."""

    @property
    def name(self) -> str: ...

    @property
    def implementation_version(self) -> str: ...

    @property
    def result_schema_version(self) -> str: ...

    @property
    def required_market_fields(self) -> tuple[str, ...]: ...

    @property
    def configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...

    def validate_dataset(self, dataset: MarketDataset) -> None: ...


class OutcomeLabeler(StudyOutcomeComponent, Protocol[OutcomeValuesT]):
    """Legacy session outcome execution contract."""

    @property
    def required_future_sessions(self) -> int: ...

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[OutcomeValuesT] | None: ...


class RequestOutcomeLabeler(Protocol[OutcomeValuesT]):
    """Opt-in future labeler receiving exact temporal inputs after prediction.

    Timestamp consumers use the generic resolver's typed availability metadata.
    This protocol does not enable timestamp membership or session-engine replay.
    A session labeler may implement both protocols without changing its results.
    """

    @property
    def configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...

    def label_request(
        self, dataset: MarketDataset, request: OutcomeEvaluationRequest
    ) -> OutcomeLabel[OutcomeValuesT] | None: ...


class TimestampStudyOutcomeLabeler(StudyOutcomeComponent, Protocol[OutcomeValuesT]):
    """QF-46 request consumer with a runner-bounded canonical label source.

    The source is supplied only after causal predictions are fixed. Its identity
    is already bound by the existing OutcomeEvaluationRequest. Use the supplied
    full-source resolution for availability; resolving the bounded source alone
    cannot distinguish missing observations from the end of the full artifact.
    """

    @property
    def required_future_duration(self) -> timedelta: ...

    def label_request(
        self,
        dataset: MarketDataset,
        request: OutcomeEvaluationRequest,
        *,
        source: TimeframeBarSeries,
        resolution: OutcomeResolution,
    ) -> OutcomeLabel[OutcomeValuesT] | None: ...


def evaluate_outcome_request[OutcomeValuesT: PredictionValues](
    labeler: OutcomeLabeler[OutcomeValuesT]
    | RequestOutcomeLabeler[OutcomeValuesT]
    | TimestampStudyOutcomeLabeler[OutcomeValuesT],
    dataset: MarketDataset,
    request: OutcomeEvaluationRequest,
    *,
    source: TimeframeBarSeries | None = None,
    resolution: OutcomeResolution | None = None,
) -> OutcomeLabel[OutcomeValuesT] | None:
    """Dispatch an already-fixed prediction's request without changing legacy inputs.

    Dataset validation and causal prediction capture remain the caller's job.
    No timestamp is synthesized for a session-only caller. Elapsed requests must
    be handled explicitly by a request-aware component, never by ``label(date)``.
    """
    from quantforge.configuration import configuration_identity

    configuration = labeler.configuration()
    sessions = getattr(labeler, "required_future_sessions", None)
    if sessions is not None and type(sessions) is not int:
        raise OutcomeTemporalError("outcome session horizon must be an integer")
    temporal = outcome_temporal_configuration(
        configuration, required_future_sessions=sessions
    )
    if (
        request.dataset_id != dataset.metadata.dataset_id
        or request.dataset_fingerprint != dataset.metadata.data_sha256
        or request.outcome_configuration_id != labeler.configuration_id
        or labeler.configuration_id != configuration_identity(configuration)
        or request.temporal_configuration != temporal
    ):
        raise OutcomeTemporalError(
            "outcome evaluation request differs from its component or dataset"
        )
    callback = getattr(labeler, "label_request", None)
    if resolution is not None and source is None:
        raise OutcomeTemporalError("outcome resolution requires its bounded source")
    if source is not None:
        if (
            source.dataset_reference != request.source_reference
            or source.timeframe != temporal.observation_timeframe
        ):
            raise OutcomeTemporalError("outcome source differs from request")
        if (
            not isinstance(resolution, OutcomeResolution)
            or resolution.request != request
            or (
                resolution.observation is not None
                and resolution.observation not in source.bars
            )
        ):
            raise OutcomeTemporalError("outcome resolution differs from request/source")
        if not callable(callback):
            raise OutcomeTemporalError("timestamp studies require label_request()")
        return cast(
            TimestampStudyOutcomeLabeler[OutcomeValuesT], labeler
        ).label_request(dataset, request, source=source, resolution=resolution)
    if callable(callback):
        return cast(
            Callable[
                [MarketDataset, OutcomeEvaluationRequest],
                OutcomeLabel[OutcomeValuesT] | None,
            ],
            callback,
        )(dataset, request)
    if request.anchor.kind is not OutcomeAnchorKind.SESSION:
        raise OutcomeTemporalError("exact timestamp outcomes require label_request()")
    return cast(OutcomeLabeler[OutcomeValuesT], labeler).label(
        dataset, request.anchor.signal_session
    )


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
    temporal_resolution: PrimitiveMappingSnapshot | None = None

    def to_primitive(self) -> PrimitiveMapping:
        primitive: PrimitiveMapping = {
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
        if self.temporal_resolution is not None:
            primitive["temporal_resolution"] = self.temporal_resolution.to_primitive()
        return primitive


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
    outcome_labeler: (
        OutcomeLabeler[OutcomeValuesT] | TimestampStudyOutcomeLabeler[OutcomeValuesT]
    )
    evaluator: PredictionEvaluator[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]
    feature_configuration_snapshot: PrimitiveMappingSnapshot
    result_schema_version: str = "1"
    outcome_source: TimeframeBarSeries | None = None

    @classmethod
    def create(
        cls,
        strategy: PredictionRule[PredictionRecordT]
        | MultiTimeframePredictionRule[PredictionRecordT],
        outcome_labeler: OutcomeLabeler[OutcomeValuesT]
        | TimestampStudyOutcomeLabeler[OutcomeValuesT],
        evaluator: PredictionEvaluator[
            PredictionRecordT, OutcomeValuesT, EvaluationValuesT
        ],
        *,
        feature_configuration: PrimitiveMapping | None = None,
        result_schema_version: str = "1",
        outcome_source: TimeframeBarSeries | None = None,
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
            outcome_source,
        )

    @property
    def feature_configuration(self) -> PrimitiveMapping:
        return self.feature_configuration_snapshot.to_primitive()
