import ast
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data import MarketDataset
from quantforge.indicators import Indicator
from quantforge.prediction import (
    AlwaysUpParameters,
    AlwaysUpPredictionStrategy,
    InvalidPredictionConfigurationError,
    InvalidPredictionOutputError,
    OutcomeLabel,
    PredictionEvaluation,
    PredictionFeature,
    PredictionOutcome,
    PredictionParameter,
    PredictionStudy,
    create_overnight_gap_prediction_study,
    run_prediction_study,
)

from ..helpers import make_dataset


@dataclass(frozen=True, slots=True)
class RecordingParameters:
    mode: str = "recording"

    def to_primitive(self) -> PrimitiveMapping:
        return {"mode": self.mode}


@dataclass(frozen=True, slots=True)
class NumericPrediction:
    symbol: str
    signal_session: date
    predicted_change: Decimal
    strategy_id: str
    strategy_implementation_version: str
    strategy_configuration_id: str
    strategy_parameters: tuple[PredictionParameter, ...]
    feature_values: tuple[PredictionFeature, ...]

    def parameters_primitive(self) -> PrimitiveMapping:
        return {item.name: item.value for item in self.strategy_parameters}

    def features_primitive(self) -> PrimitiveMapping:
        return {
            item.name: decimal_to_primitive(item.value) for item in self.feature_values
        }

    def prediction_primitive(self) -> PrimitiveMapping:
        return {"predicted_change": decimal_to_primitive(self.predicted_change)}


@dataclass(frozen=True, slots=True)
class FirstSerializationMutatingNumericPrediction(NumericPrediction):
    serialized_parameters: bool = False

    def parameters_primitive(self) -> PrimitiveMapping:
        primitive = super().parameters_primitive()
        if not self.serialized_parameters:
            object.__setattr__(
                self,
                "strategy_parameters",
                (PredictionParameter("mode", "changed_after_serialization"),),
            )
            object.__setattr__(self, "serialized_parameters", True)
        return primitive


@dataclass(frozen=True, slots=True)
class NumericPredictionOutput:
    strategy_id: str
    strategy_configuration_id: str
    dataset_id: str
    signals: tuple[NumericPrediction, ...]
    contract_version: str = "1"


class RecordingPredictionStrategy:
    name = "recording_prediction_rule"
    implementation_version = "1"
    required_indicators: tuple[Indicator, ...] = ()
    warm_up_observations = 1

    def __init__(self, events: list[str], feature_offset: Decimal = Decimal(0)) -> None:
        self._events = events
        self._feature_offset = feature_offset
        self._parameters = RecordingParameters()

    @property
    def parameters(self) -> RecordingParameters:
        return self._parameters

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_strategy",
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": self.parameters.to_primitive(),
            "required_indicators": [],
            "warm_up_observations": self.warm_up_observations,
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def generate(self, dataset: MarketDataset) -> NumericPredictionOutput:
        self._events.append("prediction_rule")
        parameters = (PredictionParameter("mode", self.parameters.mode),)
        signals = tuple(
            NumericPrediction(
                symbol=dataset.metadata.canonical_symbol,
                signal_session=bar.session_date,
                predicted_change=Decimal(1),
                strategy_id=self.name,
                strategy_implementation_version=self.implementation_version,
                strategy_configuration_id=self.configuration_id,
                strategy_parameters=parameters,
                feature_values=(
                    PredictionFeature("signal_close", bar.close + self._feature_offset),
                ),
            )
            for bar in dataset.bars
        )
        return NumericPredictionOutput(
            self.name,
            self.configuration_id,
            dataset.metadata.dataset_id,
            signals,
        )


class FirstSerializationMutatingPredictionStrategy(RecordingPredictionStrategy):
    name = "first_serialization_mutating_prediction_rule"

    def generate(self, dataset: MarketDataset) -> NumericPredictionOutput:
        output = super().generate(dataset)
        signals = tuple(
            FirstSerializationMutatingNumericPrediction(
                signal.symbol,
                signal.signal_session,
                signal.predicted_change,
                signal.strategy_id,
                signal.strategy_implementation_version,
                signal.strategy_configuration_id,
                signal.strategy_parameters,
                signal.feature_values,
            )
            for signal in output.signals
        )
        return NumericPredictionOutput(
            output.strategy_id,
            output.strategy_configuration_id,
            output.dataset_id,
            signals,
        )


class DatasetMutatingPredictionStrategy(RecordingPredictionStrategy):
    name = "dataset_mutating_prediction_rule"

    def generate(self, dataset: MarketDataset) -> NumericPredictionOutput:
        output = super().generate(dataset)
        object.__setattr__(dataset.bars[0], "close", Decimal(999))
        return output


class SharedSignalPredictionStrategy(RecordingPredictionStrategy):
    def __init__(
        self, events: list[str], shared_signals: list[NumericPrediction]
    ) -> None:
        super().__init__(events)
        self._shared_signals = shared_signals

    def generate(self, dataset: MarketDataset) -> NumericPredictionOutput:
        output = super().generate(dataset)
        self._shared_signals.extend(output.signals)
        return output


class SharedOutputPredictionStrategy(RecordingPredictionStrategy):
    def __init__(
        self, events: list[str], shared_outputs: list[NumericPredictionOutput]
    ) -> None:
        super().__init__(events)
        self._shared_outputs = shared_outputs

    def generate(self, dataset: MarketDataset) -> NumericPredictionOutput:
        output = super().generate(dataset)
        self._shared_outputs.append(output)
        return output


@dataclass(frozen=True, slots=True)
class FutureCloseValues:
    reference_close: Decimal
    future_close: Decimal

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "future_close": decimal_to_primitive(self.future_close),
            "reference_close": decimal_to_primitive(self.reference_close),
        }


class FutureCloseOutcomeLabeler:
    implementation_version = "1"
    result_schema_version = "1"
    required_market_fields = ("close",)

    def __init__(
        self,
        events: list[str],
        horizon: int = 1,
        returned_horizon: int | None = None,
    ) -> None:
        self._events = events
        self.required_future_sessions = horizon
        self._returned_horizon = (
            horizon if returned_horizon is None else returned_horizon
        )
        self.name = "future_close"

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_outcome_labeler",
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": {"future_sessions": self.required_future_sessions},
            "required_market_fields": list(self.required_market_fields),
            "result_schema_version": self.result_schema_version,
        }

    def validate_dataset(self, dataset: MarketDataset) -> None:
        assert dataset.bars
        self._events.append("outcome_validation")

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[FutureCloseValues] | None:
        self._events.append("outcome_label")
        indexes = {bar.session_date: index for index, bar in enumerate(dataset.bars)}
        index = indexes[signal_session]
        outcome_index = index + self._returned_horizon
        if outcome_index >= len(dataset.bars):
            return None
        return OutcomeLabel(
            signal_session,
            dataset.bars[outcome_index].session_date,
            FutureCloseValues(
                dataset.bars[index].close,
                dataset.bars[outcome_index].close,
            ),
        )


class SelectivelyUnavailableFutureCloseOutcomeLabeler(FutureCloseOutcomeLabeler):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.name = "selectively_unavailable_future_close"

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[FutureCloseValues] | None:
        if signal_session == dataset.bars[0].session_date:
            self._events.append("outcome_label")
            return None
        return super().label(dataset, signal_session)


class SignalMutatingFutureCloseOutcomeLabeler(FutureCloseOutcomeLabeler):
    def __init__(
        self,
        events: list[str],
        shared_signals: list[NumericPrediction],
        mutation_phase: str,
    ) -> None:
        super().__init__(events)
        self._shared_signals = shared_signals
        self._mutation_phase = mutation_phase
        self.name = f"{mutation_phase}_signal_mutating_future_close"

    def validate_dataset(self, dataset: MarketDataset) -> None:
        super().validate_dataset(dataset)
        if self._mutation_phase == "validation":
            self._mutate_signal(dataset)

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[FutureCloseValues] | None:
        if self._mutation_phase == "labeling":
            self._mutate_signal(dataset)
        return super().label(dataset, signal_session)

    def _mutate_signal(self, dataset: MarketDataset) -> None:
        object.__setattr__(
            self._shared_signals[0],
            "predicted_change",
            dataset.bars[-1].close,
        )


class DatasetMutatingFutureCloseOutcomeLabeler(FutureCloseOutcomeLabeler):
    def __init__(self, events: list[str], mutation_phase: str) -> None:
        super().__init__(events)
        self._mutation_phase = mutation_phase
        self.name = f"{mutation_phase}_dataset_mutating_future_close"

    def validate_dataset(self, dataset: MarketDataset) -> None:
        super().validate_dataset(dataset)
        if self._mutation_phase == "validation":
            self._mutate_dataset(dataset)

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[FutureCloseValues] | None:
        label = super().label(dataset, signal_session)
        if self._mutation_phase == "labeling":
            self._mutate_dataset(dataset)
        return label

    @staticmethod
    def _mutate_dataset(dataset: MarketDataset) -> None:
        object.__setattr__(dataset.bars[0], "close", Decimal(999))


class CallbackConfigurationMutatingOutcomeLabeler(FutureCloseOutcomeLabeler):
    def __init__(self, events: list[str], mutation_phase: str) -> None:
        super().__init__(events)
        self._mutation_phase = mutation_phase
        self._configuration_state = "original"
        self._label_call_count = 0
        self.name = f"{mutation_phase}_configuration_mutating_future_close"

    def configuration(self) -> PrimitiveMapping:
        base = super().configuration()
        parameters = cast(PrimitiveMapping, base["parameters"])
        return {
            **base,
            "parameters": {
                **parameters,
                "callback_configuration_state": self._configuration_state,
            },
        }

    def validate_dataset(self, dataset: MarketDataset) -> None:
        super().validate_dataset(dataset)
        if self._mutation_phase == "validation":
            self._configuration_state = "changed"

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[FutureCloseValues] | None:
        self._label_call_count += 1
        if self._mutation_phase == "validation" and self._label_call_count == 1:
            self._configuration_state = "original"
        elif self._mutation_phase == "labeling":
            self._configuration_state = (
                "changed" if self._label_call_count == 1 else "original"
            )
        return super().label(dataset, signal_session)


@dataclass(frozen=True, slots=True)
class FirstSerializationMutatingFutureCloseValues:
    reference_close: Decimal
    future_close: Decimal
    serialized: bool = False

    def to_primitive(self) -> PrimitiveMapping:
        primitive: PrimitiveMapping = {
            "future_close": decimal_to_primitive(self.future_close),
            "reference_close": decimal_to_primitive(self.reference_close),
        }
        if not self.serialized:
            object.__setattr__(self, "future_close", Decimal(999))
            object.__setattr__(self, "serialized", True)
        return primitive


class FirstSerializationMutatingOutcomeLabeler(FutureCloseOutcomeLabeler):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.name = "first_serialization_mutating_future_close"

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[FutureCloseValues] | None:
        label = super().label(dataset, signal_session)
        if label is None:
            return None
        return OutcomeLabel(
            label.signal_session,
            label.outcome_session,
            cast(
                FutureCloseValues,
                FirstSerializationMutatingFutureCloseValues(
                    label.values.reference_close,
                    label.values.future_close,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class FutureCloseChangeValues:
    close_change: Decimal

    def to_primitive(self) -> PrimitiveMapping:
        return {"close_change": decimal_to_primitive(self.close_change)}


class FutureCloseChangeEvaluator:
    name = "future_close_change"
    implementation_version = "1"
    result_schema_version = "1"

    def __init__(self, events: list[str], multiplier: Decimal = Decimal(1)) -> None:
        self._events = events
        self._multiplier = multiplier

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
                "multiplier": decimal_to_primitive(self._multiplier),
                "value": "future_close_minus_reference_close",
            },
            "result_schema_version": self.result_schema_version,
        }

    def evaluate(
        self,
        signal: NumericPrediction,
        outcome: PredictionOutcome[FutureCloseValues],
    ) -> FutureCloseChangeValues:
        assert signal.signal_session == outcome.signal_session
        self._events.append("evaluation")
        return FutureCloseChangeValues(
            (outcome.values.future_close - outcome.values.reference_close)
            * self._multiplier
        )


class SignalMutatingEvaluator(FutureCloseChangeEvaluator):
    name = "signal_mutating_future_close_change"

    def evaluate(
        self,
        signal: NumericPrediction,
        outcome: PredictionOutcome[FutureCloseValues],
    ) -> FutureCloseChangeValues:
        object.__setattr__(signal, "predicted_change", Decimal(999))
        return super().evaluate(signal, outcome)


class OutcomeMutatingEvaluator(FutureCloseChangeEvaluator):
    name = "outcome_mutating_future_close_change"

    def evaluate(
        self,
        signal: NumericPrediction,
        outcome: PredictionOutcome[FutureCloseValues],
    ) -> FutureCloseChangeValues:
        object.__setattr__(outcome.values, "future_close", Decimal(999))
        return super().evaluate(signal, outcome)


class OutputSignalCollectionMutatingEvaluator(FutureCloseChangeEvaluator):
    name = "output_signal_collection_mutating_future_close_change"

    def __init__(
        self, events: list[str], shared_outputs: list[NumericPredictionOutput]
    ) -> None:
        super().__init__(events)
        self._shared_outputs = shared_outputs

    def evaluate(
        self,
        signal: NumericPrediction,
        outcome: PredictionOutcome[FutureCloseValues],
    ) -> FutureCloseChangeValues:
        object.__setattr__(self._shared_outputs[0], "signals", ())
        return super().evaluate(signal, outcome)


class CallbackConfigurationMutatingEvaluator(FutureCloseChangeEvaluator):
    name = "callback_configuration_mutating_future_close_change"

    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self._configuration_state = "original"
        self._evaluation_call_count = 0

    def configuration(self) -> PrimitiveMapping:
        base = super().configuration()
        parameters = cast(PrimitiveMapping, base["parameters"])
        return {
            **base,
            "parameters": {
                **parameters,
                "callback_configuration_state": self._configuration_state,
            },
        }

    def evaluate(
        self,
        signal: NumericPrediction,
        outcome: PredictionOutcome[FutureCloseValues],
    ) -> FutureCloseChangeValues:
        self._evaluation_call_count += 1
        self._configuration_state = (
            "changed" if self._evaluation_call_count == 1 else "original"
        )
        return super().evaluate(signal, outcome)


class DelayedMutatingEvaluator(FutureCloseChangeEvaluator):
    name = "delayed_mutating_future_close_change"

    def __init__(self, events: list[str], mutation_target: str) -> None:
        super().__init__(events)
        self._mutation_target = mutation_target
        self._retained_signal: NumericPrediction | None = None
        self._retained_outcome: PredictionOutcome[FutureCloseValues] | None = None
        self._retained_evaluation: FutureCloseChangeValues | None = None

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_evaluator",
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": {
                "multiplier": decimal_to_primitive(self._multiplier),
                "mutation_target": self._mutation_target,
                "value": "future_close_minus_reference_close",
            },
            "result_schema_version": self.result_schema_version,
        }

    def evaluate(
        self,
        signal: NumericPrediction,
        outcome: PredictionOutcome[FutureCloseValues],
    ) -> FutureCloseChangeValues:
        if (
            self._retained_signal is not None
            and self._retained_outcome is not None
            and self._retained_evaluation is not None
        ):
            if self._mutation_target == "signal":
                object.__setattr__(
                    self._retained_signal, "predicted_change", Decimal(999)
                )
            elif self._mutation_target == "outcome":
                object.__setattr__(
                    self._retained_outcome.values, "future_close", Decimal(999)
                )
            else:
                object.__setattr__(
                    self._retained_evaluation, "close_change", Decimal(999)
                )

        evaluation = super().evaluate(signal, outcome)
        if self._retained_signal is None:
            self._retained_signal = signal
            self._retained_outcome = outcome
            self._retained_evaluation = evaluation
        return evaluation


def _study(
    events: list[str],
    horizon: int = 1,
    *,
    multiplier: Decimal = Decimal(1),
    feature_schema_version: str = "1",
    feature_offset: Decimal = Decimal(0),
    returned_horizon: int | None = None,
) -> PredictionStudy[NumericPrediction, FutureCloseValues, FutureCloseChangeValues]:
    return PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        RecordingPredictionStrategy(events, feature_offset),
        FutureCloseOutcomeLabeler(events, horizon, returned_horizon),
        FutureCloseChangeEvaluator(events, multiplier),
        feature_configuration={
            "feature_schema_version": feature_schema_version,
            "source": "fixture_features",
        },
    )


def test_generic_runner_executes_non_gap_outcome_without_gap_fields() -> None:
    events: list[str] = []
    dataset = make_dataset(
        ("100", "102", "101", "104"),
        opens=("90", "91", "92", "93"),
        lows=("89", "90", "91", "92"),
    )

    result = run_prediction_study(dataset, _study(events))
    primitive = result.to_primitive()
    rendered = str(primitive)

    assert len(result.rows) == 3
    assert result.unavailable_outcome_count == 1
    assert [row.evaluation.values.close_change for row in result.rows] == [
        Decimal(2),
        Decimal(-1),
        Decimal(3),
    ]
    assert "next_open" not in rendered
    assert "overnight_gap" not in rendered
    assert "correct" not in rendered
    assert "direction" not in rendered


def test_market_data_provenance_preserves_boundary_missing_sessions() -> None:
    missing_sessions = (date(2024, 7, 1), date(2024, 7, 5))
    dataset = make_dataset(
        ("100", "102"),
        sessions=(date(2024, 7, 2), date(2024, 7, 3)),
        requested_start=missing_sessions[0],
        requested_end=missing_sessions[-1],
        missing_sessions=missing_sessions,
    )

    result = run_prediction_study(dataset, _study([]))

    assert result.market_data.missing_sessions == missing_sessions
    assert result.market_data.to_primitive()["missing_sessions"] == [
        "2024-07-01",
        "2024-07-05",
    ]


def test_prediction_label_evaluation_order_and_contract_are_causal() -> None:
    events: list[str] = []
    result = run_prediction_study(make_dataset(("100", "102", "101")), _study(events))

    assert events[0] == "prediction_rule"
    assert events[1] == "outcome_validation"
    assert events[2:] == [
        "outcome_label",
        "evaluation",
        "outcome_label",
        "evaluation",
        "outcome_label",
    ]
    assert all(isinstance(row.outcome, PredictionOutcome) for row in result.rows)
    assert all(isinstance(row.evaluation, PredictionEvaluation) for row in result.rows)


def test_study_identity_includes_outcome_configuration() -> None:
    dataset = make_dataset(("100", "102", "101", "104"))
    one_session = run_prediction_study(dataset, _study([], horizon=1))
    two_sessions = run_prediction_study(dataset, _study([], horizon=2))
    scaled_evaluation = run_prediction_study(
        dataset, _study([], horizon=1, multiplier=Decimal(2))
    )
    changed_features = run_prediction_study(
        dataset, _study([], horizon=1, feature_schema_version="2")
    )
    gap_study = run_prediction_study(
        dataset,
        create_overnight_gap_prediction_study(
            AlwaysUpPredictionStrategy(AlwaysUpParameters(excluded_weekdays=()))
        ),
    )

    assert one_session.study_id != two_sessions.study_id
    assert one_session.study_id != scaled_evaluation.study_id
    assert one_session.study_id != changed_features.study_id
    assert one_session.study_id != gap_study.study_id
    assert one_session.configuration.required_future_sessions == 1
    assert two_sessions.configuration.required_future_sessions == 2


@pytest.mark.parametrize("warm_up", [0, -1, True, "1"])
def test_strategy_warm_up_must_be_a_positive_integer(warm_up: object) -> None:
    events: list[str] = []
    strategy = RecordingPredictionStrategy(events)
    strategy.warm_up_observations = cast(int, warm_up)
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        strategy,
        FutureCloseOutcomeLabeler(events),
        FutureCloseChangeEvaluator(events),
    )

    with pytest.raises(InvalidPredictionConfigurationError, match="warm-up"):
        run_prediction_study(make_dataset(("100", "102", "101")), study)


def test_signal_cannot_precede_the_declared_strategy_warm_up() -> None:
    events: list[str] = []
    strategy = RecordingPredictionStrategy(events)
    strategy.warm_up_observations = 2
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        strategy,
        FutureCloseOutcomeLabeler(events),
        FutureCloseChangeEvaluator(events),
    )

    with pytest.raises(InvalidPredictionOutputError, match="warm-up completed"):
        run_prediction_study(make_dataset(("100", "102", "101")), study)


def test_label_session_must_match_declared_future_session_horizon() -> None:
    dataset = make_dataset(("100", "102", "101", "104"))

    with pytest.raises(InvalidPredictionOutputError, match=r"declared.*horizon"):
        run_prediction_study(
            dataset,
            _study([], horizon=2, returned_horizon=1),
        )


def test_labeler_cannot_censor_an_available_declared_outcome() -> None:
    events: list[str] = []
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        RecordingPredictionStrategy(events),
        SelectivelyUnavailableFutureCloseOutcomeLabeler(events),
        FutureCloseChangeEvaluator(events),
    )

    with pytest.raises(
        InvalidPredictionOutputError, match=r"declared future session is available"
    ):
        run_prediction_study(make_dataset(("100", "102", "101")), study)


@pytest.mark.parametrize("mutation_phase", ["validation", "labeling"])
def test_labeler_cannot_mutate_signals_after_they_are_generated(
    mutation_phase: str,
) -> None:
    events: list[str] = []
    shared_signals: list[NumericPrediction] = []
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        SharedSignalPredictionStrategy(events, shared_signals),
        SignalMutatingFutureCloseOutcomeLabeler(events, shared_signals, mutation_phase),
        FutureCloseChangeEvaluator(events),
    )

    with pytest.raises(InvalidPredictionOutputError, match="prediction signal"):
        run_prediction_study(make_dataset(("100", "102", "101")), study)


def test_signal_validation_rejects_first_serialization_parameter_drift() -> None:
    events: list[str] = []
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        FirstSerializationMutatingPredictionStrategy(events),
        FutureCloseOutcomeLabeler(events),
        FutureCloseChangeEvaluator(events),
    )

    with pytest.raises(InvalidPredictionOutputError, match="prediction signal"):
        run_prediction_study(make_dataset(("100", "102", "101")), study)


def test_generated_signal_collection_is_snapshotted_once() -> None:
    events: list[str] = []
    shared_outputs: list[NumericPredictionOutput] = []
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        SharedOutputPredictionStrategy(events, shared_outputs),
        FutureCloseOutcomeLabeler(events),
        OutputSignalCollectionMutatingEvaluator(events, shared_outputs),
    )

    result = run_prediction_study(make_dataset(("100", "102", "101")), study)

    assert result.generated_prediction_count == 3
    assert len(result.rows) == 2
    assert result.unavailable_outcome_count == 1


@pytest.mark.parametrize("mutation_phase", ["strategy", "validation", "labeling"])
def test_components_cannot_mutate_the_caller_owned_market_dataset(
    mutation_phase: str,
) -> None:
    events: list[str] = []
    dataset = make_dataset(("100", "102", "101"))
    original_first_close = dataset.bars[0].close
    strategy = (
        DatasetMutatingPredictionStrategy(events)
        if mutation_phase == "strategy"
        else RecordingPredictionStrategy(events)
    )
    labeler = (
        FutureCloseOutcomeLabeler(events)
        if mutation_phase == "strategy"
        else DatasetMutatingFutureCloseOutcomeLabeler(events, mutation_phase)
    )
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        strategy,
        labeler,
        FutureCloseChangeEvaluator(events),
    )

    with pytest.raises(InvalidPredictionOutputError, match="market dataset"):
        run_prediction_study(dataset, study)

    assert dataset.bars[0].close == original_first_close


@pytest.mark.parametrize(
    ("mutation_phase", "changed_component"),
    [
        ("validation", "outcome labeler"),
        ("labeling", "outcome labeler"),
        ("evaluation", "prediction evaluator"),
    ],
)
def test_component_configuration_is_checked_after_each_callback(
    mutation_phase: str,
    changed_component: str,
) -> None:
    events: list[str] = []
    labeler = (
        FutureCloseOutcomeLabeler(events)
        if mutation_phase == "evaluation"
        else CallbackConfigurationMutatingOutcomeLabeler(events, mutation_phase)
    )
    evaluator = (
        CallbackConfigurationMutatingEvaluator(events)
        if mutation_phase == "evaluation"
        else FutureCloseChangeEvaluator(events)
    )
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        RecordingPredictionStrategy(events),
        labeler,
        evaluator,
    )

    with pytest.raises(InvalidPredictionOutputError, match=changed_component):
        run_prediction_study(make_dataset(("100", "102", "101")), study)


def test_outcome_identity_rejects_first_serialization_value_drift() -> None:
    events: list[str] = []
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        RecordingPredictionStrategy(events),
        FirstSerializationMutatingOutcomeLabeler(events),
        FutureCloseChangeEvaluator(events),
    )

    with pytest.raises(InvalidPredictionOutputError, match="prediction outcome values"):
        run_prediction_study(make_dataset(("100", "102", "101")), study)


@pytest.mark.parametrize(
    ("evaluator", "changed_component"),
    [
        (SignalMutatingEvaluator([]), "prediction signal"),
        (OutcomeMutatingEvaluator([]), "prediction outcome"),
    ],
)
def test_evaluator_cannot_mutate_fixed_inputs(
    evaluator: FutureCloseChangeEvaluator,
    changed_component: str,
) -> None:
    events: list[str] = []
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        RecordingPredictionStrategy(events),
        FutureCloseOutcomeLabeler(events),
        evaluator,
    )

    with pytest.raises(InvalidPredictionOutputError, match=changed_component):
        run_prediction_study(make_dataset(("100", "102", "101")), study)


@pytest.mark.parametrize(
    ("mutation_target", "changed_component"),
    [
        ("signal", "prediction signal"),
        ("outcome", "prediction outcome"),
        ("evaluation", "prediction evaluation values"),
    ],
)
def test_evaluator_cannot_mutate_an_earlier_completed_row(
    mutation_target: str,
    changed_component: str,
) -> None:
    events: list[str] = []
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        RecordingPredictionStrategy(events),
        FutureCloseOutcomeLabeler(events),
        DelayedMutatingEvaluator(events, mutation_target),
    )

    with pytest.raises(InvalidPredictionOutputError, match=changed_component):
        run_prediction_study(make_dataset(("100", "102", "101")), study)


@pytest.mark.parametrize("mutation_target", ["signal", "outcome", "evaluation"])
def test_returned_rows_are_detached_from_values_retained_across_runs(
    mutation_target: str,
) -> None:
    events: list[str] = []
    study = PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        RecordingPredictionStrategy(events),
        FutureCloseOutcomeLabeler(events),
        DelayedMutatingEvaluator(events, mutation_target),
    )
    dataset = make_dataset(("100", "102"))

    first = run_prediction_study(dataset, study)
    first_primitive = first.to_primitive()
    first_typed_values = (
        first.rows[0].signal.predicted_change,
        first.rows[0].outcome.values.future_close,
        first.rows[0].evaluation.values.close_change,
    )

    run_prediction_study(dataset, study)

    assert first.to_primitive() == first_primitive
    assert (
        first.rows[0].signal.predicted_change,
        first.rows[0].outcome.values.future_close,
        first.rows[0].evaluation.values.close_change,
    ) == first_typed_values


def test_row_identity_includes_fixed_feature_values() -> None:
    dataset = make_dataset(("100", "102", "101", "104"))

    original = run_prediction_study(dataset, _study([], feature_offset=Decimal(0)))
    changed = run_prediction_study(dataset, _study([], feature_offset=Decimal(1)))

    assert original.study_id == changed.study_id
    assert tuple(row.row_id for row in original.rows) != tuple(
        row.row_id for row in changed.rows
    )


def test_repeated_generic_studies_have_stable_component_and_row_ids() -> None:
    dataset = make_dataset(("100", "102", "101", "104"))

    first = run_prediction_study(dataset, _study([]))
    second = run_prediction_study(dataset, _study([]))

    assert first == second
    assert tuple(row.row_id for row in first.rows) == tuple(
        row.row_id for row in second.rows
    )
    assert tuple(row.outcome.outcome_id for row in first.rows) == tuple(
        row.outcome.outcome_id for row in second.rows
    )
    assert tuple(row.evaluation.evaluation_id for row in first.rows) == tuple(
        row.evaluation.evaluation_id for row in second.rows
    )


def test_generic_orchestrator_has_no_gap_provider_or_backtest_imports() -> None:
    module_path = (
        Path(__file__).parents[3] / "src" / "quantforge" / "prediction" / "study.py"
    )
    parsed = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(parsed)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any("overnight_gap" in module for module in imported_modules)
    assert not any("backtesting" in module for module in imported_modules)
    assert not any("tiingo" in module for module in imported_modules)
