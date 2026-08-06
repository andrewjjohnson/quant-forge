import ast
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

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

    def __init__(self, events: list[str]) -> None:
        self._events = events
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
                feature_values=(PredictionFeature("signal_close", bar.close),),
            )
            for bar in dataset.bars
        )
        return NumericPredictionOutput(
            self.name,
            self.configuration_id,
            dataset.metadata.dataset_id,
            signals,
        )


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

    def __init__(self, events: list[str], horizon: int = 1) -> None:
        self._events = events
        self.required_future_sessions = horizon
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
        outcome_index = index + self.required_future_sessions
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


def _study(
    events: list[str],
    horizon: int = 1,
    *,
    multiplier: Decimal = Decimal(1),
    feature_schema_version: str = "1",
) -> PredictionStudy[NumericPrediction, FutureCloseValues, FutureCloseChangeValues]:
    return PredictionStudy[
        NumericPrediction, FutureCloseValues, FutureCloseChangeValues
    ].create(
        RecordingPredictionStrategy(events),
        FutureCloseOutcomeLabeler(events, horizon),
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
