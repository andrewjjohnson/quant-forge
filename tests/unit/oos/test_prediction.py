"""Stored-row arithmetic, explicit availability, and real prediction OOS example."""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.oos import (
    OOSIntegrityError,
    PredictionMetricFields,
    aggregate_backtest,
    aggregate_prediction,
    export_oos_aggregate,
    load_oos_aggregate,
    load_oos_source,
)
from quantforge.oos._records import mapping
from quantforge.oos.prediction import summarize_prediction_observations
from quantforge.prediction import OvernightGapDirectionEvaluator, PredictionStudy
from quantforge.prediction.comparison_metrics import wilson_interval
from quantforge.prediction.contracts import PredictionOutcome
from quantforge.prediction.models import PredictionSignal
from quantforge.prediction.outcomes.overnight_gap import (
    NextSessionOpenGapValues,
    OvernightGapDirectionEvaluationValues,
)
from quantforge.walk_forward import WalkForwardStudy
from tests.unit.walk_forward.fixtures import StudyFactory, prediction_fixture

from .conftest import CompletedStudy


def observation(
    direction: str, values: PrimitiveMapping | None, *, eligible: bool = True
) -> PrimitiveMappingSnapshot:
    return PrimitiveMappingSnapshot.capture(
        {
            "signal": {"prediction": {"values": {"direction": direction}}},
            "row": None if values is None else {"evaluation": {"values": values}},
            "eligible": eligible,
        }
    )


def test_weighted_prediction_metrics_and_availability() -> None:
    rows = (
        observation(
            "up",
            {
                "direction_correct": True,
                "baseline": False,
                "signed_prediction_return": "0.2",
                "mfe_percentage": "0.3",
                "mae_percentage": "-0.1",
                "label": "target_first",
            },
        ),
        observation(
            "down",
            {
                "direction_correct": False,
                "baseline": True,
                "signed_prediction_return": "-0.1",
                "mfe_percentage": "0.2",
                "mae_percentage": "-0.2",
                "label": "stop_first",
            },
        ),
        observation(
            "up",
            {
                "direction_correct": True,
                "signed_prediction_return": "0.05",
                "label": "both_same_session",
            },
        ),
        observation("up", None),
        observation("down", {"direction_correct": True}, eligible=False),
    )
    summary = summarize_prediction_observations(
        rows,
        8,
        PredictionMetricFields(baseline_correct="baseline", baseline_name="fixture"),
    )
    assert summary["prediction_count"] == 4
    assert summary["excluded_signal_count"] == 1
    assert summary["prediction_frequency"] == "0.5"
    assert summary["direction_distribution"] == {"up": 3, "down": 1}
    assert summary["accuracy_interval"] == wilson_interval(2, 3).to_primitive()
    assert summary["accuracy_unavailable_count"] == 1
    assert mapping(summary["matched_baseline"])["accuracy"] == "0.5"
    assert mapping(summary["matched_baseline"])["accuracy_difference"] == "0"
    assert mapping(summary["signed_outcome"])["mean"] == "0.05"
    assert mapping(summary["signed_outcome"])["median"] == "0.05"
    assert mapping(summary["mfe"])["mean"] == "0.25"
    assert mapping(summary["mae"])["mean"] == "-0.15"
    assert mapping(summary["mfe"])["status"] == "partial"
    assert mapping(summary["event_outcomes"])["counts"] == {
        "target_first": 1,
        "stop_first": 1,
        "both_same_session": 1,
    }
    with localcontext() as context:
        context.prec = 3
        context.rounding = "ROUND_DOWN"
        assert (
            summarize_prediction_observations(
                rows,
                8,
                PredictionMetricFields(
                    baseline_correct="baseline", baseline_name="fixture"
                ),
            )
            == summary
        )


def test_empty_and_no_prediction_windows() -> None:
    summary = summarize_prediction_observations((), 4)
    assert summary["prediction_count"] == 0
    assert summary["prediction_frequency"] == "0"
    assert summary["accuracy"] is None
    assert summary["accuracy_interval"] == wilson_interval(0, 0).to_primitive()
    assert mapping(summary["mfe"])["status"] == "unavailable"
    assert summarize_prediction_observations((), 0)["prediction_frequency"] is None


@pytest.mark.parametrize("invalid", ["True", 1, "NaN"])
def test_invalid_accuracy_is_not_coerced(invalid: str | int) -> None:
    with pytest.raises(OOSIntegrityError, match="boolean"):
        summarize_prediction_observations(
            (observation("up", {"direction_correct": invalid}),), 1
        )


def test_invalid_numeric_metric_and_baseline_contract() -> None:
    with pytest.raises(ValueError, match="finite"):
        summarize_prediction_observations(
            (observation("up", {"mfe_percentage": "NaN"}),), 1
        )
    with pytest.raises(OOSIntegrityError, match="both"):
        PredictionMetricFields(baseline_correct="baseline")


def test_real_prediction_oos_example_and_roundtrip(
    prediction_study: CompletedStudy, tmp_path: Path
) -> None:
    result = aggregate_prediction(prediction_study.source)
    summary = result.summary.to_primitive()
    assert summary["prediction_count"] == 8
    assert summary["accuracy"] == "1"
    assert summary["accuracy_interval"] == wilson_interval(8, 8).to_primitive()
    assert mapping(summary["matched_baseline"])["status"] == "unavailable"
    assert len(result.observations) == 8
    assert (
        not {"equity", "trades", "profit_factor", "commission", "total_return"}
        & summary.keys()
    )
    for item in result.observations:
        assert item.to_primitive()["row"] is not None
        assert item.to_primitive()["decision_timestamp"] is not None
    path = export_oos_aggregate(result, tmp_path / "oos")
    assert load_oos_aggregate(path).to_primitive() == result.to_primitive()
    assert export_oos_aggregate(result, tmp_path / "oos") == path
    assert (
        aggregate_prediction(
            load_oos_source(
                prediction_study.source.plan, prediction_study.study.study_path
            )
        )
        == result
    )
    with pytest.raises(OOSIntegrityError, match="trading study"):
        aggregate_backtest(prediction_study.source)


@dataclass(frozen=True, slots=True)
class RichValues(OvernightGapDirectionEvaluationValues):
    baseline_correct: bool = False
    mfe_percentage: Decimal = Decimal("0.04")
    mae_percentage: Decimal = Decimal("-0.02")
    label: str = "target_first"

    def to_primitive(self) -> PrimitiveMapping:
        return {
            **OvernightGapDirectionEvaluationValues.to_primitive(self),
            "baseline_correct": self.baseline_correct,
            "mfe_percentage": str(self.mfe_percentage),
            "mae_percentage": str(self.mae_percentage),
            "label": self.label,
        }


class RichEvaluator(OvernightGapDirectionEvaluator):
    name = "fixture_rich_oos_evaluator"

    def evaluate(
        self,
        signal: PredictionSignal,
        outcome: PredictionOutcome[NextSessionOpenGapValues],
    ) -> RichValues:
        base = super().evaluate(signal, outcome)
        return RichValues(
            base.predicted_direction,
            base.actual_direction,
            base.signed_prediction_return,
            base.direction_correct,
        )


class RichFactory(StudyFactory):
    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        original = super().build(parameters)
        return PredictionStudy[
            PredictionSignal, NextSessionOpenGapValues, RichValues
        ].create(original.strategy, original.outcome_labeler, RichEvaluator())


def test_supported_metrics_flow_through_real_qf39_artifacts(tmp_path: Path) -> None:
    config, evaluator = prediction_fixture(tmp_path, factory=RichFactory())
    study = WalkForwardStudy(config, evaluator, tmp_path / "study")
    study.run()
    source = load_oos_source(config.plan, study.study_path)
    summary = aggregate_prediction(
        source,
        PredictionMetricFields(
            baseline_correct="baseline_correct", baseline_name="fixed_fixture"
        ),
    ).summary.to_primitive()
    assert summary["prediction_count"] == 8
    assert mapping(summary["mfe"])["mean"] == "0.04"
    assert mapping(summary["mae"])["mean"] == "-0.02"
    assert mapping(summary["matched_baseline"])["accuracy_difference"] == "1"
    assert mapping(summary["event_outcomes"])["rates"] == {"target_first": "1"}
