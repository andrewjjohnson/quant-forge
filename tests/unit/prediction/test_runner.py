import csv
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data.models import AdjustmentMode, MarketDataset
from quantforge.indicators import Indicator
from quantforge.prediction import (
    InvalidPredictionDataError,
    PredictionAnalysisResult,
    PredictionDirection,
    PredictionExportError,
    PredictionFeature,
    PredictionParameter,
    PredictionSignal,
    PredictionStrategyOutput,
    export_prediction_analysis,
    load_prediction_manifest,
    run_prediction_analysis,
    validate_prediction_analysis_export,
)

from ..helpers import make_dataset


@dataclass(frozen=True, slots=True)
class FixedParameters:
    directions: str

    def to_primitive(self) -> PrimitiveMapping:
        return {"directions": self.directions}


class FixedPredictionStrategy:
    name = "fixed_prediction"
    implementation_version = "1"
    required_indicators: tuple[Indicator, ...] = ()
    warm_up_observations = 1

    def __init__(self, directions: tuple[PredictionDirection, ...]) -> None:
        self.emitted_directions = directions
        self.reason = "fixed test direction"
        self.feature_values: tuple[PredictionFeature, ...] = ()
        self._parameters = FixedParameters(
            ",".join(direction.value for direction in directions)
        )

    @property
    def parameters(self) -> FixedParameters:
        return self._parameters

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_type": "prediction_strategy",
            "component_name": self.name,
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": self.parameters.to_primitive(),
            "required_indicators": [],
            "warm_up_observations": self.warm_up_observations,
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def generate(self, dataset: MarketDataset) -> PredictionStrategyOutput:
        parameter_snapshot = (
            PredictionParameter("directions", self.parameters.directions),
        )
        signals = tuple(
            PredictionSignal(
                symbol=dataset.metadata.canonical_symbol,
                signal_session=bar.session_date,
                direction=direction,
                strategy_id=self.name,
                strategy_implementation_version=self.implementation_version,
                strategy_configuration_id=self.configuration_id,
                strategy_parameters=parameter_snapshot,
                reason=self.reason,
                feature_values=self.feature_values,
            )
            for bar, direction in zip(
                dataset.bars, self.emitted_directions, strict=True
            )
        )
        return PredictionStrategyOutput(
            self.name,
            self.configuration_id,
            dataset.metadata.dataset_id,
            signals,
        )


def _configured_result() -> PredictionAnalysisResult:
    dataset = make_dataset(
        ("100", "100", "100", "100"),
        opens=("100", "102", "99", "100"),
        highs=("101", "103", "101", "101"),
        lows=("99", "99", "98", "99"),
    )
    strategy = FixedPredictionStrategy(
        (
            PredictionDirection.UP,
            PredictionDirection.DOWN,
            PredictionDirection.UP,
            PredictionDirection.DOWN,
        )
    )
    return run_prediction_analysis(dataset, strategy)


def test_next_session_labels_and_requested_gap_metrics() -> None:
    result = _configured_result()

    assert result.generated_signal_count == 4
    assert result.unlabeled_end_of_data_count == 1
    assert [row.overnight_gap_percentage for row in result.rows] == [
        Decimal("0.02"),
        Decimal("-0.01"),
        Decimal(0),
    ]
    assert [row.correct for row in result.rows] == [True, True, False]
    assert all(row.outcome_session > row.signal_session for row in result.rows)
    assert [row.outcome_session for row in result.rows] == [
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
    ]
    assert result.metrics.prediction_count == 3
    assert result.metrics.correct_count == 2
    assert result.metrics.incorrect_count == 1
    assert result.metrics.accuracy == Decimal("0.6666666666666666666666666666666667")
    assert result.metrics.average_gap_size_correct == Decimal("0.015")
    assert result.metrics.average_gap_size_incorrect == Decimal(0)
    assert result.metrics.average_signed_return_correct == Decimal("0.015")
    assert result.metrics.average_signed_return_incorrect == Decimal(0)


def test_internal_missing_session_is_not_used_as_a_later_outcome() -> None:
    dataset = make_dataset(
        ("100", "100"),
        sessions=(date(2024, 7, 1), date(2024, 7, 3)),
        missing_sessions=(date(2024, 7, 2),),
    )
    strategy = FixedPredictionStrategy(
        (PredictionDirection.UP, PredictionDirection.DOWN)
    )

    with pytest.raises(InvalidPredictionDataError, match="missing sessions"):
        run_prediction_analysis(dataset, strategy)


def test_raw_stock_split_dataset_is_rejected_as_a_mechanical_gap() -> None:
    dataset = make_dataset(
        ("100", "50"),
        splits=((date(2024, 7, 2), "2"),),
    )
    strategy = FixedPredictionStrategy(
        (PredictionDirection.DOWN, PredictionDirection.UP)
    )

    with pytest.raises(InvalidPredictionDataError, match="mechanical split"):
        run_prediction_analysis(dataset, strategy)


def test_repeated_inputs_are_deterministic_and_json_safe() -> None:
    first = _configured_result()
    second = _configured_result()

    assert first == second
    assert first.analysis_id == (
        "10ae28f6b24c77382da3d6303ac044f8e61a4191850dee8c077aa4c800d8e974"
    )
    assert tuple(row.prediction_id for row in first.rows) == (
        "5bb97b8b64d1cbd0ea93fe2a819d7a8d231678236b632b8f8fa60845de115fbd",
        "00657471c607907970150f6861c6765eec723002b3104b58725a23cb1e15ba5a",
        "339a441ed11e6431b46d1b988ef94c2a4309eda719be1f006be0170be742d8f1",
    )
    assert first.engine_version == "5"
    assert first.result_schema_version == "4"
    assert first.analysis_id == second.analysis_id
    assert first.to_primitive() == second.to_primitive()
    json.dumps(first.to_primitive(), allow_nan=False, sort_keys=True)


def test_market_data_provenance_preserves_adjustment_and_volume_policy() -> None:
    dataset = make_dataset(
        ("100", "102"),
        adjustment_mode=AdjustmentMode.SPLIT_ADJUSTED,
        adjusted_fields_used=True,
    )
    result = run_prediction_analysis(
        dataset,
        FixedPredictionStrategy((PredictionDirection.UP, PredictionDirection.DOWN)),
    )

    assert result.market_data.ohlc_basis == "split_adjusted"
    assert result.market_data.volume_basis == "split_adjusted"
    assert result.market_data.adjusted_fields_used is True
    assert result.market_data.corporate_action_policy == (
        "separate_provider_reported_cash_dividends_and_splits"
    )
    assert result.market_data.to_primitive()["volume_basis"] == "split_adjusted"
    assert result.market_data.to_primitive()["adjusted_fields_used"] is True
    assert result.market_data.to_primitive()["corporate_action_policy"] == (
        "separate_provider_reported_cash_dividends_and_splits"
    )


@pytest.mark.parametrize("changed_signal_field", ["direction", "reason", "features"])
def test_legacy_prediction_ids_include_the_fixed_signal_snapshot(
    changed_signal_field: str,
) -> None:
    dataset = make_dataset(
        ("100", "100"),
        opens=("100", "102"),
        highs=("101", "103"),
        lows=("99", "99"),
    )
    strategy = FixedPredictionStrategy(
        (PredictionDirection.UP, PredictionDirection.DOWN)
    )
    original = run_prediction_analysis(dataset, strategy)

    if changed_signal_field == "direction":
        strategy.emitted_directions = (
            PredictionDirection.DOWN,
            PredictionDirection.DOWN,
        )
    elif changed_signal_field == "reason":
        strategy.reason = "changed without changing declared configuration"
    else:
        strategy.feature_values = (PredictionFeature("changed_feature", Decimal(1)),)

    changed = run_prediction_analysis(dataset, strategy)

    assert original.analysis_id == changed.analysis_id
    assert original.rows[0].prediction_id != changed.rows[0].prediction_id


def test_export_writes_manifest_and_labeled_prediction_csv(tmp_path: Path) -> None:
    result = _configured_result()

    destination = export_prediction_analysis(result, tmp_path)

    manifest = load_prediction_manifest(destination / "manifest.json")
    assert manifest["analysis_id"] == result.analysis_id
    assert result.market_data.retrieved_at.isoformat() == "2024-07-15T00:00:00+00:00"
    assert result.market_data.provider_timezone == "America/New_York"
    assert manifest["market_data"] == result.market_data.to_primitive()
    assert manifest["record_counts"] == {
        "generated_signals": 4,
        "labeled_predictions": 3,
        "unlabeled_end_of_data": 1,
    }
    with (destination / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert rows[0]["signal_session"] == "2024-07-01"
    assert rows[0]["outcome_session"] == "2024-07-02"
    assert "fill" not in rows[0]
    assert "order" not in rows[0]

    with pytest.raises(PredictionExportError, match="already exists"):
        export_prediction_analysis(result, tmp_path)


def test_existing_export_must_exactly_match_the_result(tmp_path: Path) -> None:
    result = _configured_result()
    destination = export_prediction_analysis(result, tmp_path)

    assert validate_prediction_analysis_export(result, destination) == destination

    (destination / "predictions.csv").write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(PredictionExportError, match="expected immutable result"):
        validate_prediction_analysis_export(result, destination)
