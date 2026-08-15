from decimal import Decimal
from pathlib import Path

from quantforge.data.models import MarketDataset
from quantforge.indicators import (
    NATIVE_INDICATOR_BACKEND,
    TALIB_INDICATOR_BACKEND,
    IndicatorComparisonTolerances,
)
from quantforge.prediction import (
    PREDICTION_BACKEND_COMPARISON_ARTIFACT_FILENAMES,
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    export_overnight_gap_backend_comparison,
    run_overnight_gap_backend_comparison,
    validate_overnight_gap_backend_comparison_export,
)

from ..helpers import make_dataset


def _overnight_dataset() -> MarketDataset:
    closes = (
        "100",
        "102",
        "101",
        "103",
        "99",
        "101",
        "100",
        "102",
        "98",
        "100",
        "99",
        "101",
        "100",
        "102",
        "101",
    )
    opens = tuple(str(int(value) - 1) for value in closes)
    highs = tuple(str(int(value) + 1) for value in closes)
    lows = tuple(str(int(value) - 2) for value in closes)
    return make_dataset(closes, opens=opens, highs=highs, lows=lows)


def test_implicit_overnight_strategy_remains_the_legacy_native_path() -> None:
    default = OvernightGapPredictionStrategy(OvernightGapPredictionParameters())
    explicit_native = OvernightGapPredictionStrategy(
        OvernightGapPredictionParameters(), backend_id=NATIVE_INDICATOR_BACKEND
    )

    assert all(
        "backend" not in indicator.configuration()
        for indicator in default.required_indicators
    )
    assert all(
        "backend" in indicator.configuration()
        for indicator in explicit_native.required_indicators
    )
    assert default.configuration()["contract_version"] == "1"


def test_overnight_gap_report_quantifies_value_signal_and_metric_impact() -> None:
    dataset = _overnight_dataset()

    result = run_overnight_gap_backend_comparison(
        dataset,
        tolerances=IndicatorComparisonTolerances(
            Decimal("0.000000000001"), Decimal("0.000000000001")
        ),
    )
    prediction = result.prediction_comparison

    assert tuple(item.definition.name for item in result.indicator_comparisons) == (
        "wilder_relative_strength_index",
        "wilder_directional_movement",
    )
    assert all(
        item.backend_a_identity.backend_id == NATIVE_INDICATOR_BACKEND
        and item.backend_b_identity.backend_id == TALIB_INDICATOR_BACKEND
        for item in result.indicator_comparisons
    )
    assert all(
        item.backend_b_identity.library_version == "0.7.1"
        for item in result.indicator_comparisons
    )
    assert prediction.backend_a_prediction_count == (
        len(prediction.backend_a_only_prediction_dates)
        + prediction.shared_prediction_date_count
    )
    assert prediction.backend_b_prediction_count == (
        len(prediction.backend_b_only_prediction_dates)
        + prediction.shared_prediction_date_count
    )
    assert tuple(item.metric_name for item in prediction.metrics) == (
        "accuracy",
        "average_signed_return",
    )
    assert dataset.metadata.data_sha256 in result.source_snapshot.canonical_json
    assert prediction.matched_prediction_count >= 0


def test_overnight_gap_backend_artifacts_are_deterministic(tmp_path: Path) -> None:
    result = run_overnight_gap_backend_comparison(
        _overnight_dataset(),
        tolerances=IndicatorComparisonTolerances(
            Decimal("0.000000000001"), Decimal("0.000000000001")
        ),
    )

    first = export_overnight_gap_backend_comparison(result, tmp_path / "first")
    second = export_overnight_gap_backend_comparison(result, tmp_path / "second")

    assert {item.name for item in first.iterdir()} == set(
        PREDICTION_BACKEND_COMPARISON_ARTIFACT_FILENAMES
    )
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes()
        for name in PREDICTION_BACKEND_COMPARISON_ARTIFACT_FILENAMES
    )
    assert validate_overnight_gap_backend_comparison_export(result, first) == first
    summary = (first / "summary.txt").read_text(encoding="utf-8")
    assert "Matched predictions:" in summary
    assert "Changed directions:" in summary
    assert "Comparison only; native studies remain unchanged." in summary
