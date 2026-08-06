from pathlib import Path

from quantforge.prediction import (
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    export_prediction_analysis,
    run_prediction_analysis,
)

from ..unit.helpers import make_dataset


def test_local_spy_gap_prediction_runs_and_exports_end_to_end(tmp_path: Path) -> None:
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
    dataset = make_dataset(
        closes,
        dataset_id="immutable-local-spy-gap-fixture",
        opens=opens,
        highs=highs,
        lows=lows,
    )

    result = run_prediction_analysis(
        dataset,
        OvernightGapPredictionStrategy(OvernightGapPredictionParameters()),
    )
    destination = export_prediction_analysis(result, tmp_path)

    assert result.market_data.symbol == "SPY"
    assert result.strategy_id == "overnight_gap_direction"
    assert result.rows
    assert all(row.signal_session.weekday() != 4 for row in result.rows)
    assert result.metrics.prediction_count == len(result.rows)
    assert (destination / "manifest.json").is_file()
    assert (destination / "predictions.csv").is_file()
