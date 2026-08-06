import csv
import json
from pathlib import Path

import pytest

from quantforge.prediction import (
    COMPARISON_ARTIFACT_FILENAMES,
    PredictionExportError,
    export_prediction_comparison,
    run_prediction_comparison,
    validate_prediction_comparison_export,
)

from ..unit.helpers import make_dataset


def test_local_spy_comparison_runs_and_exports_every_artifact(tmp_path: Path) -> None:
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
    dataset = make_dataset(
        closes,
        dataset_id="immutable-local-spy-comparison-fixture",
        opens=tuple(str(int(value) - 1) for value in closes),
        highs=tuple(str(int(value) + 1) for value in closes),
        lows=tuple(str(int(value) - 2) for value in closes),
    )

    first = run_prediction_comparison(dataset)
    second = run_prediction_comparison(dataset)
    destination = export_prediction_comparison(first, tmp_path)

    assert first == second
    assert {item.configuration_name for item in first.configuration_summaries} == {
        "combined_original",
        "focused_rules",
        "rsi_oversold_up",
        "always_up",
    }
    assert {path.name for path in destination.iterdir()} == set(
        COMPARISON_ARTIFACT_FILENAMES
    )
    assert validate_prediction_comparison_export(second, destination) == destination
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["feature_outcome_boundary"].startswith(
        "features are completed-session values"
    )
    with (destination / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert "rsi" in rows[0]
    assert "baseline_signed_return" in rows[0]
    assert "fill" not in rows[0]
    assert "order" not in rows[0]

    with pytest.raises(PredictionExportError, match="already exists"):
        export_prediction_comparison(second, tmp_path)


def test_comparison_export_validation_rejects_mutation(tmp_path: Path) -> None:
    closes = tuple("100" for _ in range(15))
    result = run_prediction_comparison(make_dataset(closes))
    destination = export_prediction_comparison(result, tmp_path)

    (destination / "metrics.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PredictionExportError, match="expected immutable result"):
        validate_prediction_comparison_export(result, destination)
