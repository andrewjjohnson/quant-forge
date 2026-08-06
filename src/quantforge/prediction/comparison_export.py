"""Atomic immutable exports for prediction comparison studies."""

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.prediction.comparison_models import PredictionComparisonStudyResult
from quantforge.prediction.errors import PredictionExportError

COMPARISON_ARTIFACT_FILENAMES = (
    "manifest.json",
    "configuration_summary.csv",
    "predictions.csv",
    "rule_summary.csv",
    "weekday_summary.csv",
    "annual_summary.csv",
    "period_summary.csv",
    "threshold_sensitivity.csv",
    "feature_bin_summary.csv",
    "baseline_comparison.csv",
    "best_outcomes.csv",
    "worst_outcomes.csv",
    "metrics.json",
)


def export_prediction_comparison(
    result: PredictionComparisonStudyResult, output_root: Path
) -> Path:
    """Atomically create one complete immutable comparison-study directory."""
    destination = output_root / result.study_id
    if destination.exists():
        raise PredictionExportError(
            f"prediction comparison already exists: {destination}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{result.study_id}.", dir=str(output_root))
    )
    try:
        _write_json(temporary / "manifest.json", result.manifest_primitive())
        _write_rows(
            temporary / "configuration_summary.csv",
            [item.to_primitive() for item in result.configuration_summaries],
        )
        _write_rows(
            temporary / "predictions.csv",
            [item.to_primitive() for item in result.predictions],
        )
        _write_rows(
            temporary / "rule_summary.csv",
            [item.to_primitive() for item in result.rule_summaries],
        )
        _write_rows(
            temporary / "weekday_summary.csv",
            [item.to_primitive() for item in result.weekday_summaries],
        )
        _write_rows(
            temporary / "annual_summary.csv",
            [item.to_primitive() for item in result.annual_summaries],
        )
        _write_rows(
            temporary / "period_summary.csv",
            [item.to_primitive() for item in result.period_summaries],
        )
        _write_rows(
            temporary / "threshold_sensitivity.csv",
            [item.to_primitive() for item in result.threshold_summaries],
        )
        _write_rows(
            temporary / "feature_bin_summary.csv",
            [item.to_primitive() for item in result.feature_bin_summaries],
        )
        _write_rows(
            temporary / "baseline_comparison.csv",
            [item.to_primitive() for item in result.baseline_comparisons],
        )
        _write_rows(
            temporary / "best_outcomes.csv",
            [item.to_primitive() for item in result.best_outcomes],
        )
        _write_rows(
            temporary / "worst_outcomes.csv",
            [item.to_primitive() for item in result.worst_outcomes],
        )
        _write_json(temporary / "metrics.json", result.metrics_primitive())
        os.rename(temporary, destination)
    except (OSError, TypeError, ValueError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise PredictionExportError(
            "failed to export immutable prediction comparison"
        ) from error
    return destination


def validate_prediction_comparison_export(
    result: PredictionComparisonStudyResult, path: Path
) -> Path:
    """Require an existing comparison export to match the result byte-for-byte."""
    try:
        if path.name != result.study_id or not path.is_dir():
            raise PredictionExportError(
                "comparison export does not match the expected immutable result"
            )
        entries = {entry.name: entry for entry in path.iterdir()}
        if set(entries) != set(COMPARISON_ARTIFACT_FILENAMES):
            raise PredictionExportError(
                "comparison export does not match the expected immutable result"
            )
        with tempfile.TemporaryDirectory(
            prefix="quantforge-comparison-export-validation-"
        ) as temporary_root:
            expected = export_prediction_comparison(result, Path(temporary_root))
            if any(
                entries[filename].read_bytes() != (expected / filename).read_bytes()
                for filename in COMPARISON_ARTIFACT_FILENAMES
            ):
                raise PredictionExportError(
                    "comparison export does not match the expected immutable result"
                )
    except PredictionExportError:
        raise
    except OSError as error:
        raise PredictionExportError(
            "failed to validate immutable prediction comparison export"
        ) from error
    return path


def _write_json(path: Path, value: PrimitiveMapping) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_rows(path: Path, rows: list[PrimitiveMapping]) -> None:
    if not rows:
        raise PredictionExportError(f"comparison artifact has no rows: {path.name}")
    fieldnames = tuple(rows[0])
    if any(tuple(row) != fieldnames for row in rows):
        raise PredictionExportError(
            f"comparison artifact rows have inconsistent schemas: {path.name}"
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row[name]) for name in fieldnames})
        stream.flush()
        os.fsync(stream.fileno())


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
