"""Atomic immutable JSON and CSV export for prediction analyses."""

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import cast

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.prediction.errors import PredictionExportError
from quantforge.prediction.models import PredictionAnalysisResult

_PREDICTION_FIELDS = (
    "prediction_id",
    "dataset_id",
    "dataset_fingerprint",
    "symbol",
    "signal_session",
    "outcome_session",
    "direction",
    "strategy_id",
    "strategy_implementation_version",
    "strategy_configuration_id",
    "strategy_parameters",
    "reason",
    "feature_values",
    "signal_close",
    "next_open",
    "overnight_gap_percentage",
    "gap_size_percentage",
    "signed_prediction_return",
    "correct",
)

PREDICTION_ARTIFACT_FILENAMES = ("manifest.json", "predictions.csv")


def export_prediction_analysis(
    result: PredictionAnalysisResult, output_root: Path
) -> Path:
    """Atomically export a new analysis directory without overwriting results."""
    destination = output_root / result.analysis_id
    if destination.exists():
        raise PredictionExportError(
            f"prediction analysis already exists: {destination}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{result.analysis_id}.", dir=str(output_root))
    )
    try:
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                result.manifest_primitive(),
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        _write_predictions_csv(temporary / "predictions.csv", result)
        os.rename(temporary, destination)
    except (OSError, TypeError, ValueError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise PredictionExportError(
            "failed to export immutable prediction analysis"
        ) from error
    return destination


def load_prediction_manifest(path: Path) -> PrimitiveMapping:
    """Load an exported manifest and reject non-object JSON."""
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PredictionExportError("failed to load prediction manifest") from error
    if not isinstance(loaded, dict):
        raise PredictionExportError("prediction manifest must be a JSON object")
    loaded_mapping = cast(dict[object, object], loaded)
    if any(not isinstance(key, str) for key in loaded_mapping):
        raise PredictionExportError("prediction manifest keys must be strings")
    return cast(PrimitiveMapping, loaded_mapping)


def validate_prediction_analysis_export(
    result: PredictionAnalysisResult, path: Path
) -> Path:
    """Require an existing export to exactly match this immutable result."""
    try:
        if path.name != result.analysis_id or not path.is_dir():
            raise PredictionExportError(
                "prediction export does not match the expected immutable result"
            )
        entries = {entry.name: entry for entry in path.iterdir()}
        if set(entries) != set(PREDICTION_ARTIFACT_FILENAMES):
            raise PredictionExportError(
                "prediction export does not match the expected immutable result"
            )
        manifest = load_prediction_manifest(entries["manifest.json"])
        if manifest.get("analysis_id") != result.analysis_id:
            raise PredictionExportError(
                "prediction export does not match the expected immutable result"
            )
        with tempfile.TemporaryDirectory(
            prefix="quantforge-prediction-export-validation-"
        ) as temporary_root:
            expected_path = export_prediction_analysis(result, Path(temporary_root))
            if any(
                entries[filename].read_bytes()
                != (expected_path / filename).read_bytes()
                for filename in PREDICTION_ARTIFACT_FILENAMES
            ):
                raise PredictionExportError(
                    "prediction export does not match the expected immutable result"
                )
    except PredictionExportError:
        raise
    except OSError as error:
        raise PredictionExportError(
            "failed to validate immutable prediction export"
        ) from error
    return path


def _write_predictions_csv(path: Path, result: PredictionAnalysisResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=_PREDICTION_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for row in result.rows:
            primitive = row.to_primitive()
            writer.writerow(
                {name: _csv_value(primitive.get(name)) for name in _PREDICTION_FIELDS}
            )
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
