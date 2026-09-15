"""Bind native feature tables to validated saved checkpoints, without research."""

from pathlib import Path

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments._feature_integrity import validate_feature_rows
from quantforge.experiments._feature_row_integrity import validate_feature_schema
from quantforge.experiments._json import ManifestError, text
from quantforge.experiments._producer_snapshot import ProducerReadSet
from quantforge.prediction.feature_dataset import (
    _render_csv_rows,  # pyright: ignore[reportPrivateUsage]
    _render_parquet_rows,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.signal_feature_models import (
    SignalFeatureRow,
    SignalFeatureSchema,
)


def validate_feature_tables(
    source: Path,
    manifest: PrimitiveMapping,
    schema_record: PrimitiveMapping,
    reads: ProducerReadSet,
) -> tuple[Path, ...]:
    """Reuse only producer serialization, retaining the bytes read in the index."""
    directory = source / "rows"
    if not directory.is_dir():
        raise ManifestError("feature checkpoint directory is missing")
    paths = tuple(sorted(directory.glob("*.json")))
    records: list[PrimitiveMapping] = []
    for path in paths:
        record, _ = reads.read(path)
        if record.get("row_id") != path.stem:
            raise ManifestError("feature checkpoint filename differs from row identity")
        records.append(record)
    records.sort(key=lambda row: text(row.get("signal_session")))
    rows: list[Primitive] = list(records)
    validate_feature_rows(manifest, {"schema": schema_record, "rows": rows})
    schema = SignalFeatureSchema(
        text(schema_record["feature_schema_version"]),
        text(schema_record["outcome_schema_version"]),
        validate_feature_schema(manifest, schema_record),
    )
    checkpoints = tuple(SignalFeatureRow.capture(row) for row in records)
    tables = {"features.csv": _render_csv_rows(schema, checkpoints).encode("utf-8")}
    if manifest["engine_version"] == "35":
        tables["features.parquet"] = _render_parquet_rows(
            text(manifest["dataset_id"]), schema, checkpoints
        )
    for name, expected in tables.items():
        path = source / name
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ManifestError(f"cannot read feature table {name}") from error
        if content != expected:
            raise ManifestError(f"feature table {name} differs from saved checkpoints")
        reads.expect(path, content)
    return paths
