"""Bind native feature tables to validated saved checkpoints, without research."""

from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Protocol, cast

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments._feature_integrity import validate_feature_rows
from quantforge.experiments._feature_row_integrity import validate_feature_schema
from quantforge.experiments._json import ManifestError, parse_json, text
from quantforge.experiments._producer_snapshot import ProducerReadSet
from quantforge.prediction.feature_dataset import (
    _parquet_table,  # pyright: ignore[reportPrivateUsage]
    _render_csv_rows,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.signal_feature_models import (
    SignalFeatureRow,
    SignalFeatureSchema,
)


class _ArrowSchema(Protocol):
    @property
    def metadata(self) -> dict[bytes, bytes] | None: ...


class _ArrowTable(Protocol):
    @property
    def schema(self) -> _ArrowSchema: ...

    def equals(self, other: object, *, check_metadata: bool) -> bool: ...


class _ParquetReader(Protocol):
    def read_table(self, source: BytesIO) -> _ArrowTable: ...


def _validate_parquet(
    content: bytes,
    dataset_id: str,
    schema: SignalFeatureSchema,
    checkpoints: tuple[SignalFeatureRow, ...],
) -> None:
    parquet = cast(_ParquetReader, import_module("pyarrow.parquet"))
    try:
        table = parquet.read_table(BytesIO(content))
        metadata = table.schema.metadata or {}
        saved_schema = parse_json(metadata.get(b"quantforge_schema", b"{}"))
    except (OSError, ValueError, TypeError) as error:
        raise ManifestError("cannot decode feature table features.parquet") from error
    if (
        metadata.get(b"quantforge_dataset_id") != dataset_id.encode("ascii")
        or configuration_identity(saved_schema)
        != configuration_identity(schema.to_primitive())
        or not table.equals(
            _parquet_table(dataset_id, schema, checkpoints), check_metadata=False
        )
    ):
        raise ManifestError(
            "feature table features.parquet differs from saved checkpoints"
        )


def validate_feature_tables(
    source: Path,
    manifest: PrimitiveMapping,
    schema_record: PrimitiveMapping,
    reads: ProducerReadSet,
) -> tuple[Path, ...]:
    """Check logical rows/schema, retaining the original table bytes in the index."""
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
    records.sort(
        key=lambda row: (
            text(row.get("signal_session")),
            ""
            if row.get("decision_timestamp") is None
            else text(row["decision_timestamp"]),
        )
    )
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
        path = source / "features.parquet"
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ManifestError("cannot read feature table features.parquet") from error
        _validate_parquet(content, text(manifest["dataset_id"]), schema, checkpoints)
        reads.expect(path, content)
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
