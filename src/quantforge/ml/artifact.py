"""Event ML dataset artifacts: write once, validate offline, read typed (QF-67).

Layout (one immutable directory named by the scientific dataset ID)::

    <output-root>/<dataset-id>/
        manifest.json   canonical {"payload": ..., "fingerprint": sha256} line
        rows.parquet    compact rows (pyarrow, zstd, no dictionary encoding)
        rows.csv        optional human inspection copy (Parquet is authoritative)

Scientific identity (``dataset_id``) hashes the population, feature schema,
bound target, partition plan, column layout and the ordered logical rows.
Physical integrity is separate: the manifest fingerprint plus each file's
SHA-256 and byte size. Paths, timings and physical window IDs never enter the
scientific identity. ``read_event_dataset`` re-validates everything before
returning, including label consistency, ordering, membership and summaries.
"""

import csv
import hashlib
import io
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from importlib import import_module
from itertools import pairwise
from pathlib import Path
from typing import Protocol, cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.ml.dataset import (
    EVENT_DATASET_COMPONENT,
    EVENT_DATASET_SCHEMA_VERSION,
    EventDataset,
    EventRow,
    column_layout,
    label_summary,
    logical_rows_sha256,
)
from quantforge.ml.errors import (
    EventDatasetError,
    EventDatasetIntegrityError,
    EventHoldoutError,
)
from quantforge.ml.features import EventFeatureSchema, FeatureValue
from quantforge.ml.sources import WORKSPACE_HOLDOUT_LEDGER
from quantforge.ml.targets import ForwardReturnBinaryTarget, TargetLabel
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_encoding import canonical, decode
from quantforge.validation import PartitionRole

MANIFEST_FILE = "manifest.json"
ROWS_PARQUET = "rows.parquet"
ROWS_CSV = "rows.csv"
_ARTIFACT_FILES = frozenset({ROWS_PARQUET, ROWS_CSV})
_PAYLOAD_FIELDS = frozenset(
    {
        "component",
        "schema_version",
        "dataset_id",
        "scientific",
        "summaries",
        "provenance",
        "interpretation",
        "files",
    }
)


class _ArrowBuffer(Protocol):
    def to_pybytes(self) -> bytes: ...


class _ArrowSink(Protocol):
    def getvalue(self) -> _ArrowBuffer: ...


class _ArrowSchema(Protocol):
    def equals(self, other: object, *, check_metadata: bool) -> bool: ...


class _ArrowTable(Protocol):
    @property
    def schema(self) -> _ArrowSchema: ...

    @property
    def num_rows(self) -> int: ...

    def to_pylist(self) -> list[dict[str, object]]: ...


class _ArrowTableFactory(Protocol):
    def from_pylist(
        self, rows: list[dict[str, object]], *, schema: object
    ) -> _ArrowTable: ...


class _ArrowModule(Protocol):
    Table: _ArrowTableFactory
    BufferOutputStream: Callable[[], _ArrowSink]

    def bool_(self) -> object: ...

    def int32(self) -> object: ...

    def int64(self) -> object: ...

    def string(self) -> object: ...

    def date32(self) -> object: ...

    def timestamp(self, unit: str, *, tz: str) -> object: ...

    def field(
        self,
        name: str,
        data_type: object,
        *,
        nullable: bool,
        metadata: dict[bytes, bytes],
    ) -> object: ...

    def schema(
        self, fields: list[object], *, metadata: dict[bytes, bytes]
    ) -> object: ...


class _ParquetModule(Protocol):
    def write_table(
        self,
        table: object,
        where: object,
        *,
        compression: str,
        use_dictionary: bool,
        write_statistics: bool,
    ) -> None: ...

    def read_table(self, source: io.BytesIO) -> _ArrowTable: ...


def _arrow() -> tuple[_ArrowModule, _ParquetModule]:
    return (
        cast(_ArrowModule, import_module("pyarrow")),
        cast(_ParquetModule, import_module("pyarrow.parquet")),
    )


def _arrow_schema(dataset_id: str, layout: PrimitiveMapping) -> object:
    pa, _ = _arrow()
    types: dict[str, Callable[[], object]] = {
        "int64": pa.int64,
        "int32": pa.int32,
        "string": pa.string,
        "boolean": pa.bool_,
        "date32": pa.date32,
        "timestamp_us_utc": lambda: pa.timestamp("us", tz="UTC"),
    }
    fields: list[object] = []
    for item in cast(list[PrimitiveMapping], layout["columns"]):
        fields.append(
            pa.field(
                cast(str, item["name"]),
                types[cast(str, item["arrow_type"])](),
                nullable=cast(bool, item["nullable"]),
                metadata={
                    b"quantforge_column_group": cast(str, item["group"]).encode()
                },
            )
        )
    return pa.schema(
        fields,
        metadata={
            b"quantforge_event_dataset_id": dataset_id.encode("ascii"),
            b"quantforge_column_layout": canonical(layout),
        },
    )


def _arrow_rows(dataset: EventDataset) -> list[dict[str, object]]:
    names = dataset.feature_columns
    return [
        {
            "row_index": index,
            "source_observation_id": row.source_observation_id,
            "source_index": row.source_index,
            "decision_timestamp": row.decision_timestamp,
            "signal_session": row.signal_session,
            "decision_sequence": row.decision_sequence,
            "signal_index": row.signal_index,
            "context_id": row.context_id,
            "prediction_study_id": row.prediction_study_id,
            "direction": row.direction,
            "disposition": row.disposition,
            "partition_role": row.partition_role.value,
            "fold_id": row.fold_id,
            "fold_index": row.fold_index,
            **dict(zip(names, row.features, strict=True)),
            "target": row.label.value,
            "target_status": row.label.status,
            "target_source_value": row.label.source_value,
            "target_outcome_id": row.label.outcome_id,
        }
        for index, row in enumerate(dataset.rows)
    ]


def render_parquet(dataset: EventDataset) -> bytes:
    """Deterministic Parquet bytes for one pyarrow version (QF-29 settings)."""
    pa, pq = _arrow()
    layout = cast(PrimitiveMapping, dataset.scientific.to_primitive()["columns"])
    table = pa.Table.from_pylist(
        _arrow_rows(dataset), schema=_arrow_schema(dataset.dataset_id, layout)
    )
    sink = pa.BufferOutputStream()
    pq.write_table(
        table, sink, compression="zstd", use_dictionary=False, write_statistics=True
    )
    return sink.getvalue().to_pybytes()


def _csv_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def render_csv(dataset: EventDataset) -> bytes:
    """Inspection-only CSV; empty cells are nulls, booleans are true/false."""
    layout = cast(PrimitiveMapping, dataset.scientific.to_primitive()["columns"])
    names = [
        cast(str, item["name"])
        for item in cast(list[PrimitiveMapping], layout["columns"])
    ]
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(names)
    for record in _arrow_rows(dataset):
        writer.writerow([_csv_text(record[name]) for name in names])
    return stream.getvalue().encode("utf-8")


def _write(path: Path, content: bytes) -> PrimitiveMapping:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def export_event_dataset(
    dataset: EventDataset, output_root: Path, *, include_csv: bool = False
) -> Path:
    """Publish one immutable dataset directory atomically; return its path.

    Files are written and fsynced in a private staging directory and fully
    validated there; only then is it renamed to ``<output-root>/<dataset-id>``
    and the parent fsynced. An existing directory with the same scientific
    identity is validated and reused as is (its optional CSV is not added); a
    conflicting one is an error. Output is refused inside the permanent
    holdout ledger.
    """
    if type(cast(object, dataset)) is not EventDataset:
        raise EventDatasetError("only verified event datasets can be exported")
    root = Path(output_root).resolve()
    if any(part.lower() == WORKSPACE_HOLDOUT_LEDGER.name for part in root.parts):
        raise EventHoldoutError("event datasets cannot be written inside a ledger")
    final = root / dataset.dataset_id
    if final.exists():
        if read_event_dataset(final).dataset_id != dataset.dataset_id:
            raise EventDatasetIntegrityError("existing dataset directory differs")
        return final
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset.dataset_id}.", dir=root))
    try:
        files: PrimitiveMapping = {
            ROWS_PARQUET: _write(staging / ROWS_PARQUET, render_parquet(dataset))
        }
        if include_csv:
            files[ROWS_CSV] = _write(staging / ROWS_CSV, render_csv(dataset))
        payload: PrimitiveMapping = {**dataset.manifest(), "files": files}
        _write(
            staging / MANIFEST_FILE,
            canonical(
                {"payload": payload, "fingerprint": configuration_identity(payload)}
            )
            + b"\n",
        )
        _sync_directory(staging)
        # Validate the staged bytes before publication: nothing unverifiable
        # ever appears under the scientific dataset ID.
        if _read(staging, dataset.dataset_id) != dataset:
            raise EventDatasetIntegrityError("staged dataset differs from the source")
        try:
            staging.rename(final)
        except OSError as error:
            if not final.exists():
                raise EventDatasetIntegrityError("cannot publish dataset") from error
            if read_event_dataset(final).dataset_id != dataset.dataset_id:
                raise EventDatasetIntegrityError(
                    "conflicting dataset directory"
                ) from error
        _sync_directory(root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EventDatasetIntegrityError(message)


def _manifest(path: Path) -> PrimitiveMapping:
    try:
        envelope = decode((path / MANIFEST_FILE).read_bytes(), canonical_line=True)
    except (OSError, InvalidPredictionOutputError) as error:
        raise EventDatasetIntegrityError(
            "dataset manifest is missing or invalid"
        ) from error
    payload = envelope.get("payload")
    _require(
        set(envelope) == {"payload", "fingerprint"}
        and isinstance(payload, dict)
        and envelope["fingerprint"] == configuration_identity(payload),
        "dataset manifest fingerprint is inconsistent",
    )
    return cast(PrimitiveMapping, payload)


def _file(path: Path, record: Primitive) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise EventDatasetIntegrityError(f"missing dataset file {path.name}") from error
    _require(
        isinstance(record, dict)
        and set(record) == {"sha256", "bytes"}
        and record["sha256"] == hashlib.sha256(content).hexdigest()
        and record["bytes"] == len(content),
        f"dataset file {path.name} differs from its manifest",
    )
    return content


def _parse_rows(
    table: _ArrowTable,
    schema: EventFeatureSchema,
    target: ForwardReturnBinaryTarget,
) -> list[EventRow]:
    names = schema.column_names
    rows: list[EventRow] = []
    for index, record in enumerate(table.to_pylist()):
        try:
            _require(record["row_index"] == index, "row index differs from position")
            stamp = cast(datetime, record["decision_timestamp"])
            features: tuple[FeatureValue, ...] = tuple(
                definition.value(
                    {definition.source_field: cast(Primitive, record[name])}
                )
                for definition, name in zip(schema.features, names, strict=True)
            )
            value, status = record["target"], record["target_status"]
            source_value = record["target_source_value"]
            available = status == "available"
            _require(
                (value is None) is (not available)
                and (source_value is None) is (not available)
                and (
                    source_value is None
                    or value == (Decimal(cast(str, source_value)) > target.threshold)
                ),
                "target label is inconsistent with its outcome value",
            )
            fold_index = record["fold_index"]
            rows.append(
                EventRow(
                    source_observation_id=cast(str, record["source_observation_id"]),
                    source_index=cast(int, record["source_index"]),
                    decision_timestamp=stamp.astimezone(UTC),
                    signal_session=cast(date, record["signal_session"]),
                    decision_sequence=cast(int, record["decision_sequence"]),
                    signal_index=cast(int, record["signal_index"]),
                    context_id=cast(str | None, record["context_id"]),
                    prediction_study_id=cast(str, record["prediction_study_id"]),
                    direction=cast(str | None, record["direction"]),
                    disposition=cast(str | None, record["disposition"]),
                    partition_role=PartitionRole(cast(str, record["partition_role"])),
                    fold_id=cast(str | None, record["fold_id"]),
                    fold_index=cast(int | None, fold_index),
                    features=features,
                    label=TargetLabel(
                        cast(bool | None, value),
                        cast(str, status),
                        cast(str | None, source_value),
                        cast(str, record["target_outcome_id"]),
                    ),
                )
            )
        except (KeyError, TypeError, ValueError, ArithmeticError) as error:
            if isinstance(error, EventDatasetError):
                raise
            raise EventDatasetIntegrityError("dataset row is malformed") from error
    return rows


def read_event_dataset(path: Path) -> EventDataset:
    """Validate a dataset directory offline and return the typed dataset."""
    path = Path(path)
    return _read(path, path.name)


def _read(path: Path, name: str) -> EventDataset:
    """Validate a directory whose published name must be ``name``."""
    payload = _manifest(path)
    _require(
        frozenset(payload) == _PAYLOAD_FIELDS, "dataset manifest fields are invalid"
    )
    scientific = payload["scientific"]
    _require(
        payload["component"] == EVENT_DATASET_COMPONENT
        and payload["schema_version"] == EVENT_DATASET_SCHEMA_VERSION
        and isinstance(scientific, dict)
        and scientific.get("component") == EVENT_DATASET_COMPONENT
        and scientific.get("schema_version") == EVENT_DATASET_SCHEMA_VERSION,
        "unsupported event dataset schema",
    )
    scientific = cast(PrimitiveMapping, scientific)
    dataset_id = configuration_identity(scientific)
    _require(
        payload["dataset_id"] == dataset_id == name,
        "dataset identity differs from its scientific content or directory",
    )
    try:
        schema = EventFeatureSchema.from_primitive(scientific["feature_schema"])
        bound = cast(PrimitiveMapping, scientific["target"])
        target = ForwardReturnBinaryTarget.from_primitive(
            cast(PrimitiveMapping, bound["target"])
        )
    except (KeyError, TypeError) as error:
        raise EventDatasetIntegrityError("dataset schemas are missing") from error
    _require(
        scientific.get("feature_schema_id") == schema.schema_id
        and bound.get("target_id") == target.target_id
        and scientific.get("target_configuration_id") == configuration_identity(bound)
        and scientific.get("columns") == column_layout(schema),
        "dataset schema identities are inconsistent",
    )
    files = payload["files"]
    _require(
        isinstance(files, dict)
        and ROWS_PARQUET in files
        and set(files) <= _ARTIFACT_FILES,
        "dataset file list is invalid",
    )
    files = cast(PrimitiveMapping, files)
    for name in _ARTIFACT_FILES - set(files):
        _require(not (path / name).exists(), f"unlisted dataset file {name} exists")
    content = _file(path / ROWS_PARQUET, files[ROWS_PARQUET])
    _, pq = _arrow()
    try:
        table = pq.read_table(io.BytesIO(content))
    except (OSError, ValueError) as error:
        raise EventDatasetIntegrityError("rows.parquet cannot be decoded") from error
    layout = cast(PrimitiveMapping, scientific["columns"])
    _require(
        table.schema.equals(_arrow_schema(dataset_id, layout), check_metadata=True),
        "rows.parquet schema differs from the declared column layout",
    )
    rows = _parse_rows(table, schema, target)
    plan = cast(PrimitiveMapping, scientific["partition_plan"])
    sources = cast(list[PrimitiveMapping], plan["sources"])
    declared = cast(PrimitiveMapping, scientific["rows"])
    _require(
        declared.get("row_count") == len(rows) == table.num_rows
        and declared.get("logical_rows_sha256") == logical_rows_sha256(rows),
        "dataset rows differ from their scientific identity",
    )
    keys = [row.sort_key() for row in rows]
    _require(
        all(left < right for left, right in pairwise(keys)),
        "dataset rows are not in their documented order",
    )
    _require(
        len({row.source_observation_id for row in rows}) == len(rows),
        "duplicate source observation IDs",
    )
    for row in rows:
        _require(0 <= row.source_index < len(sources), "row references no source")
        source = sources[row.source_index]
        _require(
            source.get("role") == row.partition_role.value
            and source.get("fold_id") == row.fold_id
            and source.get("fold_index") == row.fold_index,
            "row membership differs from its source partition",
        )
    _require(
        payload["summaries"] == label_summary(rows),
        "dataset summaries differ from their rows",
    )
    dataset = EventDataset(
        dataset_id,
        PrimitiveMappingSnapshot.capture(scientific),
        PrimitiveMappingSnapshot.capture(cast(PrimitiveMapping, payload["summaries"])),
        PrimitiveMappingSnapshot.capture(cast(PrimitiveMapping, payload["provenance"])),
        schema,
        target,
        tuple(rows),
    )
    _require(
        {k: v for k, v in payload.items() if k != "files"} == dataset.manifest(),
        "dataset manifest differs from its typed dataset",
    )
    if ROWS_CSV in files:
        _require(
            _file(path / ROWS_CSV, files[ROWS_CSV]) == render_csv(dataset),
            "rows.csv differs from rows.parquet",
        )
    return dataset


def validate_event_dataset(path: Path) -> str:
    """Offline validation only; returns the verified scientific dataset ID."""
    return read_event_dataset(path).dataset_id


__all__ = [
    "MANIFEST_FILE",
    "ROWS_CSV",
    "ROWS_PARQUET",
    "export_event_dataset",
    "read_event_dataset",
    "render_csv",
    "render_parquet",
    "validate_event_dataset",
]
