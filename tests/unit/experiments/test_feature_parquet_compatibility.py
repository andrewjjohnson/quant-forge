"""Historical Parquet encodings retain their original bytes and logical data."""

import json
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.prediction import PredictionRuleContext, SignalFeatureCandidateOutput
from tests.unit.experiments.test_adapters import block_research
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _build,  # pyright: ignore[reportPrivateUsage]
    _FixtureCandidateRule,  # pyright: ignore[reportPrivateUsage]
)


def export_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty: bool
) -> Path:
    original = _FixtureCandidateRule.generate_with_context

    def generate_empty(
        rule: _FixtureCandidateRule, context: PredictionRuleContext
    ) -> SignalFeatureCandidateOutput:
        return replace(original(rule, context), signals=())

    with monkeypatch.context() as patch:
        if empty:
            patch.setattr(
                _FixtureCandidateRule, "generate_with_context", generate_empty
            )
        result, _, _ = _build(tmp_path / "features")
    assert bool(result.rows) != empty
    return tmp_path / "features" / result.dataset_id


@pytest.mark.parametrize("empty", [False, True], ids=["populated", "empty"])
@pytest.mark.parametrize("compression", [None, "snappy", "gzip", "zstd"])
def test_equivalent_parquet_encodings_preserve_original_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty: bool,
    compression: str | None,
) -> None:
    root = export_fixture(tmp_path, monkeypatch, empty)
    path = root / "features.parquet"
    original = path.read_bytes()
    pa, pq = import_module("pyarrow"), import_module("pyarrow.parquet")
    table = pq.read_table(pa.BufferReader(original))
    metadata = dict(table.schema.metadata)
    # Formatting and unrelated writer annotations are not scientific provenance.
    metadata[b"quantforge_schema"] = json.dumps(
        json.loads(metadata[b"quantforge_schema"]), indent=2
    ).encode()
    metadata[b"writer_annotation"] = b"historical export"
    table = table.replace_schema_metadata(metadata)
    pq.write_table(
        table,
        path,
        compression=compression,
        use_dictionary=True,
        write_statistics=False,
        row_group_size=1,
        version="1.0",
    )
    historical = path.read_bytes()
    assert historical != original
    block_research(monkeypatch)

    def no_writer(*args: Any, **kwargs: Any) -> None:
        pytest.fail("inspection attempted to regenerate Parquet")

    monkeypatch.setattr(pq, "write_table", no_writer)
    bundle = inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)
    entry = next(
        entry
        for entry in bundle.index.entries
        if entry.path.endswith("features.parquet")
    )
    assert entry.sha256 == sha256(historical).hexdigest()
    assert path.read_bytes() == historical
    assert verify_artifacts(bundle.index, tmp_path).valid


@pytest.mark.parametrize(
    "change",
    [
        "column_type",
        "column_order",
        "nullability",
        "null_value",
        "missing_column",
        "extra_column",
        "schema_metadata",
        "missing_metadata",
        "invalid_metadata",
        "corrupt",
    ],
)
def test_logical_parquet_corruption_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    root = export_fixture(tmp_path, monkeypatch, False)
    path = root / "features.parquet"
    pa, pq = import_module("pyarrow"), import_module("pyarrow.parquet")
    table = pq.read_table(pa.BufferReader(path.read_bytes()))
    position = table.schema.get_field_index("direction")
    if change == "column_type":
        table = table.set_column(position, "direction", pa.array([1] * len(table)))
    elif change == "column_order":
        table = table.select(list(reversed(table.column_names)))
    elif change == "nullability":
        field = table.schema.field(position)
        table = table.set_column(
            position, field.with_nullable(not field.nullable), table.column(position)
        )
    elif change == "null_value":
        table = table.set_column(
            position,
            table.schema.field(position),
            pa.array([None] * len(table), type=pa.string()),
        )
    elif change == "missing_column":
        table = table.drop(["direction"])
    elif change == "extra_column":
        table = table.append_column("foreign", pa.array([1] * len(table)))
    elif change in {"schema_metadata", "missing_metadata", "invalid_metadata"}:
        metadata = dict(table.schema.metadata)
        if change == "schema_metadata":
            schema = json.loads(metadata[b"quantforge_schema"])
            schema["feature_schema_version"] = "foreign"
            metadata[b"quantforge_schema"] = json.dumps(schema).encode()
        elif change == "missing_metadata":
            metadata.pop(b"quantforge_schema")
        else:
            metadata[b"quantforge_schema"] = b"not-json"
        table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="snappy", use_dictionary=True)
    if change == "corrupt":
        path.write_bytes(b"not-parquet")
    before = path.read_bytes()
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="feature table"):
        inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)
    assert path.read_bytes() == before
