"""Directory table bytes must retain the producer's checkpointed rows."""

import csv
import io
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
from quantforge.prediction import SignalFeatureDatasetResult
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_feature_integrity import (
    feature_result as feature_result,
)
from tests.unit.experiments.test_grid_integrity import read_record
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _build,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize("change", ["value", "truncate", "header", "duplicate"])
def test_feature_csv_matches_checkpoint_rows(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult, change: str
) -> None:
    root = tmp_path / "features" / feature_result.dataset_id
    path = root / "features.csv"
    rows = list(csv.reader(io.StringIO(path.read_text())))
    if change == "value":
        rows[1][rows[0].index("direction")] = "corrupt"
    elif change == "truncate":
        rows.pop()
    elif change == "header":
        rows[0][0] = "foreign_column"
    else:
        rows.append(rows[1])
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    path.write_text(stream.getvalue())
    with pytest.raises(ManifestError, match="feature table"):
        inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)


@pytest.mark.parametrize("change", ["value", "truncate", "metadata", "duplicate"])
def test_feature_parquet_matches_checkpoint_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    result, _, _ = _build(tmp_path / "features")
    root = tmp_path / "features" / result.dataset_id
    path = root / "features.parquet"
    pa = import_module("pyarrow")
    pq = import_module("pyarrow.parquet")
    table = pq.read_table(pa.BufferReader(path.read_bytes()))
    if change == "value":
        position = table.schema.get_field_index("direction")
        table = table.set_column(
            position, table.schema.field(position), pa.array(["corrupt"] * len(table))
        )
    elif change == "truncate":
        table = table.slice(0, len(table) - 1)
    elif change == "metadata":
        table = table.replace_schema_metadata({b"quantforge_dataset_id": b"foreign"})
    else:
        table = pa.concat_tables([table, table])
    pq.write_table(
        table, path, compression="zstd", use_dictionary=False, write_statistics=True
    )
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="feature table"):
        inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "change", ["missing", "rename", "duplicate", "value", "missing_directory"]
)
def test_feature_checkpoint_evidence_is_required_and_validated(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult, change: str
) -> None:
    root = tmp_path / "features" / feature_result.dataset_id
    path = next((root / "rows").glob("*.json"))
    if change == "missing":
        path.unlink()
    elif change == "missing_directory":
        (root / "rows").rename(root / "untrusted_rows")
    elif change == "rename":
        path.rename(path.with_name("foreign.json"))
    elif change == "duplicate":
        path.with_name("duplicate.json").write_bytes(path.read_bytes())
    else:
        row = read_record(path)
        row["direction"] = "corrupt"
        write_json(path, row)
    with pytest.raises(ManifestError, match="feature"):
        inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)


def test_feature_checkpoints_are_indexed_and_remain_immutable(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult
) -> None:
    root = tmp_path / "features" / feature_result.dataset_id
    bundle = inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)
    rows = list((root / "rows").glob("*.json"))
    assert {p.relative_to(tmp_path).as_posix() for p in rows} <= {
        e.path for e in bundle.index.entries
    }
    assert verify_artifacts(bundle.index, tmp_path).valid
    rows[0].write_text("{}")
    assert not verify_artifacts(bundle.index, tmp_path).valid


@pytest.mark.parametrize("target", ["row", "csv", "parquet"])
def test_feature_files_cannot_change_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    from quantforge.experiments import adapters
    from quantforge.experiments.artifacts import ArtifactEntry

    result, _, _ = _build(tmp_path / "features")
    root = tmp_path / "features" / result.dataset_id
    path = (
        next((root / "rows").glob("*.json"))
        if target == "row"
        else root / ("features.csv" if target == "csv" else "features.parquet")
    )
    original = adapters.index_artifact
    changed = False

    def replace_file(artifact_root: Path, **arguments: Any) -> ArtifactEntry:
        nonlocal changed
        if artifact_root / arguments["path"] == path and not changed:
            path.write_bytes(path.read_bytes() + b"\n")
            changed = True
        return original(artifact_root, **arguments)

    block_research(monkeypatch)
    monkeypatch.setattr(adapters, "index_artifact", replace_file)
    with pytest.raises(ManifestError, match="changed during indexing"):
        inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)
    assert changed


def test_empty_feature_directory_preserves_header_and_zero_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.helpers import make_dataset
    from tests.unit.prediction.test_feature_dataset import (
        FixtureCandidateRule,
        _build_fixture,  # pyright: ignore[reportPrivateUsage]
    )

    result = _build_fixture(
        make_dataset(("100", "102")), FixtureCandidateRule(()), tmp_path / "features"
    )
    block_research(monkeypatch)
    root = tmp_path / "features" / result.dataset_id
    bundle = inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)
    assert verify_artifacts(bundle.index, tmp_path).valid
    (root / "rows").rmdir()
    with pytest.raises(ManifestError, match="feature checkpoint directory"):
        inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)
