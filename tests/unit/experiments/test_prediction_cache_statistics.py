"""Persisted QF-32 cache diagnostics retain the producer's counter schema."""

from dataclasses import fields
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.prediction.grid import PredictionGridCacheStatistics
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import build_grid_export, read_record


@pytest.mark.parametrize("invalid", ["missing", None, [], "counters", {}, {"extra": 0}])
def test_prediction_summary_requires_cache_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: Primitive
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    path = root / "summary.json"
    summary = read_record(path)
    if invalid == "missing":
        summary.pop("cache_statistics")
    else:
        summary["cache_statistics"] = invalid
    write_json(path, summary)
    before = path.read_bytes()
    with pytest.raises(ManifestError, match="prediction summary cache statistics"):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "field", [item.name for item in fields(PredictionGridCacheStatistics)] + ["extra"]
)
@pytest.mark.parametrize("invalid", ["missing", -1, True, 1.5, "1", None])
def test_prediction_cache_statistics_require_exact_nonnegative_integer_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, invalid: Primitive
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    path = root / "summary.json"
    summary = read_record(path)
    counters = cast(PrimitiveMapping, summary["cache_statistics"])
    if invalid == "missing" and field != "extra":
        counters.pop(field)
    else:
        counters[field] = invalid
    write_json(path, summary)
    before = path.read_bytes()
    with pytest.raises(ManifestError, match="prediction summary cache statistics"):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("counter", [0, 10**20])
def test_prediction_cache_counters_remain_observational(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, counter: int
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    path = root / "summary.json"
    summary = read_record(path)
    summary["cache_statistics"] = {
        item.name: counter for item in fields(PredictionGridCacheStatistics)
    }
    write_json(path, summary)
    before = path.read_bytes()
    inspected = inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert verify_artifacts(inspected.index, tmp_path).valid
    assert path.read_bytes() == before
