"""QF-6 stability remains complete even when dependent projections are refreshed."""

import csv
import json
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
from quantforge.optimization.export import (
    _csv_text,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.optimization.models import StabilitySummary
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import build_grid_export, read_record


def rewrite_projections(root: Path, document: PrimitiveMapping) -> None:
    records = cast(list[PrimitiveMapping], document["summaries"])
    write_json(root / "stability.json", document)
    summary = read_record(root / "summary.json")
    summary["top_stability_trials"] = cast(list[Primitive], records[:10])
    write_json(root / "summary.json", summary)
    with (root / "stability.csv").open() as handle:
        parameters = [
            cast(Primitive, json.loads(row["parameters"]))
            for row in csv.DictReader(handle)
        ]
    rows = [
        {**record, "parameters": parameter}
        for record, parameter in zip(records, parameters, strict=True)
    ]
    (root / "stability.csv").write_text(
        _csv_text(
            rows, (*(item.name for item in fields(StabilitySummary)), "parameters")
        )
    )


@pytest.mark.parametrize(
    "field", [item.name for item in fields(StabilitySummary)] + ["extra"]
)
def test_optimization_stability_requires_exact_fields_after_projection_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    document = read_record(root / "stability.json")
    record = cast(list[PrimitiveMapping], document["summaries"])[0]
    if field == "extra":
        record[field] = "unexpected"
    else:
        record.pop(field)
    rewrite_projections(root, document)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ManifestError, match="optimization stability"):
        inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        (field, invalid)
        for field in (
            "valid_neighbor_count",
            "excluded_neighbor_count",
            "successful_eligible_neighbor_count",
        )
        for invalid in (-1, True, "1", None)
    ]
    + [
        (field, invalid)
        for field in (
            "mean_neighbor_objective",
            "worst_neighbor_objective",
            "objective_standard_deviation",
            "stability_score",
            "constraint_pass_fraction",
        )
        for invalid in (True, 1, "NaN", "invalid")
    ]
    + [
        ("objective_standard_deviation", "-1"),
        ("stability_score", "-0.1"),
        ("stability_score", "1.1"),
        ("constraint_pass_fraction", "1.1"),
        ("stability_rank", None),
        ("stability_rank", True),
        ("is_boundary", "false"),
        ("is_isolated_peak", 0),
        ("classification", "unknown"),
        ("isolation_reason", False),
        ("neighbor_objective_values", [True]),
    ],
)
def test_optimization_stability_rejects_invalid_domains_after_projection_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, invalid: Primitive
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    document = read_record(root / "stability.json")
    cast(list[PrimitiveMapping], document["summaries"])[0][field] = invalid
    rewrite_projections(root, document)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ManifestError, match="optimization stability"):
        inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    assert {path: path.read_bytes() for path in before} == before


def test_native_optimization_stability_preserves_nullable_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    records = cast(
        list[PrimitiveMapping], read_record(root / "stability.json")["summaries"]
    )
    assert any(record["objective_standard_deviation"] is None for record in records)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    inspected = inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    assert verify_artifacts(inspected.index, tmp_path).valid
    assert {path: path.read_bytes() for path in before} == before
