"""Parameter-summary schemas remain required after their CSV is regenerated."""

from copy import deepcopy
from dataclasses import fields
from decimal import Decimal
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
from quantforge.experiments._stability_integrity import validate_parameter_summaries
from quantforge.optimization import GridSearchStudy, MovingAverageCrossoverFactory
from quantforge.optimization.export import (
    _csv_text,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.optimization.models import ParameterSummary
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record, selective_runner
from tests.unit.optimization.test_study import (
    _dataset,  # pyright: ignore[reportPrivateUsage]
    _study_config,  # pyright: ignore[reportPrivateUsage]
)

type Captured = tuple[Path, PrimitiveMapping]


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory) -> Captured:
    study = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path_factory.mktemp("parameter-summary")),
        backtest_runner=selective_runner,
    )
    study.run()
    return study.study_path, read_record(study.study_path / "summary.json")


@pytest.fixture(autouse=True)
def no_research(captured: Captured, monkeypatch: pytest.MonkeyPatch) -> None:
    block_research(monkeypatch)


def inspect(captured: Captured, summary: PrimitiveMapping) -> None:
    root, _ = captured
    write_json(root / "summary.json", summary)
    raw = summary.get("parameter_summaries")
    rows = (
        [cast(PrimitiveMapping, row) for row in raw if isinstance(row, dict)]
        if isinstance(raw, list)
        else []
    )
    (root / "parameter_summary.csv").write_text(
        _csv_text(rows, tuple(field.name for field in fields(ParameterSummary)))
    )
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    try:
        bundle = inspect_study(StudyType.OPTIMIZATION, root, artifact_root=root.parent)
        assert verify_artifacts(bundle.index, root.parent).valid
    finally:
        assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "field", [field.name for field in fields(ParameterSummary)] + ["extra", "empty"]
)
def test_parameter_summary_requires_exact_fields(
    captured: Captured, field: str
) -> None:
    summary = deepcopy(captured[1])
    record = cast(list[PrimitiveMapping], summary["parameter_summaries"])[0]
    if field == "extra":
        record[field] = None
    elif field == "empty":
        summary["parameter_summaries"] = [{}]
    else:
        record.pop(field)
    with pytest.raises(ManifestError, match="optimization parameter summary"):
        inspect(captured, summary)


@pytest.mark.parametrize("invalid", [None, {}, "", 1, [None], [1], ["invalid"]])
def test_parameter_summary_collection_requires_records(
    captured: Captured,
    invalid: Primitive,
) -> None:
    summary = deepcopy(captured[1])
    summary["parameter_summaries"] = invalid
    with pytest.raises(ManifestError, match="optimization parameter summary"):
        inspect(captured, summary)


def test_parameter_summary_collection_is_required(captured: Captured) -> None:
    summary = deepcopy(captured[1])
    del summary["parameter_summaries"]
    with pytest.raises(ManifestError, match="optimization parameter summary"):
        inspect(captured, summary)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("parameter_name", invalid)
        for invalid in (None, "", "  ", True, 1, cast(Primitive, {}))
    ]
    + [
        ("parameter_value", invalid)
        for invalid in (None, 1.5, cast(Primitive, []), cast(Primitive, {}))
    ]
    + [
        (field, invalid)
        for field in ("successful_count", "eligible_count")
        for invalid in (-1, True, "1", 1.0, None)
    ]
    + [
        ("constraint_pass_fraction", invalid)
        for invalid in (None, True, 1, "NaN", "Infinity", "invalid", "-0.1", "1.1")
    ]
    + [
        (field, invalid)
        for field in ("mean_objective", "median_objective", "best_objective")
        for invalid in (None, True, 1, "NaN", "Infinity", "invalid")
    ],
)
def test_parameter_summary_domains_survive_csv_projection_refresh(
    captured: Captured,
    field: str,
    invalid: Primitive,
) -> None:
    summary = deepcopy(captured[1])
    record = next(
        item
        for item in cast(list[PrimitiveMapping], summary["parameter_summaries"])
        if cast(int, item["eligible_count"]) > 0
    )
    record[field] = invalid
    with pytest.raises(ManifestError, match="optimization parameter summary"):
        inspect(captured, summary)


@pytest.mark.parametrize(
    "field", ["eligible_count", "mean_objective", "median_objective", "best_objective"]
)
def test_parameter_summary_counts_and_null_availability_agree(
    captured: Captured,
    field: str,
) -> None:
    summary = deepcopy(captured[1])
    record = next(
        item
        for item in cast(list[PrimitiveMapping], summary["parameter_summaries"])
        if item["successful_count"] == 0
    )
    record[field] = 1 if field == "eligible_count" else "0.1"
    with pytest.raises(ManifestError, match="optimization parameter summary"):
        inspect(captured, summary)


def test_native_parameter_summaries_preserve_unavailable_statistics(
    captured: Captured,
) -> None:
    summary = captured[1]
    records = cast(list[PrimitiveMapping], summary["parameter_summaries"])
    assert any(
        item["eligible_count"] == 0 and item["mean_objective"] is None
        for item in records
    )
    assert any(cast(int, item["eligible_count"]) > 0 for item in records)
    inspect(captured, summary)


@pytest.mark.parametrize("value", [True, False, -3, 0, "1.25", "close", ""])
def test_parameter_values_preserve_native_primitive_types(
    value: str | int | bool,
) -> None:
    record = ParameterSummary(
        "parameter",
        value,
        2,
        1,
        Decimal("0.5"),
        Decimal("-0.3"),
        Decimal("-0.3"),
        Decimal("-0.3"),
    ).to_primitive()
    before = deepcopy(record)
    assert validate_parameter_summaries([record]) == [before]
    assert type(record["parameter_value"]) is type(value)
    assert record == before
