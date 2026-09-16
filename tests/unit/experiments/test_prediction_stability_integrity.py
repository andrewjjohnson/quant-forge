"""QF-32 stability records retain the producer schema without recalculation."""

from dataclasses import fields
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.optimization.models import StabilityConfig
from quantforge.prediction import PredictionStudyResult, PredictionTrialAnalysis
from quantforge.prediction.grid import PredictionStabilitySummary
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import build_grid_export, read_record
from tests.unit.prediction.test_prediction_grid import (
    FixtureAnalyzer,
    _grid,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def prediction_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)


@pytest.mark.parametrize(
    "field", [field.name for field in fields(PredictionStabilitySummary)] + ["unknown"]
)
def test_stability_requires_exact_producer_fields(
    tmp_path: Path, prediction_root: Path, field: str
) -> None:
    path = prediction_root / "summary.json"
    summary = read_record(path)
    record = cast(list[PrimitiveMapping], summary["stability"])[0]
    if field == "unknown":
        record[field] = "unexpected"
    else:
        record.pop(field)
    write_json(path, summary)
    before = path.read_bytes()
    with pytest.raises(ManifestError, match="prediction stability"):
        inspect_study(
            StudyType.PARAMETER_STUDY, prediction_root, artifact_root=tmp_path
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        (field, invalid)
        for field in (
            "valid_neighbor_count",
            "excluded_neighbor_count",
            "eligible_neighbor_count",
        )
        for invalid in (-1, True, 1.5, "1", None)
    ]
    + [
        (field, invalid)
        for field in (
            "objective_value",
            "median_neighbor_objective",
            "relative_dispersion",
            "constraint_pass_fraction",
            "center_to_neighbor_difference",
            "relative_center_to_neighbor_difference",
        )
        for invalid in (True, 0, 0.5, "NaN", "Infinity", "invalid")
    ]
    + [
        (field, invalid)
        for field in ("is_boundary", "is_isolated_peak")
        for invalid in (None, 0, "false")
    ]
    + [
        ("classification", invalid)
        for invalid in (None, 1, "foreign", cast(Primitive, []))
    ]
    + [("isolation_reason", invalid) for invalid in (True, 1, "", cast(Primitive, []))]
    + [
        ("neighbor_objective_values", invalid)
        for invalid in (None, 1, cast(Primitive, {}), [True], ["NaN"], [1], ["invalid"])
    ]
    + [("objective_rank", invalid) for invalid in (0, -1, True)]
    + [
        ("constraint_pass_fraction", "-0.1"),
        ("constraint_pass_fraction", "1.1"),
        ("relative_dispersion", "-0.1"),
    ],
)
def test_stability_rejects_invalid_field_types_and_domains(
    tmp_path: Path, prediction_root: Path, field: str, invalid: Primitive
) -> None:
    path = prediction_root / "summary.json"
    summary = read_record(path)
    cast(list[PrimitiveMapping], summary["stability"])[0][field] = invalid
    write_json(path, summary)
    before = path.read_bytes()
    with pytest.raises(ManifestError, match="prediction stability"):
        inspect_study(
            StudyType.PARAMETER_STUDY, prediction_root, artifact_root=tmp_path
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["array_count", "valid_count", "isolation_reason"])
def test_stability_retains_consistent_neighbor_and_isolation_evidence(
    tmp_path: Path, prediction_root: Path, change: str
) -> None:
    path = prediction_root / "summary.json"
    summary = read_record(path)
    record = cast(list[PrimitiveMapping], summary["stability"])[0]
    if change == "array_count":
        record["neighbor_objective_values"] = ["0.6"]
    elif change == "valid_count":
        record.update(
            valid_neighbor_count=0,
            eligible_neighbor_count=1,
            neighbor_objective_values=["0.6"],
        )
    else:
        record.update(is_isolated_peak=True, isolation_reason=None)
    write_json(path, summary)
    before = path.read_bytes()
    with pytest.raises(ManifestError, match="prediction stability"):
        inspect_study(
            StudyType.PARAMETER_STUDY, prediction_root, artifact_root=tmp_path
        )
    assert path.read_bytes() == before


class StabilityAnalyzer(FixtureAnalyzer):
    def __init__(self, isolated: bool) -> None:
        self.isolated = isolated
        self.calls = 0

    def analyze(
        self, result: PredictionStudyResult[Any, Any, Any]
    ) -> PredictionTrialAnalysis:
        self.calls += 1
        accuracy = ("1" if self.calls == 2 else "0.1") if self.isolated else "0.6"
        return PredictionTrialAnalysis.create(
            prediction_count=12,
            metrics={"accuracy": accuracy, "quality": 2},
            period_comparisons=({"period": "development", "accuracy": accuracy},),
            weekday_comparisons=({"weekday": 1, "accuracy": accuracy},),
            matched_baseline_comparisons=(
                {"baseline_name": "always_up", "accuracy_delta": "0.1"},
            ),
        )


@pytest.mark.parametrize("scenario", ["empty_neighbors", "neighbors", "isolated"])
def test_native_stability_records_remain_readable_without_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    if scenario == "empty_neighbors":
        root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    else:
        study = _grid(
            tmp_path / "prediction",
            analyzer=StabilityAnalyzer(isolated=scenario == "isolated"),
            stability=StabilityConfig(
                minimum_eligible_neighbors=2,
                isolated_peak_top_fraction=Decimal("0.5"),
                isolated_peak_absolute_drop=Decimal("0.5"),
                isolated_peak_relative_drop=Decimal("0.5"),
                isolated_peak_maximum_constraint_pass_fraction=Decimal("1"),
            )
            if scenario == "isolated"
            else None,
        )
        result = study.run()
        root = tmp_path / "prediction" / result.study_id
        block_research(monkeypatch)
    path = root / "summary.json"
    before = path.read_bytes()
    records = cast(list[PrimitiveMapping], read_record(path)["stability"])
    if scenario == "empty_neighbors":
        assert all(record["neighbor_objective_values"] == [] for record in records)
        assert all(record["relative_dispersion"] is None for record in records)
    else:
        assert any(record["neighbor_objective_values"] for record in records)
        if scenario == "isolated":
            assert any(record["is_isolated_peak"] is True for record in records)
    inspected = inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert verify_artifacts(inspected.index, tmp_path).valid
    assert path.read_bytes() == before
