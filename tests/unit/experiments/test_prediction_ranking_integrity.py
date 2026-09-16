from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import PredictionStudyResult, PredictionTrialAnalysis
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    FixtureTrialAnalyzer,
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.prediction.test_prediction_grid import (
    _grid,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def prediction_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)


@pytest.mark.parametrize("section", ["rankings", "stability"])
@pytest.mark.parametrize(
    "change",
    [
        "remove",
        "duplicate",
        "foreign",
        "failed",
        "excluded",
        "combination_id",
        "rank",
        "objective_value",
    ],
)
def test_prediction_summary_selections_must_match_trials(
    tmp_path: Path, prediction_root: Path, section: str, change: str
) -> None:
    summary = read_record(prediction_root / "summary.json")
    rows = cast(list[PrimitiveMapping], summary[section])
    if change == "remove":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "foreign":
        rows[0]["trial_id"] = "0" * 64
    elif change in {"failed", "excluded"}:
        trial = read_record(trial_path(prediction_root, change))
        rows[0]["trial_id"], rows[0]["combination_id"] = (
            trial["trial_id"],
            trial["combination_id"],
        )
    elif change == "combination_id":
        rows[0][change] = "0" * 64
    elif change == "rank":
        rows[0]["rank" if section == "rankings" else "objective_rank"] = 99
    else:
        rows[0][change] = "999"
    write_json(prediction_root / "summary.json", summary)
    with pytest.raises(ManifestError, match=r"(prediction|optimization)"):
        inspect_study(
            StudyType.PARAMETER_STUDY, prediction_root, artifact_root=tmp_path
        )


@pytest.mark.parametrize("count", [-1, 999, True, 2.0, "2", None])
def test_prediction_summary_eligible_count_is_exact(
    tmp_path: Path, prediction_root: Path, count: Primitive
) -> None:
    summary = read_record(prediction_root / "summary.json")
    cast(PrimitiveMapping, summary["counts"])["eligible"] = count
    write_json(prediction_root / "summary.json", summary)
    with pytest.raises(ManifestError, match=r"prediction summary.*counts"):
        inspect_study(
            StudyType.PARAMETER_STUDY, prediction_root, artifact_root=tmp_path
        )


@pytest.mark.parametrize(
    "change", ["objective_metric", "schema_version", "reordered_ties"]
)
def test_prediction_summary_matches_configured_objective_schema_and_order(
    tmp_path: Path, prediction_root: Path, change: str
) -> None:
    summary = read_record(prediction_root / "summary.json")
    rows = cast(list[PrimitiveMapping], summary["rankings"])
    if change == "schema_version":
        summary[change] = "999"
    elif change == "objective_metric":
        rows[0][change] = "quality"
    else:
        assert len(rows) >= 2
        rows.reverse()
        for rank, record in enumerate(rows, 1):
            record["rank"] = rank
        stable = cast(list[PrimitiveMapping], summary["stability"])
        stable.reverse()
        for rank, record in enumerate(stable, 1):
            record["objective_rank"] = rank
    write_json(prediction_root / "summary.json", summary)
    with pytest.raises(ManifestError, match="prediction"):
        inspect_study(
            StudyType.PARAMETER_STUDY, prediction_root, artifact_root=tmp_path
        )


class IneligibleAnalyzer(FixtureTrialAnalyzer):
    def __init__(self, *, all_ineligible: bool = False) -> None:
        super().__init__()
        self.all_ineligible = all_ineligible

    def analyze(
        self, result: PredictionStudyResult[Any, Any, Any]
    ) -> PredictionTrialAnalysis:
        analysis = super().analyze(result)
        return replace(
            analysis,
            prediction_count=0 if self.all_ineligible or self.calls == 3 else 12,
        )


@pytest.mark.parametrize(
    "change",
    [None, "remove", "duplicate", "foreign", "combination_id", "reasons", "overlap"],
)
def test_prediction_ineligible_references_cover_remaining_successful_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str | None
) -> None:
    study = _grid(tmp_path / "prediction", analyzer=IneligibleAnalyzer())
    result = study.run()
    root = tmp_path / "prediction" / result.study_id
    summary = read_record(root / "summary.json")
    rows = cast(list[PrimitiveMapping], summary["ineligible_trials"])
    assert len(rows) == 1
    if change == "remove":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "foreign":
        rows[0]["trial_id"] = "0" * 64
    elif change == "combination_id":
        rows[0][change] = "0" * 64
    elif change == "reasons":
        rows[0][change] = []
    elif change == "overlap":
        ranked = cast(list[PrimitiveMapping], summary["rankings"])[0]
        rows[0]["trial_id"], rows[0]["combination_id"] = (
            ranked["trial_id"],
            ranked["combination_id"],
        )
    write_json(root / "summary.json", summary)
    block_research(monkeypatch)
    if change is None:
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match=r"(prediction|optimization)"):
            inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)


def test_prediction_all_ineligible_preserves_empty_rankings_and_stability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _grid(
        tmp_path / "prediction", analyzer=IneligibleAnalyzer(all_ineligible=True)
    )
    result = study.run()
    root = tmp_path / "prediction" / result.study_id
    assert result.rankings == ()
    assert result.stability == ()
    assert len(result.ineligible_trials) == 2
    block_research(monkeypatch)
    inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
