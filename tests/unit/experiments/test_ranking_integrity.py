from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.optimization import (
    GridSearchStudy,
    IntegerValues,
    MetricName,
    MovingAverageCrossoverFactory,
    ParameterSearchSpace,
    RankingConfig,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.optimization.test_study import (
    _dataset,  # pyright: ignore[reportPrivateUsage]
    _study_config,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def optimization_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)


@pytest.mark.parametrize("filename", ["ranking.json", "stability.json"])
def test_derived_configuration_cannot_change_with_current_study_id(
    tmp_path: Path, optimization_root: Path, filename: str
) -> None:
    document = read_record(optimization_root / filename)
    cast(PrimitiveMapping, document["configuration"])["changed"] = True
    write_json(optimization_root / filename, document)
    with pytest.raises(ManifestError, match="summary configuration"):
        inspect_study(StudyType.OPTIMIZATION, optimization_root, artifact_root=tmp_path)


@pytest.mark.parametrize("filename", ["ranking.json", "stability.json"])
@pytest.mark.parametrize(
    "change",
    [
        "remove",
        "duplicate",
        "failed_trial",
        "combination_id",
        "rank",
        "objective_value",
    ],
)
def test_derived_trial_records_must_match_existing_trials_and_summary(
    tmp_path: Path, optimization_root: Path, filename: str, change: str
) -> None:
    document = read_record(optimization_root / filename)
    rows = cast(
        list[PrimitiveMapping],
        document["eligible_rankings" if filename == "ranking.json" else "summaries"],
    )
    if change == "remove":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "failed_trial":
        failed = read_record(trial_path(optimization_root, "failed"))
        rows[0]["trial_id"], rows[0]["combination_id"] = (
            failed["trial_id"],
            failed["combination_id"],
        )
    elif change == "combination_id":
        rows[0][change] = "0" * 64
    elif change == "rank":
        rows[0]["rank" if filename == "ranking.json" else "stability_rank"] = 99
    else:
        rows[0][change] = "999"
    write_json(optimization_root / filename, document)
    with pytest.raises(ManifestError, match="optimization"):
        inspect_study(StudyType.OPTIMIZATION, optimization_root, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "top_objective_trials",
        "top_stability_trials",
        "best_objective_trial_id",
        "best_stability_trial_id",
        "recommended_robust_trial_id",
    ],
)
def test_summary_selection_references_must_match_derived_records(
    tmp_path: Path, optimization_root: Path, field: str
) -> None:
    document = read_record(optimization_root / "summary.json")
    document[field] = [] if field.startswith("top_") else "0" * 64
    write_json(optimization_root / "summary.json", document)
    with pytest.raises(ManifestError, match="optimization"):
        inspect_study(StudyType.OPTIMIZATION, optimization_root, artifact_root=tmp_path)


def test_all_ineligible_trials_preserve_empty_rankings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        _study_config(
            tmp_path / "optimization",
            ranking=RankingConfig(MetricName.TOTAL_RETURN, minimum_successful_trials=4),
        ),
    )
    study.export(study.run())
    block_research(monkeypatch)
    ranking = read_record(study.study_path / "ranking.json")
    assert ranking["eligible_rankings"] == []
    assert len(cast(list[PrimitiveMapping], ranking["ineligible_trials"])) == 3
    inspect_study(StudyType.OPTIMIZATION, study.study_path, artifact_root=tmp_path)


def test_trial_references_beyond_top_ten_are_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        replace(
            _study_config(tmp_path / "optimization"),
            search_space=ParameterSearchSpace(
                {
                    "fast_window": IntegerValues(range(1, 13)),
                    "slow_window": IntegerValues([14]),
                }
            ),
        ),
    )
    study.export(study.run())
    block_research(monkeypatch)
    inspect_study(StudyType.OPTIMIZATION, study.study_path, artifact_root=tmp_path)
    ranking = read_record(study.study_path / "ranking.json")
    rows = cast(list[PrimitiveMapping], ranking["eligible_rankings"])
    assert len(rows) == 12
    summary = read_record(study.study_path / "summary.json")
    assert summary["top_objective_trials"] == rows[:10]
    rows[-1]["trial_id"] = "0" * 64
    write_json(study.study_path / "ranking.json", ranking)
    with pytest.raises(ManifestError, match=r"optimization summary.*trial references"):
        inspect_study(StudyType.OPTIMIZATION, study.study_path, artifact_root=tmp_path)
