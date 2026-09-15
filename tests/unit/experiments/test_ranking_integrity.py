from dataclasses import replace
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


@pytest.mark.parametrize(
    "schema", ["current", "2", "missing", None, "", 1, True, {}, []]
)
def test_optimization_summary_schema_matches_recorded_study(
    tmp_path: Path, optimization_root: Path, schema: Primitive
) -> None:
    manifest = read_record(optimization_root / "manifest.json")
    recorded_schema = cast(PrimitiveMapping, manifest["identity_inputs"])[
        "study_schema_version"
    ]
    path = optimization_root / "summary.json"
    summary = read_record(path)
    assert summary["study_schema_version"] == recorded_schema
    if schema == "missing":
        del summary["study_schema_version"]
    else:
        summary["study_schema_version"] = (
            recorded_schema if schema == "current" else schema
        )
    write_json(path, summary)
    before = path.read_bytes()
    if schema == "current":
        inspected = inspect_study(
            StudyType.OPTIMIZATION, optimization_root, artifact_root=tmp_path
        )
        entry = next(
            item
            for item in inspected.index.entries
            if Path(item.path).name == "summary.json"
        )
        assert entry.schema_version == recorded_schema
        assert verify_artifacts(inspected.index, tmp_path).valid
    else:
        with pytest.raises(
            ManifestError, match="optimization summary schema differs from study"
        ):
            inspect_study(
                StudyType.OPTIMIZATION, optimization_root, artifact_root=tmp_path
            )
    assert path.read_bytes() == before


@pytest.mark.parametrize("status", ["pending", "running"])
def test_unfinished_optimization_omits_stale_summary_schema(
    tmp_path: Path, optimization_root: Path, status: str
) -> None:
    path = trial_path(optimization_root, "failed")
    trial = read_record(path)
    trial.update(
        status=status,
        failure_category=None,
        failure_type=None,
        failure_message=None,
        finished_at=None,
    )
    write_json(path, trial)
    summary_path = optimization_root / "summary.json"
    summary = read_record(summary_path)
    summary["study_schema_version"] = "unsupported stale schema"
    write_json(summary_path, summary)
    before = {item: item.read_bytes() for item in (path, summary_path)}
    inspected = inspect_study(
        StudyType.OPTIMIZATION, optimization_root, artifact_root=tmp_path
    )
    assert "trial_counts" not in inspected.provenance.observations.to_primitive()
    assert not any(
        Path(item.path).name == "summary.json" for item in inspected.index.entries
    )
    assert verify_artifacts(inspected.index, tmp_path).valid
    assert {item: item.read_bytes() for item in before} == before


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


@pytest.mark.parametrize("invalid_field", [None, "minimum", "maximum", "count"])
def test_all_ineligible_trials_preserve_empty_rankings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_field: str | None
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
    summary = read_record(study.study_path / "summary.json")
    assert summary["objective_distribution"] == {
        "count": 0,
        "minimum": None,
        "maximum": None,
    }
    if invalid_field is not None:
        cast(PrimitiveMapping, summary["objective_distribution"])[invalid_field] = 1
        write_json(study.study_path / "summary.json", summary)
        with pytest.raises(ManifestError, match="objective distribution"):
            inspect_study(
                StudyType.OPTIMIZATION, study.study_path, artifact_root=tmp_path
            )


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


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("count", 999),
        ("count", True),
        ("count", "3"),
        ("count", 3.0),
        ("minimum", "-999"),
        ("maximum", "999"),
        ("minimum", None),
        ("maximum", None),
        ("minimum", "NaN"),
        ("maximum", True),
    ],
)
def test_objective_distribution_matches_saved_eligible_values(
    tmp_path: Path, optimization_root: Path, field: str, invalid: Primitive
) -> None:
    document = read_record(optimization_root / "summary.json")
    cast(PrimitiveMapping, document["objective_distribution"])[field] = invalid
    write_json(optimization_root / "summary.json", document)
    with pytest.raises(ManifestError):
        inspect_study(StudyType.OPTIMIZATION, optimization_root, artifact_root=tmp_path)
