import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.backtesting import BacktestConfig, BacktestResult, run_backtest
from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data import MarketDataset
from quantforge.experiments import (
    ArtifactType,
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.optimization import (
    CategoricalValues,
    FloatValues,
    GridSearchStudy,
    IntegerValues,
    MovingAverageCrossoverFactory,
    ParameterSearchSpace,
)
from quantforge.prediction import PredictionStudyResult, PredictionTrialAnalysis
from quantforge.strategies import Strategy
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.optimization.test_study import (
    _dataset,  # pyright: ignore[reportPrivateUsage]
    _study_config,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_grid import (
    FixtureAnalyzer,
    _grid,  # pyright: ignore[reportPrivateUsage]
)


class FixtureTrialAnalyzer(FixtureAnalyzer):
    """Fixed analysis over real prediction exports, with one persisted failure."""

    def __init__(self) -> None:
        self.calls = 0

    def analyze(
        self, result: PredictionStudyResult[Any, Any, Any]
    ) -> PredictionTrialAnalysis:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("fixture analysis failure")
        return PredictionTrialAnalysis.create(
            prediction_count=12,
            metrics={"accuracy": "0.6", "quality": 2},
            period_comparisons=({"period": "development", "accuracy": "0.6"},),
            weekday_comparisons=({"weekday": 1, "accuracy": "0.6"},),
            matched_baseline_comparisons=(
                {
                    "baseline_name": "always_up",
                    "scope": "matched_prediction_sessions",
                    "accuracy_delta": "0.1",
                },
            ),
        )


def selective_runner(
    dataset: MarketDataset, strategy: Strategy, config: BacktestConfig
) -> BacktestResult:
    if strategy.parameters.to_primitive()["fast_window"] == 3:
        raise RuntimeError("fixture backtest failure")
    return run_backtest(dataset, strategy, config)


def build_grid_export(
    tmp_path: Path, study_type: StudyType, monkeypatch: pytest.MonkeyPatch
) -> Path:
    if study_type is StudyType.PARAMETER_STUDY:
        grid = _grid(tmp_path / "prediction", analyzer=FixtureTrialAnalyzer())
        result = grid.run()
        path = tmp_path / "prediction" / result.study_id
    else:
        optimization = GridSearchStudy(
            _dataset(),
            MovingAverageCrossoverFactory(),
            _study_config(tmp_path / "optimization"),
            backtest_runner=selective_runner,
        )
        optimization.export(optimization.run())
        path = optimization.study_path
    assert {
        str(read_record(p)["status"]) for p in (path / "trials").glob("*.json")
    } == {"succeeded", "failed", "excluded"}
    block_research(monkeypatch)
    return path


@pytest.fixture(params=[StudyType.PARAMETER_STUDY, StudyType.OPTIMIZATION])
def grid_export(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> tuple[StudyType, Path]:
    study_type = cast(StudyType, request.param)
    return study_type, build_grid_export(tmp_path, study_type, monkeypatch)


def read_record(path: Path) -> PrimitiveMapping:
    return cast(PrimitiveMapping, json.loads(path.read_bytes()))


def trial_path(root: Path, status: str = "succeeded") -> Path:
    return next(
        path
        for path in sorted((root / "trials").glob("*.json"))
        if read_record(path)["status"] == status
    )


@pytest.mark.parametrize("status", ["succeeded", "failed", "excluded"])
@pytest.mark.parametrize(
    "change", ["parameters", "combination_id", "combination_index", "trial_id"]
)
def test_optimization_trial_must_match_grid_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, change: str
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    record_path = trial_path(root, status)
    trial = read_record(record_path)
    if change == "parameters":
        cast(PrimitiveMapping, trial["parameters"])["fast_window"] = 99
    elif change == "combination_index":
        trial[change] = cast(int, trial[change]) + 1
    else:
        trial[change] = "0" * 64
    if change == "trial_id":
        # Keep the filename and result location checks satisfied, so identity
        # validation must detect the renamed trial itself.
        new_path = record_path.with_name(f"{trial['trial_id']}.json")
        if status == "succeeded":
            location = cast(str, trial["artifact_location"])
            old_directory = root / "backtests" / record_path.stem
            old_directory.rename(root / "backtests" / cast(str, trial["trial_id"]))
            trial["artifact_location"] = location.replace(
                record_path.stem, cast(str, trial["trial_id"])
            )
        record_path.unlink()
        record_path = new_path
    write_json(record_path, trial)
    with pytest.raises(ManifestError, match=r"trial.*(coordinates|identity)"):
        inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)


@pytest.mark.parametrize("fast_window", [99, 3])
def test_rehashed_combination_cannot_change_its_declared_grid_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast_window: int
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    record_path = trial_path(root)
    trial = read_record(record_path)
    parameters = cast(PrimitiveMapping, trial["parameters"])
    parameters["fast_window"] = fast_window
    configuration = cast(
        PrimitiveMapping, read_record(root / "manifest.json")["identity_inputs"]
    )
    trial["combination_id"] = configuration_identity(
        {
            "component": "quantforge_parameter_combination",
            "combination_schema_version": "1",
            "strategy_name": configuration["strategy_name"],
            "strategy_version": configuration["strategy_version"],
            "strategy_factory": configuration["strategy_factory"],
            "parameters": parameters,
        }
    )
    write_json(record_path, trial)
    with pytest.raises(ManifestError, match="parameters do not match grid coordinates"):
        inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)


@pytest.mark.parametrize("invalid", [-1, 4, True, 0.0, "0", None])
def test_trial_position_must_be_a_bounded_integer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: Primitive
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    record_path = trial_path(root)
    trial = read_record(record_path)
    trial["combination_index"] = invalid
    write_json(record_path, trial)
    with pytest.raises(ManifestError, match="coordinates"):
        inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)


def test_grid_position_preserves_declared_order_and_serialized_decimal_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        _study_config(tmp_path / "grid"),
        search_space=ParameterSearchSpace(
            {
                "target_long_weight": FloatValues(["0.75", "0.25"]),
                "source_field": CategoricalValues(["close"]),
                "slow_window": IntegerValues([4]),
                "fast_window": IntegerValues([2, 1]),
            }
        ),
    )
    study = GridSearchStudy(_dataset(), MovingAverageCrossoverFactory(), config)
    result = study.run()
    study.export(result)
    block_research(monkeypatch)
    bundle = inspect_study(
        StudyType.OPTIMIZATION, study.study_path, artifact_root=tmp_path
    )
    assert bundle.provenance.producer_study_id == result.study_id
    assert (
        len(
            cast(
                list[Primitive],
                bundle.provenance.observations.to_primitive()["trial_ids"],
            )
        )
        == 4
    )


def test_intact_grid_exports_remain_indexable(
    grid_export: tuple[StudyType, Path], tmp_path: Path
) -> None:
    study_type, path = grid_export
    bundle = inspect_study(study_type, path, artifact_root=tmp_path)
    assert verify_artifacts(bundle.index, tmp_path).valid


@pytest.mark.parametrize("change", ["delete", "modify"])
def test_optimization_ranking_artifact_is_integrity_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    path = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    bundle = inspect_study(StudyType.OPTIMIZATION, path, artifact_root=tmp_path)
    ranking = path / "ranking.json"
    entries = [
        entry
        for entry in bundle.index.entries
        if entry.path == ranking.relative_to(tmp_path).as_posix()
    ]
    assert len(entries) == 1
    assert entries[0].artifact_type is ArtifactType.PARAMETER_SUMMARY
    if change == "delete":
        ranking.unlink()
    else:
        content = read_record(ranking)
        content["eligible_rankings"] = []
        write_json(ranking, content)
    assert not verify_artifacts(bundle.index, tmp_path).valid


def test_completed_optimization_cannot_omit_ranking_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    (path / "ranking.json").unlink()
    with pytest.raises(ManifestError, match="ranking"):
        inspect_study(StudyType.OPTIMIZATION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("status", ["succeeded", "failed", "excluded"])
def test_missing_trial_is_rejected_against_completed_summary(
    grid_export: tuple[StudyType, Path], tmp_path: Path, status: str
) -> None:
    study_type, path = grid_export
    trial_path(path, status).unlink()
    with pytest.raises(ManifestError, match=r"trial.*summary"):
        inspect_study(study_type, path, artifact_root=tmp_path)


def test_changed_status_counts_are_rejected_with_unchanged_trial_total(
    grid_export: tuple[StudyType, Path], tmp_path: Path
) -> None:
    study_type, path = grid_export
    summary_path = path / "summary.json"
    summary = read_record(summary_path)
    counts = cast(PrimitiveMapping, summary["counts"])
    counts["failed"], counts["excluded"] = 0, 2
    write_json(summary_path, summary)
    with pytest.raises(ManifestError, match=r"trial.*summary"):
        inspect_study(study_type, path, artifact_root=tmp_path)


def test_prediction_artifact_must_match_persisted_trial_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    trial = read_record(trial_path(path))
    artifact_path = path / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    analysis = cast(PrimitiveMapping, artifact["analysis"])
    cast(PrimitiveMapping, analysis["metrics"])["accuracy"] = "0.9"
    artifact["artifact_fingerprint"] = configuration_identity(
        {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
    )
    assert artifact["artifact_fingerprint"] != trial["artifact_fingerprint"]
    write_json(artifact_path, artifact)
    with pytest.raises(ManifestError, match="fingerprint"):
        inspect_study(StudyType.PARAMETER_STUDY, path, artifact_root=tmp_path)


def test_successful_trial_cannot_omit_its_result_reference(
    grid_export: tuple[StudyType, Path], tmp_path: Path
) -> None:
    study_type, path = grid_export
    record_path = trial_path(path)
    trial = read_record(record_path)
    trial["artifact_location"] = None
    write_json(record_path, trial)
    with pytest.raises(ManifestError, match="missing its result artifact"):
        inspect_study(study_type, path, artifact_root=tmp_path)


@pytest.mark.parametrize("field", ["analysis", "schema_version"])
def test_prediction_trial_metadata_must_match_unchanged_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    path = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    record_path = trial_path(path)
    trial = read_record(record_path)
    trial[field] = {"altered": True} if field == "analysis" else "altered"
    write_json(record_path, trial)
    with pytest.raises(ManifestError, match="metadata does not match"):
        inspect_study(StudyType.PARAMETER_STUDY, path, artifact_root=tmp_path)


def test_extra_trial_is_rejected_against_completed_summary(
    grid_export: tuple[StudyType, Path], tmp_path: Path
) -> None:
    study_type, path = grid_export
    trial = read_record(trial_path(path, "excluded"))
    trial["trial_id"] = "unrecorded-trial"
    write_json(path / "trials" / "unrecorded-trial.json", trial)
    with pytest.raises(ManifestError, match=r"trial.*summary"):
        inspect_study(study_type, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("status", "field"),
    [
        ("failed", "failure_type"),
        ("failed", "failure_message"),
        ("excluded", "exclusion_code"),
        ("excluded", "exclusion_reason"),
    ],
)
@pytest.mark.parametrize("invalid", [None, "", " ", 17])
def test_trial_status_requires_diagnostic_or_exclusion_context(
    grid_export: tuple[StudyType, Path],
    tmp_path: Path,
    status: str,
    field: str,
    invalid: Primitive,
) -> None:
    study_type, path = grid_export
    record_path = trial_path(path, status)
    trial = read_record(record_path)
    trial[field] = invalid
    write_json(record_path, trial)
    with pytest.raises(ManifestError, match=r"trial.*context"):
        inspect_study(study_type, path, artifact_root=tmp_path)


def test_failed_qf6_trial_requires_failure_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    record_path = trial_path(path, "failed")
    trial = read_record(record_path)
    trial["failure_category"] = None
    write_json(record_path, trial)
    with pytest.raises(ManifestError, match=r"trial.*context"):
        inspect_study(StudyType.OPTIMIZATION, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("status", "field"),
    [
        ("succeeded", "failure_message"),
        ("failed", "exclusion_reason"),
        ("excluded", "failure_type"),
        ("failed", "artifact_location"),
    ],
)
def test_trial_status_rejects_contradictory_outcome_fields(
    grid_export: tuple[StudyType, Path], tmp_path: Path, status: str, field: str
) -> None:
    study_type, path = grid_export
    record_path = trial_path(path, status)
    trial = read_record(record_path)
    trial[field] = "contradictory-outcome"
    write_json(record_path, trial)
    with pytest.raises(ManifestError, match="trial status contradicts"):
        inspect_study(study_type, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "metrics",
        "dataset",
        "backtest_configuration",
        "strategy_name",
        "strategy_version",
        "strategy_configuration_id",
        "strategy_parameters",
    ],
)
def test_qf6_trial_provenance_must_match_linked_backtest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    path = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    record_path = trial_path(path)
    trial = read_record(record_path)
    original_run_id = trial["qf5_run_id"]
    trial[field] = {"altered": True} if isinstance(trial[field], dict) else "altered"
    write_json(record_path, trial)
    assert read_record(record_path)["qf5_run_id"] == original_run_id
    with pytest.raises(ManifestError, match="linked QF-5"):
        inspect_study(StudyType.OPTIMIZATION, path, artifact_root=tmp_path)
