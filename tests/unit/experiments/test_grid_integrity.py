import json
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.backtesting import BacktestConfig, BacktestResult, run_backtest
from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import MarketDataset
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.optimization import GridSearchStudy, MovingAverageCrossoverFactory
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


def test_intact_grid_exports_remain_indexable(
    grid_export: tuple[StudyType, Path], tmp_path: Path
) -> None:
    study_type, path = grid_export
    bundle = inspect_study(study_type, path, artifact_root=tmp_path)
    assert verify_artifacts(bundle.index, tmp_path).valid


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
