"""Real two-fold engine fixtures shared by aggregation and holdout tests."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from quantforge.oos import OOSSource, load_oos_source
from quantforge.walk_forward import (
    BacktestEvaluator,
    PredictionEvaluator,
    WalkForwardStudy,
)
from tests.unit.walk_forward.fixtures import backtest_fixture, prediction_fixture


@dataclass(frozen=True)
class CompletedStudy:
    study: WalkForwardStudy
    source: OOSSource
    evaluator: PredictionEvaluator | BacktestEvaluator


def complete_study(root: Path, *, prediction: bool) -> CompletedStudy:
    config, evaluator = (
        prediction_fixture(root) if prediction else backtest_fixture(root)
    )
    study = WalkForwardStudy(config, evaluator, root / "walk-forward")
    study.run()
    source = load_oos_source(config.plan, study.study_path)
    assert all(fold.artifact for fold in source.folds)
    return CompletedStudy(study, source, evaluator)


@pytest.fixture
def prediction_study(tmp_path: Path) -> CompletedStudy:
    return complete_study(tmp_path, prediction=True)


@pytest.fixture
def backtest_study(tmp_path: Path) -> CompletedStudy:
    return complete_study(tmp_path, prediction=False)


@pytest.fixture(params=[True, False], ids=["prediction", "backtest"])
def completed_study(tmp_path: Path, request: pytest.FixtureRequest) -> CompletedStudy:
    return complete_study(tmp_path, prediction=bool(request.param))
