"""End-to-end QF-39 acceptance tests with unchanged production evaluators."""

from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.validation import TrainingWindowMode
from quantforge.walk_forward import (
    BacktestOOSArtifact,
    FoldStatus,
    PredictionOOSArtifact,
    WalkForwardStudy,
)
from tests.unit.helpers import SESSIONS

from .fixtures import backtest_fixture, prediction_fixture


@pytest.mark.parametrize("mode", list(TrainingWindowMode))
def test_backtest_two_windows_and_deterministic_resume(
    tmp_path: Path, mode: TrainingWindowMode
) -> None:
    config, adapter = backtest_fixture(tmp_path, mode=mode)
    study = WalkForwardStudy(config, adapter, tmp_path / "study")
    result = study.run()
    assert [fold.status for fold in result.folds] == [FoldStatus.COMPLETED] * 2, result
    for index, fold in enumerate(result.folds):
        artifact = fold.artifact
        assert isinstance(artifact, BacktestOOSArtifact)
        rows = cast(
            list[PrimitiveMapping], artifact.snapshot.to_primitive()["daily_equity"]
        )
        sessions = SESSIONS[7:10] if index == 0 else SESSIONS[10:13]
        assert [row["session"] for row in rows] == [
            session.isoformat() for session in sessions
        ]
        assert rows[0]["cash"] == "100000"
        assert rows[0]["shares"] == 0
        assert fold.selection is not None
    assert study.resume() == result
    rerun = WalkForwardStudy(config, adapter, tmp_path / "repeat").run()
    assert rerun == result


@pytest.mark.parametrize("mode", list(TrainingWindowMode))
def test_prediction_uses_complete_historical_windows(
    tmp_path: Path, mode: TrainingWindowMode
) -> None:
    config, adapter = prediction_fixture(tmp_path, mode=mode)
    study = WalkForwardStudy(config, adapter, tmp_path / "study")
    result = study.run()
    assert [fold.status for fold in result.folds] == [FoldStatus.COMPLETED] * 2, result
    for fold in result.folds:
        artifact = fold.artifact
        assert isinstance(artifact, PredictionOOSArtifact)
        snapshot = artifact.snapshot.to_primitive()
        decisions = cast(list[PrimitiveMapping], snapshot["decisions"])
        assert len(decisions) == 4  # Two purged-safe sessions, two decisions each.
        ids = [
            cast(PrimitiveMapping, decision["prediction_study"])["manifest"]
            for decision in decisions
        ]
        assert len(ids) == 4
        assert fold.selection is not None
        frozen = fold.selection.snapshot.to_primitive()
        assert frozen["plan_id"] == config.plan.plan_id
    assert study.resume() == result
    assert WalkForwardStudy(config, adapter, tmp_path / "repeat").run() == result
