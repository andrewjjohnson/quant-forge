"""Backtest aggregate totals must agree with captured native count evidence."""

import shutil
from copy import deepcopy
from dataclasses import replace

import pytest

from quantforge.experiments import ManifestError
from quantforge.experiments._backtest_summary_integrity import (
    validate_backtest_aggregate_summary,
)
from quantforge.experiments._json import mapping
from quantforge.oos import aggregate_backtest, load_oos_source
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_aggregate_schemas import Captured, inspect_record
from tests.unit.oos.conftest import complete_study


@pytest.fixture(scope="module", params=["complete", "partial", "empty"])
def captured(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> Captured:
    root = tmp_path_factory.mktemp("backtest-counts")
    completed = complete_study(root, prediction=False)
    if request.param != "complete":
        folds = completed.study.study_path / "folds"
        shutil.rmtree(folds / completed.source.folds[1].fold_id)
        if request.param == "empty":
            path = folds / completed.source.folds[0].fold_id / "state.json"
            state = read_record(path)
            state["status"] = "failed"
            state["artifact_id"] = None
            state["failures"] = [{"stage": "test", "error_type": "FixtureFailure"}]
            write_record(path, state)
        completed = replace(
            completed,
            source=load_oos_source(completed.source.plan, completed.study.study_path),
        )
    return completed, aggregate_backtest(completed.source).to_primitive(), root


@pytest.mark.parametrize(
    "field",
    [
        "trade_count",
        "winning_trades",
        "losing_trades",
        "open_trade_count",
        "oos_session_count",
        "coordinated",
    ],
)
def test_rehashed_summary_cannot_replace_captured_counts(
    captured: Captured, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    block_research(monkeypatch)
    inspect_record(captured, captured[1])
    document = deepcopy(captured[1])
    summary = mapping(document["summary"])
    for key in (
        ("trade_count", "winning_trades") if field == "coordinated" else (field,)
    ):
        count = summary[key]
        assert type(count) is int
        summary[key] = count + 7
    # Keep the forged declaration internally well typed and available; it must
    # fail reconciliation with saved evidence, not the summary schema check.
    if summary["trade_count"] != 0 and summary["win_rate"] is None:
        summary["win_rate"] = "0"
    if summary["oos_session_count"] != 0 and summary["exposure"] is None:
        summary["exposure"] = "0"
    validate_backtest_aggregate_summary(summary)
    with pytest.raises(ManifestError, match=r"backtest counts.*captured"):
        inspect_record(captured, document)
