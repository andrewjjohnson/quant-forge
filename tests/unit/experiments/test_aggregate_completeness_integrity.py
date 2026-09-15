"""Completeness labels must describe the captured folds, even after rehashing."""

import shutil
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, inspect_validation, verify_artifacts
from quantforge.experiments._json import mapping
from quantforge.oos import (
    OOSSource,
    aggregate_backtest,
    aggregate_prediction,
    load_oos_source,
)
from quantforge.walk_forward.models import FoldStatus
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_adapters import block_research
from tests.unit.oos.conftest import complete_study

type CapturedAggregate = tuple[OOSSource, Path, Path, PrimitiveMapping]


@pytest.fixture(scope="module", params=[False, True], ids=["backtest", "prediction"])
def prediction(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


@pytest.fixture(
    scope="module", params=["missing", *(status.value for status in FoldStatus)]
)
def captured(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
    prediction: bool,
) -> CapturedAggregate:
    root = tmp_path_factory.mktemp("aggregate-completeness")
    completed = complete_study(root, prediction=prediction)
    fold_root = (
        completed.study.study_path / "folds" / completed.source.folds[-1].fold_id
    )
    status = cast(str, request.param)
    if status == "missing":
        shutil.rmtree(fold_root)
    elif status != "completed":
        state_path = fold_root / "state.json"
        state = read_record(state_path)
        state["status"] = status
        state["artifact_id"] = None
        if status == "failed":
            state["failures"] = [{"stage": "test", "error_type": "FixtureFailure"}]
        write_record(state_path, state)
    source = load_oos_source(completed.source.plan, completed.study.study_path)
    aggregate = (aggregate_prediction if prediction else aggregate_backtest)(source)
    return source, completed.study.study_path, root, aggregate.to_primitive()


def inspect(captured: CapturedAggregate, aggregate: PrimitiveMapping) -> None:
    source, study_path, root, _ = captured
    path = root / f"{configuration_identity(aggregate)}.json"
    write_record(path, aggregate)
    before = path.read_bytes()
    try:
        bundle = inspect_validation(
            source, study_path, artifact_root=root, aggregate_path=path
        )
        assert verify_artifacts(bundle.index, root).valid
    finally:
        assert path.read_bytes() == before


@pytest.mark.parametrize(
    "field",
    [
        "expected_windows",
        "completed_windows",
        "complete",
        "missing_windows",
        "failed_windows",
        "incomplete_windows",
        "interpretation",
        "promote_complete",
    ],
)
def test_rehashed_completeness_must_match_captured_folds(
    captured: CapturedAggregate, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    aggregate = deepcopy(captured[3])
    completeness = mapping(mapping(aggregate["summary"])["completeness"])
    if field == "promote_complete":
        # Preserve every validated window; falsify only the reporting claims.
        # For an already-complete source, fabricate partial status instead.
        if completeness["complete"]:
            completeness["complete"] = False
            completeness["completed_windows"] = 1
            completeness["incomplete_windows"] = [captured[0].folds[-1].fold_id]
            completeness["interpretation"] = (
                "partial_observed_windows_only; failures_are_not_zero_returns"
            )
        else:
            completeness.update(
                {
                    "completed_windows": completeness["expected_windows"],
                    "complete": True,
                    "missing_windows": [],
                    "failed_windows": [],
                    "incomplete_windows": [],
                    "interpretation": "all_planned_windows",
                }
            )
    elif field in {"expected_windows", "completed_windows"}:
        completeness[field] = cast(int, completeness[field]) + 1
    elif field == "complete":
        completeness[field] = not completeness[field]
    elif field == "interpretation":
        completeness[field] = (
            "partial_observed_windows_only; failures_are_not_zero_returns"
            if completeness["complete"]
            else "all_planned_windows"
        )
    else:
        completeness[field] = [] if completeness[field] else ["foreign-fold"]
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=r"completeness.*captured folds"):
        inspect(captured, aggregate)


def test_native_completeness_preserves_each_fold_state(
    captured: CapturedAggregate, monkeypatch: pytest.MonkeyPatch
) -> None:
    block_research(monkeypatch)
    inspect(captured, captured[3])
