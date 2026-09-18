import shutil
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, inspect_validation
from quantforge.oos import (
    aggregate_backtest,
    aggregate_prediction,
    export_oos_aggregate,
    load_oos_source,
)
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.study_fixtures import CapturedStudy, copy_study
from tests.unit.experiments.study_fixtures import study_baselines as study_baselines
from tests.unit.experiments.test_adapters import block_research

type AggregateFixture = tuple[CapturedStudy, PrimitiveMapping]


@pytest.fixture(scope="module")
def aggregate_baselines(
    study_baselines: dict[bool, CapturedStudy],
) -> dict[bool, AggregateFixture]:
    baselines: dict[bool, AggregateFixture] = {}
    for prediction, completed in study_baselines.items():
        aggregate = (aggregate_prediction if prediction else aggregate_backtest)(
            completed.source
        )
        root = completed.study_path.parent.parent
        path = export_oos_aggregate(aggregate, root / "oos")
        with pytest.MonkeyPatch.context() as monkeypatch:
            block_research(monkeypatch)
            inspect_validation(
                completed.source,
                completed.study_path,
                artifact_root=root,
                aggregate_path=path,
            )
        baselines[prediction] = completed, aggregate.to_primitive()
    return baselines


@pytest.mark.parametrize("prediction", [False, True], ids=["backtest", "prediction"])
@pytest.mark.parametrize("empty", [False, True], ids=["partial", "empty"])
def test_aggregate_preserves_failed_and_missing_windows_without_fabricating_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_baselines: dict[bool, CapturedStudy],
    prediction: bool,
    empty: bool,
) -> None:
    completed = copy_study(study_baselines[prediction], tmp_path)
    root = completed.study_path / "folds"
    shutil.rmtree(root / completed.source.folds[1].fold_id)
    if empty:
        state_path = root / completed.source.folds[0].fold_id / "state.json"
        state = read_record(state_path)
        state["status"] = "failed"
        state["artifact_id"] = None
        state["failures"] = [{"stage": "test", "error_type": "FixtureFailure"}]
        write_record(state_path, state)
    source = load_oos_source(completed.source.plan, completed.study_path)
    aggregate = (aggregate_prediction if prediction else aggregate_backtest)(source)
    path = export_oos_aggregate(aggregate, tmp_path / "oos")
    block_research(monkeypatch)
    inspect_validation(
        source, completed.study_path, artifact_root=tmp_path, aggregate_path=path
    )
    assert sum(fold.artifact is not None for fold in source.folds) == (
        0 if empty else 1
    )


def captured_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    baselines: dict[bool, AggregateFixture],
    *,
    prediction: bool,
) -> AggregateFixture:
    block_research(monkeypatch)
    completed, aggregate = baselines[prediction]
    return copy_study(completed, tmp_path), deepcopy(aggregate)


def reject_rehashed_aggregate(
    fixture: AggregateFixture, tmp_path: Path, *, message: str = "OOS aggregate"
) -> None:
    completed, aggregate = fixture
    path = tmp_path / "oos" / f"{configuration_identity(aggregate)}.json"
    write_record(path, aggregate)
    with pytest.raises(ManifestError, match=message):
        inspect_validation(
            completed.source,
            completed.study_path,
            artifact_root=tmp_path,
            aggregate_path=path,
        )


@pytest.fixture
def backtest_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregate_baselines: dict[bool, AggregateFixture],
) -> AggregateFixture:
    return captured_aggregate(
        tmp_path, monkeypatch, aggregate_baselines, prediction=False
    )


@pytest.fixture
def prediction_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregate_baselines: dict[bool, AggregateFixture],
) -> AggregateFixture:
    return captured_aggregate(
        tmp_path, monkeypatch, aggregate_baselines, prediction=True
    )


@pytest.mark.parametrize("prediction", [False, True], ids=["backtest", "prediction"])
def test_aggregate_copies_keep_nested_mutations_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregate_baselines: dict[bool, AggregateFixture],
    prediction: bool,
) -> None:
    original = deepcopy(aggregate_baselines[prediction][1])
    first = captured_aggregate(
        tmp_path / "first", monkeypatch, aggregate_baselines, prediction=prediction
    )
    second = captured_aggregate(
        tmp_path / "second", monkeypatch, aggregate_baselines, prediction=prediction
    )
    windows = cast(
        list[PrimitiveMapping], cast(PrimitiveMapping, first[1]["summary"])["windows"]
    )
    windows.clear()
    assert aggregate_baselines[prediction][1] == original
    assert second[1] == original
    assert first[0].study_path != second[0].study_path


@pytest.mark.parametrize(
    "change",
    [
        "fold_id",
        "selection_id",
        "export_location",
        "export_fingerprint",
        "foreign_result",
        "foreign_window",
        "remove",
        "duplicate",
        "reverse",
    ],
)
def test_native_windows_remain_bound_to_captured_results(
    tmp_path: Path,
    backtest_aggregate: AggregateFixture,
    change: str,
) -> None:
    native = cast(list[PrimitiveMapping], backtest_aggregate[1]["native_windows"])
    assert len(native) == 2
    if change == "foreign_result":
        native[0]["result"] = deepcopy(native[1]["result"])
    elif change == "foreign_window":
        fold_id = native[0]["fold_id"]
        native[0] = deepcopy(native[1])
        native[0]["fold_id"] = fold_id
    elif change == "remove":
        native.pop()
    elif change == "duplicate":
        native.append(deepcopy(native[0]))
    elif change == "reverse":
        native.reverse()
    else:
        native[0][change] = "0" * 64
    reject_rehashed_aggregate(backtest_aggregate, tmp_path, message="native windows")


@pytest.mark.parametrize(
    "change",
    ["fold_id", "run_id", "session", "window_start", "remove", "duplicate", "reverse"],
)
def test_normalized_equity_preserves_native_window_and_session_membership(
    tmp_path: Path,
    backtest_aggregate: AggregateFixture,
    change: str,
) -> None:
    rows = cast(list[PrimitiveMapping], backtest_aggregate[1]["normalized_equity"])
    if change == "remove":
        rows.pop()
    elif change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change == "reverse":
        rows.reverse()
    else:
        rows[0][change] = False if change == "window_start" else "foreign"
    reject_rehashed_aggregate(backtest_aggregate, tmp_path, message="equity references")


@pytest.mark.parametrize(
    "change",
    [
        "fold_id",
        "selection_id",
        "window_result_id",
        "context_id",
        "prediction_study_id",
        "foreign_payload",
        "row",
        "eligible",
        "remove",
        "duplicate",
        "reverse",
    ],
)
def test_prediction_observations_remain_bound_to_captured_signals_and_rows(
    tmp_path: Path,
    prediction_aggregate: AggregateFixture,
    change: str,
) -> None:
    rows = cast(list[PrimitiveMapping], prediction_aggregate[1]["observations"])
    assert len(rows) > 1
    if change == "foreign_payload":
        assert rows[0]["signal"] != rows[-1]["signal"]
        for key in ("signal", "row"):
            rows[0][key] = deepcopy(rows[-1][key])
    elif change == "remove":
        rows.pop()
    elif change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change == "reverse":
        rows.reverse()
    elif change == "eligible":
        rows[0]["eligible"] = not rows[0]["eligible"]
    elif change == "row":
        rows[0]["row"] = {"foreign": True}
    else:
        rows[0][change] = "0" * 64
    reject_rehashed_aggregate(
        prediction_aggregate, tmp_path, message="prediction observations"
    )


@pytest.mark.parametrize("prediction", [False, True], ids=["backtest", "prediction"])
@pytest.mark.parametrize(
    "change", ["kind", "fold_id", "status", "remove", "missing_payload"]
)
def test_aggregate_family_and_window_summaries_retain_captured_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregate_baselines: dict[bool, AggregateFixture],
    prediction: bool,
    change: str,
) -> None:
    fixture = captured_aggregate(
        tmp_path, monkeypatch, aggregate_baselines, prediction=prediction
    )
    aggregate = fixture[1]
    if change == "kind":
        aggregate["kind"] = (
            "backtest_oos_aggregate" if prediction else "prediction_oos_aggregate"
        )
    else:
        windows = cast(
            list[PrimitiveMapping],
            cast(PrimitiveMapping, aggregate["summary"])["windows"],
        )
        if change == "remove":
            windows.pop()
        elif change == "missing_payload":
            windows[0]["summary" if prediction else "performance"] = None
        else:
            windows[0][change] = "foreign"
    reject_rehashed_aggregate(fixture, tmp_path)


def test_backtest_window_performance_is_the_captured_native_performance(
    tmp_path: Path,
    backtest_aggregate: AggregateFixture,
) -> None:
    windows = cast(
        list[PrimitiveMapping],
        cast(PrimitiveMapping, backtest_aggregate[1]["summary"])["windows"],
    )
    cast(PrimitiveMapping, windows[0]["performance"])["total_return"] = "999"
    reject_rehashed_aggregate(backtest_aggregate, tmp_path, message="window summaries")
