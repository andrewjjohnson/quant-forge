"""Indexed QF-11 context through canonical QF-52 inputs and QF-56 resume."""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from quantforge.prediction import (
    PredictionStudy,
    intraday_excursion_outcome,
    intraday_forward_return_outcome,
    intraday_target_stop_outcome,
)
from quantforge.prediction.study import prepare_prediction_study_dataset
from quantforge.prediction.window import run_prediction_window_in_session
from quantforge.prediction.window_compact import CompactPredictionWindowResult
from quantforge.prediction.window_execution import (
    run_incremental_prediction_window_in_session,
)
from quantforge.prediction.window_incremental import IncrementalPredictionWindowWriter
from quantforge.walk_forward.prediction import (
    _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_bounded_prediction_workflow import workflow_fixture
from tests.integration.test_intraday_prediction_provenance import cached_fixture
from tests.unit.helpers import SESSIONS


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("indexed-execution")
    fixture = cached_fixture(root / "cache", session_dates=SESSIONS[:7])
    config, adapter = workflow_fixture(root, fixture)
    permitted = adapter._partition(config, 0, test=False)  # pyright: ignore[reportPrivateUsage]
    schedule = adapter._schedule(permitted)  # pyright: ignore[reportPrivateUsage]
    return fixture, config, adapter, permitted, schedule


def arguments(inputs: Any, *, reference: bool = False) -> dict[str, Any]:
    _, config, adapter, permitted, schedule = inputs
    provider = _PermittedContextProvider(
        config.plan, permitted, adapter.series, schedule
    )
    assert provider._prepared_context is not None  # pyright: ignore[reportPrivateUsage]
    if reference:
        object.__setattr__(provider, "_prepared_context", None)
    return {
        "schedule": schedule,
        "context_provider": provider,
        "dataset_family_fingerprint": adapter.series[0].dataset_reference.family_id,
        "context_environment": adapter._environment(
            config.plan, permitted
        ).to_primitive(),  # pyright: ignore[reportPrivateUsage]
    }


def study_for(inputs: Any, outcome_name: str) -> Any:
    fixture, _, adapter, _, _ = inputs
    original = adapter.factory.build(
        adapter.universe.candidates[0].parameters.to_primitive()
    )
    if outcome_name == "forward":
        outcome = intraday_forward_return_outcome(
            timedelta(minutes=30), fixture.primary
        )
    elif outcome_name == "excursion":
        outcome = intraday_excursion_outcome(timedelta(minutes=60), fixture.primary)
    else:
        outcome = intraday_target_stop_outcome(
            timedelta(minutes=60), fixture.primary, Decimal("0.003"), Decimal("0.002")
        )
    return PredictionStudy[Any, Any, Any].create(
        original.strategy,
        outcome.labeler,
        outcome.evaluator,
        outcome_source=fixture.primary,
    )


@pytest.mark.parametrize("outcome_name", ["forward", "excursion", "target_stop"])
def test_exact_prediction_outcome_context_and_compact_identity(
    inputs: Any, outcome_name: str
) -> None:
    _, _, _, permitted, _ = inputs
    session = prepare_prediction_study_dataset(permitted.dataset)
    study = study_for(inputs, outcome_name)
    old = run_prediction_window_in_session(
        session, study, **arguments(inputs, reference=True)
    )
    new = run_prediction_window_in_session(session, study, **arguments(inputs))
    assert old.serialize() == new.serialize()
    assert any(decision.result.rows for decision in new.decisions)
    assert CompactPredictionWindowResult.from_window(old).serialize() == (
        CompactPredictionWindowResult.from_window(new).serialize()
    )


def test_reference_checkpoint_resumes_with_zero_prefix_recomputation(
    inputs: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _, _, permitted, schedule = inputs
    session = prepare_prediction_study_dataset(permitted.dataset)
    study = study_for(inputs, "forward")
    old_args = arguments(inputs, reference=True)
    baseline = run_prediction_window_in_session(session, study, **old_args)
    path = tmp_path / "window.jsonl"
    append = IncrementalPredictionWindowWriter.append

    def interrupt(self: IncrementalPredictionWindowWriter, decision: Any) -> None:
        append(self, decision)
        if self.completed_count == 2:
            raise KeyboardInterrupt("bounded reference prefix")

    with monkeypatch.context() as stopped:
        stopped.setattr(IncrementalPredictionWindowWriter, "append", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_incremental_prediction_window_in_session(
                session,
                study,
                path=path,
                canonical_metadata=fixture.dataset.metadata,
                **old_args,
            )
    staging = path.with_name(path.name + ".in-progress")
    journal = staging / "decisions.jsonl"
    prefix = journal.read_bytes()
    assert (
        json.loads((staging / "checkpoint.json").read_bytes())["checkpoint"][
            "completed_count"
        ]
        == 2
    )
    # Rebuild indexes as a new process would; no operational state is persisted.
    new_args = arguments(inputs)
    provider = new_args["context_provider"]
    executed: list[datetime] = []

    class Recorder:
        def get_context_at(self, requirements: Any, *, as_of: datetime) -> Any:
            executed.append(as_of)
            return provider.get_context_at(requirements, as_of=as_of)

    new_args["context_provider"] = Recorder()

    def verify(self: IncrementalPredictionWindowWriter, decision: Any) -> None:
        assert self.completed_count == 2
        assert journal.read_bytes() == prefix
        append(self, decision)
        assert journal.read_bytes().startswith(prefix)

    with monkeypatch.context() as resumed:
        resumed.setattr(IncrementalPredictionWindowWriter, "append", verify)
        run_incremental_prediction_window_in_session(
            session,
            study,
            path=path,
            canonical_metadata=fixture.dataset.metadata,
            **new_args,
        )
    assert executed == [schedule.decision_timestamps[2]]
    assert (
        path.read_bytes()
        == CompactPredictionWindowResult.from_window(baseline).serialize()
    )
    run_incremental_prediction_window_in_session(
        session,
        study,
        path=path,
        canonical_metadata=fixture.dataset.metadata,
        **new_args,
    )
    assert executed == [schedule.decision_timestamps[2]]


def test_prepared_provider_rejects_nonmember_without_changing_schedule(
    inputs: Any,
) -> None:
    _, _, adapter, _, schedule = inputs
    provider = arguments(inputs)["context_provider"]
    study = adapter.factory.build(
        adapter.universe.candidates[0].parameters.to_primitive()
    )
    with pytest.raises(ValueError, match="outside permitted membership"):
        provider.get_context_at(
            study.strategy.context_requirements,
            as_of=schedule.decision_timestamps[0] + timedelta(seconds=1),
        )


def test_two_fold_grid_ranking_and_scope_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantforge.validation import PreparedPredictionContext
    from tests.unit.walk_forward.test_incremental_prediction import compact_adapter
    from tests.unit.walk_forward.timestamp_fixtures import timestamp_fixture

    config, original = timestamp_fixture(tmp_path)
    adapter = compact_adapter(original)
    reference: list[Any] = []

    def reference_capture(*args: Any, **kwargs: Any) -> None:
        return None

    with monkeypatch.context() as old:
        old.setattr(
            PreparedPredictionContext,
            "capture",
            classmethod(reference_capture),
        )
        for fold in range(2):
            reference.append(
                adapter.select(config, fold, tmp_path / "reference" / str(fold))
            )
    captured: list[PreparedPredictionContext] = []
    original_capture = PreparedPredictionContext.capture.__func__

    def record(cls: Any, *args: Any, **kwargs: Any) -> Any:
        prepared = original_capture(cls, *args, **kwargs)
        assert prepared is not None
        captured.append(prepared)
        return prepared

    monkeypatch.setattr(PreparedPredictionContext, "capture", classmethod(record))
    for fold in range(2):
        result = adapter.select(config, fold, tmp_path / "indexed" / str(fold))
        assert result == reference[fold]
    # One preparation per selection scope, shared across that grid's two trials.
    assert len(captured) == 2
    assert captured[0].window_id != captured[1].window_id
    assert captured[0].input_identity != captured[1].input_identity
