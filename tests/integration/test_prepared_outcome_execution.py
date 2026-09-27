"""Source counts, leakage isolation, exact outcomes and pre-QF-61 resume bytes."""

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pytest

from quantforge.data import IntradayBar
from quantforge.prediction import prepared_outcomes
from quantforge.prediction import study as study_module
from quantforge.prediction.prepared_outcomes import (
    PreparedOutcomeSource,
    PreparedOutcomeSources,
)
from quantforge.prediction.study import prepare_prediction_study_dataset
from quantforge.prediction.window import run_prediction_window_in_session
from quantforge.prediction.window_compact import CompactPredictionWindowResult
from quantforge.prediction.window_execution import (
    run_incremental_prediction_window_in_session,
)
from quantforge.prediction.window_incremental import IncrementalPredictionWindowWriter
from tests.integration.test_prepared_prediction_execution import (
    arguments,
    study_for,
)
from tests.integration.test_prepared_prediction_execution import (
    inputs as inputs,
)
from tests.unit.walk_forward.fixtures import StudyRule


def reference_preparation(*args: Any, **kwargs: Any) -> None:
    return None


def forbidden_preparation(*args: Any, **kwargs: Any) -> Any:
    pytest.fail("offline preparation")


@pytest.mark.parametrize(
    "dimension",
    ["ancestry", "manifest", "symbol", "adjustment", "session", "timeframe"],
)
def test_canonical_source_incompatibility_fails_closed(
    inputs: Any, dimension: str
) -> None:
    from quantforge.data.exceptions import ValidationError
    from quantforge.prediction.outcome_temporal import OutcomeTemporalError
    from quantforge.timeframes import BarLabel

    fixture, _, _, permitted, _ = inputs
    registry = PreparedOutcomeSources()
    prepared = registry.prepare(permitted.dataset, fixture.primary)
    assert prepared is not None
    source = deepcopy(prepared.source)
    if dimension == "ancestry":
        object.__setattr__(
            source,
            "dataset_reference",
            replace(source.dataset_reference, canonical_source_snapshot_id="foreign"),
        )
    elif dimension == "manifest":
        object.__setattr__(source, "dataset_family_manifest_id", "foreign")
    elif dimension == "session":
        object.__setattr__(source.timeframe.session_policy, "timezone_name", "UTC")
    elif dimension == "timeframe":
        object.__setattr__(
            source, "timeframe", replace(source.timeframe, bar_label=BarLabel.END)
        )
    else:
        bar = cast(IntradayBar, source.bars[-1])
        changed = (
            replace(bar, symbol="QQQ")
            if dimension == "symbol"
            else replace(
                bar,
                provenance=replace(
                    bar.provenance,
                    adjustment_basis=replace(
                        bar.provenance.adjustment_basis, adjusted_fields_used=True
                    ),
                ),
            )
        )
        object.__setattr__(source, "bars", (*source.bars[:-1], changed))
    with pytest.raises((ValidationError, OutcomeTemporalError)):
        registry.prepare(permitted.dataset, source)


@pytest.mark.parametrize(
    "mode", ["empty", "mixed", "forward", "excursion", "target_stop"]
)
def test_requests_only_for_candidates_and_exact_output(
    inputs: Any,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    _, _, _, permitted, schedule = inputs
    study = study_for(
        inputs, mode if mode in ("excursion", "target_stop") else "forward"
    )
    generate = StudyRule.generate_with_context

    def guarded(self: Any, context: Any) -> Any:
        # Rule capability contains only causal bars and normalized features. The
        # future-bearing registry and source are never attached to that context.
        assert not hasattr(context, "outcome_sources")
        assert not hasattr(context, "outcome_source")
        assert not hasattr(context, "prepared_source")
        for requirement in context.requirements.all_timeframes:
            assert all(
                bar.end_timestamp <= context.as_of
                for bar in context.bars_for(requirement.timeframe)
            )
        output = generate(self, context)
        if mode == "empty" or (
            mode == "mixed" and context.as_of != schedule.decision_timestamps[1]
        ):
            return replace(output, signals=())
        return output

    monkeypatch.setattr(StudyRule, "generate_with_context", guarded)
    with monkeypatch.context() as reference:
        reference.setattr(PreparedOutcomeSources, "prepare", reference_preparation)
        old = run_prediction_window_in_session(
            prepare_prediction_study_dataset(permitted.dataset),
            study,
            **arguments(inputs),
        )
    counts = {"validation": 0, "index": 0, "request": 0, "labels": 0}
    label_seconds = 0.0
    decision_seconds = 0.0
    original_label = study_module.evaluate_outcome_request
    validate, build, request = (
        prepared_outcomes.validate_prediction_source,
        prepared_outcomes._build_indexes,  # pyright: ignore[reportPrivateUsage]
        PreparedOutcomeSource.validate_request,
    )  # pyright: ignore[reportPrivateUsage]

    def validation(*args: Any, **kwargs: Any) -> Any:
        counts["validation"] += 1
        return validate(*args, **kwargs)

    def indexes(*args: Any, **kwargs: Any) -> Any:
        counts["index"] += 1
        return build(*args, **kwargs)

    def requested(*args: Any, **kwargs: Any) -> Any:
        counts["request"] += 1
        return request(*args, **kwargs)

    def labeled(*args: Any, **kwargs: Any) -> Any:
        nonlocal label_seconds
        counts["labels"] += 1
        begin = perf_counter()
        result = original_label(*args, **kwargs)
        label_seconds += perf_counter() - begin
        return result

    monkeypatch.setattr(prepared_outcomes, "validate_prediction_source", validation)
    monkeypatch.setattr(prepared_outcomes, "_build_indexes", indexes)
    monkeypatch.setattr(PreparedOutcomeSource, "validate_request", requested)
    monkeypatch.setattr(study_module, "evaluate_outcome_request", labeled)
    session = prepare_prediction_study_dataset(permitted.dataset)
    # Repeated compatible studies share the dataset-session source preparation.
    for _ in range(10):
        kwargs = arguments(inputs)
        begin = perf_counter()
        new = run_prediction_window_in_session(session, study, **kwargs)
        decision_seconds += perf_counter() - begin
        assert new.serialize() == old.serialize()
        assert (
            CompactPredictionWindowResult.from_window(new).serialize()
            == CompactPredictionWindowResult.from_window(old).serialize()
        )
    candidates = sum(item.result.generated_prediction_count for item in old.decisions)
    assert counts == {
        "validation": 1,
        "index": 1,
        "request": candidates * 10,
        "labels": candidates * 10,
    }
    assert candidates == (0 if mode == "empty" else 1 if mode == "mixed" else 3)
    record_property(
        "seconds_per_decision_including_window_setup", decision_seconds / 30
    )
    record_property("label_dispatch_seconds", label_seconds / max(1, counts["labels"]))
    for name, count in counts.items():
        record_property(name + "_count", count)


def test_shared_configured_outcomes_prepare_once(
    inputs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, permitted, _ = inputs
    session = prepare_prediction_study_dataset(permitted.dataset)
    validate = prepared_outcomes.validate_prediction_source
    calls: list[int] = []

    def tracked(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return validate(*args, **kwargs)

    monkeypatch.setattr(prepared_outcomes, "validate_prediction_source", tracked)
    for name in ("forward", "excursion", "target_stop"):
        result = run_prediction_window_in_session(
            session, study_for(inputs, name), **arguments(inputs)
        )
        assert all(item.result.rows for item in result.decisions)
    assert calls == [1]


def test_restart_from_reference_preserves_prefix_and_final_bytes(
    inputs: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, _, permitted, schedule = inputs
    study = study_for(inputs, "target_stop")
    path = tmp_path / "window.jsonl"
    append = IncrementalPredictionWindowWriter.append

    def interrupted(self: Any, decision: Any) -> None:
        append(self, decision)
        if self.completed_count == 2:
            raise KeyboardInterrupt("reference prefix")

    with monkeypatch.context() as reference:
        reference.setattr(PreparedOutcomeSources, "prepare", reference_preparation)
        baseline = run_prediction_window_in_session(
            prepare_prediction_study_dataset(permitted.dataset),
            study,
            **arguments(inputs),
        )
        reference.setattr(IncrementalPredictionWindowWriter, "append", interrupted)
        with pytest.raises(KeyboardInterrupt):
            run_incremental_prediction_window_in_session(
                prepare_prediction_study_dataset(permitted.dataset),
                study,
                path=path,
                canonical_metadata=fixture.dataset.metadata,
                **arguments(inputs),
            )
    staging = path.with_name(path.name + ".in-progress")
    prefix = (staging / "decisions.jsonl").read_bytes()
    assert (
        json.loads((staging / "checkpoint.json").read_bytes())["checkpoint"][
            "completed_count"
        ]
        == 2
    )
    calls: list[int] = []
    validate = prepared_outcomes.validate_prediction_source

    def validation(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return validate(*args, **kwargs)

    monkeypatch.setattr(prepared_outcomes, "validate_prediction_source", validation)
    executed: list[Any] = []
    generate = StudyRule.generate_with_context

    def tracked(self: Any, context: Any) -> Any:
        executed.append(context.as_of)
        assert (staging / "decisions.jsonl").read_bytes() == prefix
        return generate(self, context)

    monkeypatch.setattr(StudyRule, "generate_with_context", tracked)

    def preserved_append(self: Any, decision: Any) -> None:
        append(self, decision)
        assert (staging / "decisions.jsonl").read_bytes().startswith(prefix)

    monkeypatch.setattr(IncrementalPredictionWindowWriter, "append", preserved_append)
    run_incremental_prediction_window_in_session(
        prepare_prediction_study_dataset(permitted.dataset),
        study,
        path=path,
        canonical_metadata=fixture.dataset.metadata,
        **arguments(inputs),
    )
    assert calls == [1]
    assert executed == [schedule.decision_timestamps[2]]
    assert (
        path.read_bytes()
        == CompactPredictionWindowResult.from_window(baseline).serialize()
    )
    # A finalized retry needs neither source preparation nor prediction execution.
    monkeypatch.setattr(
        PreparedOutcomeSources,
        "prepare",
        forbidden_preparation,
    )
    run_incremental_prediction_window_in_session(
        prepare_prediction_study_dataset(permitted.dataset),
        study,
        path=path,
        canonical_metadata=fixture.dataset.metadata,
        **arguments(inputs),
    )
    assert executed == [schedule.decision_timestamps[2]]
