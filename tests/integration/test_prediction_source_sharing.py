"""Old-copy/new-sharing parity with real QF-11/42/48/52/55/56 contracts."""

import json
from dataclasses import FrozenInstanceError, dataclass, field, replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

import quantforge.prediction.window as window_module
from quantforge.data import DatasetFamily, IntradayMarketDataCache, TimeframeBarSeries
from quantforge.data.models import BoundedPredictionProvenance
from quantforge.data.prediction_views import bounded_prediction_view
from quantforge.indicators import (
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    TALIB_INDICATOR_BACKEND,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
)
from quantforge.prediction import (
    InvalidPredictionOutputError,
    PredictionDecisionSchedule,
    PredictionIndicatorRequirement,
    PredictionRuleContext,
    PredictionStudy,
    SignalFeatureCandidateOutput,
    SignalFeatureValue,
    intraday_excursion_outcome,
    intraday_forward_return_outcome,
    intraday_target_stop_outcome,
)
from quantforge.prediction.intraday_forward_return import (
    IntradayForwardReturnOutcomeLabeler,
)
from quantforge.prediction.source_sharing import prediction_source_copy_memo
from quantforge.prediction.study import (
    PredictionStudyDatasetSession,
    PredictionStudyResult,
    prepare_prediction_study_dataset,
)
from quantforge.prediction.window_compact import CompactPredictionWindowResult
from quantforge.prediction.window_execution import (
    run_incremental_prediction_window_in_session,
)
from quantforge.prediction.window_incremental import IncrementalPredictionWindowWriter
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.validation import PredictionMembershipSource
from tests.integration.test_compact_prediction_provenance import WindowProvider
from tests.integration.test_intraday_prediction_provenance import (
    DAILY,
    DECISION,
    TWO_MINUTES,
    Fixture,
    study_inputs,
)
from tests.integration.test_intraday_prediction_provenance import fixture as fixture
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _FixtureCandidateRule,  # pyright: ignore[reportPrivateUsage]
)


class StatefulRule(_FixtureCandidateRule):
    """Mutable state must start pristine even after a previous outcome callback."""

    def __init__(self, fixture: Fixture) -> None:
        rule, _ = study_inputs(fixture)
        primary = replace(
            rule.context_requirements.primary,
            indicators=(
                PredictionIndicatorRequirement(
                    "trend",
                    SimpleMovingAverage(
                        SimpleMovingAverageParameters(2),
                        backend_id=TALIB_INDICATOR_BACKEND,
                    ),
                ),
            ),
        )
        super().__init__(replace(rule.context_requirements, primary=primary))
        self.diagnostics: dict[str, list[datetime]] = {"decisions": []}
        self.contexts: list[PredictionRuleContext] = []

    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> SignalFeatureCandidateOutput:
        assert self.generate_calls == 0
        assert self.diagnostics == {"decisions": []}
        assert self.contexts == []
        self.diagnostics["decisions"].append(context.as_of)
        self.contexts.append(context)
        assert all(
            bar.end_timestamp <= context.as_of
            for requirement in self.context_requirements.all_timeframes
            for bar in context.bars_for(requirement.timeframe)
        )
        assert context.latest_bar_for(TWO_MINUTES).end_timestamp == context.as_of
        assert context.latest_bar_for(DAILY).end_timestamp.date() < context.as_of.date()
        with pytest.raises(FrozenInstanceError):
            setattr(context, "as_of", context.as_of + timedelta(days=1))
        # Mutating a detached serialization must not modify this or later views.
        primitive = context.values_primitive()
        primitive["as_of"] = "changed"
        output = super().generate_with_context(context)
        if context.as_of == DECISION + timedelta(minutes=2):
            return replace(output, signals=())
        trend = context.indicator_for(TWO_MINUTES, "trend").values_for(
            SIMPLE_MOVING_AVERAGE_OUTPUT
        )[-1]
        assert trend is not None
        return replace(
            output,
            signals=tuple(
                replace(
                    signal,
                    strategy_features=(
                        *signal.strategy_features,
                        SignalFeatureValue("trend", trend),
                    ),
                )
                for signal in output.signals
            ),
        )


@dataclass(frozen=True)
class StatefulStudy(PredictionStudy[Any, Any, Any]):
    diagnostics: dict[str, list[datetime]] = field(
        default_factory=lambda: {"decisions": []}
    )


def prepare(
    fixture: Fixture, outcome_name: str = "forward30"
) -> tuple[PredictionStudyDatasetSession, StatefulStudy, PredictionDecisionSchedule]:
    if outcome_name.startswith("forward"):
        outcome = intraday_forward_return_outcome(
            timedelta(minutes=int(outcome_name.removeprefix("forward"))),
            fixture.primary,
        )
    elif outcome_name == "excursion":
        outcome = intraday_excursion_outcome(timedelta(minutes=60), fixture.primary)
    else:
        outcome = intraday_target_stop_outcome(
            timedelta(minutes=60), fixture.primary, Decimal("0.003"), Decimal("0.002")
        )
    study = StatefulStudy.create(
        StatefulRule(fixture),
        outcome.labeler,
        outcome.evaluator,
        outcome_source=fixture.primary,
    )
    view = bounded_prediction_view(fixture.dataset, DECISION)
    assert isinstance(view.metadata.intraday_provenance, BoundedPredictionProvenance)
    assert view.metadata.actual_last_session < DECISION.date()
    return (
        prepare_prediction_study_dataset(view),
        cast(StatefulStudy, study),
        PredictionDecisionSchedule(
            TWO_MINUTES, DECISION, DECISION + timedelta(minutes=4)
        ),
    )


def arguments(fixture: Fixture, schedule: PredictionDecisionSchedule) -> dict[str, Any]:
    return {
        "schedule": schedule,
        "context_provider": WindowProvider(fixture),
        "dataset_family_fingerprint": fixture.primary.dataset_reference.family_id,
        "context_environment": {"provider": "immutable_synthetic_fixture"},
    }


def old_copy_memo(source: TimeframeBarSeries | None) -> dict[int, object]:
    return {}


def test_canonical_source_developing_evidence_is_immutable(fixture: Fixture) -> None:
    provenance = fixture.dataset.metadata.intraday_provenance
    assert provenance is not None
    source = TimeframeBarSeries.from_source_dataset(
        fixture.source,
        family=DatasetFamily.from_manifest(provenance.family_manifest.to_primitive()),
        cache=IntradayMarketDataCache(fixture.cache.root),
    )
    memo = prediction_source_copy_memo(source)
    assert memo[id(source)] == source


@pytest.mark.parametrize(
    "outcome_name",
    ["forward10", "forward30", "forward60", "forward120", "excursion", "target_stop"],
)
def test_exact_old_copy_equivalence_and_mutable_isolation(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, outcome_name: str
) -> None:
    prepared, study, schedule = prepare(fixture, outcome_name)
    executed: list[StatefulStudy] = []
    source_copies: list[int] = []
    original_run = window_module.run_prediction_study_in_session
    original_reduce = TimeframeBarSeries.__reduce_ex__

    def count_source_copy(self: TimeframeBarSeries, protocol: int) -> Any:
        source_copies.append(len(self.bars))
        return original_reduce(self, protocol)

    def inspect_study(
        prepared: PredictionStudyDatasetSession,
        decision_study: StatefulStudy,
        **kwargs: Any,
    ) -> PredictionStudyResult[Any, Any, Any]:
        assert decision_study.diagnostics == {"decisions": []}
        decision_study.diagnostics["decisions"].append(DECISION)
        executed.append(decision_study)
        return original_run(prepared, decision_study, **kwargs)

    monkeypatch.setattr(TimeframeBarSeries, "__reduce_ex__", count_source_copy)
    monkeypatch.setattr(window_module, "run_prediction_study_in_session", inspect_study)
    with monkeypatch.context() as old:
        old.setattr(window_module, "prediction_source_copy_memo", old_copy_memo)
        baseline = window_module.run_prediction_window_in_session(
            prepared, study, **arguments(fixture, schedule)
        )
    baseline_studies = executed[:]
    assert (
        source_copies.count(len(fixture.primary.bars))
        == len(schedule.decision_timestamps) + 1
    )
    source_copies.clear()
    executed.clear()
    optimized = window_module.run_prediction_window_in_session(
        prepared, study, **arguments(fixture, schedule)
    )
    assert source_copies.count(len(fixture.primary.bars)) == 0
    # The small bounded labeler inputs still receive pristine and component copies.
    assert len(source_copies) == 4  # two candidate decisions, two copies each
    assert optimized.serialize() == baseline.serialize()
    old_compact = CompactPredictionWindowResult.from_window(baseline)
    new_compact = CompactPredictionWindowResult.from_window(optimized)
    assert new_compact.serialize() == old_compact.serialize()
    assert new_compact.window_result_id == old_compact.window_result_id
    assert [d.decision_id for d in new_compact.decisions] == [
        d.decision_id for d in old_compact.decisions
    ]
    membership = PredictionMembershipSource.capture(schedule, fixture.primary)
    assert tuple(d.decision_timestamp for d in optimized.decisions) == (
        membership.schedule.decision_timestamps
    )
    for previous, current in zip(baseline_studies, executed, strict=True):
        assert current.outcome_source is executed[0].outcome_source
        assert current.outcome_source == fixture.primary
        assert previous.outcome_source is not fixture.primary
        assert previous.outcome_source == current.outcome_source
        before_rule, after_rule = (
            cast(StatefulRule, previous.strategy),
            cast(StatefulRule, current.strategy),
        )
        assert before_rule.contexts[0].values_primitive() == (
            after_rule.contexts[0].values_primitive()
        )
        assert after_rule.generate_calls == 1
    for attribute in ("strategy", "outcome_labeler", "evaluator", "diagnostics"):
        assert len({id(getattr(item, attribute)) for item in executed}) == 3
        assert all(
            getattr(item, attribute) is not getattr(study, attribute)
            for item in executed
        )
    contexts = [cast(StatefulRule, item.strategy).contexts[0] for item in executed]
    assert len({id(context) for context in contexts}) == 3
    assert tuple(context.as_of for context in contexts) == schedule.decision_timestamps
    assert [len(context.bars_for(TWO_MINUTES)) for context in contexts] == [
        240,
        241,
        242,
    ]
    assert cast(StatefulRule, study.strategy).diagnostics == {"decisions": []}
    assert study.diagnostics == {"decisions": []}


def test_old_checkpoint_resumes_without_reexecution_or_prefix_changes(
    fixture: Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, study, schedule = prepare(fixture)
    path = tmp_path / "window.jsonl"
    kwargs = arguments(fixture, schedule)
    kwargs["canonical_metadata"] = fixture.dataset.metadata
    append = IncrementalPredictionWindowWriter.append

    def stop_after_prefix(
        self: IncrementalPredictionWindowWriter, decision: Any
    ) -> None:
        append(self, decision)
        if self.completed_count == 2:
            raise KeyboardInterrupt("durable old-copy prefix")

    with monkeypatch.context() as old:
        old.setattr(window_module, "prediction_source_copy_memo", old_copy_memo)
        baseline = window_module.run_prediction_window_in_session(
            prepared, study, **arguments(fixture, schedule)
        )
        old.setattr(IncrementalPredictionWindowWriter, "append", stop_after_prefix)
        with pytest.raises(KeyboardInterrupt):
            run_incremental_prediction_window_in_session(
                prepared, study, path=path, **kwargs
            )
    staging = path.with_name(path.name + ".in-progress")
    journal = staging / "decisions.jsonl"
    prefix = journal.read_bytes()
    checkpoint = json.loads((staging / "checkpoint.json").read_bytes())["checkpoint"]
    assert checkpoint["completed_count"] == 2
    executed: list[datetime] = []
    provider = cast(WindowProvider, kwargs["context_provider"])

    class RecordingProvider:
        def get_context_at(self, requirements: Any, *, as_of: datetime) -> Any:
            executed.append(as_of)
            return provider.get_context_at(requirements, as_of=as_of)

    def verify_prefix(self: IncrementalPredictionWindowWriter, decision: Any) -> None:
        assert self.completed_count == 2
        assert decision.to_primitive()["sequence"] == 2
        assert journal.read_bytes() == prefix
        append(self, decision)
        assert journal.read_bytes().startswith(prefix)

    kwargs["context_provider"] = RecordingProvider()
    with monkeypatch.context() as resumed:
        resumed.setattr(IncrementalPredictionWindowWriter, "append", verify_prefix)
        reader = run_incremental_prediction_window_in_session(
            prepared, study, path=path, **kwargs
        )
    assert executed == [schedule.decision_timestamps[2]]

    assert (
        path.read_bytes()
        == CompactPredictionWindowResult.from_window(baseline).serialize()
    )
    reader.verify_integrity()
    assert PredictionWindowReader.open(path).header() == reader.header()
    run_incremental_prediction_window_in_session(prepared, study, path=path, **kwargs)
    assert executed == [schedule.decision_timestamps[2]]


def test_labeler_source_mutation_still_rejects_without_changing_backing(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, study, schedule = prepare(fixture)
    original_bars = tuple(bar.to_primitive() for bar in fixture.primary.bars)
    original_label = IntradayForwardReturnOutcomeLabeler.label_request

    def mutate_source(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_label(self, *args, **kwargs)
        source = cast(TimeframeBarSeries, kwargs["source"])
        # Deliberately bypass the frozen API to exercise QF-11's stronger
        # component-facing mutation detection and detached bounded inputs.
        object.__setattr__(source.bars[0], "close", Decimal("999"))
        return result

    with monkeypatch.context() as malicious:
        malicious.setattr(
            IntradayForwardReturnOutcomeLabeler, "label_request", mutate_source
        )
        with pytest.raises(
            InvalidPredictionOutputError, match="mutated its bounded source"
        ):
            window_module.run_prediction_window_in_session(
                prepared, study, **arguments(fixture, schedule)
            )
    assert tuple(bar.to_primitive() for bar in fixture.primary.bars) == original_bars
    # A fresh historical execution still sees the pristine canonical prices.
    result = window_module.run_prediction_window_in_session(
        prepared, study, **arguments(fixture, schedule)
    )
    assert len(result.decisions) == 3
