"""Deterministic SPY windows using QF-20 alignment and the real QF-11 runner."""

import ast
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import (
    AggregatedSessionBar,
    ContextCompletionPolicy,
    IntradayBar,
    MarketDataset,
    MultiTimeframeContext,
    TimeframeBarSeries,
    build_multi_timeframe_context,
)
from quantforge.indicators import (
    NATIVE_INDICATOR_BACKEND,
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    TALIB_INDICATOR_BACKEND,
)
from quantforge.optimization import IntegerValues, ParameterSearchSpace
from quantforge.optimization.models import StabilityConfig, TrialStatus
from quantforge.prediction import (
    InvalidPredictionConfigurationError,
    InvalidPredictionDataError,
    InvalidPredictionGridConfigurationError,
    InvalidPredictionOutputError,
    NextSessionOpenGapOutcomeLabeler,
    NextSessionOpenGapValues,
    OutcomeLabel,
    PredictionContextEnvironment,
    PredictionContextFailurePolicy,
    PredictionContextRequirements,
    PredictionDecisionSchedule,
    PredictionDirection,
    PredictionFeature,
    PredictionGridConfig,
    PredictionGridPersistenceError,
    PredictionGridStudy,
    PredictionRankingConfig,
    PredictionRuleContext,
    PredictionStrategyOutput,
    PredictionStudy,
    PredictionTrialAnalysis,
    PredictionWindowResult,
    run_prediction_study,
    run_prediction_window,
)
from quantforge.timeframes import (
    BarLabel,
    CrossSessionPolicy,
    DevelopingBarExposure,
    ExchangeSessionPolicy,
    IntradayAnchor,
    IntradayInterval,
    SessionScope,
    Timeframe,
)
from tests.unit.data.test_multi_timeframe import (
    _family,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.helpers import make_dataset
from tests.unit.indicators.test_timeframe_evaluation import (
    _adjustment_basis,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_multi_timeframe_study import (
    FixtureContextProvider,
    FixtureMultiTimeframeRule,
    FixtureParameters,
    _prediction_context,  # pyright: ignore[reportPrivateUsage]
    _prediction_dataset,  # pyright: ignore[reportPrivateUsage]
    _requirements,  # pyright: ignore[reportPrivateUsage]
    _study,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_grid import (
    FixtureAnalyzer,
    FixtureStudyFactory,
    _backend_environment,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_technical_confluence import (
    _rule as confluence_rule,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_technical_confluence import (
    _study as confluence_study,  # pyright: ignore[reportPrivateUsage]
)

NEW_YORK = ZoneInfo("America/New_York")
START = datetime(2024, 7, 11, 13, 35, tzinfo=UTC)
END = START + timedelta(minutes=15)


class WindowProvider:
    """Use existing validated bar fixtures with the public context builder."""

    def __init__(self, *, append_future: bool = False) -> None:
        self.family = replace(_family(), adjustment_basis=_adjustment_basis())
        original = _prediction_context()
        self.series: tuple[TimeframeBarSeries, ...] = ()
        for lineage in self.family.datasets:
            bars = tuple(
                bar
                for bar in original.bars_for(lineage.timeframe)
                if isinstance(bar, (IntradayBar, AggregatedSessionBar))
            )
            if lineage.timeframe == original.primary_timeframe:
                last = cast(IntradayBar, bars[-1])
                extra = tuple(
                    replace(
                        last,
                        start_timestamp=START + timedelta(minutes=5 * index),
                        end_timestamp=START + timedelta(minutes=5 * (index + 1)),
                        open=Decimal(close),
                        high=Decimal(close + 1),
                        low=Decimal(close - 1),
                        close=Decimal(close),
                    )
                    for index, close in enumerate(
                        (10, 11, 9, 999) if append_future else (10, 11, 9)
                    )
                )
                bars = (*bars, *extra)
            self.series += (
                TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
                    self.family.reference(lineage.dataset_id),
                    lineage.timeframe,
                    bars,
                    dataset_family_manifest_id=self.family.manifest_id,
                ),
            )
        self.requests: list[datetime] = []
        self.interrupt_at: datetime | None = None

    def get_context_at(
        self, requirements: PredictionContextRequirements, *, as_of: datetime
    ) -> MultiTimeframeContext:
        self.requests.append(as_of)
        if as_of == self.interrupt_at:
            self.interrupt_at = None
            raise KeyboardInterrupt("deterministic mid-window interruption")
        return build_multi_timeframe_context(
            as_of=as_of,
            primary_timeframe=requirements.primary.timeframe,
            required_timeframes=requirements.context_timeframe_requirements(),
            completion_policy=requirements.context_completion_policy,
            series=self.series,
        )


class WindowRule(FixtureMultiTimeframeRule):
    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> PredictionStrategyOutput:
        output = super().generate_with_context(context)
        primary = self.context_requirements.primary.timeframe
        trend = context.indicator_for(primary, "trend").values_for(
            SIMPLE_MOVING_AVERAGE_OUTPUT
        )[-1]
        assert trend is not None
        close = context.latest_bar_for(primary).close
        return replace(
            output,
            signals=(
                replace(
                    output.signals[0],
                    direction=PredictionDirection.UP
                    if close >= trend
                    else PredictionDirection.DOWN,
                    feature_values=(
                        PredictionFeature("calls", Decimal(self.calls)),
                        PredictionFeature("primary_trend", trend),
                    ),
                ),
            ),
        )


class WindowFactory(FixtureStudyFactory):
    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        return _study(WindowRule(_requirements(window=cast(int, parameters["window"]))))


class WindowAnalyzer:
    name = "fixture_window_direction_accuracy"
    version = "1"

    def __init__(self) -> None:
        self.seen: list[tuple[str, ...]] = []

    def configuration(self) -> PrimitiveMapping:
        return {"name": self.name, "version": self.version, "baseline": "always_up"}

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def analyze_window(
        self, result: PredictionWindowResult[Any, Any, Any]
    ) -> PredictionTrialAnalysis:
        self.seen.append(tuple(item.result.study_id for item in result.decisions))
        rows = [row for decision in result.decisions for row in decision.result.rows]
        correct = sum(row.evaluation.values.direction_correct for row in rows)
        accuracy = Decimal(correct) / len(rows) if rows else Decimal(0)
        return PredictionTrialAnalysis.create(
            prediction_count=len(rows),
            metrics={"accuracy": str(accuracy)},
            period_comparisons=({"period": "fixture", "count": len(rows)},),
            weekday_comparisons=({"weekday": 3, "count": len(rows)},),
            matched_baseline_comparisons=(
                {"baseline_name": "always_up", "count": len(rows)},
            ),
            artifacts={"row_ids": [row.row_id for row in rows]},
        )


def schedule(
    start: datetime = START, end: datetime = END
) -> PredictionDecisionSchedule:
    return PredictionDecisionSchedule(_requirements().primary.timeframe, start, end)


def run_window(
    provider: WindowProvider | None = None,
    *,
    requirements: PredictionContextRequirements | None = None,
    decision_schedule: PredictionDecisionSchedule | None = None,
) -> PredictionWindowResult[Any, Any, Any]:
    provider = provider or WindowProvider()
    return run_prediction_window(
        _prediction_dataset(),
        _study(WindowRule(requirements or _requirements())),
        schedule=decision_schedule or schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment={"provider": "immutable_fixture", "version": "1"},
    )


def grid(
    output_root: Path,
    provider: WindowProvider,
    *,
    analyzer: WindowAnalyzer | None = None,
    decision_schedule: PredictionDecisionSchedule | None = None,
    factory: FixtureStudyFactory | None = None,
    backend_configuration: PrimitiveMapping | None = None,
) -> PredictionGridStudy:
    return PredictionGridStudy(
        dataset=_prediction_dataset(),
        dataset_family_fingerprint=provider.family.family_id,
        study_factory=factory or WindowFactory(),
        analyzer=analyzer or WindowAnalyzer(),
        context_provider=provider,
        context_environment=PredictionContextEnvironment.create("fixture", "1", {}),
        indicator_backend=_backend_environment(configuration=backend_configuration),
        decision_schedule=decision_schedule or schedule(),
        config=PredictionGridConfig(
            label="SPY historical decision fixture",
            search_space=ParameterSearchSpace({"window": IntegerValues((2, 3))}),
            ranking=PredictionRankingConfig(
                "accuracy", "always_up", minimum_prediction_count=4
            ),
            stability=StabilityConfig(minimum_eligible_neighbors=1),
            output_root=output_root,
        ),
    )


def test_schedule_is_deterministic_closed_and_normalized_to_utc() -> None:
    baseline = schedule()
    localized = schedule(START.astimezone(NEW_YORK), END.astimezone(NEW_YORK))
    assert baseline == localized
    assert baseline.schedule_id == localized.schedule_id
    assert baseline.decision_timestamps == tuple(
        START + timedelta(minutes=5 * i) for i in range(4)
    )
    assert (
        schedule(START + timedelta(seconds=1), END).decision_timestamps
        == baseline.decision_timestamps[1:]
    )
    assert schedule(START, START).decision_timestamps == (START,)
    assert (
        schedule(
            START + timedelta(seconds=1), START + timedelta(seconds=2)
        ).decision_timestamps
        == ()
    )
    assert (
        replace(
            baseline,
            primary_timeframe=replace(
                baseline.primary_timeframe, bar_label=BarLabel.END
            ),
        ).decision_timestamps
        == baseline.decision_timestamps
    )


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (
            datetime(2024, 7, 3, tzinfo=NEW_YORK),
            datetime(2024, 7, 5, 23, tzinfo=NEW_YORK),
            (
                "2024-07-03T17:00:00+00:00",
                "2024-07-05T17:30:00+00:00",
                "2024-07-05T20:00:00+00:00",
            ),
        ),
        (
            datetime(2024, 3, 8, tzinfo=NEW_YORK),
            datetime(2024, 3, 11, 23, tzinfo=NEW_YORK),
            (
                "2024-03-08T18:30:00+00:00",
                "2024-03-08T21:00:00+00:00",
                "2024-03-11T17:30:00+00:00",
                "2024-03-11T20:00:00+00:00",
            ),
        ),
    ],
)
def test_exchange_holiday_early_close_and_dst(
    start: datetime, end: datetime, expected: tuple[str, ...]
) -> None:
    result = PredictionDecisionSchedule(
        Timeframe.us_equity(IntradayInterval(timedelta(hours=4))), start, end
    )
    assert tuple(item.isoformat() for item in result.decision_timestamps) == expected


def test_clock_anchor_and_extended_hours_follow_existing_session_windows() -> None:
    timeframe = Timeframe(
        IntradayInterval(timedelta(hours=1), IntradayAnchor.CLOCK, time(9)),
        ExchangeSessionPolicy(
            scope=SessionScope.EXTENDED_HOURS,
            extended_hours_start=time(8, 30),
            extended_hours_end=time(17, 15),
        ),
    )
    result = PredictionDecisionSchedule(
        timeframe,
        datetime(2024, 7, 3, tzinfo=NEW_YORK),
        datetime(2024, 7, 3, 23, tzinfo=NEW_YORK),
    )
    assert tuple(
        item.astimezone(NEW_YORK).time() for item in result.decision_timestamps
    ) == (*tuple(time(hour) for hour in range(9, 18)), time(17, 15))


@pytest.mark.parametrize(
    "timeframe",
    [
        Timeframe.us_equity(),
        Timeframe.us_equity(
            IntradayInterval(
                timedelta(minutes=5), cross_session_policy=CrossSessionPolicy.PERMITTED
            )
        ),
        replace(
            _requirements().primary.timeframe,
            developing_bar_exposure=DevelopingBarExposure.INCLUDE,
        ),
    ],
)
def test_schedule_rejects_unsupported_primary_semantics(timeframe: Timeframe) -> None:
    with pytest.raises(InvalidPredictionConfigurationError, match="completed intraday"):
        PredictionDecisionSchedule(timeframe, START, END)


def test_schedule_rejects_naive_and_reversed_intervals() -> None:
    with pytest.raises(InvalidPredictionConfigurationError, match="timezone-aware"):
        schedule(START.replace(tzinfo=None), END)
    with pytest.raises(InvalidPredictionConfigurationError, match="start"):
        schedule(END, START)


def test_all_decisions_reuse_qf11_and_preserve_original_context_result_and_rows() -> (
    None
):
    provider = WindowProvider()
    result = run_window(provider)
    assert provider.requests == list(schedule().decision_timestamps)
    for decision in result.decisions:
        context = provider.get_context_at(
            _requirements(), as_of=decision.decision_timestamp
        )
        single = run_prediction_study(
            _prediction_dataset(),
            _study(WindowRule(_requirements())),
            context_provider=FixtureContextProvider(context),
        )
        assert decision.context_id == context.context_id
        assert decision.result == single
        assert decision.to_primitive()["prediction_study"] == single.to_primitive()
        assert decision.result.signals[0].feature_values[0].value == 1
    assert len({item.result.study_id for item in result.decisions}) == 4
    assert (
        len({row.row_id for item in result.decisions for row in item.result.rows}) == 4
    )
    assert result.results == tuple(item.result for item in result.decisions)
    assert result.counts_primitive()["valid_decisions"] == 4
    assert result.serialize() == run_window().serialize()
    assert json.loads(result.serialize()) == result.to_primitive()
    assert not hasattr(result, "study_id")
    with pytest.raises(
        InvalidPredictionOutputError, match="incomplete or out of order"
    ):
        replace(result, decisions=result.decisions[:-1])
    with pytest.raises(
        InvalidPredictionOutputError, match="incomplete or out of order"
    ):
        replace(result, decisions=tuple(reversed(result.decisions)))


def test_future_append_preserves_schedule_features_results_and_identity() -> None:
    original = run_window()
    appended = run_window(WindowProvider(append_future=True))
    assert original.schedule == appended.schedule
    assert original.serialize() == appended.serialize()
    assert original.window_result_id == appended.window_result_id


def test_missing_primary_is_scheduled_and_explicitly_skipped_or_failed() -> None:
    extended = schedule(end=END + timedelta(minutes=5))
    with pytest.raises(
        InvalidPredictionDataError, match="missing the scheduled primary"
    ):
        run_window(decision_schedule=extended)
    result = run_window(
        requirements=_requirements(failure_policy=PredictionContextFailurePolicy.SKIP),
        decision_schedule=extended,
    )
    assert result.counts_primitive()["scheduled_decisions"] == 5
    assert result.counts_primitive()["skipped_decisions"] == 1
    assert result.decisions[-1].to_primitive()["status"] == "skipped"
    assert result.decisions[-1].decision_timestamp == END + timedelta(minutes=5)


def test_stale_context_keeps_its_original_qf11_skip_result_and_identity() -> None:
    requirements = _requirements(
        daily_maximum_age=timedelta(hours=1),
        failure_policy=PredictionContextFailurePolicy.SKIP,
    )
    provider = WindowProvider()
    result = run_window(provider, requirements=requirements)
    assert result.counts_primitive()["skipped_decisions"] == 4
    for item in result.decisions:
        source = provider.get_context_at(requirements, as_of=item.decision_timestamp)
        single = run_prediction_study(
            _prediction_dataset(),
            _study(WindowRule(requirements)),
            context_provider=FixtureContextProvider(source),
        )
        assert item.result == single
        assert item.context_id == source.context_id


def test_wrong_timestamp_and_family_cannot_enter_historical_results() -> None:
    class WrongTimestampProvider(WindowProvider):
        def get_context_at(
            self, requirements: PredictionContextRequirements, *, as_of: datetime
        ) -> MultiTimeframeContext:
            return super().get_context_at(
                requirements, as_of=as_of + timedelta(minutes=5)
            )

    with pytest.raises(InvalidPredictionDataError, match="wrong decision timestamp"):
        run_window(WrongTimestampProvider())
    provider = WindowProvider()
    with pytest.raises(InvalidPredictionDataError, match="wrong dataset family"):
        run_prediction_window(
            _prediction_dataset(),
            _study(WindowRule(_requirements())),
            schedule=schedule(),
            context_provider=provider,
            dataset_family_fingerprint="different-source",
            context_environment={},
        )


def test_empty_schedule_and_no_signal_are_valid_and_distinct_from_skip() -> None:
    class EmptyRule(WindowRule):
        def generate_with_context(
            self, context: PredictionRuleContext
        ) -> PredictionStrategyOutput:
            return replace(super().generate_with_context(context), signals=())

    provider = WindowProvider()
    result = run_prediction_window(
        _prediction_dataset(),
        _study(EmptyRule(_requirements())),
        schedule=schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment={},
    )
    assert result.counts_primitive()["no_prediction_decisions"] == 4
    assert result.counts_primitive()["skipped_decisions"] == 0
    empty = run_window(
        decision_schedule=schedule(
            START + timedelta(seconds=1), START + timedelta(seconds=2)
        )
    )
    assert empty.decisions == ()
    assert empty.counts_primitive()["scheduled_decisions"] == 0


def test_grid_analyzes_full_collection_and_preserves_ranking_resume_and_sources(
    tmp_path: Path,
) -> None:
    provider, analyzer = WindowProvider(), WindowAnalyzer()
    study = grid(tmp_path, provider, analyzer=analyzer)
    result = study.run()
    assert all(item.status is TrialStatus.SUCCEEDED for item in result.trials)
    assert [item.objective_value for item in result.rankings] == [
        Decimal("0.5"),
        Decimal("0.25"),
    ]
    assert result.rankings[0].trial_id == result.trials[0].trial_id
    assert len(analyzer.seen) == 2
    assert all(len(ids) == len(set(ids)) == 4 for ids in analyzer.seen)
    assert provider.requests == list(schedule().decision_timestamps) * 2
    for trial in result.trials:
        assert trial.analysis is not None
        assert trial.analysis.prediction_count == 4
        evidence = trial.analysis.artifacts_snapshot.to_primitive()
        assert len(cast(list[str], evidence["row_ids"])) == 4
        references = cast(PrimitiveMapping, evidence["prediction_window_sources"])
        assert len(cast(list[PrimitiveMapping], references["decisions"])) == 4
        assert trial.artifact_location is not None
        artifact = json.loads(
            (tmp_path / study.study_id / trial.artifact_location).read_text()
        )
        assert "prediction_study" not in artifact
        assert len(artifact["prediction_window"]["decisions"]) == 4
    assert study.resume().rankings == result.rankings
    assert study.load_result().rankings == result.rankings
    assert len(provider.requests) == 8
    assert len(analyzer.seen) == 2


def test_interrupted_window_is_not_complete_and_resume_reruns_entire_candidate(
    tmp_path: Path,
) -> None:
    provider = WindowProvider()
    provider.interrupt_at = START + timedelta(minutes=5)
    study = grid(tmp_path, provider)
    with pytest.raises(KeyboardInterrupt):
        study.run()
    records = [
        json.loads(path.read_text())
        for path in (tmp_path / study.study_id / "trials").glob("*.json")
    ]
    assert [item["status"] for item in records] == ["running"]
    assert records[0]["analysis"] is None
    assert not list(
        (tmp_path / study.study_id / "artifacts").rglob("prediction-window.json")
    )
    with pytest.raises(PredictionGridPersistenceError):
        study.load_result()
    resumed = study.resume()
    uninterrupted = grid(tmp_path / "fresh", WindowProvider()).run()
    assert resumed.rankings == uninterrupted.rankings
    assert provider.requests == [
        START,
        START + timedelta(minutes=5),
        *schedule().decision_timestamps,
        *schedule().decision_timestamps,
    ]


@pytest.mark.parametrize(
    "mutation", ["remove_decision", "context_id", "timestamp", "backend", "result_id"]
)
def test_modified_or_partial_window_artifact_cannot_be_loaded_or_resumed(
    tmp_path: Path, mutation: str
) -> None:
    study = grid(tmp_path, WindowProvider())
    result = study.run()
    path = tmp_path / study.study_id / cast(str, result.trials[0].artifact_location)
    artifact = json.loads(path.read_text())
    window = artifact["prediction_window"]
    if mutation == "remove_decision":
        window["decisions"].pop()
    elif mutation == "context_id":
        window["decisions"][0]["context_id"] = "changed"
    elif mutation == "timestamp":
        window["decisions"][0]["decision_timestamp"] = END.isoformat()
    elif mutation == "backend":
        window["manifest"]["indicator_backend_environment"]["library_version"] = (
            "changed"
        )
    else:
        window["decisions"][0]["prediction_study_id"] = "changed"
    path.write_text(json.dumps(artifact))
    with pytest.raises(PredictionGridPersistenceError, match="incompatible"):
        study.load_result()
    with pytest.raises(PredictionGridPersistenceError, match="incompatible"):
        study.resume()


def test_schedule_backend_and_analyzer_identity_prevent_cross_configuration_resume(
    tmp_path: Path,
) -> None:
    provider = WindowProvider()
    original = grid(tmp_path, provider)
    original.run()
    changed_schedule = grid(
        tmp_path, provider, decision_schedule=schedule(end=END + timedelta(minutes=5))
    )
    changed_backend = grid(
        tmp_path, provider, backend_configuration={"selection_policy": "changed"}
    )
    analyzer = WindowAnalyzer()
    analyzer.version = "2"
    changed_analyzer = grid(tmp_path, provider, analyzer=analyzer)
    assert (
        len(
            {
                item.study_id
                for item in (
                    original,
                    changed_schedule,
                    changed_backend,
                    changed_analyzer,
                )
            }
        )
        == 4
    )
    for item in (changed_schedule, changed_backend, changed_analyzer):
        with pytest.raises(PredictionGridPersistenceError, match="no manifest"):
            item.resume()


def test_window_orchestration_uses_no_indicator_backend_directly() -> None:
    path = Path(__file__).parents[3] / "src/quantforge/prediction/window.py"
    imports = [
        node.module
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom)
    ]
    assert not any("talib" in (module or "").lower() for module in imports)


@pytest.mark.parametrize(
    ("threshold", "disposition"), [("0", "accepted"), ("12.5", "rejected")]
)
def test_qf31_candidates_preserve_acceptance_and_explicit_no_prediction(
    threshold: str, disposition: str
) -> None:
    provider = WindowProvider()
    rule = confluence_rule(
        threshold=Decimal(threshold), backend_id=NATIVE_INDICATOR_BACKEND
    )
    study, _ = confluence_study(rule)
    result = run_prediction_window(
        _prediction_dataset(),
        study,
        schedule=schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment={},
    )
    assert result.counts_primitive()["signal_dispositions"] == {disposition: 4}
    assert result.counts_primitive()["skipped_decisions"] == 0
    for decision in result.decisions:
        signal = decision.result.signals[0]
        assert signal.disposition.value == disposition
        if disposition == "rejected":
            assert signal.reason_codes == ("technical_confluence_no_prediction",)
            assert signal.direction is None
        assert decision.to_primitive()["generated_signals"]


def test_unavailable_outcomes_keep_generated_signals_and_original_result_identity() -> (
    None
):
    provider = WindowProvider()
    tomorrow = START + timedelta(days=1)
    # Select the primary by timeframe, independent of the family's canonical order.
    primary = next(
        item
        for item in provider.series
        if item.timeframe == schedule().primary_timeframe
    )
    last = cast(IntradayBar, primary.bars[-1])
    extra = replace(
        last,
        session_date=tomorrow.date(),
        start_timestamp=tomorrow - timedelta(minutes=5),
        end_timestamp=tomorrow,
    )
    extended = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        primary.dataset_reference,
        primary.timeframe,
        (*primary.bars, extra),
        dataset_family_manifest_id=provider.family.manifest_id,
    )
    provider.series = tuple(
        extended if item is primary else item for item in provider.series
    )
    result = run_window(provider, decision_schedule=schedule(tomorrow, tomorrow))
    assert result.counts_primitive()["generated_predictions"] == 1
    assert result.counts_primitive()["unavailable_outcomes"] == 1
    decision = result.decisions[0]
    assert decision.result.rows == ()
    assert len(decision.result.signals) == 1
    assert (
        len(cast(list[PrimitiveMapping], decision.to_primitive()["generated_signals"]))
        == 1
    )


def test_future_bearing_component_state_cannot_reach_the_next_decision() -> None:
    class StatefulLabeler(NextSessionOpenGapOutcomeLabeler):
        def __init__(self, rule: WindowRule) -> None:
            self.rule = rule
            self.validated = False

        def validate_dataset(self, dataset: MarketDataset) -> None:
            super().validate_dataset(dataset)
            self.validated = True

        def label(
            self, dataset: MarketDataset, signal_session: date
        ) -> OutcomeLabel[NextSessionOpenGapValues] | None:
            assert self.validated
            self.rule.calls = (
                999  # Invisible configuration state from future-bearing code.
            )
            return super().label(dataset, signal_session)

    rule = WindowRule(_requirements())
    study = replace(_study(rule), outcome_labeler=StatefulLabeler(rule))
    provider = WindowProvider()
    result = run_prediction_window(
        _prediction_dataset(),
        study,
        schedule=schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment={},
    )
    assert [
        item.result.signals[0].features_primitive()["calls"]
        for item in result.decisions
    ] == ["1"] * 4
    assert rule.calls == 0


def test_all_material_study_configuration_changes_window_identity() -> None:
    provider = WindowProvider()
    baseline_study = _study(WindowRule(_requirements()))
    studies = [baseline_study]
    for field in ("strategy", "outcome_labeler", "evaluator"):
        changed = deepcopy(baseline_study)
        setattr(getattr(changed, field), "implementation_version", "2")
        studies.append(changed)
    studies.extend(
        [
            _study(WindowRule(_requirements(window=3))),
            _study(WindowRule(_requirements(backend_id=TALIB_INDICATOR_BACKEND))),
            replace(baseline_study, result_schema_version="2"),
        ]
    )
    results = [
        run_prediction_window(
            _prediction_dataset(),
            study,
            schedule=schedule(),
            context_provider=provider,
            dataset_family_fingerprint=provider.family.family_id,
            context_environment={},
        )
        for study in studies
    ]
    assert len({result.window_id for result in results}) == len(studies)
    assert len({result.window_result_id for result in results}) == len(studies)
    assert (
        results[0].decisions[0].result.configuration.strategy_configuration_id
        != results[5].decisions[0].result.configuration.strategy_configuration_id
    )


def test_developing_context_policy_remains_identity_bearing_and_uses_qf28_skip() -> (
    None
):
    baseline = _requirements(failure_policy=PredictionContextFailurePolicy.SKIP)
    developing = replace(
        baseline,
        contextual=tuple(
            replace(
                item, completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
            )
            for item in baseline.contextual
        ),
    )
    completed_result = run_window(requirements=baseline)
    # These isolated fixtures lack QF-21's cache-validated canonical source
    # evidence. The public builder rejects developing reconstruction explicitly.
    developing_result = run_window(requirements=developing)
    assert developing_result.counts_primitive()["skipped_decisions"] == 4
    assert completed_result.window_id != developing_result.window_id


def test_cache_reuses_same_timestamp_across_candidates_without_aliasing_other_decisions(
    tmp_path: Path,
) -> None:
    class SameIndicatorFactory(WindowFactory):
        def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
            rule = WindowRule(_requirements(window=2))
            rule._parameters = FixtureParameters(mode=str(parameters["window"]))  # pyright: ignore[reportPrivateUsage]
            return _study(rule)

    provider = WindowProvider()
    result = grid(tmp_path, provider, factory=SameIndicatorFactory()).run()
    assert len(result.rankings) == 2
    assert provider.requests == list(schedule().decision_timestamps)
    assert result.cache_statistics.context_hits == 4
    assert result.cache_statistics.context_misses == 4
    assert result.cache_statistics.indicator_hits == 16
    assert result.cache_statistics.indicator_misses == 16


def test_window_grid_rejects_a_single_decision_analyzer(tmp_path: Path) -> None:
    with pytest.raises(InvalidPredictionGridConfigurationError, match="analyze_window"):
        grid(
            tmp_path, WindowProvider(), analyzer=cast(WindowAnalyzer, FixtureAnalyzer())
        )


def test_new_outcome_dataset_changes_identity_without_changing_causal_predictions() -> (
    None
):
    provider = WindowProvider()
    baseline = run_window(provider)
    changed_dataset = make_dataset(
        ("100", "101", "105"),
        sessions=(date(2024, 7, 10), date(2024, 7, 11), date(2024, 7, 12)),
        opens=("99", "100", "104"),
        highs=("101", "102", "106"),
        lows=("98", "99", "101"),
    )
    changed = run_prediction_window(
        changed_dataset,
        _study(WindowRule(_requirements())),
        schedule=schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment={"provider": "immutable_fixture", "version": "1"},
    )
    assert baseline.window_id != changed.window_id
    assert baseline.window_result_id != changed.window_result_id
    for before, after in zip(baseline.decisions, changed.decisions, strict=True):
        assert before.context_id == after.context_id
        assert before.result.signals == after.result.signals
        assert before.result.study_id != after.result.study_id
    with pytest.raises(InvalidPredictionOutputError, match="configuration or dataset"):
        replace(baseline, decisions=changed.decisions)


def test_partial_window_cannot_claim_success_with_recomputed_checksums(
    tmp_path: Path,
) -> None:
    study = grid(tmp_path, WindowProvider())
    result = study.run()
    trial = result.trials[0]
    artifact_path = tmp_path / study.study_id / cast(str, trial.artifact_location)
    artifact = json.loads(artifact_path.read_text())
    window = artifact["prediction_window"]
    window["decisions"].pop()
    result_id = configuration_identity(
        {"window_id": window["manifest"]["window_id"], "decisions": window["decisions"]}
    )
    artifact["prediction_window_id"] = result_id
    window["manifest"]["window_result_id"] = result_id
    artifact.pop("artifact_fingerprint")
    fingerprint = configuration_identity(artifact)
    artifact["artifact_fingerprint"] = fingerprint
    artifact_path.write_text(json.dumps(artifact))
    trial_path = tmp_path / study.study_id / "trials" / f"{trial.trial_id}.json"
    record = json.loads(trial_path.read_text())
    record["artifact_fingerprint"] = fingerprint
    trial_path.write_text(json.dumps(record))
    with pytest.raises(
        PredictionGridPersistenceError, match="window artifact is incomplete"
    ):
        study.load_result()


def test_candidate_primary_mismatch_is_excluded_before_context_execution(
    tmp_path: Path,
) -> None:
    provider = WindowProvider()
    incompatible = replace(
        schedule(),
        primary_timeframe=Timeframe.us_equity(IntradayInterval(timedelta(minutes=10))),
    )
    result = grid(tmp_path, provider, decision_schedule=incompatible).run()
    assert all(trial.status is TrialStatus.EXCLUDED for trial in result.trials)
    assert result.rankings == ()
    assert provider.requests == []
