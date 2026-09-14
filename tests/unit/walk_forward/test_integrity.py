"""Temporal leakage, QF-8 membership, fixed provenance, and failure cases."""

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from quantforge.backtesting import BacktestConfig, BacktestResult, run_backtest
from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.data import MarketDataset, TimeframeBarSeries, validate_market_dataset
from quantforge.optimization import (
    CategoricalValues,
    GridSearchConfig,
    GridSearchStudy,
    IntegerValues,
    MinimumTrades,
    ParameterSearchSpace,
)
from quantforge.optimization.factories import StrategyFactory
from quantforge.prediction import (
    NextSessionOpenGapOutcomeLabeler,
    PredictionRuleContext,
    PredictionStrategyOutput,
    PredictionStudy,
    PredictionTrialAnalysis,
    PredictionWindowResult,
)
from quantforge.strategies import Strategy
from quantforge.validation import (
    DatasetProvenance,
    PartitionRole,
    ValidationPlanError,
    purge_partition_observations,
)
from quantforge.walk_forward import (
    BacktestEvaluator,
    FoldStatus,
    SelectionPolicy,
    WalkForwardError,
    WalkForwardStudy,
)
from quantforge.walk_forward.partitions import (
    observation_keys,
    partition,
    project_dataset,
)
from tests.unit.helpers import SESSIONS, make_dataset
from tests.unit.prediction.test_prediction_window import WindowAnalyzer

from .fixtures import StudyFactory, backtest_fixture, prediction_fixture
from .test_resume import ProbeEvaluator


@pytest.mark.parametrize("selection", [False, True])
def test_backtest_grid_only_receives_qf8_permitted_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: bool,
) -> None:
    from quantforge.walk_forward import backtest as module

    config, adapter = backtest_fixture(tmp_path, selection=selection)
    calls: list[tuple[date, ...]] = []

    def observed(
        dataset: MarketDataset, strategy: Strategy, execution: BacktestConfig
    ) -> BacktestResult:
        validate_market_dataset(dataset)
        interval = execution.evaluation_interval
        assert interval is not None
        assert dataset.metadata.actual_last_session == interval.end_session
        assert dataset.metadata.requested_end == interval.end_session
        assert dataset.metadata.actual_first_session == dataset.bars[0].session_date
        assert all(bar.session_date <= interval.end_session for bar in dataset.bars)
        assert interval.end_session < SESSIONS[13]
        calls.append(tuple(bar.session_date for bar in dataset.bars))
        result = run_backtest(dataset, strategy, execution)
        assert all(row.session >= interval.start_session for row in result.daily_equity)
        assert all(
            row.session >= interval.start_session
            for row in result.benchmark.daily_equity
        )
        assert all(
            signal.decision.signal_session >= interval.start_session
            for signal in result.signals
        )
        return result

    class ObservedGrid(GridSearchStudy):
        def __init__(
            self,
            dataset: MarketDataset,
            strategy_factory: StrategyFactory,
            config: GridSearchConfig,
        ) -> None:
            super().__init__(
                dataset, strategy_factory, config, backtest_runner=observed
            )

    monkeypatch.setattr(module, "GridSearchStudy", ObservedGrid)
    monkeypatch.setattr(module, "run_backtest", observed)
    study = WalkForwardStudy(config, adapter, tmp_path)
    result = study.run()
    assert all(fold.status is FoldStatus.COMPLETED for fold in result.folds)
    expected_endpoints = {SESSIONS[6], SESSIONS[9], SESSIONS[12]}
    assert {sessions[-1] for sessions in calls} == expected_endpoints
    if selection:
        assert (
            calls[0][0] == SESSIONS[2]
        )  # Selection begins at 5, plus exactly 3 warm-up.
    else:
        assert calls[0][0] == SESSIONS[0]


@pytest.mark.parametrize("family", ["prediction", "backtest"])
def test_qf8_purging_embargo_and_holdout_membership_are_exact(
    tmp_path: Path, family: str
) -> None:
    config, adapter = (
        prediction_fixture(tmp_path)
        if family == "prediction"
        else backtest_fixture(tmp_path)
    )
    # Exercise stronger embargo with one-row eligibility so the tiny fixture fits.
    offset = config.plan.purge_policy.embargo
    config = replace(
        config,
        minimum_training_observations=1,
        minimum_test_observations=1,
        plan=replace(
            config.plan,
            purge_policy=replace(
                config.plan.purge_policy, embargo=replace(offset, exchange_sessions=1)
            ),
        ),
    )
    keys = observation_keys(adapter.dataset, config.plan)
    for index in range(2):
        for role in (PartitionRole.DEVELOPMENT, PartitionRole.WALK_FORWARD_TEST):
            actual = partition(
                adapter.dataset, config.plan, index, role, minimum_observations=1
            )
            expected = purge_partition_observations(
                config.plan, index, role, keys, source=adapter.dataset
            )
            assert actual.purge == expected
            assert all(
                key not in actual.purge.retained
                for key in actual.membership.warm_up_context
            )
            protected_start = (
                SESSIONS[7]
                if index == 0 and role is PartitionRole.DEVELOPMENT
                else (
                    SESSIONS[10]
                    if index == 0 or role is PartitionRole.DEVELOPMENT
                    else SESSIONS[13]
                )
            )
            assert actual.dataset.bars[-1].session_date < protected_start
    with pytest.raises(WalkForwardError, match="holdout"):
        partition(
            adapter.dataset,
            config.plan,
            0,
            PartitionRole.FINAL_HOLDOUT,
            minimum_observations=1,
        )


class GuardAnalyzer(WindowAnalyzer):
    seen_windows: ClassVar[list[tuple[date, ...]]] = []

    def analyze_window(
        self, result: PredictionWindowResult[Any, Any, Any]
    ) -> PredictionTrialAnalysis:
        sessions = result.schedule.decision_sessions
        # Only selection windows reach the analyzer, never either test window.
        assert sessions[-1] < (
            SESSIONS[7] if sessions[0] == SESSIONS[4] else SESSIONS[10]
        )
        assert all(decision.result.rows for decision in result.decisions)
        assert len(result.decisions) >= 4
        self.seen_windows.append(sessions)
        return super().analyze_window(result)


def test_qf32_analyzes_all_permitted_decisions_and_never_test(tmp_path: Path) -> None:
    GuardAnalyzer.seen_windows = []
    config, adapter = prediction_fixture(tmp_path, analyzer=GuardAnalyzer())
    result = WalkForwardStudy(config, adapter, tmp_path).run()
    assert all(fold.status is FoldStatus.COMPLETED for fold in result.folds)
    # Two candidates per fold. Four and eight decisions respectively after purge.
    assert [len(sessions) for sessions in GuardAnalyzer.seen_windows] == [4, 4, 8, 8]


class LeakingLabeler(NextSessionOpenGapOutcomeLabeler):
    def validate_dataset(self, dataset: MarketDataset) -> None:
        super().validate_dataset(dataset)
        # Deliberately demand information from beyond every selection boundary.
        assert any(bar.session_date == SESSIONS[13] for bar in dataset.bars), (
            "protected data unavailable"
        )


class LeakingFactory(StudyFactory):
    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        study = super().build(parameters)
        return PredictionStudy[Any, Any, Any].create(
            study.strategy, LeakingLabeler(), study.evaluator
        )


def test_injecting_protected_prices_into_prediction_selection_fails(
    tmp_path: Path,
) -> None:
    config, adapter = prediction_fixture(tmp_path, factory=LeakingFactory())
    probe = ProbeEvaluator(adapter)
    result = WalkForwardStudy(config, probe, tmp_path).run()
    assert all(fold.status is FoldStatus.FAILED for fold in result.folds)
    assert not probe.test_calls


def test_backtest_attempt_to_read_protected_test_price_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantforge.walk_forward import backtest as module

    config, adapter = backtest_fixture(tmp_path)

    def leaking(
        dataset: MarketDataset, strategy: Strategy, execution: BacktestConfig
    ) -> BacktestResult:
        assert any(bar.session_date == SESSIONS[13] for bar in dataset.bars), (
            "test price inaccessible"
        )
        return run_backtest(dataset, strategy, execution)

    class LeakingGrid(GridSearchStudy):
        def __init__(
            self,
            dataset: MarketDataset,
            strategy_factory: StrategyFactory,
            config: GridSearchConfig,
        ) -> None:
            super().__init__(dataset, strategy_factory, config, backtest_runner=leaking)

    monkeypatch.setattr(module, "GridSearchStudy", LeakingGrid)
    probe = ProbeEvaluator(adapter)
    result = WalkForwardStudy(config, probe, tmp_path).run()
    assert all(fold.status is FoldStatus.FAILED for fold in result.folds)
    assert not probe.test_calls


def test_changed_test_prices_do_not_change_current_fold_selection(
    tmp_path: Path,
) -> None:
    config, adapter = backtest_fixture(tmp_path)
    original = WalkForwardStudy(config, adapter, tmp_path / "original").run()
    closes = tuple(
        str(bar.close) if index < 7 else str(1000 + index)
        for index, bar in enumerate(adapter.dataset.bars)
    )
    changed = make_dataset(closes)
    environment = replace(
        config.plan.environment, dataset=DatasetProvenance.from_market_dataset(changed)
    )
    revised = replace(config, plan=replace(config.plan, environment=environment))
    result = WalkForwardStudy(
        revised,
        BacktestEvaluator(changed, adapter.factory, adapter.grid_config),
        tmp_path / "changed",
    ).run()
    assert original.study_id != result.study_id
    first, second = original.folds[0].selection, result.folds[0].selection
    assert first is not None
    assert second is not None
    assert (
        first.snapshot.to_primitive()["candidate"]
        == second.snapshot.to_primitive()["candidate"]
    )
    assert (
        first.snapshot.to_primitive()["selection_grid_study_id"]
        == second.snapshot.to_primitive()["selection_grid_study_id"]
    )


def test_appending_available_future_context_preserves_historical_freezes(
    tmp_path: Path,
) -> None:
    config, adapter = prediction_fixture(tmp_path)
    original = WalkForwardStudy(config, adapter, tmp_path / "first").run()
    # Same immutable artifact provenance, with only a shorter available prefix.
    adapter.series = tuple(
        TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
            series.dataset_reference,
            series.timeframe,
            tuple(
                bar for bar in series.bars if bar.end_timestamp.date() <= SESSIONS[12]
            ),
            dataset_family_manifest_id=series.dataset_family_manifest_id,
        )
        for series in adapter.series
    )
    truncated = WalkForwardStudy(config, adapter, tmp_path / "prefix").run()
    assert truncated == original


@pytest.mark.parametrize(
    "change",
    ["dataset", "candidate", "ranking", "execution", "validation", "selection_policy"],
)
def test_material_changes_create_distinct_study_identities(
    tmp_path: Path, change: str
) -> None:
    config, adapter = backtest_fixture(tmp_path)
    original = WalkForwardStudy(config, adapter, tmp_path).study_id
    if change == "dataset":
        dataset = make_dataset(
            tuple(str(bar.close) for bar in adapter.dataset.bars),
            dataset_id="new-source",
        )
        config = replace(
            config,
            plan=replace(
                config.plan,
                environment=replace(
                    config.plan.environment,
                    dataset=DatasetProvenance.from_market_dataset(dataset),
                ),
            ),
        )
        adapter = BacktestEvaluator(dataset, adapter.factory, adapter.grid_config)
    elif change in {"candidate", "ranking", "execution"}:
        grid = adapter.grid_config
        if change == "candidate":
            grid = replace(
                grid,
                search_space=ParameterSearchSpace(
                    {
                        "fast_window": IntegerValues([2]),
                        "slow_window": IntegerValues([3, 4]),
                    }
                ),
            )
        elif change == "ranking":
            grid = replace(
                grid,
                ranking=replace(grid.ranking, hard_constraints=(MinimumTrades(999),)),
            )
        else:
            from quantforge.validation import BacktestProvenance

            grid = replace(
                grid, backtest=replace(grid.backtest, initial_capital=Decimal("200000"))
            )
            config = replace(
                config,
                plan=replace(
                    config.plan,
                    environment=replace(
                        config.plan.environment,
                        execution=BacktestProvenance.capture(grid.backtest),
                    ),
                ),
            )
        adapter = BacktestEvaluator(adapter.dataset, adapter.factory, grid)
    elif change == "validation":
        config = replace(config, plan=replace(config.plan, name="different-validation"))
    else:
        config = replace(config, selection_policy=SelectionPolicy.FIRST_STABLE)
    assert WalkForwardStudy(config, adapter, tmp_path).study_id != original


def test_candidate_identity_ignores_mapping_insertion_order_and_output_path(
    tmp_path: Path,
) -> None:
    _, adapter = backtest_fixture(tmp_path)
    grid = replace(
        adapter.grid_config,
        search_space=ParameterSearchSpace(
            {"fast_window": IntegerValues([2, 3]), "slow_window": IntegerValues([3, 4])}
        ),
    )
    other = BacktestEvaluator(adapter.dataset, adapter.factory, grid)
    assert adapter.universe == other.universe


def test_backend_is_not_a_search_axis(tmp_path: Path) -> None:
    _, adapter = backtest_fixture(tmp_path)
    grid = replace(
        adapter.grid_config,
        search_space=ParameterSearchSpace(
            {"backend_id": CategoricalValues(["native_v1", "talib_v1"])}
        ),
    )
    with pytest.raises(WalkForwardError, match="backend"):
        BacktestEvaluator(adapter.dataset, adapter.factory, grid)


@pytest.mark.parametrize(
    "minimum", ["minimum_training_observations", "minimum_test_observations"]
)
def test_insufficient_observations_fail_before_selection(
    tmp_path: Path, minimum: str
) -> None:
    config, adapter = backtest_fixture(tmp_path)
    config = replace(config, **{minimum: 99})
    probe = ProbeEvaluator(adapter)
    result = WalkForwardStudy(config, probe, tmp_path).run()
    assert all(fold.status is FoldStatus.FAILED for fold in result.folds)
    assert not probe.selection_calls
    assert not probe.test_calls


def test_no_eligible_candidate_has_no_favorable_fallback(tmp_path: Path) -> None:
    config, adapter = backtest_fixture(tmp_path)
    grid = replace(
        adapter.grid_config,
        ranking=replace(
            adapter.grid_config.ranking, hard_constraints=(MinimumTrades(999),)
        ),
    )
    probe = ProbeEvaluator(BacktestEvaluator(adapter.dataset, adapter.factory, grid))
    result = WalkForwardStudy(config, probe, tmp_path).run()
    assert all(
        fold.status is FoldStatus.FAILED and fold.selection is None
        for fold in result.folds
    )
    assert not probe.test_calls


def test_declared_warmup_cannot_be_smaller_than_a_candidate(tmp_path: Path) -> None:
    config, adapter = backtest_fixture(tmp_path)
    grid = replace(
        adapter.grid_config,
        search_space=ParameterSearchSpace(
            {"fast_window": IntegerValues([2]), "slow_window": IntegerValues([10])}
        ),
    )
    with pytest.raises(ValidationPlanError, match="indicator"):
        WalkForwardStudy(
            config, BacktestEvaluator(adapter.dataset, adapter.factory, grid), tmp_path
        )


def test_bounded_dataset_hides_future_metadata_and_corporate_actions(
    tmp_path: Path,
) -> None:
    del tmp_path
    dataset = make_dataset(
        tuple("100" for _ in SESSIONS), dividends=((SESSIONS[12], "2"),)
    )
    bounded = project_dataset(dataset, SESSIONS[3], SESSIONS[6])
    validate_market_dataset(bounded)
    assert not bounded.corporate_actions
    assert bounded.metadata.dividend_count == 0
    assert bounded.metadata.dividend_sessions == ()
    assert bounded.metadata.requested_start == SESSIONS[3]
    assert bounded.metadata.requested_end == SESSIONS[6]
    assert bounded.metadata.retrieved_at.year == 1970
    assert bounded.bars == dataset.bars[3:7]


class EmptyTestFactory(StudyFactory):
    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        from .fixtures import StudyRule

        class EmptyTestRule(StudyRule):
            def configuration(self) -> PrimitiveMapping:
                return {
                    **super().configuration(),
                    "abstain_from": SESSIONS[7].isoformat(),
                }

            def generate_with_context(
                self, context: PredictionRuleContext
            ) -> PredictionStrategyOutput:
                output = super().generate_with_context(context)
                if (
                    context.latest_bar_for(
                        self.context_requirements.primary.timeframe
                    ).end_timestamp.date()
                    >= SESSIONS[7]
                ):
                    return replace(output, signals=())
                return output

        study = super().build(parameters)
        return PredictionStudy[Any, Any, Any].create(
            EmptyTestRule(getattr(study.strategy, "context_requirements")),
            study.outcome_labeler,
            study.evaluator,
        )


def test_prediction_test_without_predictions_fails_after_freeze(tmp_path: Path) -> None:
    config, adapter = prediction_fixture(tmp_path, factory=EmptyTestFactory())
    result = WalkForwardStudy(config, adapter, tmp_path).run()
    assert all(fold.status is FoldStatus.FAILED for fold in result.folds)
    assert all(fold.selection is not None for fold in result.folds)
    assert all(
        fold.failures[-1].to_primitive()["stage"] == "test" for fold in result.folds
    )
    assert all(
        "no valid predictions" in str(fold.failures[-1].to_primitive()["message"])
        for fold in result.folds
    )


def test_changed_prediction_backend_environment_changes_identity(
    tmp_path: Path,
) -> None:
    config, adapter = prediction_fixture(tmp_path)
    before = WalkForwardStudy(config, adapter, tmp_path).study_id
    from quantforge.walk_forward import PredictionEvaluator

    backend = replace(
        adapter.backend,
        configuration_snapshot=PrimitiveMappingSnapshot.capture(
            {"policy_revision": "2"}
        ),
    )
    changed = PredictionEvaluator(
        dataset=adapter.dataset,
        series=adapter.series,
        primary_timeframe=adapter.primary_timeframe,
        study_factory=adapter.factory,
        analyzer=adapter.analyzer,
        indicator_backend=backend,
        grid_config=adapter.grid_config,
    )
    assert WalkForwardStudy(config, changed, tmp_path).study_id != before
    assert changed.universe.universe_id != adapter.universe.universe_id


def test_changed_prediction_outcome_cannot_reuse_plan(tmp_path: Path) -> None:
    class OtherHorizon(NextSessionOpenGapOutcomeLabeler):
        required_future_sessions = 2

    class OtherFactory(StudyFactory):
        def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
            study = super().build(parameters)
            return PredictionStudy[Any, Any, Any].create(
                study.strategy, OtherHorizon(), study.evaluator
            )

    from quantforge.walk_forward import PredictionEvaluator

    config, adapter = prediction_fixture(tmp_path)
    changed = PredictionEvaluator(
        dataset=adapter.dataset,
        series=adapter.series,
        primary_timeframe=adapter.primary_timeframe,
        study_factory=OtherFactory(),
        analyzer=adapter.analyzer,
        indicator_backend=adapter.backend,
        grid_config=adapter.grid_config,
    )
    assert changed.universe.universe_id != adapter.universe.universe_id
    with pytest.raises(WalkForwardError, match="outcome"):
        WalkForwardStudy(config, changed, tmp_path)


def test_state_freeze_is_deeply_immutable(tmp_path: Path) -> None:
    config, adapter = backtest_fixture(tmp_path)
    result = WalkForwardStudy(config, adapter, tmp_path).run()
    frozen = result.folds[0].selection
    assert frozen is not None
    identity = frozen.selection_id
    detached = frozen.snapshot.to_primitive()
    cast(PrimitiveMapping, detached["candidate"])["parameters"] = {
        "injected": "test-return"
    }
    assert frozen.selection_id == identity
    assert frozen.snapshot.to_primitive() != detached
