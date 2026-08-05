import csv
import json
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Self, cast

import pytest

from quantforge.backtesting import (
    BacktestConfig,
    BacktestResult,
    BasisPointSlippage,
    ExplicitZeroFees,
    FixedCommission,
    run_backtest,
)
from quantforge.data import MarketDataset, ValidationError
from quantforge.indicators import MarketField
from quantforge.optimization import (
    CombinationLimitExceededError,
    ExecutionConfig,
    ExecutionMode,
    FilePersistenceConfig,
    GridSearchConfig,
    GridSearchStudy,
    InvalidStudyConfigurationError,
    MetricName,
    MovingAverageCrossoverFactory,
    ParameterLessThan,
    ParameterSearchSpace,
    RankingConfig,
    StabilityConfig,
    StudyPersistenceError,
    StudyResult,
    TrialStatus,
)
from quantforge.optimization.errors import InvalidTrialTransitionError
from quantforge.optimization.spaces import IntegerValues
from quantforge.strategies import Strategy

from ..helpers import make_dataset


def _backtest_config(commission: str = "1") -> BacktestConfig:
    return BacktestConfig(
        Decimal("100000"),
        FixedCommission(Decimal(commission)),
        ExplicitZeroFees(),
        BasisPointSlippage(Decimal("5")),
    )


def _study_config(
    output_root: Path,
    *,
    execution: ExecutionConfig = ExecutionConfig(),
    commission: str = "1",
    fast_values: tuple[int, ...] = (2, 3),
    ranking: RankingConfig | None = None,
    stability: StabilityConfig | None = None,
    maximum_combinations: int = 100,
) -> GridSearchConfig:
    return GridSearchConfig(
        label="QF-6 deterministic SPY grid",
        search_space=ParameterSearchSpace(
            {
                "slow_window": IntegerValues([3, 4]),
                "fast_window": IntegerValues(fast_values),
            }
        ),
        parameter_constraints=(ParameterLessThan("fast_window", "slow_window"),),
        backtest=_backtest_config(commission),
        execution=execution,
        ranking=(
            RankingConfig(MetricName.TOTAL_RETURN) if ranking is None else ranking
        ),
        stability=(
            StabilityConfig(minimum_eligible_neighbors=1)
            if stability is None
            else stability
        ),
        persistence=FilePersistenceConfig(output_root),
        maximum_combinations=maximum_combinations,
    )


def _dataset(seed: str = "study") -> MarketDataset:
    return make_dataset(
        ("100", "99", "98", "99", "101", "103", "102", "99", "97"),
        dataset_id=seed,
    )


def _scientific_result(result: StudyResult) -> tuple[object, ...]:
    return (
        [
            (
                trial.combination_index,
                trial.status,
                trial.parameters,
                trial.qf5_run_id,
                trial.metrics,
            )
            for trial in result.trials
        ],
        result.rankings,
        result.ineligible_trials,
        result.stability,
        result.parameter_summaries,
    )


def test_study_identity_covers_scientific_inputs_and_is_equivalent(
    tmp_path: Path,
) -> None:
    baseline = GridSearchStudy(
        _dataset("same"),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path / "a"),
    )
    equivalent = GridSearchStudy(
        _dataset("same"),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path / "b"),
    )
    assert baseline.study_id == equivalent.study_id
    assert [item.combination_id for item in baseline.candidates] == [
        item.combination_id for item in equivalent.candidates
    ]

    variants = (
        GridSearchStudy(
            _dataset("different"),
            MovingAverageCrossoverFactory(),
            _study_config(tmp_path / "dataset"),
        ),
        GridSearchStudy(
            _dataset("same"),
            MovingAverageCrossoverFactory(),
            _study_config(tmp_path / "space", fast_values=(1, 2)),
        ),
        GridSearchStudy(
            _dataset("same"),
            MovingAverageCrossoverFactory(),
            _study_config(tmp_path / "cost", commission="2"),
        ),
        GridSearchStudy(
            _dataset("same"),
            MovingAverageCrossoverFactory(),
            _study_config(
                tmp_path / "rank",
                ranking=RankingConfig(MetricName.MAXIMUM_DRAWDOWN),
            ),
        ),
        GridSearchStudy(
            _dataset("same"),
            MovingAverageCrossoverFactory(),
            _study_config(
                tmp_path / "stability",
                stability=StabilityConfig(
                    minimum_eligible_neighbors=1,
                    isolated_peak_relative_drop=Decimal("0.5"),
                ),
            ),
        ),
    )
    assert all(item.study_id != baseline.study_id for item in variants)


@pytest.mark.parametrize(
    "factory",
    [
        MovingAverageCrossoverFactory(default_source_field=MarketField.OPEN),
        MovingAverageCrossoverFactory(default_target_long_weight=Decimal("0.5")),
    ],
)
def test_factory_defaults_are_recorded_in_study_and_trial_identities(
    tmp_path: Path,
    factory: MovingAverageCrossoverFactory,
) -> None:
    baseline_factory = MovingAverageCrossoverFactory()
    baseline = GridSearchStudy(
        _dataset("factory-defaults"),
        baseline_factory,
        _study_config(tmp_path / "baseline", fast_values=(2,)),
    )
    changed = GridSearchStudy(
        _dataset("factory-defaults"),
        factory,
        _study_config(tmp_path / "changed", fast_values=(2,)),
    )

    assert baseline_factory.configuration()["default_parameters"] == {
        "source_field": "close",
        "target_long_weight": "1",
    }
    assert (
        factory.configuration()["default_parameters"]
        != (baseline_factory.configuration()["default_parameters"])
    )
    assert baseline.study_id != changed.study_id
    assert {candidate.combination_id for candidate in baseline.candidates}.isdisjoint(
        candidate.combination_id for candidate in changed.candidates
    )

    baseline_result = baseline.run()
    changed_result = changed.run()

    assert {trial.trial_id for trial in baseline_result.trials}.isdisjoint(
        trial.trial_id for trial in changed_result.trials
    )


def test_sequential_execution_persists_failures_and_resume_skips_completed(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def selective_runner(
        dataset: MarketDataset,
        strategy: Strategy,
        config: BacktestConfig,
    ) -> BacktestResult:
        parameters = strategy.parameters.to_primitive()
        calls.append(strategy.configuration_id)
        if parameters["fast_window"] == 3:
            raise RuntimeError("synthetic sanitized trial failure")
        return run_backtest(dataset, strategy, config)

    study = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path),
        backtest_runner=selective_runner,
    )
    result = study.run()
    calls_after_run = len(calls)

    assert len(result.successful_trials) == 2
    assert len(result.failed_trials) == 1
    assert len(result.excluded_trials) == 1
    failure = result.failed_trials[0]
    assert failure.failure_category == "unexpected_implementation_failure"
    assert failure.failure_type == "RuntimeError"
    assert failure.failure_message == "synthetic sanitized trial failure"
    assert all(trial.qf5_run_id for trial in result.successful_trials)
    assert all(trial.metrics is not None for trial in result.successful_trials)

    resumed = study.resume()
    assert len(calls) == calls_after_run
    assert _scientific_result(resumed) == _scientific_result(result)


def test_study_validates_dataset_bars_before_trusting_persisted_trials(
    tmp_path: Path,
) -> None:
    dataset = _dataset("resume-dataset-validation")
    original = GridSearchStudy(
        dataset,
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path, fast_values=(2,)),
    )
    original.run()
    changed_first_bar = replace(
        dataset.bars[0],
        open=Decimal("101"),
        high=Decimal("102"),
        close=Decimal("101"),
    )
    stale_dataset = MarketDataset(
        (changed_first_bar, *dataset.bars[1:]),
        dataset.metadata,
    )

    with pytest.raises(ValidationError, match="dataset identity"):
        GridSearchStudy(
            stale_dataset,
            MovingAverageCrossoverFactory(),
            _study_config(tmp_path, fast_values=(2,)),
        ).resume()


@pytest.mark.parametrize(
    ("mismatch", "expected_message"),
    [
        ("market_data", "market data does not match"),
        ("backtest_configuration", "backtest configuration does not match"),
    ],
)
def test_success_result_must_match_study_data_and_backtest_configuration(
    tmp_path: Path,
    mismatch: str,
    expected_message: str,
) -> None:
    alternate_dataset = _dataset("runner-used-other-dataset")
    alternate_backtest = _backtest_config("2")

    def mismatched_runner(
        dataset: MarketDataset,
        strategy: Strategy,
        config: BacktestConfig,
    ) -> BacktestResult:
        if mismatch == "market_data":
            return run_backtest(alternate_dataset, strategy, config)
        return run_backtest(dataset, strategy, alternate_backtest)

    study = GridSearchStudy(
        _dataset("expected-runner-inputs"),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path, fast_values=(2,)),
        backtest_runner=mismatched_runner,
    )

    result = study.run()

    assert not result.successful_trials
    assert len(result.failed_trials) == 2
    assert all(
        trial.failure_category == "persistence_failure"
        and trial.failure_message is not None
        and expected_message in trial.failure_message
        for trial in result.failed_trials
    )
    assert not (study.study_path / "backtests").exists()


def test_success_result_must_match_candidate_provenance_before_export(
    tmp_path: Path,
) -> None:
    factory = MovingAverageCrossoverFactory()

    def mismatched_runner(
        dataset: MarketDataset,
        strategy: Strategy,
        config: BacktestConfig,
    ) -> BacktestResult:
        wrong_parameters = strategy.parameters.to_primitive()
        wrong_parameters["slow_window"] = (
            4 if wrong_parameters["slow_window"] == 3 else 3
        )
        return run_backtest(dataset, factory.build(wrong_parameters), config)

    study = GridSearchStudy(
        _dataset("mismatched-result"),
        factory,
        _study_config(tmp_path, fast_values=(2,)),
        backtest_runner=mismatched_runner,
    )

    result = study.run()

    assert not result.successful_trials
    assert len(result.failed_trials) == 2
    assert all(
        trial.failure_category == "persistence_failure"
        and trial.failure_message is not None
        and "configuration does not match" in trial.failure_message
        for trial in result.failed_trials
    )
    assert not (study.study_path / "backtests").exists()


def test_failed_trial_retry_policy_is_explicit(tmp_path: Path) -> None:
    calls_by_strategy: dict[str, int] = {}

    def retrying_runner(
        dataset: MarketDataset,
        strategy: Strategy,
        config: BacktestConfig,
    ) -> BacktestResult:
        attempt = calls_by_strategy.get(strategy.configuration_id, 0) + 1
        calls_by_strategy[strategy.configuration_id] = attempt
        if attempt == 1:
            raise RuntimeError("first attempt fails")
        return run_backtest(dataset, strategy, config)

    study = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        _study_config(
            tmp_path,
            execution=ExecutionConfig(retry_failed=True),
            fast_values=(2,),
        ),
        backtest_runner=retrying_runner,
    )
    first = study.run()
    invalid_retry = replace(
        first.failed_trials[0],
        status=TrialStatus.RUNNING,
        failure_category=None,
        failure_type=None,
        failure_message=None,
        finished_at=None,
    )
    with pytest.raises(InvalidTrialTransitionError, match="failed-attempt history"):
        study.store.write_trial(invalid_retry)

    def interrupting_runner(
        dataset: MarketDataset,
        strategy: Strategy,
        config: BacktestConfig,
    ) -> BacktestResult:
        raise KeyboardInterrupt

    interrupted_study = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        _study_config(
            tmp_path,
            execution=ExecutionConfig(retry_failed=True),
            fast_values=(2,),
        ),
        backtest_runner=interrupting_runner,
    )
    with pytest.raises(KeyboardInterrupt):
        interrupted_study.resume()
    interrupted = tuple(
        trial
        for trial in interrupted_study.store.load_trials()
        if trial.status is TrialStatus.RUNNING
    )

    assert len(interrupted) == 1
    assert interrupted[0].failed_attempts[0].failure_message == "first attempt fails"

    second = study.resume()

    assert len(first.failed_trials) == 2
    assert len(second.successful_trials) == 2
    assert not second.failed_trials
    assert set(calls_by_strategy.values()) == {2}
    for trial in second.successful_trials:
        assert trial.failure_category is None
        assert len(trial.failed_attempts) == 1
        failed_attempt = trial.failed_attempts[0]
        assert failed_attempt.attempt_number == 1
        assert failed_attempt.failure_category == "unexpected_implementation_failure"
        assert failed_attempt.failure_type == "RuntimeError"
        assert failed_attempt.failure_message == "first attempt fails"
        assert failed_attempt.started_at is not None
        assert failed_attempt.finished_at is not None
    with (study.study_path / "trials.csv").open(encoding="utf-8", newline="") as stream:
        trial_rows = tuple(csv.DictReader(stream))
    assert all(
        json.loads(row["failed_attempts"])[0]["failure_message"]
        == "first attempt fails"
        for row in trial_rows
        if row["status"] == TrialStatus.SUCCEEDED.value
    )


def test_process_and_sequential_execution_have_equivalent_final_results(
    tmp_path: Path,
) -> None:
    sequential_study = GridSearchStudy(
        _dataset("parallel"),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path / "sequential"),
    )
    process_study = GridSearchStudy(
        _dataset("parallel"),
        MovingAverageCrossoverFactory(),
        _study_config(
            tmp_path / "process",
            execution=ExecutionConfig(
                mode=ExecutionMode.PROCESS,
                maximum_workers=2,
            ),
        ),
    )

    sequential = sequential_study.run()
    parallel = process_study.run()

    assert sequential_study.study_id == process_study.study_id
    assert _scientific_result(sequential) == _scientific_result(parallel)


def test_fail_fast_inspects_completed_batch_before_process_rescheduling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset("fail-fast-batch")
    factory = MovingAverageCrossoverFactory()
    successful_result = run_backtest(
        dataset,
        factory.build({"fast_window": 1, "slow_window": 3}),
        _backtest_config(),
    )
    submissions: list[tuple[object, ...]] = []

    class CompletedBatchExecutor:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def submit(self, function: object, *args: object) -> Future[BacktestResult]:
            submissions.append(args)
            future: Future[BacktestResult] = Future()
            if len(submissions) == 1:
                future.set_result(successful_result)
            else:
                future.set_exception(RuntimeError("synthetic batch failure"))
            return future

    monkeypatch.setattr(
        "quantforge.optimization.study.ProcessPoolExecutor",
        CompletedBatchExecutor,
    )
    study = GridSearchStudy(
        dataset,
        factory,
        _study_config(
            tmp_path,
            execution=ExecutionConfig(
                mode=ExecutionMode.PROCESS,
                maximum_workers=2,
                fail_fast=True,
            ),
            fast_values=(1, 2),
        ),
    )

    result = study.run()

    assert len(submissions) == 2
    assert len(result.successful_trials) == 1
    assert len(result.failed_trials) == 1
    assert len(result.pending_trials) == 2


def test_broken_process_pool_stops_scheduling_without_stale_running_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submissions: list[tuple[object, ...]] = []

    class BrokenPoolExecutor:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def submit(self, function: object, *args: object) -> Future[BacktestResult]:
            submissions.append(args)
            future: Future[BacktestResult] = Future()
            future.set_exception(BrokenProcessPool("synthetic worker-pool failure"))
            return future

    monkeypatch.setattr(
        "quantforge.optimization.study.ProcessPoolExecutor",
        BrokenPoolExecutor,
    )
    study = GridSearchStudy(
        _dataset("broken-pool"),
        MovingAverageCrossoverFactory(),
        _study_config(
            tmp_path,
            execution=ExecutionConfig(
                mode=ExecutionMode.PROCESS,
                maximum_workers=1,
            ),
            fast_values=(2,),
        ),
    )

    result = study.run()

    assert len(submissions) == 1
    assert len(result.failed_trials) == 1
    assert result.failed_trials[0].failure_category == "worker_failure"
    assert len(result.pending_trials) == 1
    assert result.pending_trials[0].status is TrialStatus.PENDING
    assert all(trial.status is not TrialStatus.RUNNING for trial in result.trials)


def test_export_reconstructs_from_trials_and_rejects_corruption(tmp_path: Path) -> None:
    study = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path, fast_values=(2,)),
    )
    result = study.run()
    reconstructed = study.load_result()

    assert _scientific_result(result) == _scientific_result(reconstructed)
    expected = {
        "manifest.json",
        "trials.csv",
        "eligible_rankings.csv",
        "ineligible_trials.csv",
        "failures.csv",
        "exclusions.csv",
        "stability.csv",
        "parameter_summary.csv",
        "ranking.json",
        "stability.json",
        "summary.json",
    }
    assert expected.issubset({path.name for path in study.study_path.iterdir()})
    json.dumps(
        json.loads((study.study_path / "summary.json").read_text(encoding="utf-8")),
        allow_nan=False,
    )

    successful = result.successful_trials[0]
    study.store.trial_path(successful.trial_id).write_text("{broken", encoding="utf-8")
    with pytest.raises(StudyPersistenceError, match="failed to load"):
        study.load_result()


@pytest.mark.parametrize(
    "failure_field",
    ["failure_category", "failure_type", "failure_message"],
)
@pytest.mark.parametrize("action_name", ["load_result", "resume"])
def test_failed_trial_requires_complete_failure_context(
    tmp_path: Path,
    failure_field: str,
    action_name: str,
) -> None:
    def failing_runner(
        dataset: MarketDataset,
        strategy: Strategy,
        config: BacktestConfig,
    ) -> BacktestResult:
        raise RuntimeError("synthetic persisted failure")

    study = GridSearchStudy(
        _dataset("incomplete-failure-context"),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path, fast_values=(2,)),
        backtest_runner=failing_runner,
    )
    result = study.run()
    failed = result.failed_trials[0]
    trial_path = study.store.trial_path(failed.trial_id)
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    trial[failure_field] = None
    trial_path.write_text(json.dumps(trial), encoding="utf-8")

    action = study.load_result if action_name == "load_result" else study.resume
    with pytest.raises(
        StudyPersistenceError,
        match="incomplete failure context",
    ):
        action()


@pytest.mark.parametrize(
    "artifact_name",
    [
        "signals.csv",
        "orders.csv",
        "fills.csv",
        "positions.csv",
        "trades.csv",
        "equity.csv",
        "benchmark_equity.csv",
    ],
)
@pytest.mark.parametrize("action_name", ["load_result", "resume"])
def test_successful_trial_requires_complete_qf5_artifact(
    tmp_path: Path,
    artifact_name: str,
    action_name: str,
) -> None:
    study = GridSearchStudy(
        _dataset("incomplete-qf5-artifact"),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path, fast_values=(2,)),
    )
    result = study.run()
    successful = result.successful_trials[0]
    artifact_path = study.study_path / cast(str, successful.artifact_location)
    (artifact_path / artifact_name).unlink()

    action = study.load_result if action_name == "load_result" else study.resume
    with pytest.raises(
        StudyPersistenceError,
        match="incomplete QF-5 artifact",
    ):
        action()


@pytest.mark.parametrize("resume", [False, True])
def test_missing_manifest_rejects_a_nonempty_orphaned_study_directory(
    tmp_path: Path,
    resume: bool,
) -> None:
    study = GridSearchStudy(
        _dataset("orphaned-store"),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path, fast_values=(2,)),
    )
    orphaned_trial = study.store.trial_path("orphaned-trial")
    orphaned_trial.parent.mkdir(parents=True)
    orphaned_trial.write_text("{}\n", encoding="utf-8")

    action = study.resume if resume else study.run
    with pytest.raises(StudyPersistenceError, match="manifest is missing"):
        action()

    assert not study.store.manifest_path.exists()
    assert orphaned_trial.read_text(encoding="utf-8") == "{}\n"


@pytest.mark.parametrize("reuse_known_combination", [False, True])
def test_result_loading_rejects_trials_outside_the_candidate_set(
    tmp_path: Path,
    reuse_known_combination: bool,
) -> None:
    study = GridSearchStudy(
        _dataset("orphaned-trial"),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path, fast_values=(2,)),
    )
    result = study.run()
    source = result.successful_trials[0]
    orphan = replace(
        source,
        trial_id="f" * 64,
        combination_id=(source.combination_id if reuse_known_combination else "e" * 64),
    )
    study.store.write_trial(orphan)

    with pytest.raises(StudyPersistenceError, match="expected candidate set"):
        study.load_result()
    with pytest.raises(StudyPersistenceError, match="expected candidate set"):
        study.resume()


@pytest.mark.parametrize("tamper", ["candidate", "metrics"])
def test_expected_trial_files_must_match_candidate_and_qf5_artifact(
    tmp_path: Path,
    tamper: str,
) -> None:
    study = GridSearchStudy(
        _dataset("tampered-expected-trial"),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path, fast_values=(2,)),
    )
    result = study.run()
    successful = result.successful_trials[0]
    trial_path = study.store.trial_path(successful.trial_id)
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    if tamper == "candidate":
        trial["combination_id"] = "e" * 64
        expected_message = "expected candidate"
    else:
        trial["metrics"]["total_return"] = "999"
        expected_message = "linked QF-5 artifact"
    trial_path.write_text(json.dumps(trial), encoding="utf-8")

    with pytest.raises(StudyPersistenceError, match=expected_message):
        study.load_result()
    with pytest.raises(StudyPersistenceError, match=expected_message):
        study.resume()


def test_grid_limit_fails_before_execution_with_multiplicative_count(
    tmp_path: Path,
) -> None:
    with pytest.raises(CombinationLimitExceededError, match=r"2 x 2 = 4"):
        GridSearchStudy(
            _dataset(),
            MovingAverageCrossoverFactory(),
            _study_config(tmp_path, maximum_combinations=3),
        )


def test_unsupported_execution_mode_fails_during_configuration() -> None:
    with pytest.raises(InvalidStudyConfigurationError, match="execution mode"):
        ExecutionConfig(mode=cast(ExecutionMode, "threads"))


def test_changed_scientific_inputs_cannot_resume_old_study(tmp_path: Path) -> None:
    original = GridSearchStudy(
        _dataset("resume"),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path, fast_values=(2,)),
    )
    original.run()
    changed = GridSearchStudy(
        _dataset("resume"),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path, fast_values=(1,)),
    )

    with pytest.raises(StudyPersistenceError, match="does not exist"):
        changed.resume()
