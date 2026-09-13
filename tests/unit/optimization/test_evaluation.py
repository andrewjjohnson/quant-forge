"""QF-43 uses the existing QF-6 execution, identity, and persistence path."""

import hashlib
import json
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import (
    BacktestConfig,
    BacktestResult,
    EvaluationInterval,
    run_backtest,
)
from quantforge.configuration import PrimitiveMapping
from quantforge.data import MarketDataset
from quantforge.indicators import SimpleMovingAverage, SimpleMovingAverageParameters
from quantforge.indicators.backends import (
    IndicatorBackendIdentity,
    NativeIndicatorBackend,
    StandardIndicatorDefinition,
)
from quantforge.optimization import (
    ExecutionConfig,
    ExecutionMode,
    GridSearchConfig,
    GridSearchStudy,
    MovingAverageCrossoverFactory,
    StudyPersistenceError,
    TrialStatus,
)
from quantforge.optimization.persistence import FileStudyStore
from quantforge.strategies import (
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
    Strategy,
)

from ..helpers import SESSIONS, make_dataset
from .test_study import (
    _dataset,  # pyright: ignore[reportPrivateUsage]
    _scientific_result,  # pyright: ignore[reportPrivateUsage]
    _study_config,  # pyright: ignore[reportPrivateUsage]
)


def bounded_config(output_root: Path) -> GridSearchConfig:
    baseline = _study_config(output_root, fast_values=(2,))
    return replace(
        baseline,
        backtest=replace(
            baseline.backtest,
            evaluation_interval=EvaluationInterval(SESSIONS[4], SESSIONS[8]),
        ),
    )


def test_default_study_and_trial_identities_remain_pre_qf43(tmp_path: Path) -> None:
    # Captured from main e4cdbdd before changing any production code.
    study = GridSearchStudy(
        _dataset(), MovingAverageCrossoverFactory(), _study_config(tmp_path)
    )
    assert (
        study.study_id
        == "c880f7c9400a3e8b33fc00e00d8ddc615d0fbabefc3e4276d8472b7217e2fbb4"
    )
    assert [trial.trial_id for trial in study.run().trials] == [
        "77ebff15f25907f1feee30a8fb2a8ac1b96761cfa152b4bb8d8e1a545392ba55",
        "eb6e1d1ff1b94f7dd0e3824d8aff742a71aabe2642ed52d72da4e01b57dbb5d8",
        "bb6614dbe45cabca4fbadf27363c37e33385e3ed440c698c628d9b57c85115fa",
        "60b767cbce0a2e7c5d128038915f9dc13b05d933f2fc29caab2db0bbb4022163",
    ]


def test_bounded_trials_resume_without_calls_and_match_process_execution(
    tmp_path: Path,
) -> None:
    calls = 0

    def counted(
        dataset: MarketDataset, strategy: Strategy, config: BacktestConfig
    ) -> BacktestResult:
        nonlocal calls
        calls += 1
        assert config.evaluation_interval is not None
        result = run_backtest(dataset, strategy, config)
        assert [row.session for row in result.daily_equity] == list(SESSIONS[4:9])
        assert result.daily_equity[0].cash == config.initial_capital
        assert result.daily_equity[0].shares == 0
        return result

    sequential = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        bounded_config(tmp_path / "sequential"),
        backtest_runner=counted,
    )
    result = sequential.run()
    assert len(result.successful_trials) == calls == 2
    for trial in result.trials:
        assert (
            trial.backtest_configuration["evaluation_interval"]
            == EvaluationInterval(SESSIONS[4], SESSIONS[8]).to_primitive()
        )
    assert _scientific_result(sequential.resume()) == _scientific_result(result)
    assert calls == 2
    processes = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        replace(
            bounded_config(tmp_path / "processes"),
            execution=ExecutionConfig(mode=ExecutionMode.PROCESS, maximum_workers=2),
        ),
    )
    parallel = processes.run()
    assert parallel.study_id == result.study_id
    assert _scientific_result(parallel) == _scientific_result(result)


@pytest.mark.parametrize(
    "change", ["start", "end", "default_policy", "capital", "source", "context"]
)
def test_identity_and_resume_reject_incompatible_boundaries_and_provenance(
    tmp_path: Path, change: str
) -> None:
    config = bounded_config(tmp_path)
    dataset = _dataset()
    original = GridSearchStudy(dataset, MovingAverageCrossoverFactory(), config)
    original_result = original.run()
    if change == "start":
        config = replace(
            config,
            backtest=replace(
                config.backtest,
                evaluation_interval=EvaluationInterval(SESSIONS[3], SESSIONS[8]),
            ),
        )
    elif change == "end":
        config = replace(
            config,
            backtest=replace(
                config.backtest,
                evaluation_interval=EvaluationInterval(SESSIONS[4], SESSIONS[7]),
            ),
        )
    elif change == "default_policy":
        config = replace(
            config, backtest=replace(config.backtest, evaluation_interval=None)
        )
    elif change == "capital":
        config = replace(
            config, backtest=replace(config.backtest, initial_capital=Decimal(123456))
        )
    elif change == "source":
        dataset = _dataset("different-source-snapshot")
    else:
        dataset = make_dataset(
            ("99", "98", "99", "101", "103", "102", "99", "97"),
            sessions=SESSIONS[1:9],
            dataset_id="study",
        )
    incompatible = GridSearchStudy(dataset, MovingAverageCrossoverFactory(), config)
    assert incompatible.study_id != original.study_id
    with pytest.raises(StudyPersistenceError, match="manifest does not exist"):
        incompatible.resume()
    # Even forcing the physical old store cannot bypass exact manifest validation.
    incompatible.store = FileStudyStore(tmp_path, original.study_id)
    with pytest.raises(StudyPersistenceError, match="incompatible"):
        incompatible.resume()
    fresh = GridSearchStudy(dataset, MovingAverageCrossoverFactory(), config).run()
    assert {trial.trial_id for trial in fresh.trials}.isdisjoint(
        trial.trial_id for trial in original_result.trials
    )


def test_trial_runner_cannot_return_results_without_requested_boundary(
    tmp_path: Path,
) -> None:
    def ignores_boundary(
        dataset: MarketDataset, strategy: Strategy, config: BacktestConfig
    ) -> BacktestResult:
        return run_backtest(
            dataset, strategy, replace(config, evaluation_interval=None)
        )

    study = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        bounded_config(tmp_path),
        backtest_runner=ignores_boundary,
    )
    result = study.run()
    assert all(trial.status is TrialStatus.FAILED for trial in result.trials)
    assert all(
        trial.failure_category == "persistence_failure" for trial in result.trials
    )
    assert not (study.study_path / "backtests").exists()


@pytest.mark.parametrize("level", ["trial", "artifact"])
@pytest.mark.parametrize(
    "field",
    [
        "start_session",
        "end_session",
        "account_initialization",
        "contract_version",
        "strategy_metadata",
    ],
)
def test_resume_rejects_modified_boundary_in_completed_artifacts(
    tmp_path: Path, level: str, field: str
) -> None:
    study = GridSearchStudy(
        _dataset(), MovingAverageCrossoverFactory(), bounded_config(tmp_path)
    )
    trial = study.run().successful_trials[0]
    if level == "trial":
        path = study.study_path / "trials" / f"{trial.trial_id}.json"
    else:
        path = study.study_path / cast(str, trial.artifact_location) / "manifest.json"
    payload = json.loads(path.read_text())
    payload["backtest_configuration"]["evaluation_interval"][field] = "incompatible"
    path.write_text(json.dumps(payload))
    if level == "artifact":
        # Rehash to prove QF-6's scientific comparison catches incompatible policy,
        # independently of QF-5's file-integrity protection.
        integrity_path = path.parent / "integrity.json"
        integrity = json.loads(integrity_path.read_text())
        integrity["files"]["manifest.json"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        integrity_path.write_text(json.dumps(integrity))
    with pytest.raises(StudyPersistenceError, match=r"does not match|do not match"):
        study.resume()


def test_previous_metadata_policy_cannot_resume_as_causal_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_serialization = EvaluationInterval.to_primitive

    def previous_serialization(interval: EvaluationInterval) -> PrimitiveMapping:
        configuration = current_serialization(interval)
        configuration["contract_version"] = "1"
        del configuration["strategy_metadata"]
        return configuration

    config = bounded_config(tmp_path)
    with monkeypatch.context() as previous:
        previous.setattr(EvaluationInterval, "to_primitive", previous_serialization)
        old_study = GridSearchStudy(_dataset(), MovingAverageCrossoverFactory(), config)
        old_result = old_study.run()

    current = GridSearchStudy(_dataset(), MovingAverageCrossoverFactory(), config)
    assert current.study_id != old_study.study_id
    current.store = old_study.store
    with pytest.raises(StudyPersistenceError, match="incompatible"):
        current.resume()
    current.store = FileStudyStore(tmp_path, current.study_id)
    current_result = current.run()
    assert len(current_result.successful_trials) == 2
    assert {trial.trial_id for trial in old_result.trials}.isdisjoint(
        trial.trial_id for trial in current_result.trials
    )


class BackendMovingAverage(MovingAverageCrossoverStrategy):
    def __init__(
        self, parameters: MovingAverageCrossoverParameters, backend_id: str
    ) -> None:
        super().__init__(parameters)
        self._required_indicators = tuple(
            SimpleMovingAverage(
                SimpleMovingAverageParameters(window, parameters.source_field),
                backend_id=backend_id,
            )
            for window in (parameters.fast_window, parameters.slow_window)
        )


@dataclass(frozen=True)
class BackendFactory(MovingAverageCrossoverFactory):
    backend_id: str = "native_v1"

    def configuration(self) -> PrimitiveMapping:
        return {**super().configuration(), "backend_id": self.backend_id}

    def build(self, parameters: PrimitiveMapping) -> Strategy:
        original = super().build(parameters)
        return BackendMovingAverage(
            cast(MovingAverageCrossoverParameters, original.parameters), self.backend_id
        )


class SameFactoryConfiguration(BackendFactory):
    """Model unchanged factory provenance resolving a different backend runtime."""

    def configuration(self) -> PrimitiveMapping:
        configuration = super().configuration()
        del configuration["backend_id"]
        return configuration


def test_study_identity_binds_resolved_candidate_configurations(tmp_path: Path) -> None:
    config = bounded_config(tmp_path)
    native = GridSearchStudy(_dataset(), SameFactoryConfiguration(), config)
    talib = GridSearchStudy(
        _dataset(), SameFactoryConfiguration(backend_id="talib_v1"), config
    )
    assert (
        native.strategy_factory.configuration()
        == talib.strategy_factory.configuration()
    )
    assert [candidate.combination_id for candidate in native.candidates] == [
        candidate.combination_id for candidate in talib.candidates
    ]
    assert native.study_id != talib.study_id
    native.run()
    talib.store = native.store
    with pytest.raises(StudyPersistenceError, match="incompatible"):
        talib.resume()


def test_resolved_backend_version_changes_study_and_trial_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = bounded_config(tmp_path)
    original = GridSearchStudy(_dataset(), BackendFactory(), config)
    original_result = original.run()
    original_identity_for = NativeIndicatorBackend.identity_for

    def revised_identity_for(
        backend: NativeIndicatorBackend, definition: StandardIndicatorDefinition
    ) -> IndicatorBackendIdentity:
        return replace(
            original_identity_for(backend, definition), library_version="qf43-test-2"
        )

    monkeypatch.setattr(NativeIndicatorBackend, "identity_for", revised_identity_for)
    revised = GridSearchStudy(_dataset(), BackendFactory(), config)
    assert (
        revised.strategy_factory.configuration()
        == original.strategy_factory.configuration()
    )
    assert revised.study_id != original.study_id
    revised.store = original.store
    with pytest.raises(StudyPersistenceError, match="incompatible"):
        revised.resume()
    revised.store = FileStudyStore(tmp_path, revised.study_id)
    revised_result = revised.run()
    assert len(revised_result.successful_trials) == 2
    assert {trial.trial_id for trial in revised_result.trials}.isdisjoint(
        trial.trial_id for trial in original_result.trials
    )


def test_indicator_backend_identity_prevents_bounded_trial_reuse(
    tmp_path: Path,
) -> None:
    config = bounded_config(tmp_path)
    legacy = GridSearchStudy(_dataset(), MovingAverageCrossoverFactory(), config).run()
    native_study = GridSearchStudy(_dataset(), BackendFactory(), config)
    native = native_study.run()
    talib_study = GridSearchStudy(
        _dataset(), BackendFactory(backend_id="talib_v1"), config
    )
    assert len({legacy.study_id, native.study_id, talib_study.study_id}) == 3
    talib_study.store = native_study.store
    with pytest.raises(StudyPersistenceError, match="incompatible"):
        talib_study.resume()
    talib_study.store = FileStudyStore(tmp_path, talib_study.study_id)
    talib = talib_study.run()
    for result in (legacy, native, talib):
        assert len(result.successful_trials) == 2
    assert (
        len(
            {
                result.trials[0].strategy_configuration_id
                for result in (legacy, native, talib)
            }
        )
        == 3
    )
    assert len({result.trials[0].trial_id for result in (legacy, native, talib)}) == 3
    assert len({result.trials[0].qf5_run_id for result in (legacy, native, talib)}) == 3
