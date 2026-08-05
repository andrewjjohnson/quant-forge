"""Deterministic, resumable grid-search study orchestration."""

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from quantforge.backtesting import (
    ENGINE_VERSION,
    RESULT_SCHEMA_VERSION,
    BacktestError,
    BacktestResult,
    MarketDataMetadata,
    export_backtest_result,
    fingerprint_market_bars,
    load_backtest_manifest,
    run_backtest,
)
from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import MarketDataset, validate_market_dataset
from quantforge.data.identity import serialize_metadata_values
from quantforge.optimization.combinations import (
    CombinationCandidate,
    CombinationExclusion,
    ParameterCombination,
    iter_combination_candidates,
    validate_combination_definition,
)
from quantforge.optimization.errors import (
    CombinationLimitExceededError,
    StudyExecutionError,
    StudyPersistenceError,
)
from quantforge.optimization.factories import StrategyFactory
from quantforge.optimization.models import (
    GridSearchConfig,
    StudyResult,
    TrialRecord,
    TrialStatus,
)
from quantforge.optimization.persistence import FileStudyStore
from quantforge.optimization.ranking import rank_trials
from quantforge.optimization.runner import (
    BacktestRunner,
    initialize_process_worker,
    run_process_trial,
)
from quantforge.optimization.stability import analyze_stability
from quantforge.strategies import StrategyError

OPTIMIZATION_ENGINE_VERSION = "1"
_QF5_ARTIFACT_FILENAMES = (
    "manifest.json",
    "signals.csv",
    "orders.csv",
    "fills.csv",
    "positions.csv",
    "trades.csv",
    "equity.csv",
    "benchmark_equity.csv",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _dataset_primitive(dataset: MarketDataset) -> PrimitiveMapping:
    values = serialize_metadata_values(asdict(dataset.metadata))
    return cast(PrimitiveMapping, values)


def _sanitized_error(error: BaseException) -> tuple[str, str]:
    message = " ".join(str(error).split())[:500]
    return error.__class__.__name__, message or "trial execution failed"


def _failure_category(error: BaseException) -> str:
    if isinstance(error, StrategyError):
        return "strategy_failure"
    if isinstance(error, BacktestError):
        return "backtest_domain_failure"
    if isinstance(error, StudyPersistenceError):
        return "persistence_failure"
    if isinstance(error, (BrokenProcessPool, ChildProcessError)):
        return "worker_failure"
    return "unexpected_implementation_failure"


class GridSearchStudy:
    """Coordinate QF-4 parameter construction and repeated deterministic QF-5 runs."""

    def __init__(
        self,
        dataset: MarketDataset,
        strategy_factory: StrategyFactory,
        config: GridSearchConfig,
        *,
        backtest_runner: BacktestRunner = run_backtest,
    ) -> None:
        validate_market_dataset(dataset)
        validate_combination_definition(
            strategy_factory,
            config.search_space,
            config.parameter_constraints,
        )
        total_combinations = config.search_space.combination_count()
        if (
            total_combinations > config.maximum_combinations
            and not config.allow_large_grid
        ):
            expression = config.search_space.count_expression(
                strategy_factory.parameter_order
            )
            raise CombinationLimitExceededError(
                f"parameter grid has {expression} combinations, exceeding the "
                f"configured safeguard of {config.maximum_combinations:,}; set "
                "allow_large_grid=True only after reviewing the multiplicative grid"
            )
        self.dataset = dataset
        self._expected_market_data = MarketDataMetadata.from_qf3(
            dataset.metadata,
            bars_fingerprint=fingerprint_market_bars(dataset.bars),
        )
        self.strategy_factory = strategy_factory
        self.config = config
        self._backtest_runner = backtest_runner
        self.total_combinations = total_combinations
        self._candidates = tuple(
            iter_combination_candidates(
                strategy_factory,
                config.search_space,
                config.parameter_constraints,
            )
        )
        if len(self._candidates) != total_combinations:
            raise StudyExecutionError(
                "normalized Cartesian combinations were unexpectedly duplicated"
            )
        self._identity_inputs = self._scientific_definition()
        self.study_id = configuration_identity(self._identity_inputs)
        self._expected_candidates_by_trial_id = {
            self._trial_id(candidate): candidate for candidate in self._candidates
        }
        self.store = FileStudyStore(
            config.persistence.output_root,
            self.study_id,
        )

    @property
    def study_path(self) -> Path:
        return self.store.study_path

    @property
    def candidates(self) -> tuple[CombinationCandidate, ...]:
        return self._candidates

    def _scientific_definition(self) -> PrimitiveMapping:
        return {
            "component": "quantforge_grid_search_study",
            "optimization_engine_version": OPTIMIZATION_ENGINE_VERSION,
            "study_schema_version": self.config.schema_version,
            "trial_schema_version": "1",
            "qf5_engine_version": ENGINE_VERSION,
            "qf5_result_schema_version": RESULT_SCHEMA_VERSION,
            "label": self.config.label,
            "dataset": _dataset_primitive(self.dataset),
            "strategy_factory": self.strategy_factory.configuration(),
            "search_space": self.config.search_space.to_primitive(
                self.strategy_factory.parameter_order
            ),
            "parameter_constraints": [
                constraint.to_primitive()
                for constraint in self.config.parameter_constraints
            ],
            "backtest_configuration": self.config.backtest.to_primitive(),
            "ranking_configuration": self.config.ranking.to_primitive(),
            "stability_configuration": self.config.stability.to_primitive(),
        }

    def manifest_primitive(self) -> PrimitiveMapping:
        valid = sum(
            isinstance(candidate, ParameterCombination)
            for candidate in self._candidates
        )
        return {
            "study_id": self.study_id,
            "study_schema_version": self.config.schema_version,
            "identity_inputs": self._identity_inputs,
            "execution_configuration": self.config.execution.to_primitive(),
            "persistence_configuration": self.config.persistence.to_primitive(),
            "scale_safeguard": {
                "maximum_combinations": self.config.maximum_combinations,
                "allow_large_grid": self.config.allow_large_grid,
                "count_expression": self.config.search_space.count_expression(
                    self.strategy_factory.parameter_order
                ),
            },
            "combination_counts": {
                "total_cartesian": self.total_combinations,
                "valid": valid,
                "excluded": self.total_combinations - valid,
            },
            "operational_timestamp_policy": (
                "trial timestamps are diagnostic and excluded from deterministic "
                "identities"
            ),
            "warnings": [
                "rankings and stability are in-sample descriptive results",
                "testing many combinations increases multiple-comparison risk",
            ],
            "limitations": [
                "no walk-forward, holdout, or out-of-sample validation",
                "stability analysis is not evidence of future profitability",
                "local bounded process parallelism only",
            ],
        }

    def _trial_id(self, candidate: CombinationCandidate) -> str:
        strategy_configuration_id = (
            candidate.strategy_configuration_id
            if isinstance(candidate, ParameterCombination)
            else None
        )
        strategy_parameters = (
            candidate.strategy_parameters
            if isinstance(candidate, ParameterCombination)
            else candidate.parameters
        )
        return configuration_identity(
            {
                "component": "quantforge_optimization_trial",
                "trial_schema_version": "1",
                "study_id": self.study_id,
                "combination_id": candidate.combination_id,
                "dataset_id": self.dataset.metadata.dataset_id,
                "strategy_name": self.strategy_factory.strategy_name,
                "strategy_version": self.strategy_factory.strategy_version,
                "strategy_configuration_id": strategy_configuration_id,
                "strategy_parameters": strategy_parameters,
                "backtest_configuration": self.config.backtest.to_primitive(),
                "qf5_engine_version": ENGINE_VERSION,
                "qf5_result_schema_version": RESULT_SCHEMA_VERSION,
                "optimization_engine_version": OPTIMIZATION_ENGINE_VERSION,
            }
        )

    def _base_record(
        self,
        candidate: CombinationCandidate,
        status: TrialStatus,
        *,
        started_at: str | None = None,
    ) -> TrialRecord:
        strategy_parameters = (
            candidate.strategy_parameters
            if isinstance(candidate, ParameterCombination)
            else candidate.parameters
        )
        strategy_configuration_id = (
            candidate.strategy_configuration_id
            if isinstance(candidate, ParameterCombination)
            else None
        )
        return TrialRecord(
            study_id=self.study_id,
            trial_id=self._trial_id(candidate),
            combination_id=candidate.combination_id,
            combination_index=candidate.index,
            status=status,
            parameters_snapshot=PrimitiveMappingSnapshot.capture(candidate.parameters),
            strategy_parameters_snapshot=PrimitiveMappingSnapshot.capture(
                strategy_parameters
            ),
            strategy_name=self.strategy_factory.strategy_name,
            strategy_version=self.strategy_factory.strategy_version,
            strategy_configuration_id=strategy_configuration_id,
            dataset_snapshot=PrimitiveMappingSnapshot.capture(
                _dataset_primitive(self.dataset)
            ),
            backtest_configuration_snapshot=PrimitiveMappingSnapshot.capture(
                self.config.backtest.to_primitive()
            ),
            started_at=started_at,
        )

    def _prepare_records(self) -> tuple[ParameterCombination, ...]:
        pending: list[ParameterCombination] = []
        for candidate in self._candidates:
            trial_id = self._trial_id(candidate)
            existing = self.store.load_trial(trial_id)
            if isinstance(candidate, CombinationExclusion):
                if existing is None:
                    record = replace(
                        self._base_record(candidate, TrialStatus.EXCLUDED),
                        exclusion_code=candidate.reason_code,
                        exclusion_reason=candidate.reason,
                        finished_at=_utc_now(),
                    )
                    self.store.write_trial(record)
                elif existing.status is not TrialStatus.EXCLUDED:
                    raise StudyPersistenceError(
                        "persisted valid/excluded combination status is incompatible"
                    )
                continue
            if existing is None:
                existing = self._base_record(candidate, TrialStatus.PENDING)
                self.store.write_trial(existing)
            if existing.status is TrialStatus.SUCCEEDED:
                continue
            if existing.status is TrialStatus.EXCLUDED:
                raise StudyPersistenceError(
                    "persisted excluded status conflicts with a valid combination"
                )
            if (
                existing.status is TrialStatus.FAILED
                and not self.config.execution.retry_failed
            ):
                continue
            pending.append(candidate)
        return tuple(pending)

    def _write_running(self, candidate: ParameterCombination) -> TrialRecord:
        existing = self.store.load_trial(self._trial_id(candidate))
        failed_attempts = () if existing is None else existing.failed_attempts
        if existing is not None and existing.status is TrialStatus.FAILED:
            try:
                failed_attempts = (*failed_attempts, existing.failed_attempt_snapshot())
            except ValueError as error:
                raise StudyPersistenceError(
                    "cannot retry a failed trial with incomplete failure context"
                ) from error
        record = self._base_record(
            candidate,
            TrialStatus.RUNNING,
            started_at=_utc_now(),
        )
        record = replace(record, failed_attempts=failed_attempts)
        self.store.write_trial(record)
        return record

    def _artifact_for_result(self, trial_id: str, result: BacktestResult) -> str:
        artifact_root = self.study_path / "backtests" / trial_id
        destination = artifact_root / result.run_id
        if destination.exists():
            manifest = load_backtest_manifest(destination / "manifest.json")
            if manifest.get("run_id") != result.run_id:
                raise StudyPersistenceError(
                    "existing per-trial backtest artifact has an incompatible run ID"
                )
        else:
            export_backtest_result(result, artifact_root)
        return str(destination.relative_to(self.study_path))

    def _write_success(
        self,
        candidate: ParameterCombination,
        running: TrialRecord,
        result: BacktestResult,
    ) -> None:
        if result.market_data != self._expected_market_data:
            raise StudyPersistenceError(
                "QF-5 result market data does not match the study dataset"
            )
        if result.backtest_configuration != self.config.backtest.to_primitive():
            raise StudyPersistenceError(
                "QF-5 result backtest configuration does not match the study"
            )
        if (
            result.strategy_id != self.strategy_factory.strategy_name
            or result.strategy_implementation_version
            != self.strategy_factory.strategy_version
        ):
            raise StudyPersistenceError(
                "QF-5 result strategy identity does not match the trial candidate"
            )
        if result.strategy_configuration_id != candidate.strategy_configuration_id:
            raise StudyPersistenceError(
                "QF-5 result strategy configuration does not match the trial candidate"
            )
        parameters = result.strategy_configuration.get("parameters")
        if not isinstance(parameters, dict):
            raise StudyPersistenceError(
                "QF-5 result strategy configuration omitted parameter provenance"
            )
        if parameters != candidate.strategy_parameters:
            raise StudyPersistenceError(
                "QF-5 result strategy parameters do not match the trial candidate"
            )
        artifact_location = self._artifact_for_result(running.trial_id, result)
        record = TrialRecord(
            study_id=self.study_id,
            trial_id=running.trial_id,
            combination_id=candidate.combination_id,
            combination_index=candidate.index,
            status=TrialStatus.SUCCEEDED,
            parameters_snapshot=PrimitiveMappingSnapshot.capture(candidate.parameters),
            strategy_parameters_snapshot=PrimitiveMappingSnapshot.capture(
                cast(PrimitiveMapping, parameters)
            ),
            strategy_name=result.strategy_id,
            strategy_version=result.strategy_implementation_version,
            strategy_configuration_id=result.strategy_configuration_id,
            dataset_snapshot=PrimitiveMappingSnapshot.capture(
                result.market_data.to_primitive()
            ),
            backtest_configuration_snapshot=PrimitiveMappingSnapshot.capture(
                result.backtest_configuration
            ),
            metrics_snapshot=PrimitiveMappingSnapshot.capture(
                result.performance.to_primitive()
            ),
            qf5_run_id=result.run_id,
            artifact_location=artifact_location,
            failed_attempts=running.failed_attempts,
            started_at=running.started_at,
            finished_at=_utc_now(),
        )
        self.store.write_trial(record)

    def _write_failure(
        self,
        candidate: ParameterCombination,
        running: TrialRecord,
        error: BaseException,
    ) -> None:
        error_type, message = _sanitized_error(error)
        record = replace(
            running,
            status=TrialStatus.FAILED,
            failure_category=_failure_category(error),
            failure_type=error_type,
            failure_message=message,
            finished_at=_utc_now(),
        )
        self.store.write_trial(record)

    def _execute_sequential(self, pending: tuple[ParameterCombination, ...]) -> None:
        for candidate in pending:
            running = self._write_running(candidate)
            try:
                strategy = self.strategy_factory.build(candidate.parameters)
                result = self._backtest_runner(
                    self.dataset,
                    strategy,
                    self.config.backtest,
                )
                self._write_success(candidate, running, result)
            except Exception as error:
                self._write_failure(candidate, running, error)
                if self.config.execution.fail_fast:
                    break

    def _execute_processes(self, pending: tuple[ParameterCombination, ...]) -> None:
        if self._backtest_runner is not run_backtest:
            raise StudyExecutionError(
                "process execution requires the real top-level QF-5 run_backtest"
            )
        remaining = iter(pending)
        futures: dict[
            Future[BacktestResult], tuple[ParameterCombination, TrialRecord]
        ] = {}
        halted = False
        with ProcessPoolExecutor(
            max_workers=self.config.execution.maximum_workers,
            initializer=initialize_process_worker,
            initargs=(self.dataset, self.strategy_factory, self.config.backtest),
        ) as executor:

            def schedule(candidate: ParameterCombination) -> bool:
                running = self._write_running(candidate)
                try:
                    future = executor.submit(run_process_trial, candidate.parameters)
                except Exception as error:
                    self._write_failure(candidate, running, error)
                    return False
                futures[future] = (candidate, running)
                return True

            for _ in range(self.config.execution.maximum_workers):
                candidate = next(remaining, None)
                if candidate is None:
                    break
                if not schedule(candidate):
                    halted = True
                    break

            while futures:
                completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                ordered_completed = sorted(
                    completed,
                    key=lambda future: futures[future][0].index,
                )
                for future in ordered_completed:
                    candidate, running = futures.pop(future)
                    try:
                        self._write_success(candidate, running, future.result())
                    except Exception as error:
                        self._write_failure(candidate, running, error)
                        halted = (
                            halted
                            or self.config.execution.fail_fast
                            or isinstance(error, BrokenProcessPool)
                        )
                if not halted:
                    for _ in ordered_completed:
                        next_candidate = next(remaining, None)
                        if next_candidate is None:
                            break
                        if not schedule(next_candidate):
                            halted = True
                            break

    def _validate_success_artifact(self, trial: TrialRecord) -> None:
        qf5_run_id = trial.qf5_run_id
        artifact_location = trial.artifact_location
        if not qf5_run_id or not artifact_location or trial.metrics is None:
            raise StudyPersistenceError(
                "successful persisted trial is missing QF-5 result provenance"
            )
        expected_location = (Path("backtests") / trial.trial_id / qf5_run_id).as_posix()
        if artifact_location != expected_location:
            raise StudyPersistenceError(
                "successful persisted trial has an incompatible artifact location"
            )
        artifact_path = self.study_path / expected_location
        if any(
            not (artifact_path / filename).is_file()
            for filename in _QF5_ARTIFACT_FILENAMES
        ):
            raise StudyPersistenceError(
                "successful persisted trial has an incomplete QF-5 artifact"
            )
        try:
            manifest = load_backtest_manifest(artifact_path / "manifest.json")
        except BacktestError as error:
            raise StudyPersistenceError(
                "successful persisted trial has an invalid QF-5 artifact"
            ) from error
        strategy_value = cast(object, manifest.get("strategy"))
        if not isinstance(strategy_value, dict):
            raise StudyPersistenceError(
                "successful persisted trial has an invalid QF-5 strategy manifest"
            )
        strategy = cast(dict[object, object], strategy_value)
        configuration_value = strategy.get("configuration")
        if not isinstance(configuration_value, dict):
            raise StudyPersistenceError(
                "successful persisted trial has an invalid QF-5 strategy configuration"
            )
        configuration = cast(dict[object, object], configuration_value)
        try:
            manifest_strategy_configuration_id = configuration_identity(
                cast(PrimitiveMapping, configuration_value)
            )
        except (TypeError, ValueError) as error:
            raise StudyPersistenceError(
                "successful persisted trial has unstable QF-5 strategy provenance"
            ) from error
        if (
            manifest.get("run_id") != qf5_run_id
            or manifest.get("engine_version") != ENGINE_VERSION
            or manifest.get("result_schema_version") != RESULT_SCHEMA_VERSION
            or manifest.get("market_data") != trial.dataset
            or manifest.get("backtest_configuration") != trial.backtest_configuration
            or manifest.get("performance") != trial.metrics
            or strategy.get("strategy_id") != trial.strategy_name
            or strategy.get("strategy_implementation_version") != trial.strategy_version
            or strategy.get("strategy_configuration_id")
            != trial.strategy_configuration_id
            or manifest_strategy_configuration_id != trial.strategy_configuration_id
            or configuration.get("parameters") != trial.strategy_parameters
        ):
            raise StudyPersistenceError(
                "successful persisted trial does not match its linked QF-5 artifact"
            )

    def _validate_trial_contents(
        self,
        trial: TrialRecord,
        candidate: CombinationCandidate,
    ) -> None:
        expected = self._base_record(candidate, trial.status)
        expected_dataset = (
            self._expected_market_data.to_primitive()
            if trial.status is TrialStatus.SUCCEEDED
            else expected.dataset
        )
        if (
            trial.schema_version != expected.schema_version
            or trial.study_id != expected.study_id
            or trial.trial_id != expected.trial_id
            or trial.combination_id != expected.combination_id
            or trial.combination_index != expected.combination_index
            or trial.parameters != expected.parameters
            or trial.strategy_parameters != expected.strategy_parameters
            or trial.strategy_name != expected.strategy_name
            or trial.strategy_version != expected.strategy_version
            or trial.strategy_configuration_id != expected.strategy_configuration_id
            or trial.dataset != expected_dataset
            or trial.backtest_configuration != expected.backtest_configuration
        ):
            raise StudyPersistenceError(
                "persisted trial contents do not match the expected candidate"
            )
        if isinstance(candidate, CombinationExclusion):
            if (
                trial.status is not TrialStatus.EXCLUDED
                or trial.exclusion_code != candidate.reason_code
                or trial.exclusion_reason != candidate.reason
            ):
                raise StudyPersistenceError(
                    "persisted exclusion does not match the expected candidate"
                )
        elif trial.status is TrialStatus.EXCLUDED:
            raise StudyPersistenceError(
                "persisted valid candidate cannot have excluded status"
            )
        if trial.status is TrialStatus.SUCCEEDED:
            if any(
                value is not None
                for value in (
                    trial.failure_category,
                    trial.failure_type,
                    trial.failure_message,
                    trial.exclusion_code,
                    trial.exclusion_reason,
                )
            ):
                raise StudyPersistenceError(
                    "successful persisted trial contains incompatible outcome fields"
                )
            self._validate_success_artifact(trial)
        elif any(
            value is not None
            for value in (trial.metrics, trial.qf5_run_id, trial.artifact_location)
        ):
            raise StudyPersistenceError(
                "non-successful persisted trial contains QF-5 result fields"
            )
        elif trial.status is TrialStatus.FAILED and any(
            value is None
            for value in (
                trial.failure_category,
                trial.failure_type,
                trial.failure_message,
            )
        ):
            raise StudyPersistenceError(
                "failed persisted trial has incomplete failure context"
            )

    def _load_validated_trials(
        self,
        *,
        require_complete: bool,
    ) -> tuple[TrialRecord, ...]:
        trials = self.store.load_trials()
        persisted_trial_ids = frozenset(trial.trial_id for trial in trials)
        expected_trial_ids = frozenset(self._expected_candidates_by_trial_id)
        if persisted_trial_ids.difference(expected_trial_ids):
            raise StudyPersistenceError(
                "persisted trial IDs include records outside the expected candidate set"
            )
        if require_complete and expected_trial_ids.difference(persisted_trial_ids):
            raise StudyPersistenceError(
                "persisted trial IDs do not cover the complete candidate set"
            )
        for trial in trials:
            self._validate_trial_contents(
                trial,
                self._expected_candidates_by_trial_id[trial.trial_id],
            )
        return trials

    def _build_result(self) -> StudyResult:
        trials = self._load_validated_trials(require_complete=True)
        ranking = rank_trials(trials, self.config.ranking)
        stability = analyze_stability(
            trials,
            self._candidates,
            ranking,
            self.config.search_space,
            self.strategy_factory,
            self.config.ranking,
            self.config.stability,
        )
        best_objective = None if not ranking.rankings else ranking.rankings[0].trial_id
        warnings = (
            *ranking.warnings,
            "all optimization rankings are in-sample",
            "parameter stability does not substitute for untouched out-of-sample "
            "testing",
        )
        return StudyResult(
            study_id=self.study_id,
            schema_version=self.config.schema_version,
            total_combinations=self.total_combinations,
            trials=trials,
            rankings=ranking.rankings,
            ineligible_trials=ranking.ineligible_trials,
            stability=stability.summaries,
            parameter_summaries=stability.parameter_summaries,
            best_objective_trial_id=best_objective,
            best_stability_trial_id=stability.best_stability_trial_id,
            recommended_robust_trial_id=stability.recommended_robust_trial_id,
            warnings=warnings,
            limitations=(
                "in-sample Cartesian grid search only",
                "no walk-forward, holdout, cross-validation, or multiple-testing "
                "correction",
                "no automatic strategy deployment or live-trading recommendation",
            ),
        )

    def _execute(self, *, resume: bool) -> StudyResult:
        self.store.initialize(self.manifest_primitive(), resume=resume)
        self._load_validated_trials(require_complete=False)
        pending = self._prepare_records()
        if self.config.execution.mode.value == "sequential":
            self._execute_sequential(pending)
        else:
            self._execute_processes(pending)
        result = self._build_result()
        self.export(result)
        return result

    def run(self) -> StudyResult:
        """Start a new study and reject accidental reuse of an existing directory."""
        return self._execute(resume=False)

    def resume(self) -> StudyResult:
        """Resume an exact persisted manifest without rerunning completed trials."""
        return self._execute(resume=True)

    def load_result(self) -> StudyResult:
        """Rebuild ranking and stability from persisted trial records only."""
        if self.store.load_manifest() != self.manifest_primitive():
            raise StudyPersistenceError(
                "persisted study manifest is incompatible with the requested study"
            )
        return self._build_result()

    def export(self, result: StudyResult | None = None) -> Path:
        """Regenerate all deterministic study-level exports without backtesting."""
        from quantforge.optimization.export import export_study_result

        selected = self.load_result() if result is None else result
        return export_study_result(
            selected,
            self.store.study_path,
            self.manifest_primitive(),
            self.config.ranking.to_primitive(),
            self.config.stability.to_primitive(),
        )
