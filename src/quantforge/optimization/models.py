"""Immutable study, ranking, trial, and analysis models."""

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

from quantforge.backtesting import BacktestConfig
from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    decimal_to_primitive,
)
from quantforge.optimization.constraints import ParameterConstraint
from quantforge.optimization.errors import InvalidStudyConfigurationError
from quantforge.optimization.factories import StrategyFactory
from quantforge.optimization.spaces import ParameterSearchSpace

STUDY_SCHEMA_VERSION = "1"
TRIAL_SCHEMA_VERSION = "1"
STABILITY_SCHEMA_VERSION = "1"


class ExecutionMode(StrEnum):
    SEQUENTIAL = "sequential"
    PROCESS = "process"


class TrialStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXCLUDED = "excluded"


class RankingDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class UndefinedMetricPolicy(StrEnum):
    EXCLUDE = "exclude"


class MetricName(StrEnum):
    STARTING_EQUITY = "starting_equity"
    ENDING_EQUITY = "ending_equity"
    TOTAL_RETURN = "total_return"
    CAGR = "cagr"
    ANNUALIZED_VOLATILITY = "annualized_volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    PROFIT_FACTOR = "profit_factor"
    EXPOSURE = "exposure"
    TRADE_COUNT = "trade_count"
    OPEN_TRADE_COUNT = "open_trade_count"
    WIN_RATE = "win_rate"
    GROSS_PROFIT = "gross_profit"
    GROSS_LOSS = "gross_loss"
    WINNING_TRADES = "winning_trades"
    LOSING_TRADES = "losing_trades"
    AVERAGE_TRADE_RETURN = "average_trade_return"
    BENCHMARK_TOTAL_RETURN = "benchmark_total_return"


class ThresholdOperator(StrEnum):
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "le"
    EQUAL = "eq"


class HardMetricConstraint(Protocol):
    @property
    def metric(self) -> MetricName: ...

    @property
    def operator(self) -> ThresholdOperator: ...

    @property
    def threshold(self) -> Decimal | int: ...

    def to_primitive(self) -> PrimitiveMapping: ...


def _finite_threshold(value: Decimal | int | str, label: str) -> Decimal:
    if isinstance(value, bool):
        raise InvalidStudyConfigurationError(f"{label} must be numeric")
    try:
        threshold = Decimal(str(value))
    except InvalidOperation as error:
        raise InvalidStudyConfigurationError(f"{label} must be numeric") from error
    if not threshold.is_finite():
        raise InvalidStudyConfigurationError(f"{label} must be finite")
    return threshold


def _optional_decimal(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_primitive(value)


@dataclass(frozen=True, slots=True)
class MetricThreshold:
    """Generic inclusive or exclusive metric threshold."""

    metric: MetricName
    operator: ThresholdOperator
    threshold: Decimal | int

    def __post_init__(self) -> None:
        normalized = _finite_threshold(self.threshold, "metric threshold")
        object.__setattr__(self, "threshold", normalized)

    def to_primitive(self) -> PrimitiveMapping:
        assert isinstance(self.threshold, Decimal)
        return {
            "type": "metric_threshold",
            "metric": self.metric.value,
            "operator": self.operator.value,
            "threshold": decimal_to_primitive(self.threshold),
        }


@dataclass(frozen=True, slots=True)
class MinimumTrades:
    """Require completed-trade count to be at least ``minimum`` (inclusive)."""

    minimum: int

    def __post_init__(self) -> None:
        minimum = cast(object, self.minimum)
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            raise InvalidStudyConfigurationError(
                "minimum completed trades must be a nonnegative integer"
            )

    @property
    def metric(self) -> MetricName:
        return MetricName.TRADE_COUNT

    @property
    def operator(self) -> ThresholdOperator:
        return ThresholdOperator.GREATER_THAN_OR_EQUAL

    @property
    def threshold(self) -> int:
        return self.minimum

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "type": "minimum_completed_trades",
            "metric": self.metric.value,
            "operator": self.operator.value,
            "threshold": self.minimum,
        }


@dataclass(frozen=True, slots=True)
class MaximumDrawdown:
    """Require QF-5's negative drawdown to be no worse than a loss magnitude."""

    maximum_loss: Decimal

    def __post_init__(self) -> None:
        normalized = _finite_threshold(self.maximum_loss, "maximum drawdown")
        if not Decimal(0) <= normalized <= Decimal(1):
            raise InvalidStudyConfigurationError(
                "maximum drawdown loss magnitude must be between 0 and 1"
            )
        object.__setattr__(self, "maximum_loss", normalized)

    @property
    def metric(self) -> MetricName:
        return MetricName.MAXIMUM_DRAWDOWN

    @property
    def operator(self) -> ThresholdOperator:
        return ThresholdOperator.GREATER_THAN_OR_EQUAL

    @property
    def threshold(self) -> Decimal:
        return -self.maximum_loss

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "type": "maximum_drawdown_loss_magnitude",
            "metric": self.metric.value,
            "operator": self.operator.value,
            "threshold": decimal_to_primitive(self.threshold),
            "maximum_loss_magnitude": decimal_to_primitive(self.maximum_loss),
            "qf5_convention": "negative_decimal",
        }


@dataclass(frozen=True, slots=True)
class PositiveReturn:
    """Require QF-5 total return to be strictly positive."""

    @property
    def metric(self) -> MetricName:
        return MetricName.TOTAL_RETURN

    @property
    def operator(self) -> ThresholdOperator:
        return ThresholdOperator.GREATER_THAN

    @property
    def threshold(self) -> Decimal:
        return Decimal(0)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "type": "positive_total_return",
            "metric": self.metric.value,
            "operator": self.operator.value,
            "threshold": "0",
        }


@dataclass(frozen=True, slots=True)
class MetricTieBreaker:
    metric: MetricName
    direction: RankingDirection

    def to_primitive(self) -> PrimitiveMapping:
        return {"metric": self.metric.value, "direction": self.direction.value}


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """One objective, deterministic tie breakers, and hard eligibility rules."""

    objective: MetricName
    direction: RankingDirection = RankingDirection.MAXIMIZE
    hard_constraints: tuple[HardMetricConstraint, ...] = ()
    tie_breakers: tuple[MetricTieBreaker, ...] = field(
        default_factory=lambda: (
            MetricTieBreaker(MetricName.MAXIMUM_DRAWDOWN, RankingDirection.MAXIMIZE),
            MetricTieBreaker(MetricName.TRADE_COUNT, RankingDirection.MAXIMIZE),
        )
    )
    undefined_metric_policy: UndefinedMetricPolicy = UndefinedMetricPolicy.EXCLUDE
    minimum_successful_trials: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.objective), MetricName):
            raise InvalidStudyConfigurationError(
                "ranking objective must be a supported QF-5 metric"
            )
        if not isinstance(cast(object, self.direction), RankingDirection):
            raise InvalidStudyConfigurationError("ranking direction is unsupported")
        if not isinstance(
            cast(object, self.undefined_metric_policy), UndefinedMetricPolicy
        ):
            raise InvalidStudyConfigurationError(
                "undefined metric policy is unsupported"
            )
        object.__setattr__(self, "hard_constraints", tuple(self.hard_constraints))
        object.__setattr__(self, "tie_breakers", tuple(self.tie_breakers))
        minimum_successful_trials = cast(object, self.minimum_successful_trials)
        if minimum_successful_trials is not None and (
            isinstance(minimum_successful_trials, bool)
            or not isinstance(minimum_successful_trials, int)
            or minimum_successful_trials < 1
        ):
            raise InvalidStudyConfigurationError(
                "minimum successful trials must be a positive integer"
            )
        for constraint_value in cast(tuple[object, ...], self.hard_constraints):
            if not callable(getattr(constraint_value, "to_primitive", None)):
                raise InvalidStudyConfigurationError(
                    "ranking hard constraints must be HardMetricConstraint records"
                )
            if not isinstance(
                cast(object, getattr(constraint_value, "metric", None)), MetricName
            ):
                raise InvalidStudyConfigurationError(
                    "ranking hard-constraint metric must be a supported QF-5 metric"
                )
            if not isinstance(
                cast(object, getattr(constraint_value, "operator", None)),
                ThresholdOperator,
            ):
                raise InvalidStudyConfigurationError(
                    "ranking hard-constraint operator is unsupported"
                )
            threshold_value = cast(
                object,
                getattr(constraint_value, "threshold", None),
            )
            if not isinstance(threshold_value, (Decimal, int, str)):
                raise InvalidStudyConfigurationError(
                    "ranking hard-constraint threshold must be numeric"
                )
            _finite_threshold(
                threshold_value,
                "ranking hard-constraint threshold",
            )
        for tie_breaker_value in cast(tuple[object, ...], self.tie_breakers):
            if not isinstance(tie_breaker_value, MetricTieBreaker):
                raise InvalidStudyConfigurationError(
                    "ranking tie breakers must be MetricTieBreaker records"
                )
            if not isinstance(cast(object, tie_breaker_value.metric), MetricName):
                raise InvalidStudyConfigurationError(
                    "ranking tie-breaker metric must be a supported QF-5 metric"
                )
            if not isinstance(
                cast(object, tie_breaker_value.direction), RankingDirection
            ):
                raise InvalidStudyConfigurationError(
                    "ranking tie-breaker direction is unsupported"
                )
        metrics = [tie_breaker.metric for tie_breaker in self.tie_breakers]
        if len(set(metrics)) != len(metrics):
            raise InvalidStudyConfigurationError(
                "ranking tie-breaker metrics must be unique"
            )
        for constraint in self.hard_constraints:
            constraint.to_primitive()

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "objective": self.objective.value,
            "direction": self.direction.value,
            "hard_constraints": [
                constraint.to_primitive() for constraint in self.hard_constraints
            ],
            "tie_breakers": [item.to_primitive() for item in self.tie_breakers],
            "final_tie_breaker": "combination_id_ascending",
            "undefined_metric_policy": self.undefined_metric_policy.value,
            "minimum_successful_trials": self.minimum_successful_trials,
        }


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    """Grid-neighborhood and isolated-peak classification thresholds."""

    minimum_eligible_neighbors: int = 2
    stable_constraint_pass_fraction: Decimal = Decimal("0.75")
    stable_maximum_relative_dispersion: Decimal = Decimal("0.25")
    isolated_peak_top_fraction: Decimal = Decimal("0.20")
    isolated_peak_absolute_drop: Decimal = Decimal("0")
    isolated_peak_relative_drop: Decimal = Decimal("0.25")
    isolated_peak_maximum_constraint_pass_fraction: Decimal = Decimal("0.50")
    robust_recommendation_top_fraction: Decimal = Decimal("0.25")
    schema_version: str = field(default=STABILITY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        minimum_eligible_neighbors = cast(object, self.minimum_eligible_neighbors)
        if (
            isinstance(minimum_eligible_neighbors, bool)
            or not isinstance(minimum_eligible_neighbors, int)
            or minimum_eligible_neighbors < 1
        ):
            raise InvalidStudyConfigurationError(
                "minimum eligible neighbors must be a positive integer"
            )
        for name in (
            "stable_constraint_pass_fraction",
            "stable_maximum_relative_dispersion",
            "isolated_peak_top_fraction",
            "isolated_peak_absolute_drop",
            "isolated_peak_relative_drop",
            "isolated_peak_maximum_constraint_pass_fraction",
            "robust_recommendation_top_fraction",
        ):
            normalized = _finite_threshold(cast(Decimal, getattr(self, name)), name)
            if normalized < 0:
                raise InvalidStudyConfigurationError(f"{name} cannot be negative")
            object.__setattr__(self, name, normalized)
        for name in (
            "stable_constraint_pass_fraction",
            "isolated_peak_top_fraction",
            "isolated_peak_maximum_constraint_pass_fraction",
            "robust_recommendation_top_fraction",
        ):
            if cast(Decimal, getattr(self, name)) > 1:
                raise InvalidStudyConfigurationError(f"{name} cannot exceed 1")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "coordinate_model": "candidate_index",
            "numeric_adjacency": "one_candidate_step_in_one_dimension",
            "categorical_boolean_adjacency": "any_alternate_value_in_one_dimension",
            "minimum_eligible_neighbors": self.minimum_eligible_neighbors,
            "stable_constraint_pass_fraction": decimal_to_primitive(
                self.stable_constraint_pass_fraction
            ),
            "stable_maximum_relative_dispersion": decimal_to_primitive(
                self.stable_maximum_relative_dispersion
            ),
            "isolated_peak_top_fraction": decimal_to_primitive(
                self.isolated_peak_top_fraction
            ),
            "isolated_peak_absolute_drop": decimal_to_primitive(
                self.isolated_peak_absolute_drop
            ),
            "isolated_peak_relative_drop": decimal_to_primitive(
                self.isolated_peak_relative_drop
            ),
            "isolated_peak_maximum_constraint_pass_fraction": decimal_to_primitive(
                self.isolated_peak_maximum_constraint_pass_fraction
            ),
            "robust_recommendation_top_fraction": decimal_to_primitive(
                self.robust_recommendation_top_fraction
            ),
        }


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    maximum_workers: int = 1
    retry_failed: bool = False
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.mode), ExecutionMode):
            raise InvalidStudyConfigurationError("execution mode is unsupported")
        maximum_workers = cast(object, self.maximum_workers)
        if (
            isinstance(maximum_workers, bool)
            or not isinstance(maximum_workers, int)
            or maximum_workers < 1
        ):
            raise InvalidStudyConfigurationError(
                "maximum worker count must be a positive integer"
            )
        retry_failed = cast(object, self.retry_failed)
        fail_fast = cast(object, self.fail_fast)
        if not isinstance(retry_failed, bool) or not isinstance(fail_fast, bool):
            raise InvalidStudyConfigurationError(
                "retry_failed and fail_fast must be booleans"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "mode": self.mode.value,
            "maximum_workers": self.maximum_workers,
            "parallelism": (
                "none"
                if self.mode is ExecutionMode.SEQUENTIAL
                else "bounded_standard_library_process_pool"
            ),
            "retry_failed": self.retry_failed,
            "stale_running_policy": "retry",
            "fail_fast": self.fail_fast,
        }


@dataclass(frozen=True, slots=True)
class FilePersistenceConfig:
    output_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "store": "atomic_local_json_files",
            "output_root": str(self.output_root),
            "successful_trial_overwrite": "forbidden",
        }


@dataclass(frozen=True, slots=True)
class GridSearchConfig:
    """Immutable typed operational and analytical study configuration."""

    label: str
    search_space: ParameterSearchSpace
    parameter_constraints: tuple[ParameterConstraint, ...]
    backtest: BacktestConfig
    execution: ExecutionConfig
    ranking: RankingConfig
    stability: StabilityConfig
    persistence: FilePersistenceConfig
    maximum_combinations: int = 10_000
    allow_large_grid: bool = False
    schema_version: str = field(default=STUDY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        label = cast(object, self.label)
        if not isinstance(label, str) or not label.strip():
            raise InvalidStudyConfigurationError("study label must be nonempty")
        object.__setattr__(
            self, "parameter_constraints", tuple(self.parameter_constraints)
        )
        maximum_combinations = cast(object, self.maximum_combinations)
        if (
            isinstance(maximum_combinations, bool)
            or not isinstance(maximum_combinations, int)
            or maximum_combinations < 1
        ):
            raise InvalidStudyConfigurationError(
                "maximum combinations must be a positive integer"
            )
        if not isinstance(cast(object, self.allow_large_grid), bool):
            raise InvalidStudyConfigurationError("allow_large_grid must be a boolean")

    def to_primitive(self, strategy_factory: StrategyFactory) -> PrimitiveMapping:
        return {
            "label": self.label,
            "schema_version": self.schema_version,
            "search_space": self.search_space.to_primitive(
                strategy_factory.parameter_order
            ),
            "parameter_constraints": [
                constraint.to_primitive() for constraint in self.parameter_constraints
            ],
            "backtest": self.backtest.to_primitive(),
            "execution": self.execution.to_primitive(),
            "ranking": self.ranking.to_primitive(),
            "stability": self.stability.to_primitive(),
            "persistence": self.persistence.to_primitive(),
            "maximum_combinations": self.maximum_combinations,
            "allow_large_grid": self.allow_large_grid,
        }


@dataclass(frozen=True, slots=True)
class FailedTrialAttempt:
    """Immutable diagnostic context for one completed failed attempt."""

    attempt_number: int
    failure_category: str
    failure_type: str
    failure_message: str
    started_at: str | None
    finished_at: str | None

    def __post_init__(self) -> None:
        attempt_number = cast(object, self.attempt_number)
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number < 1
        ):
            raise ValueError("archived trial attempt number must be positive")
        if any(
            not isinstance(cast(object, item), str)
            for item in (
                self.failure_category,
                self.failure_type,
                self.failure_message,
            )
        ):
            raise ValueError("archived trial failure context must be text")
        if any(
            item is not None and not isinstance(cast(object, item), str)
            for item in (self.started_at, self.finished_at)
        ):
            raise ValueError("archived trial attempt timestamps must be text or null")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "attempt_number": self.attempt_number,
            "status": TrialStatus.FAILED.value,
            "failure_category": self.failure_category,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_primitive(cls, value: PrimitiveMapping) -> "FailedTrialAttempt":
        if value.get("status") != TrialStatus.FAILED.value:
            raise ValueError("archived trial attempt must have failed status")
        return cls(
            attempt_number=cast(int, value.get("attempt_number")),
            failure_category=cast(str, value["failure_category"]),
            failure_type=cast(str, value["failure_type"]),
            failure_message=cast(str, value["failure_message"]),
            started_at=cast(str | None, value.get("started_at")),
            finished_at=cast(str | None, value.get("finished_at")),
        )


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """Persisted status and complete provenance for one Cartesian assignment."""

    study_id: str
    trial_id: str
    combination_id: str
    combination_index: int
    status: TrialStatus
    parameters_snapshot: PrimitiveMappingSnapshot
    strategy_parameters_snapshot: PrimitiveMappingSnapshot
    strategy_name: str
    strategy_version: str
    strategy_configuration_id: str | None
    dataset_snapshot: PrimitiveMappingSnapshot
    backtest_configuration_snapshot: PrimitiveMappingSnapshot
    metrics_snapshot: PrimitiveMappingSnapshot | None = None
    qf5_run_id: str | None = None
    artifact_location: str | None = None
    failure_category: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    failed_attempts: tuple[FailedTrialAttempt, ...] = ()
    exclusion_code: str | None = None
    exclusion_reason: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    schema_version: str = field(default=TRIAL_SCHEMA_VERSION, init=False)

    @property
    def parameters(self) -> PrimitiveMapping:
        return self.parameters_snapshot.to_primitive()

    @property
    def strategy_parameters(self) -> PrimitiveMapping:
        return self.strategy_parameters_snapshot.to_primitive()

    @property
    def dataset(self) -> PrimitiveMapping:
        return self.dataset_snapshot.to_primitive()

    @property
    def backtest_configuration(self) -> PrimitiveMapping:
        return self.backtest_configuration_snapshot.to_primitive()

    @property
    def metrics(self) -> PrimitiveMapping | None:
        return (
            None
            if self.metrics_snapshot is None
            else self.metrics_snapshot.to_primitive()
        )

    def failed_attempt_snapshot(self) -> FailedTrialAttempt:
        """Capture the current failure before a configured retry begins."""
        if self.status is not TrialStatus.FAILED or any(
            item is None
            for item in (
                self.failure_category,
                self.failure_type,
                self.failure_message,
            )
        ):
            raise ValueError("only a complete failed trial can be archived")
        return FailedTrialAttempt(
            attempt_number=len(self.failed_attempts) + 1,
            failure_category=cast(str, self.failure_category),
            failure_type=cast(str, self.failure_type),
            failure_message=cast(str, self.failure_message),
            started_at=self.started_at,
            finished_at=self.finished_at,
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "trial_id": self.trial_id,
            "combination_id": self.combination_id,
            "combination_index": self.combination_index,
            "status": self.status.value,
            "parameters": self.parameters,
            "strategy_parameters": self.strategy_parameters,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "strategy_configuration_id": self.strategy_configuration_id,
            "dataset": self.dataset,
            "backtest_configuration": self.backtest_configuration,
            "metrics": self.metrics,
            "qf5_run_id": self.qf5_run_id,
            "artifact_location": self.artifact_location,
            "failure_category": self.failure_category,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
            "failed_attempts": [
                attempt.to_primitive() for attempt in self.failed_attempts
            ],
            "exclusion_code": self.exclusion_code,
            "exclusion_reason": self.exclusion_reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_primitive(cls, value: PrimitiveMapping) -> "TrialRecord":
        try:
            if value["schema_version"] != TRIAL_SCHEMA_VERSION:
                raise ValueError("unsupported trial schema")
            parameters_value = cast(object, value["parameters"])
            strategy_parameters_value = cast(object, value["strategy_parameters"])
            dataset_value = cast(object, value["dataset"])
            backtest_configuration_value = cast(object, value["backtest_configuration"])
            metrics_value = cast(object, value["metrics"])
            failed_attempts_value = cast(object, value.get("failed_attempts", []))
            if not all(
                isinstance(item, dict)
                for item in (
                    parameters_value,
                    strategy_parameters_value,
                    dataset_value,
                    backtest_configuration_value,
                )
            ):
                raise TypeError("trial object fields must be mappings")
            if metrics_value is not None and not isinstance(metrics_value, dict):
                raise TypeError("trial metrics must be an object or null")
            if not isinstance(failed_attempts_value, list):
                raise TypeError("failed trial attempts must be a list of objects")
            failed_attempt_items = cast(list[object], failed_attempts_value)
            if any(not isinstance(item, dict) for item in failed_attempt_items):
                raise TypeError("failed trial attempts must be a list of objects")
            parameters = cast(PrimitiveMapping, parameters_value)
            strategy_parameters = cast(PrimitiveMapping, strategy_parameters_value)
            dataset = cast(PrimitiveMapping, dataset_value)
            backtest_configuration = cast(
                PrimitiveMapping, backtest_configuration_value
            )
            return cls(
                study_id=cast(str, value["study_id"]),
                trial_id=cast(str, value["trial_id"]),
                combination_id=cast(str, value["combination_id"]),
                combination_index=cast(int, value["combination_index"]),
                status=TrialStatus(cast(str, value["status"])),
                parameters_snapshot=PrimitiveMappingSnapshot.capture(parameters),
                strategy_parameters_snapshot=PrimitiveMappingSnapshot.capture(
                    strategy_parameters
                ),
                strategy_name=cast(str, value["strategy_name"]),
                strategy_version=cast(str, value["strategy_version"]),
                strategy_configuration_id=cast(
                    str | None, value["strategy_configuration_id"]
                ),
                dataset_snapshot=PrimitiveMappingSnapshot.capture(dataset),
                backtest_configuration_snapshot=PrimitiveMappingSnapshot.capture(
                    backtest_configuration
                ),
                metrics_snapshot=(
                    None
                    if metrics_value is None
                    else PrimitiveMappingSnapshot.capture(
                        cast(PrimitiveMapping, metrics_value)
                    )
                ),
                qf5_run_id=cast(str | None, value["qf5_run_id"]),
                artifact_location=cast(str | None, value["artifact_location"]),
                failure_category=cast(str | None, value["failure_category"]),
                failure_type=cast(str | None, value["failure_type"]),
                failure_message=cast(str | None, value["failure_message"]),
                failed_attempts=tuple(
                    FailedTrialAttempt.from_primitive(cast(PrimitiveMapping, attempt))
                    for attempt in failed_attempt_items
                ),
                exclusion_code=cast(str | None, value["exclusion_code"]),
                exclusion_reason=cast(str | None, value["exclusion_reason"]),
                started_at=cast(str | None, value["started_at"]),
                finished_at=cast(str | None, value["finished_at"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            from quantforge.optimization.errors import StudyPersistenceError

            raise StudyPersistenceError("invalid or corrupt trial record") from error


@dataclass(frozen=True, slots=True)
class RankedTrial:
    rank: int
    trial_id: str
    combination_id: str
    objective_metric: MetricName
    objective_value: Decimal

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "rank": self.rank,
            "trial_id": self.trial_id,
            "combination_id": self.combination_id,
            "objective_metric": self.objective_metric.value,
            "objective_value": decimal_to_primitive(self.objective_value),
        }


@dataclass(frozen=True, slots=True)
class IneligibleTrial:
    trial_id: str
    combination_id: str
    reasons: tuple[str, ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "trial_id": self.trial_id,
            "combination_id": self.combination_id,
            "reasons": list(self.reasons),
        }


class StabilityClassification(StrEnum):
    STABLE = "stable"
    MIXED = "mixed"
    FRAGILE = "fragile"
    INSUFFICIENT_NEIGHBORS = "insufficient_neighbors"


@dataclass(frozen=True, slots=True)
class StabilitySummary:
    trial_id: str
    combination_id: str
    objective_rank: int
    objective_value: Decimal
    valid_neighbor_count: int
    excluded_neighbor_count: int
    successful_eligible_neighbor_count: int
    neighbor_objective_values: tuple[Decimal, ...]
    mean_neighbor_objective: Decimal | None
    median_neighbor_objective: Decimal | None
    worst_neighbor_objective: Decimal | None
    objective_standard_deviation: Decimal | None
    constraint_pass_fraction: Decimal
    center_to_neighbor_difference: Decimal | None
    relative_center_to_neighbor_difference: Decimal | None
    is_boundary: bool
    stability_score: Decimal
    classification: StabilityClassification
    is_isolated_peak: bool
    isolation_reason: str | None
    stability_rank: int | None = None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "trial_id": self.trial_id,
            "combination_id": self.combination_id,
            "objective_rank": self.objective_rank,
            "objective_value": decimal_to_primitive(self.objective_value),
            "valid_neighbor_count": self.valid_neighbor_count,
            "excluded_neighbor_count": self.excluded_neighbor_count,
            "successful_eligible_neighbor_count": (
                self.successful_eligible_neighbor_count
            ),
            "neighbor_objective_values": [
                decimal_to_primitive(value) for value in self.neighbor_objective_values
            ],
            "mean_neighbor_objective": _optional_decimal(self.mean_neighbor_objective),
            "median_neighbor_objective": _optional_decimal(
                self.median_neighbor_objective
            ),
            "worst_neighbor_objective": _optional_decimal(
                self.worst_neighbor_objective
            ),
            "objective_standard_deviation": _optional_decimal(
                self.objective_standard_deviation
            ),
            "constraint_pass_fraction": decimal_to_primitive(
                self.constraint_pass_fraction
            ),
            "center_to_neighbor_difference": _optional_decimal(
                self.center_to_neighbor_difference
            ),
            "relative_center_to_neighbor_difference": _optional_decimal(
                self.relative_center_to_neighbor_difference
            ),
            "is_boundary": self.is_boundary,
            "stability_score": decimal_to_primitive(self.stability_score),
            "classification": self.classification.value,
            "is_isolated_peak": self.is_isolated_peak,
            "isolation_reason": self.isolation_reason,
            "stability_rank": self.stability_rank,
        }


@dataclass(frozen=True, slots=True)
class ParameterSummary:
    parameter_name: str
    parameter_value: str | int | bool
    successful_count: int
    eligible_count: int
    constraint_pass_fraction: Decimal
    mean_objective: Decimal | None
    median_objective: Decimal | None
    best_objective: Decimal | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "parameter_name": self.parameter_name,
            "parameter_value": self.parameter_value,
            "successful_count": self.successful_count,
            "eligible_count": self.eligible_count,
            "constraint_pass_fraction": decimal_to_primitive(
                self.constraint_pass_fraction
            ),
            "mean_objective": (
                None
                if self.mean_objective is None
                else decimal_to_primitive(self.mean_objective)
            ),
            "median_objective": (
                None
                if self.median_objective is None
                else decimal_to_primitive(self.median_objective)
            ),
            "best_objective": (
                None
                if self.best_objective is None
                else decimal_to_primitive(self.best_objective)
            ),
        }


@dataclass(frozen=True, slots=True)
class StudyResult:
    study_id: str
    schema_version: str
    total_combinations: int
    trials: tuple[TrialRecord, ...]
    rankings: tuple[RankedTrial, ...]
    ineligible_trials: tuple[IneligibleTrial, ...]
    stability: tuple[StabilitySummary, ...]
    parameter_summaries: tuple[ParameterSummary, ...]
    best_objective_trial_id: str | None
    best_stability_trial_id: str | None
    recommended_robust_trial_id: str | None
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    @property
    def successful_trials(self) -> tuple[TrialRecord, ...]:
        return tuple(
            item for item in self.trials if item.status is TrialStatus.SUCCEEDED
        )

    @property
    def failed_trials(self) -> tuple[TrialRecord, ...]:
        return tuple(item for item in self.trials if item.status is TrialStatus.FAILED)

    @property
    def excluded_trials(self) -> tuple[TrialRecord, ...]:
        return tuple(
            item for item in self.trials if item.status is TrialStatus.EXCLUDED
        )

    @property
    def pending_trials(self) -> tuple[TrialRecord, ...]:
        return tuple(
            item
            for item in self.trials
            if item.status in (TrialStatus.PENDING, TrialStatus.RUNNING)
        )

    def summary_primitive(self) -> PrimitiveMapping:
        objective_values = [item.objective_value for item in self.rankings]
        return {
            "study_id": self.study_id,
            "study_schema_version": self.schema_version,
            "counts": {
                "total_cartesian_combinations": self.total_combinations,
                "recorded_trials": len(self.trials),
                "successful": len(self.successful_trials),
                "failed": len(self.failed_trials),
                "excluded": len(self.excluded_trials),
                "pending_or_running": len(self.pending_trials),
                "eligible": len(self.rankings),
                "ineligible_successful": len(self.ineligible_trials),
                "isolated_peaks": sum(item.is_isolated_peak for item in self.stability),
            },
            "objective_distribution": {
                "count": len(objective_values),
                "minimum": (
                    None
                    if not objective_values
                    else decimal_to_primitive(min(objective_values))
                ),
                "maximum": (
                    None
                    if not objective_values
                    else decimal_to_primitive(max(objective_values))
                ),
            },
            "best_objective_trial_id": self.best_objective_trial_id,
            "best_stability_trial_id": self.best_stability_trial_id,
            "recommended_robust_trial_id": self.recommended_robust_trial_id,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }
