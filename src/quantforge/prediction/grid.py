"""Deterministic parameter grids for causal multi-timeframe prediction studies."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation, localcontext
from itertools import product
from pathlib import Path
from typing import Any, Protocol, cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.models import MarketDataset
from quantforge.data.multi_timeframe import (
    ContextCompletionPolicy,
    MultiTimeframeContext,
)
from quantforge.indicators import IndicatorBackendIdentity, TimeframeIndicatorOutput
from quantforge.optimization.constraints import ParameterConstraint
from quantforge.optimization.models import (
    RankingDirection,
    StabilityClassification,
    StabilityConfig,
    ThresholdOperator,
    TrialStatus,
)
from quantforge.optimization.spaces import (
    ParameterSearchSpace,
    SearchValue,
    search_value_to_primitive,
)
from quantforge.prediction.context import (
    PredictionContextError,
    PredictionContextProvider,
    PredictionContextRequirements,
    PredictionIndicatorOutputCache,
    PredictionIndicatorRequirement,
)
from quantforge.prediction.contracts import PredictionStudy
from quantforge.prediction.errors import (
    InvalidPredictionConfigurationError,
    PredictionAnalysisError,
)
from quantforge.prediction.study import (
    PredictionStudyResult,
    prepare_prediction_study_dataset,
    run_prediction_study_in_session,
)
from quantforge.timeframes import Timeframe

PREDICTION_GRID_ENGINE_VERSION = "1"
PREDICTION_GRID_SCHEMA_VERSION = "1"


class PredictionGridError(PredictionAnalysisError):
    """Base error for deterministic prediction parameter studies."""


class InvalidPredictionGridConfigurationError(PredictionGridError):
    """A prediction grid cannot be identified or executed safely."""


class InvalidPredictionGridParametersError(PredictionGridError):
    """One parameter assignment is invalid before prediction execution."""


class PredictionGridPersistenceError(PredictionGridError):
    """Persisted prediction-grid state is missing, corrupt, or incompatible."""


@dataclass(frozen=True, slots=True)
class PredictionIndicatorBackendEnvironment:
    """Fixed ordinary-study backend identity, excluding per-function names."""

    backend_id: str
    library_name: str
    library_version: str
    contract_version: str
    runtime_library_name: str | None = None
    runtime_library_version: str | None = None
    configuration_snapshot: PrimitiveMappingSnapshot = field(
        default_factory=lambda: PrimitiveMappingSnapshot.capture({})
    )

    def __post_init__(self) -> None:
        if not all(
            (
                self.backend_id,
                self.library_name,
                self.library_version,
                self.contract_version,
            )
        ):
            raise InvalidPredictionGridConfigurationError(
                "indicator backend id, library, version, and contract are required"
            )
        if (self.runtime_library_name is None) != (
            self.runtime_library_version is None
        ):
            raise InvalidPredictionGridConfigurationError(
                "runtime backend name and version must be configured together"
            )

    @classmethod
    def create(
        cls,
        *,
        backend_id: str,
        library_name: str,
        library_version: str,
        contract_version: str,
        runtime_library_name: str | None = None,
        runtime_library_version: str | None = None,
        configuration: PrimitiveMapping | None = None,
    ) -> PredictionIndicatorBackendEnvironment:
        return cls(
            backend_id,
            library_name,
            library_version,
            contract_version,
            runtime_library_name,
            runtime_library_version,
            PrimitiveMappingSnapshot.capture(
                {} if configuration is None else configuration
            ),
        )

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.to_primitive())

    def matches(self, identity: IndicatorBackendIdentity) -> bool:
        return (
            identity.backend_id == self.backend_id
            and identity.library_name == self.library_name
            and identity.library_version == self.library_version
            and identity.contract_version == self.contract_version
            and identity.runtime_library_name == self.runtime_library_name
            and identity.runtime_library_version == self.runtime_library_version
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "backend_id": self.backend_id,
            "library_name": self.library_name,
            "library_version": self.library_version,
            "contract_version": self.contract_version,
            "runtime_library_name": self.runtime_library_name,
            "runtime_library_version": self.runtime_library_version,
            "fixed_configuration": self.configuration_snapshot.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class PredictionContextEnvironment:
    """Stable identity for the fixed context-provider environment."""

    provider_id: str
    provider_version: str
    configuration_snapshot: PrimitiveMappingSnapshot

    @classmethod
    def create(
        cls,
        provider_id: str,
        provider_version: str,
        configuration: PrimitiveMapping,
    ) -> PredictionContextEnvironment:
        if not provider_id or not provider_version:
            raise InvalidPredictionGridConfigurationError(
                "context provider id and version are required"
            )
        return cls(
            provider_id,
            provider_version,
            PrimitiveMappingSnapshot.capture(configuration),
        )

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.to_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "configuration": self.configuration_snapshot.to_primitive(),
        }


class PredictionStudyFactory(Protocol):
    """Build one QF-11/QF-31 study from normalized searched parameters."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def parameter_order(self) -> tuple[str, ...]: ...

    @property
    def required_parameter_names(self) -> frozenset[str]: ...

    def configuration(self) -> PrimitiveMapping: ...

    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]: ...


class PredictionTrialAnalyzer(Protocol):
    """Convert one immutable prediction result into rankable research evidence."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...

    def analyze(
        self, result: PredictionStudyResult[Any, Any, Any]
    ) -> PredictionTrialAnalysis: ...


def _snapshots(
    values: Sequence[PrimitiveMapping],
) -> tuple[PrimitiveMappingSnapshot, ...]:
    return tuple(PrimitiveMappingSnapshot.capture(value) for value in values)


@dataclass(frozen=True, slots=True)
class PredictionTrialAnalysis:
    """Rankable metrics plus retained period, weekday, and baseline evidence."""

    prediction_count: int
    metrics_snapshot: PrimitiveMappingSnapshot
    period_comparisons: tuple[PrimitiveMappingSnapshot, ...]
    weekday_comparisons: tuple[PrimitiveMappingSnapshot, ...]
    matched_baseline_comparisons: tuple[PrimitiveMappingSnapshot, ...]
    artifacts_snapshot: PrimitiveMappingSnapshot

    def __post_init__(self) -> None:
        prediction_count = cast(object, self.prediction_count)
        if (
            isinstance(prediction_count, bool)
            or not isinstance(prediction_count, int)
            or prediction_count < 0
        ):
            raise InvalidPredictionGridConfigurationError(
                "prediction count must be a nonnegative integer"
            )
        for name, value in self.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise InvalidPredictionGridConfigurationError(
                    f"prediction metric must be numeric: {name}"
                )
            try:
                numeric = Decimal(str(value))
            except InvalidOperation as error:
                raise InvalidPredictionGridConfigurationError(
                    f"prediction metric must be numeric: {name}"
                ) from error
            if not numeric.is_finite():
                raise InvalidPredictionGridConfigurationError(
                    f"prediction metric must be finite: {name}"
                )

    @classmethod
    def create(
        cls,
        *,
        prediction_count: int,
        metrics: PrimitiveMapping,
        period_comparisons: Sequence[PrimitiveMapping] = (),
        weekday_comparisons: Sequence[PrimitiveMapping] = (),
        matched_baseline_comparisons: Sequence[PrimitiveMapping] = (),
        artifacts: PrimitiveMapping | None = None,
    ) -> PredictionTrialAnalysis:
        return cls(
            prediction_count,
            PrimitiveMappingSnapshot.capture(metrics),
            _snapshots(period_comparisons),
            _snapshots(weekday_comparisons),
            _snapshots(matched_baseline_comparisons),
            PrimitiveMappingSnapshot.capture({} if artifacts is None else artifacts),
        )

    @property
    def metrics(self) -> PrimitiveMapping:
        return self.metrics_snapshot.to_primitive()

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "prediction_count": self.prediction_count,
            "metrics": self.metrics,
            "period_comparisons": [
                item.to_primitive() for item in self.period_comparisons
            ],
            "weekday_comparisons": [
                item.to_primitive() for item in self.weekday_comparisons
            ],
            "matched_baseline_comparisons": [
                item.to_primitive() for item in self.matched_baseline_comparisons
            ],
            "artifacts": self.artifacts_snapshot.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: PrimitiveMapping) -> PredictionTrialAnalysis:
        try:
            metrics = cast(PrimitiveMapping, value["metrics"])
            periods = cast(list[PrimitiveMapping], value["period_comparisons"])
            weekdays = cast(list[PrimitiveMapping], value["weekday_comparisons"])
            baselines = cast(
                list[PrimitiveMapping], value["matched_baseline_comparisons"]
            )
            artifacts = cast(PrimitiveMapping, value["artifacts"])
            return cls.create(
                prediction_count=cast(int, value["prediction_count"]),
                metrics=metrics,
                period_comparisons=periods,
                weekday_comparisons=weekdays,
                matched_baseline_comparisons=baselines,
                artifacts=artifacts,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PredictionGridPersistenceError(
                "invalid persisted prediction trial analysis"
            ) from error


@dataclass(frozen=True, slots=True)
class PredictionMetricConstraint:
    metric: str
    operator: ThresholdOperator
    threshold: Decimal

    def __post_init__(self) -> None:
        if not self.metric:
            raise InvalidPredictionGridConfigurationError(
                "prediction quality metric name is required"
            )
        if not isinstance(cast(object, self.operator), ThresholdOperator):
            raise InvalidPredictionGridConfigurationError(
                "prediction quality threshold operator is invalid"
            )
        try:
            threshold = Decimal(str(self.threshold))
        except InvalidOperation as error:
            raise InvalidPredictionGridConfigurationError(
                "prediction quality threshold must be numeric"
            ) from error
        if not threshold.is_finite():
            raise InvalidPredictionGridConfigurationError(
                "prediction quality threshold must be finite"
            )
        object.__setattr__(self, "threshold", threshold)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "metric": self.metric,
            "operator": self.operator.value,
            "threshold": decimal_to_primitive(self.threshold),
        }


@dataclass(frozen=True, slots=True)
class PredictionRankingConfig:
    objective_metric: str
    baseline_name: str
    direction: RankingDirection = RankingDirection.MAXIMIZE
    minimum_prediction_count: int = 1
    outcome_quality_constraints: tuple[PredictionMetricConstraint, ...] = ()

    def __post_init__(self) -> None:
        if not self.objective_metric:
            raise InvalidPredictionGridConfigurationError(
                "prediction ranking objective metric is required"
            )
        if not self.baseline_name:
            raise InvalidPredictionGridConfigurationError(
                "prediction ranking baseline name is required"
            )
        if not isinstance(cast(object, self.direction), RankingDirection):
            raise InvalidPredictionGridConfigurationError(
                "prediction ranking direction is invalid"
            )
        minimum_prediction_count = cast(object, self.minimum_prediction_count)
        if (
            isinstance(minimum_prediction_count, bool)
            or not isinstance(minimum_prediction_count, int)
            or minimum_prediction_count < 1
        ):
            raise InvalidPredictionGridConfigurationError(
                "minimum prediction count must be positive"
            )
        names = tuple(item.metric for item in self.outcome_quality_constraints)
        if len(names) != len(set(names)):
            raise InvalidPredictionGridConfigurationError(
                "prediction quality constraints must use unique metrics"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "objective_metric": self.objective_metric,
            "baseline_name": self.baseline_name,
            "direction": self.direction.value,
            "minimum_prediction_count": self.minimum_prediction_count,
            "outcome_quality_constraints": [
                item.to_primitive() for item in self.outcome_quality_constraints
            ],
            "final_tie_breaker": "combination_id_ascending",
            "interpretation": "in_sample_research_ranking_not_validated_profitability",
        }


@dataclass(frozen=True, slots=True)
class PredictionGridConfig:
    label: str
    search_space: ParameterSearchSpace
    ranking: PredictionRankingConfig
    output_root: Path
    parameter_constraints: tuple[ParameterConstraint, ...] = ()
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    retry_failed: bool = False
    maximum_combinations: int = 10_000

    def __post_init__(self) -> None:
        if not self.label:
            raise InvalidPredictionGridConfigurationError("grid label is required")
        if not isinstance(cast(object, self.retry_failed), bool):
            raise InvalidPredictionGridConfigurationError(
                "retry_failed must be boolean"
            )
        maximum_combinations = cast(object, self.maximum_combinations)
        if (
            isinstance(maximum_combinations, bool)
            or not isinstance(maximum_combinations, int)
            or maximum_combinations < 1
        ):
            raise InvalidPredictionGridConfigurationError(
                "maximum combinations must be positive"
            )
        object.__setattr__(self, "output_root", Path(self.output_root))


@dataclass(frozen=True, slots=True)
class PredictionGridCombination:
    index: int
    combination_id: str
    parameters_snapshot: PrimitiveMappingSnapshot
    coordinates: tuple[int, ...]
    trial_definition_snapshot: PrimitiveMappingSnapshot
    indicator_configuration_ids: tuple[str, ...]
    study: PredictionStudy[Any, Any, Any] = field(repr=False, compare=False)

    @property
    def parameters(self) -> PrimitiveMapping:
        return self.parameters_snapshot.to_primitive()


@dataclass(frozen=True, slots=True)
class PredictionGridExclusion:
    index: int
    combination_id: str
    parameters_snapshot: PrimitiveMappingSnapshot
    coordinates: tuple[int, ...]
    reason_code: str
    reason: str

    @property
    def parameters(self) -> PrimitiveMapping:
        return self.parameters_snapshot.to_primitive()


type PredictionGridCandidate = PredictionGridCombination | PredictionGridExclusion


@dataclass(frozen=True, slots=True)
class PredictionGridFailedAttempt:
    """Sanitized diagnostic history for one completed failed attempt."""

    failure_type: str
    failure_message: str
    started_at: str
    finished_at: str

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_primitive(cls, value: PrimitiveMapping) -> PredictionGridFailedAttempt:
        try:
            attempt = cls(
                failure_type=cast(str, value["failure_type"]),
                failure_message=cast(str, value["failure_message"]),
                started_at=cast(str, value["started_at"]),
                finished_at=cast(str, value["finished_at"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PredictionGridPersistenceError(
                "invalid persisted prediction-grid failed attempt"
            ) from error
        if not all(
            (
                attempt.failure_type,
                attempt.failure_message,
                attempt.started_at,
                attempt.finished_at,
            )
        ):
            raise PredictionGridPersistenceError(
                "persisted prediction-grid failed attempt is incomplete"
            )
        return attempt


@dataclass(frozen=True, slots=True)
class PredictionGridTrialRecord:
    study_id: str
    trial_id: str
    combination_id: str
    combination_index: int
    status: TrialStatus
    parameters_snapshot: PrimitiveMappingSnapshot
    trial_definition_snapshot: PrimitiveMappingSnapshot | None
    backend_snapshot: PrimitiveMappingSnapshot
    dataset_family_fingerprint: str
    indicator_configuration_ids: tuple[str, ...]
    analysis: PredictionTrialAnalysis | None = None
    artifact_location: str | None = None
    artifact_fingerprint: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    failed_attempts: tuple[PredictionGridFailedAttempt, ...] = ()
    exclusion_code: str | None = None
    exclusion_reason: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def parameters(self) -> PrimitiveMapping:
        return self.parameters_snapshot.to_primitive()

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": PREDICTION_GRID_SCHEMA_VERSION,
            "study_id": self.study_id,
            "trial_id": self.trial_id,
            "combination_id": self.combination_id,
            "combination_index": self.combination_index,
            "status": self.status.value,
            "parameters": self.parameters,
            "trial_definition": (
                None
                if self.trial_definition_snapshot is None
                else self.trial_definition_snapshot.to_primitive()
            ),
            "indicator_backend": self.backend_snapshot.to_primitive(),
            "dataset_family_fingerprint": self.dataset_family_fingerprint,
            "indicator_configuration_ids": list(self.indicator_configuration_ids),
            "analysis": None if self.analysis is None else self.analysis.to_primitive(),
            "artifact_location": self.artifact_location,
            "artifact_fingerprint": self.artifact_fingerprint,
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
    def from_primitive(cls, value: PrimitiveMapping) -> PredictionGridTrialRecord:
        try:
            if value["schema_version"] != PREDICTION_GRID_SCHEMA_VERSION:
                raise ValueError("unsupported prediction-grid trial schema")
            parameters = cast(PrimitiveMapping, value["parameters"])
            definition = cast(PrimitiveMapping | None, value["trial_definition"])
            backend = cast(PrimitiveMapping, value["indicator_backend"])
            analysis_value = cast(PrimitiveMapping | None, value["analysis"])
            indicator_ids = cast(list[str], value["indicator_configuration_ids"])
            failed_attempts = cast(
                list[PrimitiveMapping], value.get("failed_attempts", [])
            )
            return cls(
                study_id=cast(str, value["study_id"]),
                trial_id=cast(str, value["trial_id"]),
                combination_id=cast(str, value["combination_id"]),
                combination_index=cast(int, value["combination_index"]),
                status=TrialStatus(cast(str, value["status"])),
                parameters_snapshot=PrimitiveMappingSnapshot.capture(parameters),
                trial_definition_snapshot=(
                    None
                    if definition is None
                    else PrimitiveMappingSnapshot.capture(definition)
                ),
                backend_snapshot=PrimitiveMappingSnapshot.capture(backend),
                dataset_family_fingerprint=cast(
                    str, value["dataset_family_fingerprint"]
                ),
                indicator_configuration_ids=tuple(indicator_ids),
                analysis=(
                    None
                    if analysis_value is None
                    else PredictionTrialAnalysis.from_primitive(analysis_value)
                ),
                artifact_location=cast(str | None, value["artifact_location"]),
                artifact_fingerprint=cast(
                    str | None, value.get("artifact_fingerprint")
                ),
                failure_type=cast(str | None, value["failure_type"]),
                failure_message=cast(str | None, value["failure_message"]),
                failed_attempts=tuple(
                    PredictionGridFailedAttempt.from_primitive(item)
                    for item in failed_attempts
                ),
                exclusion_code=cast(str | None, value["exclusion_code"]),
                exclusion_reason=cast(str | None, value["exclusion_reason"]),
                started_at=cast(str | None, value["started_at"]),
                finished_at=cast(str | None, value["finished_at"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PredictionGridPersistenceError(
                "invalid or corrupt prediction-grid trial"
            ) from error


@dataclass(frozen=True, slots=True)
class PredictionRankedTrial:
    rank: int
    trial_id: str
    combination_id: str
    objective_metric: str
    objective_value: Decimal

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "rank": self.rank,
            "trial_id": self.trial_id,
            "combination_id": self.combination_id,
            "objective_metric": self.objective_metric,
            "objective_value": decimal_to_primitive(self.objective_value),
        }


@dataclass(frozen=True, slots=True)
class PredictionIneligibleTrial:
    trial_id: str
    combination_id: str
    reasons: tuple[str, ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "trial_id": self.trial_id,
            "combination_id": self.combination_id,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class PredictionStabilitySummary:
    trial_id: str
    combination_id: str
    objective_rank: int
    objective_value: Decimal
    valid_neighbor_count: int
    excluded_neighbor_count: int
    eligible_neighbor_count: int
    neighbor_objective_values: tuple[Decimal, ...]
    median_neighbor_objective: Decimal | None
    relative_dispersion: Decimal | None
    constraint_pass_fraction: Decimal
    center_to_neighbor_difference: Decimal | None
    relative_center_to_neighbor_difference: Decimal | None
    is_boundary: bool
    classification: StabilityClassification
    is_isolated_peak: bool
    isolation_reason: str | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "trial_id": self.trial_id,
            "combination_id": self.combination_id,
            "objective_rank": self.objective_rank,
            "objective_value": decimal_to_primitive(self.objective_value),
            "valid_neighbor_count": self.valid_neighbor_count,
            "excluded_neighbor_count": self.excluded_neighbor_count,
            "eligible_neighbor_count": self.eligible_neighbor_count,
            "neighbor_objective_values": [
                decimal_to_primitive(value) for value in self.neighbor_objective_values
            ],
            "median_neighbor_objective": (
                None
                if self.median_neighbor_objective is None
                else decimal_to_primitive(self.median_neighbor_objective)
            ),
            "relative_dispersion": (
                None
                if self.relative_dispersion is None
                else decimal_to_primitive(self.relative_dispersion)
            ),
            "constraint_pass_fraction": decimal_to_primitive(
                self.constraint_pass_fraction
            ),
            "center_to_neighbor_difference": (
                None
                if self.center_to_neighbor_difference is None
                else decimal_to_primitive(self.center_to_neighbor_difference)
            ),
            "relative_center_to_neighbor_difference": (
                None
                if self.relative_center_to_neighbor_difference is None
                else decimal_to_primitive(self.relative_center_to_neighbor_difference)
            ),
            "is_boundary": self.is_boundary,
            "classification": self.classification.value,
            "is_isolated_peak": self.is_isolated_peak,
            "isolation_reason": self.isolation_reason,
        }


@dataclass(frozen=True, slots=True)
class PredictionGridCacheStatistics:
    context_hits: int
    context_misses: int
    indicator_hits: int
    indicator_misses: int

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "context_hits": self.context_hits,
            "context_misses": self.context_misses,
            "indicator_hits": self.indicator_hits,
            "indicator_misses": self.indicator_misses,
        }


@dataclass(frozen=True, slots=True)
class PredictionGridResult:
    study_id: str
    trials: tuple[PredictionGridTrialRecord, ...]
    rankings: tuple[PredictionRankedTrial, ...]
    ineligible_trials: tuple[PredictionIneligibleTrial, ...]
    stability: tuple[PredictionStabilitySummary, ...]
    cache_statistics: PredictionGridCacheStatistics
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    def summary_primitive(self) -> PrimitiveMapping:
        return {
            "study_id": self.study_id,
            "schema_version": PREDICTION_GRID_SCHEMA_VERSION,
            "counts": {
                "trials": len(self.trials),
                "succeeded": sum(
                    item.status is TrialStatus.SUCCEEDED for item in self.trials
                ),
                "failed": sum(
                    item.status is TrialStatus.FAILED for item in self.trials
                ),
                "excluded": sum(
                    item.status is TrialStatus.EXCLUDED for item in self.trials
                ),
                "eligible": len(self.rankings),
            },
            "rankings": [item.to_primitive() for item in self.rankings],
            "ineligible_trials": [
                item.to_primitive() for item in self.ineligible_trials
            ],
            "stability": [item.to_primitive() for item in self.stability],
            "cache_statistics": self.cache_statistics.to_primitive(),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


class PredictionGridExecutionCache(PredictionIndicatorOutputCache):
    """Study-local cache bound to dataset, backend, and provider identities."""

    def __init__(
        self,
        *,
        dataset_family_fingerprint: str,
        backend: PredictionIndicatorBackendEnvironment,
        context_environment: PredictionContextEnvironment,
    ) -> None:
        self._dataset_family_fingerprint = dataset_family_fingerprint
        self._backend = backend
        self._context_environment = context_environment
        self._contexts: dict[str, MultiTimeframeContext] = {}
        self._indicators: dict[str, TimeframeIndicatorOutput] = {}
        self._context_hits = 0
        self._context_misses = 0
        self._indicator_hits = 0
        self._indicator_misses = 0

    def context(
        self,
        requirements: PredictionContextRequirements,
        provider: PredictionContextProvider,
    ) -> MultiTimeframeContext:
        key = configuration_identity(
            {
                "component": "prediction_grid_context_cache_key",
                "dataset_family_fingerprint": self._dataset_family_fingerprint,
                "context_environment": self._context_environment.to_primitive(),
                "request": _context_request_primitive(requirements),
            }
        )
        cached = self._contexts.get(key)
        if cached is not None:
            self._context_hits += 1
            return cached
        context = provider.get_context(requirements)
        self._contexts[key] = context
        self._context_misses += 1
        return context

    def resolve(
        self,
        requirement: PredictionIndicatorRequirement,
        context: MultiTimeframeContext,
        timeframe: Timeframe,
        completion_policy: ContextCompletionPolicy,
    ) -> TimeframeIndicatorOutput:
        requirement.validate_unchanged()
        backend_identity = requirement.backend_identity
        if backend_identity is not None and not self._backend.matches(backend_identity):
            raise PredictionContextError(
                "indicator backend does not match the fixed prediction-grid environment"
            )
        key = configuration_identity(
            {
                "component": "prediction_grid_indicator_cache_key",
                "dataset_family_fingerprint": self._dataset_family_fingerprint,
                "backend_environment": self._backend.to_primitive(),
                "context_id": context.context_id,
                "timeframe_configuration_id": timeframe.configuration_id,
                "completion_policy": completion_policy.value,
                "indicator_configuration_id": requirement.configuration_id,
                "indicator_backend": (
                    None
                    if backend_identity is None
                    else backend_identity.to_primitive()
                ),
            }
        )
        cached = self._indicators.get(key)
        if cached is not None:
            if (
                cached.backend_identity != backend_identity
                or cached.source_timeframe.configuration_id
                != timeframe.configuration_id
                or cached.completion_policy is not completion_policy
            ):
                raise PredictionContextError(
                    "cached indicator output is identity-incompatible"
                )
            self._indicator_hits += 1
            return cached
        output = requirement.evaluate(context, timeframe, completion_policy)
        if output.backend_identity != backend_identity:
            raise PredictionContextError(
                "normalized indicator output identity does not match its request"
            )
        self._indicators[key] = output
        self._indicator_misses += 1
        return output

    @property
    def statistics(self) -> PredictionGridCacheStatistics:
        return PredictionGridCacheStatistics(
            self._context_hits,
            self._context_misses,
            self._indicator_hits,
            self._indicator_misses,
        )


@dataclass(slots=True)
class _CachedContextProvider:
    source: PredictionContextProvider
    cache: PredictionGridExecutionCache

    def get_context(
        self, requirements: PredictionContextRequirements
    ) -> MultiTimeframeContext:
        return self.cache.context(requirements, self.source)


def _context_request_primitive(
    requirements: PredictionContextRequirements,
) -> PrimitiveMapping:
    def without_indicators(value: PrimitiveMapping) -> PrimitiveMapping:
        detached = PrimitiveMappingSnapshot.capture(value).to_primitive()
        detached["indicators"] = []
        return detached

    return {
        "primary": without_indicators(requirements.primary.to_primitive()),
        "contextual": [
            without_indicators(item.to_primitive()) for item in requirements.contextual
        ],
        "context_completion_policy": requirements.context_completion_policy.value,
    }


def _component_primitive(component: object, label: str) -> PrimitiveMapping:
    try:
        name = cast(str, getattr(component, "name"))
        version = cast(str, getattr(component, "implementation_version"))
        configuration_method = cast(
            Callable[[], PrimitiveMapping], getattr(component, "configuration")
        )
        configuration = configuration_method()
        configuration_id = cast(str, getattr(component, "configuration_id"))
    except (AttributeError, TypeError, ValueError) as error:
        raise InvalidPredictionGridConfigurationError(
            f"{label} does not implement the prediction component contract"
        ) from error
    if (
        not name
        or not version
        or configuration_identity(configuration) != configuration_id
    ):
        raise InvalidPredictionGridConfigurationError(
            f"{label} identity or configuration is invalid"
        )
    return {
        "name": name,
        "implementation_version": version,
        "configuration_id": configuration_id,
        "configuration": configuration,
    }


def _trial_definition(
    study: PredictionStudy[Any, Any, Any],
    backend: PredictionIndicatorBackendEnvironment,
) -> tuple[PrimitiveMapping, tuple[str, ...]]:
    requirements = getattr(study.strategy, "context_requirements", None)
    if not isinstance(requirements, PredictionContextRequirements):
        raise InvalidPredictionGridParametersError(
            "prediction grids require a QF-28 multi-timeframe prediction rule"
        )
    indicator_ids: list[str] = []
    indicator_requirements: list[PrimitiveMapping] = []
    for timeframe_requirement in requirements.all_timeframes:
        for indicator in timeframe_requirement.indicators:
            identity = indicator.backend_identity
            if identity is not None and not backend.matches(identity):
                raise InvalidPredictionGridParametersError(
                    "ordinary prediction grids cannot vary the indicator backend"
                )
            indicator_ids.append(indicator.configuration_id)
            indicator_requirements.append(
                {
                    "timeframe_configuration_id": (
                        timeframe_requirement.timeframe.configuration_id
                    ),
                    "requirement": indicator.to_primitive(),
                }
            )
    definition: PrimitiveMapping = {
        "prediction_rule": _component_primitive(study.strategy, "prediction rule"),
        "prediction_rule_parameters": study.strategy.parameters.to_primitive(),
        "prediction_context": requirements.to_primitive(),
        "indicator_configuration_ids": cast(
            list[Primitive], sorted(set(indicator_ids))
        ),
        "indicator_requirements": cast(list[Primitive], indicator_requirements),
        "indicator_backend_environment": backend.to_primitive(),
        "outcome_labeler": _component_primitive(
            study.outcome_labeler, "outcome labeler"
        ),
        "evaluator": _component_primitive(study.evaluator, "prediction evaluator"),
        "feature_configuration": study.feature_configuration,
        "result_schema_version": study.result_schema_version,
    }
    PrimitiveMappingSnapshot.capture(definition)
    return definition, tuple(sorted(set(indicator_ids)))


def _validate_factory(
    factory: PredictionStudyFactory,
    config: PredictionGridConfig,
    configuration_snapshot: PrimitiveMappingSnapshot,
    parameter_order: tuple[str, ...],
    required_parameter_names: frozenset[str],
) -> None:
    if not factory.name or not factory.version:
        raise InvalidPredictionGridConfigurationError(
            "prediction study factory identity and version are required"
        )
    order = parameter_order
    if not order or len(order) != len(set(order)):
        raise InvalidPredictionGridConfigurationError(
            "prediction study factory parameter order must be nonempty and unique"
        )
    if not required_parameter_names.issubset(order):
        raise InvalidPredictionGridConfigurationError(
            "factory required parameters must belong to its parameter contract"
        )
    missing = required_parameter_names.difference(config.search_space.names)
    if missing:
        raise InvalidPredictionGridConfigurationError(
            "search space omits required prediction parameters: "
            + ", ".join(sorted(missing))
        )
    config.search_space.ordered_items(order)
    try:
        configuration_identity(configuration_snapshot.to_primitive())
    except (TypeError, ValueError) as error:
        raise InvalidPredictionGridConfigurationError(
            "prediction study factory configuration is not deterministic"
        ) from error
    for constraint in config.parameter_constraints:
        constraint.validate(config.search_space.names)
        configuration_identity(constraint.to_primitive())


def _combination_id(
    *,
    factory_name: str,
    factory_version: str,
    factory_configuration: PrimitiveMapping,
    parameters: PrimitiveMapping,
) -> str:
    return configuration_identity(
        {
            "component": "quantforge_prediction_grid_combination",
            "schema_version": PREDICTION_GRID_SCHEMA_VERSION,
            "factory_name": factory_name,
            "factory_version": factory_version,
            "factory_configuration": factory_configuration,
            "parameters": parameters,
        }
    )


def _iter_candidates(
    factory: PredictionStudyFactory,
    config: PredictionGridConfig,
    backend: PredictionIndicatorBackendEnvironment,
    *,
    factory_name: str,
    factory_version: str,
    factory_configuration: PrimitiveMapping,
    parameter_order: tuple[str, ...],
    validate_components: Callable[[], None],
) -> Iterator[PredictionGridCandidate]:
    ordered = config.search_space.ordered_items(parameter_order)
    axes = tuple(values.values for _, values in ordered)
    coordinate_axes = tuple(range(len(axis)) for axis in axes)
    for index, coordinates in enumerate(product(*coordinate_axes)):
        validate_components()
        search_values: dict[str, SearchValue] = {
            name: axes[axis][coordinate]
            for axis, ((name, _), coordinate) in enumerate(
                zip(ordered, coordinates, strict=True)
            )
        }
        parameters: PrimitiveMapping = {
            name: search_value_to_primitive(value)
            for name, value in search_values.items()
        }
        combination_id = _combination_id(
            factory_name=factory_name,
            factory_version=factory_version,
            factory_configuration=factory_configuration,
            parameters=parameters,
        )
        snapshot = PrimitiveMappingSnapshot.capture(parameters)
        failed = next(
            (
                decision
                for constraint in config.parameter_constraints
                if not (decision := constraint.evaluate(search_values.copy())).passed
            ),
            None,
        )
        if failed is not None:
            validate_components()
            yield PredictionGridExclusion(
                index,
                combination_id,
                snapshot,
                tuple(coordinates),
                failed.code,
                failed.message,
            )
            continue
        try:
            study = factory.build(parameters.copy())
            definition, indicator_ids = _trial_definition(study, backend)
        except (
            InvalidPredictionGridParametersError,
            InvalidPredictionConfigurationError,
            PredictionContextError,
        ) as error:
            _, message = _sanitize_failure(error)
            validate_components()
            yield PredictionGridExclusion(
                index,
                combination_id,
                snapshot,
                tuple(coordinates),
                "invalid_prediction_parameters",
                message,
            )
            continue
        validate_components()
        yield PredictionGridCombination(
            index,
            combination_id,
            snapshot,
            tuple(coordinates),
            PrimitiveMappingSnapshot.capture(definition),
            indicator_ids,
            study,
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: PrimitiveMapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_mapping(path: Path) -> PrimitiveMapping:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PredictionGridPersistenceError(
            f"cannot load prediction-grid artifact: {path}"
        ) from error
    if not isinstance(value, dict):
        raise PredictionGridPersistenceError(
            f"prediction-grid artifact must contain an object: {path}"
        )
    return cast(PrimitiveMapping, value)


class _PredictionGridStore:
    def __init__(self, root: Path, study_id: str) -> None:
        self.study_id = study_id
        self.study_path = root / study_id
        self.trials_path = self.study_path / "trials"
        self.artifacts_path = self.study_path / "artifacts"
        self.manifest_path = self.study_path / "manifest.json"

    def initialize(self, manifest: PrimitiveMapping, *, resume: bool) -> None:
        if self.study_path.exists() and any(self.study_path.iterdir()):
            if not self.manifest_path.exists():
                raise PredictionGridPersistenceError(
                    "nonempty prediction-grid store has no manifest"
                )
            if _load_mapping(self.manifest_path) != manifest:
                raise PredictionGridPersistenceError(
                    "prediction-grid manifest does not match the requested study"
                )
            if not resume:
                raise PredictionGridPersistenceError(
                    "prediction-grid study already exists; use resume()"
                )
            return
        if resume:
            raise PredictionGridPersistenceError(
                "cannot resume a prediction-grid study that has no manifest"
            )
        self.trials_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.manifest_path, manifest)

    def trial_path(self, trial_id: str) -> Path:
        return self.trials_path / f"{trial_id}.json"

    def write_trial(self, record: PredictionGridTrialRecord) -> None:
        existing = self.load_trial(record.trial_id)
        if existing is not None and existing.status in (
            TrialStatus.SUCCEEDED,
            TrialStatus.EXCLUDED,
        ):
            if existing.to_primitive() == record.to_primitive():
                return
            raise PredictionGridPersistenceError(
                f"immutable {existing.status.value} prediction trial changed"
            )
        _atomic_json(self.trial_path(record.trial_id), record.to_primitive())

    def load_trial(self, trial_id: str) -> PredictionGridTrialRecord | None:
        path = self.trial_path(trial_id)
        return (
            None
            if not path.exists()
            else PredictionGridTrialRecord.from_primitive(_load_mapping(path))
        )

    def load_trials(self) -> tuple[PredictionGridTrialRecord, ...]:
        if not self.trials_path.exists():
            return ()
        records = tuple(
            PredictionGridTrialRecord.from_primitive(_load_mapping(path))
            for path in sorted(self.trials_path.glob("*.json"))
        )
        return tuple(sorted(records, key=lambda item: item.combination_index))

    def write_artifact(
        self,
        trial_id: str,
        result: PredictionStudyResult[Any, Any, Any],
        analysis: PredictionTrialAnalysis,
    ) -> tuple[str, str]:
        relative = Path("artifacts") / trial_id / "prediction-study.json"
        prediction_study = result.to_primitive()
        content: PrimitiveMapping = {
            "schema_version": PREDICTION_GRID_SCHEMA_VERSION,
            "grid_study_id": self.study_id,
            "trial_id": trial_id,
            "prediction_study_id": result.study_id,
            "prediction_study": prediction_study,
            "analysis": analysis.to_primitive(),
        }
        artifact_fingerprint = configuration_identity(content)
        _atomic_json(
            self.study_path / relative,
            {
                **content,
                "artifact_fingerprint": artifact_fingerprint,
            },
        )
        return relative.as_posix(), artifact_fingerprint

    def validate_artifact(self, record: PredictionGridTrialRecord) -> None:
        if record.artifact_location is None or record.analysis is None:
            raise PredictionGridPersistenceError(
                "completed prediction trial has incomplete artifact metadata"
            )
        artifact = _load_mapping(self.study_path / record.artifact_location)
        persisted_fingerprint = artifact.get("artifact_fingerprint")
        content: PrimitiveMapping = {
            key: value
            for key, value in artifact.items()
            if key != "artifact_fingerprint"
        }
        prediction_study = artifact.get("prediction_study")
        prediction_study_id = artifact.get("prediction_study_id")
        if not isinstance(prediction_study, dict):
            raise PredictionGridPersistenceError(
                "completed prediction trial artifact has no prediction study"
            )
        manifest = prediction_study.get("manifest")
        if (
            artifact.get("schema_version") != PREDICTION_GRID_SCHEMA_VERSION
            or not isinstance(persisted_fingerprint, str)
            or not persisted_fingerprint
            or persisted_fingerprint != record.artifact_fingerprint
            or configuration_identity(content) != persisted_fingerprint
            or artifact.get("grid_study_id") != record.study_id
            or artifact.get("trial_id") != record.trial_id
            or not isinstance(prediction_study_id, str)
            or not prediction_study_id
            or not isinstance(manifest, dict)
            or manifest.get("study_id") != prediction_study_id
            or artifact.get("analysis") != record.analysis.to_primitive()
        ):
            raise PredictionGridPersistenceError(
                "completed prediction trial artifact is incompatible with its record"
            )


def _metric_value(record: PredictionGridTrialRecord, metric: str) -> Decimal | None:
    if record.analysis is None:
        return None
    raw = record.analysis.metrics.get(metric)
    if raw is None or isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        return None
    try:
        value = Decimal(str(raw))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def _passes(actual: Decimal, constraint: PredictionMetricConstraint) -> bool:
    if constraint.operator is ThresholdOperator.GREATER_THAN:
        return actual > constraint.threshold
    if constraint.operator is ThresholdOperator.GREATER_THAN_OR_EQUAL:
        return actual >= constraint.threshold
    if constraint.operator is ThresholdOperator.LESS_THAN:
        return actual < constraint.threshold
    if constraint.operator is ThresholdOperator.LESS_THAN_OR_EQUAL:
        return actual <= constraint.threshold
    if constraint.operator is ThresholdOperator.EQUAL:
        return actual == constraint.threshold
    raise InvalidPredictionGridConfigurationError(
        "unsupported prediction metric threshold operator"
    )


def _rank(
    records: tuple[PredictionGridTrialRecord, ...],
    config: PredictionRankingConfig,
) -> tuple[tuple[PredictionRankedTrial, ...], tuple[PredictionIneligibleTrial, ...]]:
    eligible: list[tuple[PredictionGridTrialRecord, Decimal]] = []
    ineligible: list[PredictionIneligibleTrial] = []
    for record in records:
        if record.status is not TrialStatus.SUCCEEDED or record.analysis is None:
            continue
        reasons: list[str] = []
        if record.analysis.prediction_count < config.minimum_prediction_count:
            reasons.append(
                f"prediction_count={record.analysis.prediction_count} is below "
                f"{config.minimum_prediction_count}"
            )
        objective = _metric_value(record, config.objective_metric)
        if objective is None:
            reasons.append(f"objective metric {config.objective_metric} is undefined")
        for constraint in config.outcome_quality_constraints:
            actual = _metric_value(record, constraint.metric)
            if actual is None:
                reasons.append(f"quality metric {constraint.metric} is undefined")
            elif not _passes(actual, constraint):
                reasons.append(
                    f"{constraint.metric}={actual} does not satisfy "
                    f"{constraint.operator.value} {constraint.threshold}"
                )
        if reasons:
            ineligible.append(
                PredictionIneligibleTrial(
                    record.trial_id, record.combination_id, tuple(reasons)
                )
            )
        else:
            assert objective is not None
            eligible.append((record, objective))
    multiplier = (
        Decimal(-1) if config.direction is RankingDirection.MAXIMIZE else Decimal(1)
    )
    eligible.sort(key=lambda item: (item[1] * multiplier, item[0].combination_id))
    rankings = tuple(
        PredictionRankedTrial(
            index,
            record.trial_id,
            record.combination_id,
            config.objective_metric,
            objective,
        )
        for index, (record, objective) in enumerate(eligible, start=1)
    )
    return rankings, tuple(ineligible)


def _median(values: tuple[Decimal, ...]) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    with localcontext() as context:
        context.prec = 34
        return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _relative_dispersion(values: tuple[Decimal, ...]) -> Decimal | None:
    if not values:
        return None
    with localcontext() as context:
        context.prec = 34
        mean = sum(values, Decimal(0)) / Decimal(len(values))
        if len(values) < 2:
            return Decimal(0)
        variance = sum((value - mean) ** 2 for value in values) / Decimal(len(values))
        deviation = variance.sqrt()
        return None if mean == 0 else deviation / abs(mean)


def _ceiling_fraction(count: int, fraction: Decimal) -> int:
    if count == 0 or fraction == 0:
        return 0
    with localcontext() as context:
        context.prec = 34
        value = (Decimal(count) * fraction).to_integral_value(rounding=ROUND_CEILING)
    return max(1, int(value))


def _directional_difference(
    center: Decimal, neighbor: Decimal, direction: RankingDirection
) -> Decimal:
    with localcontext() as context:
        context.prec = 34
        return (
            center - neighbor
            if direction is RankingDirection.MAXIMIZE
            else neighbor - center
        )


def _is_boundary(
    coordinates: tuple[int, ...],
    axis_kinds: tuple[str, ...],
    axis_lengths: tuple[int, ...],
) -> bool:
    numeric_axes = tuple(
        axis for axis, kind in enumerate(axis_kinds) if kind in ("integer", "float")
    )
    return any(
        coordinates[axis] in (0, axis_lengths[axis] - 1) for axis in numeric_axes
    )


def _is_isolated_peak(
    *,
    objective_rank: int,
    eligible_count: int,
    eligible_neighbor_count: int,
    pass_fraction: Decimal,
    difference: Decimal | None,
    relative_difference: Decimal | None,
    is_boundary: bool,
    config: StabilityConfig,
) -> tuple[bool, str | None]:
    top_count = _ceiling_fraction(eligible_count, config.isolated_peak_top_fraction)
    if objective_rank > top_count:
        return False, None
    if eligible_neighbor_count < config.minimum_eligible_neighbors:
        return False, "insufficient eligible neighbors for isolated-peak classification"
    if difference is None or difference < config.isolated_peak_absolute_drop:
        return False, None
    if (
        relative_difference is None
        or relative_difference < config.isolated_peak_relative_drop
    ):
        return False, None
    if pass_fraction > config.isolated_peak_maximum_constraint_pass_fraction:
        return False, None
    boundary_note = " on a search-space boundary" if is_boundary else ""
    return (
        True,
        "high-ranked center exceeds its eligible-neighbor median by the configured "
        "absolute and relative drops while neighbor constraint pass rate is low"
        f"{boundary_note}",
    )


def _neighbor_coordinates(
    coordinates: tuple[int, ...],
    axis_kinds: tuple[str, ...],
    axis_lengths: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    values: list[tuple[int, ...]] = []
    for axis, (kind, length) in enumerate(zip(axis_kinds, axis_lengths, strict=True)):
        positions = (
            tuple(
                value
                for value in (coordinates[axis] - 1, coordinates[axis] + 1)
                if 0 <= value < length
            )
            if kind in ("integer", "float")
            else tuple(value for value in range(length) if value != coordinates[axis])
        )
        for position in positions:
            neighbor = list(coordinates)
            neighbor[axis] = position
            values.append(tuple(neighbor))
    return tuple(values)


def _stability(
    candidates: tuple[PredictionGridCandidate, ...],
    rankings: tuple[PredictionRankedTrial, ...],
    search_space: ParameterSearchSpace,
    factory: PredictionStudyFactory,
    ranking_config: PredictionRankingConfig,
    config: StabilityConfig,
) -> tuple[PredictionStabilitySummary, ...]:
    by_coordinates = {candidate.coordinates: candidate for candidate in candidates}
    valid_by_id = {
        candidate.combination_id: candidate
        for candidate in candidates
        if isinstance(candidate, PredictionGridCombination)
    }
    ranked_by_id = {item.combination_id: item for item in rankings}
    ordered = search_space.ordered_items(factory.parameter_order)
    kinds = tuple(values.kind for _, values in ordered)
    lengths = tuple(len(values.values) for _, values in ordered)
    summaries: list[PredictionStabilitySummary] = []
    for ranked in rankings:
        center = valid_by_id[ranked.combination_id]
        neighbors = tuple(
            by_coordinates[item]
            for item in _neighbor_coordinates(center.coordinates, kinds, lengths)
            if item in by_coordinates
        )
        valid = tuple(
            item for item in neighbors if isinstance(item, PredictionGridCombination)
        )
        values = tuple(
            ranked_by_id[item.combination_id].objective_value
            for item in valid
            if item.combination_id in ranked_by_id
        )
        with localcontext() as context:
            context.prec = 34
            pass_fraction = (
                Decimal(0) if not valid else Decimal(len(values)) / Decimal(len(valid))
            )
        dispersion = _relative_dispersion(values)
        median = _median(values)
        difference = (
            None
            if median is None
            else _directional_difference(
                ranked.objective_value, median, ranking_config.direction
            )
        )
        relative_difference = (
            None
            if difference is None or ranked.objective_value == 0
            else difference / abs(ranked.objective_value)
        )
        classification = (
            StabilityClassification.INSUFFICIENT_NEIGHBORS
            if len(values) < config.minimum_eligible_neighbors
            else StabilityClassification.STABLE
            if (
                pass_fraction >= config.stable_constraint_pass_fraction
                and dispersion is not None
                and dispersion <= config.stable_maximum_relative_dispersion
            )
            else StabilityClassification.MIXED
            if pass_fraction >= Decimal("0.5")
            else StabilityClassification.FRAGILE
        )
        boundary = _is_boundary(center.coordinates, kinds, lengths)
        isolated, isolation_reason = _is_isolated_peak(
            objective_rank=ranked.rank,
            eligible_count=len(rankings),
            eligible_neighbor_count=len(values),
            pass_fraction=pass_fraction,
            difference=difference,
            relative_difference=relative_difference,
            is_boundary=boundary,
            config=config,
        )
        if isolated and classification is StabilityClassification.STABLE:
            classification = StabilityClassification.FRAGILE
        summaries.append(
            PredictionStabilitySummary(
                trial_id=ranked.trial_id,
                combination_id=ranked.combination_id,
                objective_rank=ranked.rank,
                objective_value=ranked.objective_value,
                valid_neighbor_count=len(valid),
                excluded_neighbor_count=sum(
                    isinstance(item, PredictionGridExclusion) for item in neighbors
                ),
                eligible_neighbor_count=len(values),
                neighbor_objective_values=values,
                median_neighbor_objective=median,
                relative_dispersion=dispersion,
                constraint_pass_fraction=pass_fraction,
                center_to_neighbor_difference=difference,
                relative_center_to_neighbor_difference=relative_difference,
                is_boundary=boundary,
                classification=classification,
                is_isolated_peak=isolated,
                isolation_reason=isolation_reason,
            )
        )
    return tuple(summaries)


def _sanitize_failure(error: Exception) -> tuple[str, str]:
    failure_type = error.__class__.__name__
    return (
        failure_type,
        f"{failure_type}: prediction trial failed; raw exception text was not "
        "persisted",
    )


def _validate_analysis_for_grid(
    analysis: PredictionTrialAnalysis,
    ranking: PredictionRankingConfig,
) -> None:
    if analysis.prediction_count > 0 and (
        not analysis.period_comparisons
        or not analysis.weekday_comparisons
        or not analysis.matched_baseline_comparisons
    ):
        raise InvalidPredictionGridConfigurationError(
            "nonempty prediction trials must retain period, weekday, and matched-"
            "baseline comparisons"
        )
    for comparison in analysis.matched_baseline_comparisons:
        if comparison.to_primitive().get("baseline_name") != ranking.baseline_name:
            raise InvalidPredictionGridConfigurationError(
                "matched-baseline comparison does not match the configured baseline"
            )


class PredictionGridStudy:
    """Run, persist, resume, rank, and analyze one deterministic prediction grid."""

    def __init__(
        self,
        *,
        dataset: MarketDataset,
        dataset_family_fingerprint: str,
        study_factory: PredictionStudyFactory,
        analyzer: PredictionTrialAnalyzer,
        context_provider: PredictionContextProvider,
        context_environment: PredictionContextEnvironment,
        indicator_backend: PredictionIndicatorBackendEnvironment,
        config: PredictionGridConfig,
    ) -> None:
        if not dataset_family_fingerprint:
            raise InvalidPredictionGridConfigurationError(
                "dataset-family fingerprint is required"
            )
        prepared = prepare_prediction_study_dataset(dataset)
        try:
            factory_configuration_snapshot = PrimitiveMappingSnapshot.capture(
                study_factory.configuration()
            )
            analyzer_configuration_snapshot = PrimitiveMappingSnapshot.capture(
                analyzer.configuration()
            )
            factory_name = study_factory.name
            factory_version = study_factory.version
            parameter_order = tuple(study_factory.parameter_order)
            required_parameter_names = frozenset(study_factory.required_parameter_names)
            analyzer_name = analyzer.name
            analyzer_version = analyzer.version
            analyzer_configuration_id = analyzer.configuration_id
        except (AttributeError, TypeError, ValueError) as error:
            raise InvalidPredictionGridConfigurationError(
                "prediction factory or analyzer configuration is invalid"
            ) from error
        _validate_factory(
            study_factory,
            config,
            factory_configuration_snapshot,
            parameter_order,
            required_parameter_names,
        )
        if (
            not analyzer_name
            or not analyzer_version
            or configuration_identity(analyzer_configuration_snapshot.to_primitive())
            != analyzer_configuration_id
        ):
            raise InvalidPredictionGridConfigurationError(
                "prediction trial analyzer identity is invalid"
            )
        count = config.search_space.combination_count()
        if count > config.maximum_combinations:
            raise InvalidPredictionGridConfigurationError(
                f"prediction grid contains {count} combinations, exceeding "
                f"maximum {config.maximum_combinations}"
            )
        identity: PrimitiveMapping = {
            "component": "quantforge_prediction_parameter_grid",
            "engine_version": PREDICTION_GRID_ENGINE_VERSION,
            "schema_version": PREDICTION_GRID_SCHEMA_VERSION,
            "label": config.label,
            "dataset": prepared.market_data.to_primitive(),
            "dataset_family_fingerprint": dataset_family_fingerprint,
            "study_factory": {
                "name": factory_name,
                "version": factory_version,
                "configuration": factory_configuration_snapshot.to_primitive(),
            },
            "analyzer": {
                "name": analyzer_name,
                "version": analyzer_version,
                "configuration_id": analyzer_configuration_id,
                "configuration": analyzer_configuration_snapshot.to_primitive(),
            },
            "context_environment": context_environment.to_primitive(),
            "indicator_backend": indicator_backend.to_primitive(),
            "search_space": config.search_space.to_primitive(parameter_order),
            "parameter_constraints": [
                item.to_primitive() for item in config.parameter_constraints
            ],
            "ranking": config.ranking.to_primitive(),
            "stability": config.stability.to_primitive(),
        }
        self.study_id = configuration_identity(identity)
        self._manifest: PrimitiveMapping = {
            **identity,
            "study_id": self.study_id,
            "execution": {
                "mode": "sequential",
                "retry_failed": config.retry_failed,
                "failure_policy": "persist_and_continue",
            },
            "cache_policy": {
                "context": "dataset_family_context_environment_and_request_identity",
                "normalized_indicators": (
                    "dataset_family_context_timeframe_indicator_backend_and_configuration"
                ),
            },
            "interpretation": (
                "rankings are in-sample research comparisons and are not validated "
                "profitability or financial advice"
            ),
        }
        self._prepared = prepared
        self._dataset_family_fingerprint = dataset_family_fingerprint
        self._factory = study_factory
        self._factory_name = factory_name
        self._factory_version = factory_version
        self._factory_configuration_snapshot = factory_configuration_snapshot
        self._factory_parameter_order = parameter_order
        self._factory_required_parameter_names = required_parameter_names
        self._analyzer = analyzer
        self._analyzer_name = analyzer_name
        self._analyzer_version = analyzer_version
        self._analyzer_configuration_id = analyzer_configuration_id
        self._analyzer_configuration_snapshot = analyzer_configuration_snapshot
        self._context_provider = context_provider
        self._context_environment = context_environment
        self._backend = indicator_backend
        self._config = config
        self._store = _PredictionGridStore(config.output_root, self.study_id)

    def run(self) -> PredictionGridResult:
        return self._execute(resume=False)

    def resume(self) -> PredictionGridResult:
        return self._execute(resume=True)

    def load_result(self) -> PredictionGridResult:
        self._validate_components_unchanged()
        if not self._store.manifest_path.exists():
            raise PredictionGridPersistenceError(
                "prediction-grid study has not been persisted"
            )
        if _load_mapping(self._store.manifest_path) != self._manifest:
            raise PredictionGridPersistenceError(
                "persisted prediction-grid manifest is incompatible"
            )
        candidates = tuple(self._iter_candidates())
        return self._result(
            self._store.load_trials(),
            PredictionGridCacheStatistics(0, 0, 0, 0),
            candidates,
        )

    def _validate_components_unchanged(self) -> None:
        try:
            factory_configuration = PrimitiveMappingSnapshot.capture(
                self._factory.configuration()
            )
            analyzer_configuration = PrimitiveMappingSnapshot.capture(
                self._analyzer.configuration()
            )
            factory_required_parameter_names = frozenset(
                self._factory.required_parameter_names
            )
            analyzer_configuration_id = self._analyzer.configuration_id
        except (AttributeError, TypeError, ValueError) as error:
            raise InvalidPredictionGridConfigurationError(
                "prediction factory or analyzer changed after grid construction"
            ) from error
        if (
            self._factory.name != self._factory_name
            or self._factory.version != self._factory_version
            or tuple(self._factory.parameter_order) != self._factory_parameter_order
            or factory_required_parameter_names
            != self._factory_required_parameter_names
            or factory_configuration != self._factory_configuration_snapshot
            or self._analyzer.name != self._analyzer_name
            or self._analyzer.version != self._analyzer_version
            or analyzer_configuration_id != self._analyzer_configuration_id
            or analyzer_configuration != self._analyzer_configuration_snapshot
        ):
            raise InvalidPredictionGridConfigurationError(
                "prediction factory or analyzer changed after grid construction"
            )

    def _iter_candidates(self) -> Iterator[PredictionGridCandidate]:
        self._validate_components_unchanged()
        return _iter_candidates(
            self._factory,
            self._config,
            self._backend,
            factory_name=self._factory_name,
            factory_version=self._factory_version,
            factory_configuration=(self._factory_configuration_snapshot.to_primitive()),
            parameter_order=self._factory_parameter_order,
            validate_components=self._validate_components_unchanged,
        )

    def _trial_id(self, candidate: PredictionGridCandidate) -> str:
        return configuration_identity(
            {
                "component": "quantforge_prediction_grid_trial",
                "schema_version": PREDICTION_GRID_SCHEMA_VERSION,
                "study_id": self.study_id,
                "combination_id": candidate.combination_id,
                "dataset_family_fingerprint": self._dataset_family_fingerprint,
                "indicator_backend": self._backend.to_primitive(),
                "trial_definition": (
                    None
                    if isinstance(candidate, PredictionGridExclusion)
                    else candidate.trial_definition_snapshot.to_primitive()
                ),
            }
        )

    def _initial_record(
        self, candidate: PredictionGridCandidate
    ) -> PredictionGridTrialRecord:
        exclusion = isinstance(candidate, PredictionGridExclusion)
        return PredictionGridTrialRecord(
            self.study_id,
            self._trial_id(candidate),
            candidate.combination_id,
            candidate.index,
            TrialStatus.EXCLUDED if exclusion else TrialStatus.PENDING,
            candidate.parameters_snapshot,
            (None if exclusion else candidate.trial_definition_snapshot),
            PrimitiveMappingSnapshot.capture(self._backend.to_primitive()),
            self._dataset_family_fingerprint,
            (() if exclusion else candidate.indicator_configuration_ids),
            exclusion_code=(candidate.reason_code if exclusion else None),
            exclusion_reason=(candidate.reason if exclusion else None),
        )

    def _execute(self, *, resume: bool) -> PredictionGridResult:
        self._store.initialize(self._manifest, resume=resume)
        self._validate_components_unchanged()
        cache = PredictionGridExecutionCache(
            dataset_family_fingerprint=self._dataset_family_fingerprint,
            backend=self._backend,
            context_environment=self._context_environment,
        )
        provider = _CachedContextProvider(self._context_provider, cache)
        candidates: list[PredictionGridCandidate] = []
        for candidate in self._iter_candidates():
            candidates.append(candidate)
            initial = self._initial_record(candidate)
            existing = self._store.load_trial(initial.trial_id)
            if existing is None:
                self._store.write_trial(initial)
                existing = initial
            self._validate_record(candidate, existing)
            if isinstance(candidate, PredictionGridExclusion):
                continue
            if existing.status is TrialStatus.SUCCEEDED:
                continue
            if existing.status is TrialStatus.FAILED and not self._config.retry_failed:
                continue
            failed_attempts = existing.failed_attempts
            if existing.status is TrialStatus.FAILED:
                if (
                    existing.failure_type is None
                    or existing.failure_message is None
                    or existing.started_at is None
                    or existing.finished_at is None
                ):
                    raise PredictionGridPersistenceError(
                        "failed prediction trial cannot be archived for retry"
                    )
                failed_attempts = (
                    *failed_attempts,
                    PredictionGridFailedAttempt(
                        existing.failure_type,
                        existing.failure_message,
                        existing.started_at,
                        existing.finished_at,
                    ),
                )
            started = replace(
                existing,
                status=TrialStatus.RUNNING,
                analysis=None,
                artifact_location=None,
                artifact_fingerprint=None,
                failure_type=None,
                failure_message=None,
                failed_attempts=failed_attempts,
                started_at=_utc_now(),
                finished_at=None,
            )
            self._store.write_trial(started)
            try:
                result = run_prediction_study_in_session(
                    self._prepared,
                    candidate.study,
                    context_provider=provider,
                    indicator_output_cache=cache,
                )
                analysis = self._analyzer.analyze(result)
                _validate_analysis_for_grid(analysis, self._config.ranking)
            except Exception as error:
                failure_type, failure_message = _sanitize_failure(error)
                completed = replace(
                    started,
                    status=TrialStatus.FAILED,
                    failure_type=failure_type,
                    failure_message=failure_message,
                    finished_at=_utc_now(),
                )
            else:
                self._validate_components_unchanged()
                artifact, artifact_fingerprint = self._store.write_artifact(
                    started.trial_id, result, analysis
                )
                completed = replace(
                    started,
                    status=TrialStatus.SUCCEEDED,
                    analysis=analysis,
                    artifact_location=artifact,
                    artifact_fingerprint=artifact_fingerprint,
                    finished_at=_utc_now(),
                )
            self._store.write_trial(completed)
        candidate_snapshot = tuple(candidates)
        records = self._store.load_trials()
        self._validate_components_unchanged()
        result = self._result(records, cache.statistics, candidate_snapshot)
        _atomic_json(
            self._store.study_path / "summary.json", result.summary_primitive()
        )
        return result

    def _result(
        self,
        records: tuple[PredictionGridTrialRecord, ...],
        cache_statistics: PredictionGridCacheStatistics,
        candidates: tuple[PredictionGridCandidate, ...],
    ) -> PredictionGridResult:
        expected_ids = {self._trial_id(candidate) for candidate in candidates}
        actual_ids = {record.trial_id for record in records}
        if actual_ids != expected_ids:
            raise PredictionGridPersistenceError(
                "persisted prediction trials do not match the deterministic grid"
            )
        candidates_by_trial_id = {
            self._trial_id(candidate): candidate for candidate in candidates
        }
        for record in records:
            self._validate_record(candidates_by_trial_id[record.trial_id], record)
        rankings, ineligible = _rank(records, self._config.ranking)
        stability = _stability(
            candidates,
            rankings,
            self._config.search_space,
            self._factory,
            self._config.ranking,
            self._config.stability,
        )
        warnings = [
            f"searched {len(candidates)} parameter combinations without a "
            "multiple-comparison correction; in-sample rankings may contain "
            "false discoveries"
        ]
        if not rankings:
            warnings.append(
                "no successful prediction trial satisfied ranking eligibility"
            )
        limitations = (
            "Rankings are in-sample research comparisons, not validated profitability.",
            "No QF-5 order, fill, fee, slippage, or portfolio simulation was "
            "performed.",
            "Walk-forward and untouched holdout validation remain separate "
            "follow-up work.",
        )
        return PredictionGridResult(
            self.study_id,
            records,
            rankings,
            ineligible,
            stability,
            cache_statistics,
            tuple(warnings),
            limitations,
        )

    def _validate_record(
        self,
        candidate: PredictionGridCandidate,
        record: PredictionGridTrialRecord,
    ) -> None:
        expected = self._initial_record(candidate)
        if (
            record.study_id != expected.study_id
            or record.trial_id != expected.trial_id
            or record.combination_id != expected.combination_id
            or record.combination_index != expected.combination_index
            or record.parameters_snapshot != expected.parameters_snapshot
            or record.trial_definition_snapshot != expected.trial_definition_snapshot
            or record.backend_snapshot != expected.backend_snapshot
            or record.dataset_family_fingerprint != expected.dataset_family_fingerprint
            or record.indicator_configuration_ids
            != expected.indicator_configuration_ids
        ):
            raise PredictionGridPersistenceError(
                "persisted prediction trial does not match its deterministic candidate"
            )
        if isinstance(candidate, PredictionGridExclusion):
            if (
                record.status is not TrialStatus.EXCLUDED
                or record.exclusion_code != candidate.reason_code
                or record.exclusion_reason != candidate.reason
            ):
                raise PredictionGridPersistenceError(
                    "persisted prediction exclusion changed"
                )
            return
        if record.status is TrialStatus.EXCLUDED:
            raise PredictionGridPersistenceError(
                "valid prediction candidate cannot be persisted as excluded"
            )
        if record.status is TrialStatus.SUCCEEDED:
            expected_artifact = (
                Path("artifacts") / record.trial_id / "prediction-study.json"
            ).as_posix()
            if (
                record.analysis is None
                or record.artifact_location != expected_artifact
                or not record.artifact_fingerprint
                or not (self._store.study_path / expected_artifact).is_file()
            ):
                raise PredictionGridPersistenceError(
                    "completed prediction trial artifact is missing or incompatible"
                )
            self._store.validate_artifact(record)
        if record.status is TrialStatus.FAILED and (
            not record.failure_type or not record.failure_message
        ):
            raise PredictionGridPersistenceError(
                "failed prediction trial has incomplete diagnostic context"
            )


__all__ = [
    "PREDICTION_GRID_ENGINE_VERSION",
    "PREDICTION_GRID_SCHEMA_VERSION",
    "InvalidPredictionGridConfigurationError",
    "InvalidPredictionGridParametersError",
    "PredictionContextEnvironment",
    "PredictionGridCacheStatistics",
    "PredictionGridConfig",
    "PredictionGridError",
    "PredictionGridExecutionCache",
    "PredictionGridFailedAttempt",
    "PredictionGridPersistenceError",
    "PredictionGridResult",
    "PredictionGridStudy",
    "PredictionGridTrialRecord",
    "PredictionIndicatorBackendEnvironment",
    "PredictionIneligibleTrial",
    "PredictionMetricConstraint",
    "PredictionRankedTrial",
    "PredictionRankingConfig",
    "PredictionStabilitySummary",
    "PredictionStudyFactory",
    "PredictionTrialAnalysis",
    "PredictionTrialAnalyzer",
]
