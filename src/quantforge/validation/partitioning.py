"""Leakage-safe membership, horizon purging, embargo, and warm-up selection."""

from dataclasses import dataclass
from itertools import pairwise
from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import AggregatedSessionBar, MarketDataset, TimeframeBarSeries
from quantforge.timeframes import Timeframe, resolve_exchange_session
from quantforge.validation.errors import ValidationPlanError
from quantforge.validation.models import (
    BoundaryAxis,
    DatasetProvenance,
    ExchangeSessionBoundary,
    PartitionRole,
    TimestampBoundary,
    ValidationBoundary,
    ValidationPlan,
    ValidationWindow,
    boundaries_share_semantics,
    boundary_value,
)


def _validate_observations(
    observations: tuple[ValidationBoundary, ...],
    reference: ValidationBoundary,
) -> None:
    if not isinstance(cast(object, observations), tuple):
        raise ValidationPlanError("validation observations must be a tuple")
    if any(not boundaries_share_semantics(reference, item) for item in observations):
        raise ValidationPlanError(
            "validation observations do not match the plan temporal semantics"
        )
    if any(
        boundary_value(current) >= boundary_value(following)
        for current, following in pairwise(observations)
    ):
        raise ValidationPlanError(
            "validation observations must be unique and strictly chronological"
        )


def _boundary_primitive(boundary: ValidationBoundary) -> PrimitiveMapping:
    return boundary.to_primitive()


def _validate_source_observations(
    observations: tuple[ValidationBoundary, ...],
    reference: ValidationBoundary,
    source: MarketDataset | TimeframeBarSeries,
    *,
    plan: ValidationPlan | None = None,
) -> tuple[str, Timeframe, str | None]:
    """Verify a prefix of an artifact's completed-bar chronology, never infer it."""
    _validate_observations(observations, reference)
    source_value = cast(object, source)
    keys: tuple[ValidationBoundary, ...]
    manifest_id: str | None = None
    if isinstance(source_value, MarketDataset):
        provenance = DatasetProvenance.from_market_dataset(source_value)
        assert provenance.standalone_timeframe is not None
        timeframe = provenance.standalone_timeframe
        dataset_id = source_value.metadata.dataset_id
        if plan is not None and provenance != plan.environment.dataset:
            raise ValidationPlanError("observation source does not match plan dataset")
        if reference.axis is BoundaryAxis.EXCHANGE_SESSION:
            keys = tuple(
                ExchangeSessionBoundary(bar.session_date, timeframe.session_policy)
                for bar in source_value.bars
            )
        else:
            keys = tuple(
                TimestampBoundary(
                    resolve_exchange_session(
                        bar.session_date, timeframe.session_policy
                    ).close_timestamp
                )
                for bar in source_value.bars
            )
    elif isinstance(source_value, TimeframeBarSeries):
        timeframe = source_value.timeframe
        dataset_id = source_value.dataset_reference.dataset_id
        manifest_id = source_value.dataset_family_manifest_id
        if plan is not None and (
            source_value.dataset_reference
            not in plan.environment.dataset.family_references
            or manifest_id != plan.environment.dataset.family_manifest_id
        ):
            raise ValidationPlanError("observation source does not match plan dataset")
        if any(not bar.complete for bar in source_value.bars):
            raise ValidationPlanError("validation chronology requires completed bars")
        if reference.axis is BoundaryAxis.EXCHANGE_SESSION:
            if any(
                not isinstance(bar, AggregatedSessionBar) for bar in source_value.bars
            ):
                raise ValidationPlanError(
                    "intraday observations require timestamp boundaries"
                )
            keys = tuple(
                ExchangeSessionBoundary(
                    cast(AggregatedSessionBar, bar).session_dates[-1],
                    timeframe.session_policy,
                )
                for bar in source_value.bars
            )
        else:
            keys = tuple(
                TimestampBoundary(bar.end_timestamp) for bar in source_value.bars
            )
    else:
        raise ValidationPlanError(
            "observations require a validated dataset or timeframe series"
        )
    if plan is not None:
        rule_timeframe_id = (
            plan.environment.research_rule.warm_up_timeframe_configuration_id
            or plan.environment.timeframes[0].configuration_id
        )
        if timeframe.configuration_id != rule_timeframe_id:
            raise ValidationPlanError(
                "purge observations must use the research rule source timeframe"
            )
    _validate_observations(keys, reference)
    if len(observations) > len(keys) or observations != keys[: len(observations)]:
        raise ValidationPlanError(
            "observations must be an exact prefix of the source bar chronology"
        )
    return dataset_id, timeframe, manifest_id


@dataclass(frozen=True, slots=True)
class PurgedPartitionObservations:
    """Deterministic earlier-partition membership after purging and embargo."""

    plan_id: str
    fold_id: str
    source_window_id: str
    protected_window_id: str
    retained: tuple[ValidationBoundary, ...]
    purged: tuple[ValidationBoundary, ...]
    source_dataset_id: str
    source_timeframe_configuration_id: str

    def _identity_primitive(self) -> PrimitiveMapping:
        return {
            "plan_id": self.plan_id,
            "fold_id": self.fold_id,
            "source_window_id": self.source_window_id,
            "protected_window_id": self.protected_window_id,
            "source_dataset_id": self.source_dataset_id,
            "source_timeframe_configuration_id": self.source_timeframe_configuration_id,
            "retained": [_boundary_primitive(item) for item in self.retained],
            "purged": [_boundary_primitive(item) for item in self.purged],
        }

    @property
    def result_id(self) -> str:
        return configuration_identity(self._identity_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {"result_id": self.result_id, **self._identity_primitive()}


def purge_development_observations(
    plan: ValidationPlan,
    fold_index: int,
    observations: tuple[ValidationBoundary, ...],
    *,
    source: MarketDataset | TimeframeBarSeries,
) -> PurgedPartitionObservations:
    """Purge a prefix of the plan-bound source artifact's completed observations."""
    return purge_partition_observations(
        plan,
        fold_index,
        PartitionRole.DEVELOPMENT,
        observations,
        source=source,
    )


def purge_partition_observations(
    plan: ValidationPlan,
    fold_index: int,
    source_role: PartitionRole,
    observations: tuple[ValidationBoundary, ...],
    *,
    source: MarketDataset | TimeframeBarSeries,
) -> PurgedPartitionObservations:
    """Purge one partition using keys verified against the plan-bound source.

    Observations must be an exact prefix of source bar ends (timestamps) or
    final constituent sessions (session keys). A new artifact needs a new plan.
    """
    index = cast(object, fold_index)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValidationPlanError(
            "validation fold index must be a non-negative integer"
        )
    try:
        fold = plan.folds[index]
    except IndexError as error:
        raise ValidationPlanError("validation fold index is out of range") from error
    source_window, protected = _source_and_protected_windows(plan, index, source_role)
    dataset_id, timeframe, _ = _validate_source_observations(
        observations, source_window.interval.start, source, plan=plan
    )
    membership = tuple(
        item for item in observations if source_window.interval.contains(item)
    )
    if not membership:
        raise ValidationPlanError(
            "source validation window has no observations in the supplied chronology"
        )
    purge_cutoff = _purge_cutoff(plan, source_window, protected, observations)
    retained: list[ValidationBoundary] = []
    purged: list[ValidationBoundary] = []
    for observation in membership:
        target = (
            purged
            if boundary_value(observation) >= boundary_value(purge_cutoff)
            else retained
        )
        target.append(observation)
    return PurgedPartitionObservations(
        plan.plan_id,
        fold.fold_id,
        source_window.window_id,
        protected.window_id,
        tuple(retained),
        tuple(purged),
        dataset_id,
        timeframe.configuration_id,
    )


def _source_and_protected_windows(
    plan: ValidationPlan,
    fold_index: int,
    source_role: PartitionRole,
) -> tuple[ValidationWindow, ValidationWindow]:
    fold = plan.folds[fold_index]
    if source_role is PartitionRole.DEVELOPMENT:
        return fold.development, fold.next_protected_window
    if source_role is PartitionRole.SELECTION:
        if fold.selection is None:
            raise ValidationPlanError(
                "cannot purge selection membership for a fold without selection"
            )
        return fold.selection, fold.test
    if source_role is PartitionRole.WALK_FORWARD_TEST:
        protected = (
            plan.folds[fold_index + 1].test
            if fold_index + 1 < len(plan.folds)
            else plan.final_holdout.window
        )
        return fold.test, protected
    raise ValidationPlanError(
        "final holdout is reserved and cannot be a purge source partition"
    )


def _purge_cutoff(
    plan: ValidationPlan,
    source: ValidationWindow,
    protected: ValidationWindow,
    observations: tuple[ValidationBoundary, ...],
) -> ValidationBoundary:
    protected_start = protected.interval.start
    if plan.axis is BoundaryAxis.TIMESTAMP:
        start = cast(TimestampBoundary, protected_start).timestamp
        horizon = plan.purge_policy.label_horizon.elapsed
        embargo = plan.purge_policy.embargo.elapsed
        assert horizon is not None
        assert embargo is not None
        return TimestampBoundary(start - horizon - embargo)
    source_start = cast(ExchangeSessionBoundary, source.interval.start)
    protected_session = cast(ExchangeSessionBoundary, protected_start)
    horizon_sessions = plan.purge_policy.label_horizon.exchange_sessions
    embargo_sessions = plan.purge_policy.embargo.exchange_sessions
    assert horizon_sessions is not None
    assert embargo_sessions is not None
    separation_sessions = horizon_sessions + embargo_sessions
    if separation_sessions == 0:
        return protected_session
    if not any(protected.interval.contains(item) for item in observations):
        raise ValidationPlanError(
            "protected validation window has no observations in the supplied chronology"
        )
    return _observed_session_purge_cutoff(
        observations,
        source_start,
        protected_session,
        separation_sessions,
    )


def _observed_session_purge_cutoff(
    observations: tuple[ValidationBoundary, ...],
    source_start: ExchangeSessionBoundary,
    protected_start: ExchangeSessionBoundary,
    separation_sessions: int,
) -> ExchangeSessionBoundary:
    if source_start.session_date >= protected_start.session_date:
        raise ValidationPlanError(
            "source partition must precede the protected interval"
        )
    protected_index = next(
        (
            index
            for index, observation in enumerate(observations)
            if boundary_value(observation) >= boundary_value(protected_start)
        ),
        None,
    )
    assert protected_index is not None
    cutoff_index = protected_index - separation_sessions
    assert cutoff_index < protected_index
    if cutoff_index < 0:
        return source_start
    cutoff = cast(ExchangeSessionBoundary, observations[cutoff_index])
    if cutoff.session_date < source_start.session_date:
        return source_start
    return cutoff


@dataclass(frozen=True, slots=True)
class WindowObservationSelection:
    """Warm-up-only context kept structurally separate from study membership."""

    window_id: str
    warm_up_context: tuple[ValidationBoundary, ...]
    study_observations: tuple[ValidationBoundary, ...]
    source_dataset_id: str
    source_timeframe_configuration_id: str
    source_family_manifest_id: str | None = None

    def _identity_primitive(self) -> PrimitiveMapping:
        return {
            "window_id": self.window_id,
            "source_dataset_id": self.source_dataset_id,
            "source_family_manifest_id": self.source_family_manifest_id,
            "source_timeframe_configuration_id": (
                self.source_timeframe_configuration_id
            ),
            "warm_up_context": [
                _boundary_primitive(item) for item in self.warm_up_context
            ],
            "study_observations": [
                _boundary_primitive(item) for item in self.study_observations
            ],
            "warm_up_eligible_for_selection": False,
        }

    @property
    def selection_id(self) -> str:
        return configuration_identity(self._identity_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {"selection_id": self.selection_id, **self._identity_primitive()}


def select_window_observations(
    window: ValidationWindow,
    observations: tuple[ValidationBoundary, ...],
    *,
    source: MarketDataset | TimeframeBarSeries,
    source_timeframe: Timeframe | None = None,
) -> WindowObservationSelection:
    """Select preceding context and membership from a verified artifact prefix.

    An explicit source_timeframe must match the artifact, not merely the window's
    boundary axis. The result preserves the source dataset and timeframe IDs.
    """
    dataset_id, timeframe, manifest_id = _validate_source_observations(
        observations, window.interval.start, source
    )
    if source_timeframe is not None and source_timeframe != timeframe:
        raise ValidationPlanError(
            "selected source timeframe does not match observation artifact"
        )
    warm_up_observations = window.warm_up_observations_for(source_timeframe)
    study_indexes = tuple(
        index
        for index, observation in enumerate(observations)
        if window.interval.contains(observation)
    )
    if not study_indexes:
        raise ValidationPlanError(
            "validation window has no observations in the supplied chronology"
        )
    first_index = study_indexes[0]
    warm_up_start = first_index - warm_up_observations
    if warm_up_start < 0:
        raise ValidationPlanError(
            "insufficient historical observations for validation window warm-up"
        )
    warm_up = observations[warm_up_start:first_index]
    study = tuple(observations[index] for index in study_indexes)
    if warm_up and boundary_value(warm_up[-1]) >= boundary_value(window.interval.start):
        raise ValidationPlanError(
            "validation warm-up context must precede the protected interval"
        )
    return WindowObservationSelection(
        window.window_id,
        warm_up,
        study,
        dataset_id,
        timeframe.configuration_id,
        manifest_id,
    )
