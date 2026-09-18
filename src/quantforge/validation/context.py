"""Plan-bound, timestamp-preserving feature context for prediction studies."""

from dataclasses import dataclass
from typing import cast
from zoneinfo import ZoneInfo

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import TimeframeBarSeries
from quantforge.timeframes import resolve_exchange_session
from quantforge.validation.errors import ValidationPlanError
from quantforge.validation.models import (
    BoundaryAxis,
    ExchangeSessionBoundary,
    TimestampBoundary,
    ValidationPlan,
    ValidationWindow,
)
from quantforge.validation.partitioning import (
    WindowObservationSelection,
    validate_source_observations,
)


@dataclass(frozen=True, slots=True)
class PredictionContextObservationSelection:
    """Context-only source membership at an explicit decision cutoff.

    The nested selection reuses immutable source/membership validation; its
    study_observations are feature-source bars, never outcome observations.
    """

    plan_id: str
    as_of: TimestampBoundary
    source_selection: WindowObservationSelection

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.plan_id), str)
            or len(self.plan_id) != 64
            or any(character not in "0123456789abcdef" for character in self.plan_id)
            or not isinstance(cast(object, self.as_of), TimestampBoundary)
            or not isinstance(
                cast(object, self.source_selection), WindowObservationSelection
            )
        ):
            raise ValidationPlanError(
                "prediction context selection provenance is invalid"
            )
        for key in (
            *self.source_selection.warm_up_context,
            *self.source_selection.study_observations,
        ):
            if (
                not isinstance(key, TimestampBoundary)
                or key.timestamp > self.as_of.timestamp
            ):
                raise ValidationPlanError(
                    "context bars must be timestamped and available by as_of"
                )

    def _identity_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": "1",
            "plan_id": self.plan_id,
            "as_of": self.as_of.to_primitive(),
            "source_selection": self.source_selection.to_primitive(),
            "context_only": True,
            "eligible_for_outcome_selection": False,
        }

    @property
    def selection_id(self) -> str:
        return configuration_identity(self._identity_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {"selection_id": self.selection_id, **self._identity_primitive()}


def select_prediction_context_observations(
    plan: ValidationPlan,
    window: ValidationWindow,
    *,
    source: TimeframeBarSeries,
    as_of: TimestampBoundary,
) -> PredictionContextObservationSelection:
    """Select completed context bars for one decision in a declared window.

    Study membership/purging uses the plan's separate observation source.
    Source keys remain UTC bar ends. Only bars completed by as_of are exposed;
    preceding context stays structurally separate. When no
    source bar has completed in the window (e.g. weekly context on Tuesday), an
    extra preceding anchor supplies the current indicator input. This is not a
    context provider: QF-20/QF-28 still enforce decision alignment and staleness.
    """
    if plan.environment.prediction_dataset is None or (
        plan.axis is not BoundaryAxis.EXCHANGE_SESSION
        and plan.prediction_membership is None
    ):
        raise ValidationPlanError(
            "prediction context selection requires a dual-input prediction plan"
        )
    windows = (
        *(
            item
            for fold in plan.folds
            for item in (fold.development, fold.selection, fold.test)
            if item is not None
        ),
        plan.final_holdout.window,
    )
    if window not in windows:
        raise ValidationPlanError("context window must belong to the validation plan")
    if not isinstance(cast(object, source), TimeframeBarSeries) or (
        source.dataset_reference not in plan.environment.dataset.family_references
        or source.dataset_family_manifest_id
        != plan.environment.dataset.family_manifest_id
    ):
        raise ValidationPlanError(
            "context source must match the selected plan family artifact"
        )
    if not isinstance(cast(object, as_of), TimestampBoundary):
        raise ValidationPlanError("context as_of must be a timestamp boundary")
    if plan.prediction_membership is not None:
        if not window.interval.contains(as_of):
            raise ValidationPlanError(
                "context decision must belong to the timestamp window"
            )
        plan.prediction_membership.session_for(as_of.timestamp)
        start_timestamp = cast(TimestampBoundary, window.interval.start).timestamp
    else:
        start = cast(ExchangeSessionBoundary, window.interval.start)
        decision_session = as_of.timestamp.astimezone(
            ZoneInfo(start.session_policy.timezone_name)
        ).date()
        decision = ExchangeSessionBoundary(decision_session, start.session_policy)
        if not window.interval.contains(decision):
            raise ValidationPlanError(
                "context decision must belong to the session window"
            )
        metadata = plan.environment.prediction_dataset.market_data_metadata
        assert metadata is not None
        if (
            not metadata.actual_first_session
            <= decision_session
            <= metadata.actual_last_session
            or decision_session in metadata.missing_sessions
        ):
            raise ValidationPlanError(
                "context decision must be observed in the prediction dataset"
            )
        session = resolve_exchange_session(decision_session, start.session_policy)
        if not session.open_timestamp <= as_of.timestamp <= session.close_timestamp:
            raise ValidationPlanError(
                "context decision must lie within the exchange session"
            )
        start_timestamp = resolve_exchange_session(
            start.session_date, start.session_policy
        ).open_timestamp
    keys = tuple(TimestampBoundary(bar.end_timestamp) for bar in source.bars)
    dataset_id, timeframe, manifest_id = validate_source_observations(
        keys, as_of, source
    )
    counts = {
        item.timeframe.configuration_id: item.observations
        for item in window.warm_up_by_timeframe
    }
    if timeframe.configuration_id not in counts:
        raise ValidationPlanError(
            "context source needs an explicit timeframe warm-up count"
        )
    history = tuple(key for key in keys if key.timestamp < start_timestamp)
    visible = tuple(
        key for key in keys if start_timestamp <= key.timestamp <= as_of.timestamp
    )
    required_history = counts[timeframe.configuration_id] + (0 if visible else 1)
    if len(history) < required_history:
        raise ValidationPlanError(
            "insufficient historical observations for prediction context warm-up"
        )
    warm_up = history[-required_history:] if required_history else ()
    selection = WindowObservationSelection(
        window.window_id,
        warm_up,
        visible,
        dataset_id,
        timeframe.configuration_id,
        manifest_id,
    )
    return PredictionContextObservationSelection(plan.plan_id, as_of, selection)
