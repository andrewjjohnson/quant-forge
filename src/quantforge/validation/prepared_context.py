"""Execution-local immutable preparation for exact prediction context selection."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime
from typing import cast
from zoneinfo import ZoneInfo

from quantforge.configuration import (
    PrimitiveMappingSnapshot,
)
from quantforge.data import TimeframeBarSeries
from quantforge.timeframes import (
    ExchangeSessionPolicy,
    Timeframe,
    resolve_exchange_session,
)
from quantforge.validation.context import PredictionContextObservationSelection
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
class PreparedValidationPlan:
    """Captured canonical plan identity, never a memo on a caller-owned object.

    Capturing a changed plan produces a new identity. A live plan is never used
    after context capture; explicit reuse checks compare its current identity.
    No operational preparation fields enter the QF-8 identity.
    """

    plan_id: str

    @classmethod
    def capture(cls, plan: ValidationPlan) -> "PreparedValidationPlan":
        return cls(plan.plan_id)

    def validate_compatible(self, plan: ValidationPlan) -> None:
        """Check at an execution-scope boundary, never once per decision."""
        if self.plan_id != plan.plan_id:
            raise ValidationPlanError("prepared context validation plan differs")


@dataclass(frozen=True, slots=True)
class PredictionSourceIndex:
    """Sorted timestamp keys and positions referring to one immutable artifact."""

    source: TimeframeBarSeries
    keys: tuple[TimestampBoundary, ...]
    timestamps: tuple[datetime, ...]
    window_start: int
    warm_up_count: int
    timeframe_id: str

    @classmethod
    def build(
        cls, source: TimeframeBarSeries, start_timestamp: datetime, warm_up_count: int
    ) -> "PredictionSourceIndex":
        keys = tuple(TimestampBoundary(bar.end_timestamp) for bar in source.bars)
        validate_source_observations(keys, TimestampBoundary(start_timestamp), source)
        timestamps = tuple(key.timestamp for key in keys)
        return cls(
            source,
            keys,
            timestamps,
            bisect_left(timestamps, start_timestamp),
            warm_up_count,
            source.timeframe.configuration_id,
        )

    def bounds(self, as_of: datetime) -> tuple[int, int]:
        """Return the same contiguous warm-up/visible slice as the reference path."""
        stop = bisect_right(self.timestamps, as_of)
        required = self.warm_up_count + (0 if stop > self.window_start else 1)
        if self.window_start < required:
            raise ValidationPlanError(
                "insufficient historical observations for prediction context warm-up"
            )
        return self.window_start - required, stop


@dataclass(frozen=True, slots=True)
class PreparedPredictionContext:
    """Bound plan/window/view scope with no live plan or mutable selection state.

    Context indexes depend on source, window and warm-up semantics, not indicator
    parameters. QF-20/QF-28 still own completion, alignment, freshness and indicator
    requirements. Callers without this optional preparation retain the reference
    selector. Source graphs that cannot be shared immutably also use that path.
    """

    plan: PreparedValidationPlan
    window_snapshot: PrimitiveMappingSnapshot
    input_identity: str
    indexes: tuple[PredictionSourceIndex, ...]
    window_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    timestamp_membership: frozenset[datetime] | None
    session_interval: tuple[date, date] | None
    session_policy: ExchangeSessionPolicy
    first_session: date
    last_session: date
    missing_sessions: frozenset[date]

    @classmethod
    def capture(
        cls,
        plan: ValidationPlan,
        window: ValidationWindow,
        *,
        series: tuple[TimeframeBarSeries, ...],
        input_identity: str,
    ) -> "PreparedPredictionContext | None":
        # Reuse QF-58's exact-type, recursive immutability boundary. No source
        # prices are duplicated; calendar timestamp shells may normalize once.
        from quantforge.prediction.source_sharing import prediction_source_copy_memo

        sources: list[TimeframeBarSeries] = []
        for source in series:
            memo = prediction_source_copy_memo(source)
            if not memo:
                return None
            sources.append(cast(TimeframeBarSeries, memo[id(source)]))
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
            raise ValidationPlanError(
                "context window must belong to the validation plan"
            )
        for source in sources:
            if (
                source.dataset_reference
                not in plan.environment.dataset.family_references
                or source.dataset_family_manifest_id
                != plan.environment.dataset.family_manifest_id
            ):
                raise ValidationPlanError(
                    "context source must match the selected plan family artifact"
                )
        if not input_identity or not sources:
            raise ValidationPlanError("prepared context requires source/view identity")
        prepared_plan = PreparedValidationPlan.capture(plan)
        # Retain only detached selection metadata, not another copy of the
        # potentially large canonical source evidence embedded in the plan.
        metadata = plan.environment.prediction_dataset.market_data_metadata
        assert metadata is not None
        policy = Timeframe.from_primitive(
            plan.environment.timeframes[0].to_primitive()
        ).session_policy
        window_snapshot = PrimitiveMappingSnapshot.capture(window.to_primitive())
        if plan.prediction_membership is not None:
            session_interval = None
            start = datetime.fromisoformat(
                cast(TimestampBoundary, window.interval.start).timestamp.isoformat()
            )
            end = datetime.fromisoformat(
                cast(TimestampBoundary, window.interval.end).timestamp.isoformat()
            )
            timestamps = frozenset(
                datetime.fromisoformat(t.isoformat())
                for t in plan.prediction_membership.schedule.decision_timestamps
            )
        else:
            session_interval = (
                cast(ExchangeSessionBoundary, window.interval.start).session_date,
                cast(ExchangeSessionBoundary, window.interval.end).session_date,
            )
            start = resolve_exchange_session(
                cast(ExchangeSessionBoundary, window.interval.start).session_date,
                policy,
            ).open_timestamp
            end = resolve_exchange_session(
                cast(ExchangeSessionBoundary, window.interval.end).session_date, policy
            ).close_timestamp
            timestamps = None
        counts = {
            item.timeframe.configuration_id: item.observations
            for item in window.warm_up_by_timeframe
        }
        indexes: list[PredictionSourceIndex] = []
        for source in sources:
            timeframe_id = source.timeframe.configuration_id
            if timeframe_id not in counts:
                raise ValidationPlanError(
                    "context source needs an explicit timeframe warm-up count"
                )
            indexes.append(
                PredictionSourceIndex.build(source, start, counts[timeframe_id])
            )
        return cls(
            prepared_plan,
            window_snapshot,
            input_identity,
            tuple(indexes),
            window.window_id,
            start,
            end,
            timestamps,
            session_interval,
            policy,
            date.fromisoformat(metadata.actual_first_session.isoformat()),
            date.fromisoformat(metadata.actual_last_session.isoformat()),
            frozenset(
                date.fromisoformat(s.isoformat()) for s in metadata.missing_sessions
            ),
        )

    def validate_compatible(
        self,
        plan: ValidationPlan,
        window: ValidationWindow,
        *,
        series: tuple[TimeframeBarSeries, ...],
        input_identity: str,
    ) -> None:
        """Fail closed when explicitly reusing preparation in another scope."""
        self.plan.validate_compatible(plan)
        if (
            self.window_snapshot
            != PrimitiveMappingSnapshot.capture(window.to_primitive())
            or self.input_identity != input_identity
            or tuple(index.source for index in self.indexes) != series
        ):
            raise ValidationPlanError("prepared context source/view/window differs")

    def select(
        self,
        timeframe: Timeframe,
        *,
        as_of: TimestampBoundary,
    ) -> tuple[PredictionContextObservationSelection, TimeframeBarSeries]:
        """Resolve exact causal membership and original bars without full scanning."""
        if not isinstance(cast(object, as_of), TimestampBoundary):
            raise ValidationPlanError("context as_of must be a timestamp boundary")
        timestamp = as_of.timestamp
        if self.timestamp_membership is not None:
            if not self.start_timestamp <= timestamp <= self.end_timestamp:
                raise ValidationPlanError(
                    "context decision must belong to the timestamp window"
                )
            if timestamp not in self.timestamp_membership:
                raise ValidationPlanError("timestamp is outside the captured schedule")
        else:
            session_date = timestamp.astimezone(
                ZoneInfo(self.session_policy.timezone_name)
            ).date()
            # Preserve the reference selector's calendar validation and local
            # session-label boundary checks, including overnight calendars.
            ExchangeSessionBoundary(session_date, self.session_policy)
            assert self.session_interval is not None
            if not self.session_interval[0] <= session_date <= self.session_interval[1]:
                raise ValidationPlanError(
                    "context decision must belong to the session window"
                )
            if (
                not self.first_session <= session_date <= self.last_session
                or session_date in self.missing_sessions
            ):
                raise ValidationPlanError(
                    "context decision must be observed in the prediction dataset"
                )
            session = resolve_exchange_session(session_date, self.session_policy)
            if not session.open_timestamp <= timestamp <= session.close_timestamp:
                raise ValidationPlanError(
                    "context decision must lie within the exchange session"
                )
        index = next(
            (item for item in self.indexes if item.source.timeframe == timeframe), None
        )
        if index is None:
            raise ValidationPlanError("prepared context timeframe/session differs")
        start, stop = index.bounds(timestamp)
        source = index.source
        selection = PredictionContextObservationSelection(
            self.plan.plan_id,
            as_of,
            WindowObservationSelection(
                self.window_id,
                index.keys[start : index.window_start],
                index.keys[index.window_start : stop],
                source.dataset_reference.dataset_id,
                index.timeframe_id,
                source.dataset_family_manifest_id,
            ),
        )
        selected = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
            source.dataset_reference,
            source.timeframe,
            source.bars[start:stop],
            dataset_family_manifest_id=source.dataset_family_manifest_id,
            developing_source_evidence=source._developing_source_evidence,  # pyright: ignore[reportPrivateUsage]
        )
        return selection, selected
