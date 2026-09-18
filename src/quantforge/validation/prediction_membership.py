"""Immutable QF-42 schedule evidence for exact prediction observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import DatasetFamilyReference, TimeframeBarSeries
from quantforge.validation.errors import ValidationPlanError
from quantforge.validation.models import BoundaryAxis, TimestampBoundary

if TYPE_CHECKING:
    from quantforge.prediction.window import PredictionDecisionSchedule


@dataclass(frozen=True, slots=True, init=False)
class PredictionMembershipSource:
    """A schedule certified by an immutable, completed primary source artifact.

    Missing scheduled bars fail closed. Observed rows never redefine the schedule.
    Only source identity is retained here; no future prices enter plan metadata.
    """

    schedule: PredictionDecisionSchedule
    source_reference: DatasetFamilyReference
    family_manifest_id: str
    _sessions_by_timestamp: Mapping[datetime, date] = field(
        init=False, repr=False, compare=False
    )

    def __init__(self) -> None:
        raise TypeError("prediction membership must be captured from a source")

    @classmethod
    def capture(
        cls, schedule: PredictionDecisionSchedule, source: TimeframeBarSeries
    ) -> PredictionMembershipSource:
        from quantforge.prediction.window import PredictionDecisionSchedule

        if not isinstance(
            cast(object, schedule), PredictionDecisionSchedule
        ) or not isinstance(cast(object, source), TimeframeBarSeries):
            raise ValidationPlanError(
                "prediction membership requires a schedule/source"
            )
        instance = object.__new__(cls)
        object.__setattr__(instance, "schedule", schedule)
        object.__setattr__(instance, "source_reference", source.dataset_reference)
        object.__setattr__(
            instance, "family_manifest_id", source.dataset_family_manifest_id
        )
        instance.validate_source(source)
        instance._index_sessions()
        return instance

    def _index_sessions(self) -> None:
        object.__setattr__(
            self,
            "_sessions_by_timestamp",
            MappingProxyType(
                dict(
                    zip(
                        self.schedule.decision_timestamps,
                        self.schedule.decision_sessions,
                        strict=True,
                    )
                )
            ),
        )

    def __getstate__(
        self,
    ) -> tuple[PredictionDecisionSchedule, DatasetFamilyReference, str]:
        """Copy/serialize canonical evidence; rebuild the derived read-only index."""
        return self.schedule, self.source_reference, self.family_manifest_id

    def __setstate__(
        self, state: tuple[PredictionDecisionSchedule, DatasetFamilyReference, str]
    ) -> None:
        object.__setattr__(self, "schedule", state[0])
        object.__setattr__(self, "source_reference", state[1])
        object.__setattr__(self, "family_manifest_id", state[2])
        self._index_sessions()

    def validate_source(self, source: TimeframeBarSeries) -> None:
        if (
            source.timeframe != self.schedule.primary_timeframe
            or source.dataset_reference != self.source_reference
            or source.dataset_family_manifest_id != self.family_manifest_id
            or not self.family_manifest_id
        ):
            raise ValidationPlanError("prediction membership source lineage differs")
        completed = {bar.end_timestamp: bar for bar in source.bars if bar.complete}
        if not self.schedule.decision_timestamps or any(
            timestamp not in completed
            or getattr(completed[timestamp], "session_date", None) != session
            for timestamp, session in zip(
                self.schedule.decision_timestamps,
                self.schedule.decision_sessions,
                strict=True,
            )
        ):
            raise ValidationPlanError(
                "source is missing a scheduled completed observation"
            )

    @property
    def axis(self) -> BoundaryAxis:
        return BoundaryAxis.TIMESTAMP

    @property
    def observations(self) -> tuple[TimestampBoundary, ...]:
        return tuple(TimestampBoundary(t) for t in self.schedule.decision_timestamps)

    def session_for(self, timestamp: datetime) -> date:
        """Resolve an exact decision in constant expected time without scanning."""
        try:
            return self._sessions_by_timestamp[timestamp]
        except KeyError as error:
            raise ValidationPlanError(
                "timestamp is outside the captured schedule"
            ) from error

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "axis": self.axis.value,
            "schedule_id": self.schedule.schedule_id,
            "schedule": self.schedule.to_primitive(),
            "source_reference": self.source_reference.to_primitive(
                include_feed_scope=True
            ),
            "family_manifest_id": self.family_manifest_id,
            "coverage_policy": "require_every_scheduled_completed_primary_bar",
        }

    @property
    def membership_id(self) -> str:
        return configuration_identity(self.to_primitive())
