"""Resolve future observation boundaries without calculating outcome values."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data.intraday import IntradayBar
from quantforge.data.intraday_aggregation import intraday_session_windows
from quantforge.data.lineage import DatasetFamilyReference
from quantforge.data.multi_timeframe import TimeframeBarSeries
from quantforge.prediction.outcome_temporal import (
    ElapsedDurationHorizon,
    OutcomeAnchor,
    OutcomeTemporalConfiguration,
    OutcomeTemporalError,
)
from quantforge.timeframes import BarCompletion, resolve_exchange_session


@dataclass(frozen=True, slots=True)
class OutcomeEvaluationRequest:
    """Future-label input created after causal predictions have been fixed.

    The optional source reference identifies a separately validated intraday
    artifact; dataset identity continues to identify the prediction dataset.
    Exact decisions are supplied by QF-42/QF-11, never generated here.
    """

    anchor: OutcomeAnchor
    temporal_configuration: OutcomeTemporalConfiguration
    outcome_configuration_id: str
    dataset_id: str
    dataset_fingerprint: str
    source_reference: DatasetFamilyReference | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.anchor), OutcomeAnchor)
            or not isinstance(
                cast(object, self.temporal_configuration), OutcomeTemporalConfiguration
            )
            or self.anchor.kind is not self.temporal_configuration.anchor_kind
        ):
            raise OutcomeTemporalError("outcome request anchor/configuration mismatch")
        for identity in (
            self.outcome_configuration_id,
            self.dataset_id,
            self.dataset_fingerprint,
        ):
            if not isinstance(cast(object, identity), str) or not identity.strip():
                raise OutcomeTemporalError("outcome request identities are required")
        reference = self.source_reference
        if reference is not None:
            timeframe = self.temporal_configuration.observation_timeframe
            if (
                not isinstance(cast(object, reference), DatasetFamilyReference)
                or timeframe is None
                or reference.timeframe_configuration_id != timeframe.configuration_id
            ):
                raise OutcomeTemporalError(
                    "outcome source timeframe differs from configuration"
                )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "anchor": self.anchor.to_primitive(),
            "temporal_configuration": self.temporal_configuration.to_primitive(),
            "outcome_configuration_id": self.outcome_configuration_id,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "source_reference": (
                None
                if self.source_reference is None
                else self.source_reference.to_primitive(include_feed_scope=True)
            ),
        }

    @property
    def request_id(self) -> str:
        return configuration_identity(self.to_primitive())


class OutcomeResolutionStatus(StrEnum):
    AVAILABLE = "available"
    SESSION_OVERFLOW = "session_overflow"
    MISSING_OBSERVATION = "missing_required_observation"
    INCOMPLETE = "incomplete_future_data"
    DATASET_END = "dataset_end"


@dataclass(frozen=True, slots=True)
class OutcomeResolution:
    """Time/provenance metadata only. An unavailable result contains no bar."""

    request: OutcomeEvaluationRequest
    requested_target_timestamp: datetime
    expected_observation_timestamp: datetime | None
    status: OutcomeResolutionStatus
    observation: IntradayBar | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.status), OutcomeResolutionStatus):
            raise OutcomeTemporalError("outcome resolution status is invalid")
        target, expected = _target_boundaries(self.request)
        if (
            self.requested_target_timestamp != target
            or self.expected_observation_timestamp != expected
        ):
            raise OutcomeTemporalError(
                "outcome resolution differs from requested boundaries"
            )
        object.__setattr__(self, "requested_target_timestamp", target)
        object.__setattr__(self, "expected_observation_timestamp", expected)
        if (expected is None) != (
            self.status is OutcomeResolutionStatus.SESSION_OVERFLOW
        ):
            raise OutcomeTemporalError(
                "outcome overflow status differs from session boundary"
            )
        if self.available != (self.observation is not None):
            raise OutcomeTemporalError(
                "unavailable outcome must not contain an observation"
            )
        if self.observation is not None and (
            not self.observation.complete
            or self.observation.end_timestamp != expected
            or self.observation.session_date != self.request.anchor.signal_session
            or self.observation.timeframe
            != self.request.temporal_configuration.observation_timeframe
        ):
            raise OutcomeTemporalError(
                "resolved observation differs from expected completed boundary"
            )

    @property
    def available(self) -> bool:
        return self.status is OutcomeResolutionStatus.AVAILABLE

    @property
    def resolved_observation_timestamp(self) -> datetime | None:
        return None if self.observation is None else self.observation.end_timestamp

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "request": self.request.to_primitive(),
            "requested_target_timestamp": self.requested_target_timestamp.isoformat(),
            "expected_observation_timestamp": (
                None
                if self.expected_observation_timestamp is None
                else self.expected_observation_timestamp.isoformat()
            ),
            "resolved_observation_timestamp": (
                None
                if self.resolved_observation_timestamp is None
                else self.resolved_observation_timestamp.isoformat()
            ),
            "status": self.status.value,
            "available": self.available,
            "observation_id": None
            if self.observation is None
            else self.observation.bar_id,
        }

    def metadata_primitive(self) -> PrimitiveMapping:
        """Flat, value-free fields for QF-7/QF-29 future-outcome schema hooks."""
        horizon = cast(
            ElapsedDurationHorizon, self.request.temporal_configuration.horizon
        )
        return {
            **self.request.anchor.to_primitive(),
            "horizon_kind": "elapsed_duration",
            "elapsed_duration_microseconds": horizon.duration
            // timedelta(microseconds=1),
            "outcome_configuration_id": self.request.outcome_configuration_id,
            "temporal_configuration_id": (
                self.request.temporal_configuration.configuration_id
            ),
            **{
                key: value
                for key, value in self.to_primitive().items()
                if key != "request"
            },
        }


def _target_boundaries(
    request: OutcomeEvaluationRequest,
) -> tuple[datetime, datetime | None]:
    config = request.temporal_configuration
    decision = request.anchor.decision_timestamp
    if not isinstance(config.horizon, ElapsedDurationHorizon) or decision is None:
        raise OutcomeTemporalError(
            "observation resolution requires an exact elapsed outcome request"
        )
    assert config.observation_timeframe is not None
    session = resolve_exchange_session(
        request.anchor.signal_session, config.observation_timeframe.session_policy
    )
    if not session.open_timestamp <= decision <= session.close_timestamp:
        raise OutcomeTemporalError(
            "decision timestamp is outside the declared signal session"
        )
    try:
        target = decision + config.horizon.duration
    except OverflowError as error:
        raise OutcomeTemporalError("outcome target exceeds datetime range") from error
    if target > session.close_timestamp:
        return target, None
    expected = next(
        window.end_timestamp
        for window in intraday_session_windows(
            request.anchor.signal_session, config.observation_timeframe
        )
        if window.end_timestamp >= target
    )
    return target, expected


def resolve_future_observation(
    request: OutcomeEvaluationRequest, source: TimeframeBarSeries
) -> OutcomeResolution:
    """Ceil to an expected bar end, then require that exact completed observation.

    An absent boundary within observed coverage is missing; beyond the final
    observed boundary it is dataset end. A developing required interval is
    incomplete. This certifies only the endpoint, never intermediate path coverage.
    Source construction owns immutable artifact/lineage validation.
    """
    if (
        not isinstance(cast(object, source), TimeframeBarSeries)
        or source.dataset_reference != request.source_reference
        or source.timeframe != request.temporal_configuration.observation_timeframe
    ):
        raise OutcomeTemporalError(
            "outcome source does not match the request's immutable reference"
        )
    target, expected = _target_boundaries(request)
    if expected is None:
        return OutcomeResolution(
            request, target, None, OutcomeResolutionStatus.SESSION_OVERFLOW
        )
    # Expected calendar windows drive lookup; observed rows never redefine alignment.
    for bar in source.bars:
        if not isinstance(bar, IntradayBar):
            raise OutcomeTemporalError(
                "elapsed resolution requires canonical intraday bars"
            )
        if bar.session_date != request.anchor.signal_session:
            continue
        if (
            bar.end_timestamp == expected
            and bar.completion is not BarCompletion.DEVELOPING
        ):
            return OutcomeResolution(
                request, target, expected, OutcomeResolutionStatus.AVAILABLE, bar
            )
        if (
            bar.completion is BarCompletion.DEVELOPING
            and bar.start_timestamp < expected
        ):
            windows = intraday_session_windows(
                request.anchor.signal_session, source.timeframe
            )
            if any(
                window.start_timestamp == bar.start_timestamp
                and window.end_timestamp == expected
                for window in windows
            ):
                return OutcomeResolution(
                    request, target, expected, OutcomeResolutionStatus.INCOMPLETE
                )
    status = (
        OutcomeResolutionStatus.DATASET_END
        if not source.bars or expected > source.bars[-1].end_timestamp
        else OutcomeResolutionStatus.MISSING_OBSERVATION
    )
    return OutcomeResolution(request, target, expected, status)
