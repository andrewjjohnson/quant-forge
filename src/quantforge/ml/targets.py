"""Explicit, versioned event ML targets over existing persisted outcomes (QF-67).

The first supported target is binary: an existing QF-49 intraday forward close
return is strictly greater than a threshold (default ``0``). It reuses the
labeler's unchanged definition and availability semantics:

- return convention: ``outcome_close / reference_close - 1`` (dimensionless
  arithmetic ratio, exact decimal; no costs, no fills);
- comparison: ``raw_return > threshold``; an available return equal to the
  threshold (``0``) is a **negative**;
- direction: none. The raw return is not sign-adjusted for DOWN signals;
- missing labels: an unavailable outcome (any QF-46 status other than
  ``available``) is an explicit null label with its status, never ``False``
  and never zero.

Binding a target to a population reconstructs the exact QF-49 labeler and
evaluator configurations from the persisted temporal configuration and requires
byte-equal configurations, the configured horizon, the same-session policy and
a plan whose outcome and purge horizon cover it. The ``kind`` field reserves a
place for later continuous targets (for example the return itself) without a
general target framework.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.ml.errors import EventDatasetIntegrityError, EventTargetError
from quantforge.prediction.intraday_forward_return import (
    IntradayForwardReturnEvaluator,
    IntradayForwardReturnOutcomeLabeler,
)
from quantforge.prediction.outcome_resolution import OutcomeResolutionStatus
from quantforge.prediction.outcome_temporal import (
    ElapsedDurationHorizon,
    OutcomeSessionPolicy,
    OutcomeTemporalError,
    outcome_temporal_configuration,
)
from quantforge.validation import ValidationPlan

EVENT_TARGET_CONTRACT_VERSION = "1"
_STATUSES = frozenset(status.value for status in OutcomeResolutionStatus)


class TargetKind(StrEnum):
    """Supported target value kinds; continuous targets are future work."""

    BINARY = "binary"


@dataclass(frozen=True, slots=True)
class TargetLabel:
    """One row's label. ``value`` is ``None`` exactly when unavailable."""

    value: bool | None
    status: str
    source_value: str | None
    outcome_id: str


@dataclass(frozen=True, slots=True)
class ForwardReturnBinaryTarget:
    """``raw_return > threshold`` of an existing QF-49 intraday forward return."""

    horizon: timedelta = timedelta(minutes=30)
    threshold: Decimal = Decimal(0)
    name: str = "forward_return_30m_positive"
    version: str = "1"

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.horizon), timedelta)
            or self.horizon <= timedelta(0)
            or self.horizon % timedelta(microseconds=1)
        ):
            raise EventTargetError("target horizon must be a positive duration")
        if (
            not isinstance(cast(object, self.threshold), Decimal)
            or not self.threshold.is_finite()
        ):
            raise EventTargetError("target threshold must be a finite Decimal")
        if not self.name or not self.version:
            raise EventTargetError("target name and version are required")

    @property
    def kind(self) -> TargetKind:
        return TargetKind.BINARY

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "contract_version": EVENT_TARGET_CONTRACT_VERSION,
            "kind": self.kind.value,
            "name": self.name,
            "version": self.version,
            "outcome": {
                "component_name": IntradayForwardReturnOutcomeLabeler.name,
                "implementation_version": (
                    IntradayForwardReturnOutcomeLabeler.implementation_version
                ),
                "evaluator_name": IntradayForwardReturnEvaluator.name,
                "horizon_microseconds": self.horizon // timedelta(microseconds=1),
                "value_field": "raw_return",
                "return_formula": "outcome_close / reference_close - 1",
                "return_unit": "ratio",
                "session_policy": OutcomeSessionPolicy.SAME_SESSION_ONLY.value,
            },
            "comparison": "strictly_greater_than",
            "threshold": decimal_to_primitive(self.threshold),
            "direction_adjustment": "none",
            "zero_at_threshold": "negative",
            "missing_label": "unavailable_outcome_is_null_with_status",
        }

    @property
    def target_id(self) -> str:
        return configuration_identity(self.to_primitive())

    @classmethod
    def from_primitive(cls, value: PrimitiveMapping) -> "ForwardReturnBinaryTarget":
        try:
            outcome = cast(PrimitiveMapping, value["outcome"])
            target = cls(
                horizon=timedelta(
                    microseconds=cast(int, outcome["horizon_microseconds"])
                ),
                threshold=Decimal(cast(str, value["threshold"])),
                name=cast(str, value["name"]),
                version=cast(str, value["version"]),
            )
        except (KeyError, TypeError, ValueError, InvalidOperation) as error:
            raise EventTargetError("invalid target configuration") from error
        if target.to_primitive() != value:
            raise EventTargetError("unsupported or inconsistent target configuration")
        return target

    def bind(
        self, candidate_definition: PrimitiveMapping, plan: ValidationPlan
    ) -> "BoundEventTarget":
        """Require the population's exact QF-49 outcome and plan compatibility."""
        try:
            labeler = cast(PrimitiveMapping, candidate_definition["outcome_labeler"])
            evaluator = cast(PrimitiveMapping, candidate_definition["evaluator"])
            labeler_configuration = cast(PrimitiveMapping, labeler["configuration"])
            evaluator_configuration = cast(PrimitiveMapping, evaluator["configuration"])
            temporal = outcome_temporal_configuration(labeler_configuration)
            expected = IntradayForwardReturnOutcomeLabeler(temporal)
        except (KeyError, TypeError, OutcomeTemporalError) as error:
            raise EventTargetError(
                "population outcome is not a QF-49 intraday forward return"
            ) from error
        expected_evaluator = IntradayForwardReturnEvaluator()
        if (
            labeler.get("name") != expected.name
            or labeler.get("implementation_version") != expected.implementation_version
            or labeler_configuration != expected.configuration()
            or labeler.get("configuration_id") != expected.configuration_id
            or evaluator.get("name") != expected_evaluator.name
            or evaluator_configuration != expected_evaluator.configuration()
            or evaluator.get("configuration_id") != expected_evaluator.configuration_id
        ):
            raise EventTargetError(
                "population outcome/evaluator differ from the QF-49 forward return"
            )
        if temporal.horizon != ElapsedDurationHorizon(self.horizon):
            raise EventTargetError(
                "target horizon differs from the population's outcome horizon"
            )
        if temporal.session_policy is not OutcomeSessionPolicy.SAME_SESSION_ONLY:
            raise EventTargetError("target requires same-session outcomes")
        provenance = next(
            (
                item
                for item in plan.environment.outcomes
                if item.configuration_id == expected.configuration_id
            ),
            None,
        )
        purge = plan.purge_policy.label_horizon.elapsed
        if (
            plan.prediction_membership is None
            or provenance is None
            or provenance.future_horizon.elapsed != expected.required_future_duration
            or purge is None
            or purge < expected.required_future_duration
        ):
            raise EventTargetError(
                "plan does not declare and purge this target's outcome reach"
            )
        return BoundEventTarget(
            self,
            outcome_configuration_id=expected.configuration_id,
            evaluator_configuration_id=expected_evaluator.configuration_id,
            outcome_reach=expected.required_future_duration,
        )


@dataclass(frozen=True, slots=True)
class BoundEventTarget:
    """A target bound to one population's exact outcome and evaluator."""

    target: ForwardReturnBinaryTarget
    outcome_configuration_id: str
    evaluator_configuration_id: str
    outcome_reach: timedelta

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "target": self.target.to_primitive(),
            "target_id": self.target.target_id,
            "outcome_configuration_id": self.outcome_configuration_id,
            "evaluator_configuration_id": self.evaluator_configuration_id,
            "outcome_reach_microseconds": self.outcome_reach
            // timedelta(microseconds=1),
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.to_primitive())

    def label(self, row: PrimitiveMapping | None, decision: datetime) -> TargetLabel:
        """Label one observation from its persisted row; never infer a value."""
        if row is None:
            raise EventDatasetIntegrityError(
                "QF-49 observations require an explicit availability row"
            )
        try:
            outcome = cast(PrimitiveMapping, row["outcome"])
            evaluation = cast(PrimitiveMapping, row["evaluation"])
            values = cast(PrimitiveMapping, outcome["values"])
            outcome_id = outcome["outcome_id"]
            status, available = values["status"], values["available"]
            raw = values["raw_return"]
        except (KeyError, TypeError) as error:
            raise EventDatasetIntegrityError("incomplete persisted outcome") from error
        if (
            outcome.get("outcome_configuration_id") != self.outcome_configuration_id
            or evaluation.get("evaluator_configuration_id")
            != self.evaluator_configuration_id
            or evaluation.get("outcome_id") != outcome_id
            or evaluation.get("values") != values
            or not isinstance(outcome_id, str)
            or values.get("decision_timestamp") != decision.isoformat()
            or values.get("elapsed_duration_microseconds")
            != self.target.horizon // timedelta(microseconds=1)
        ):
            raise EventDatasetIntegrityError(
                "persisted outcome differs from the bound target configuration"
            )
        if status not in _STATUSES or type(available) is not bool:
            raise EventDatasetIntegrityError("unknown outcome availability status")
        if (available is (status == OutcomeResolutionStatus.AVAILABLE.value)) is False:
            raise EventDatasetIntegrityError("outcome availability is contradictory")
        if not available:
            if raw is not None:
                raise EventDatasetIntegrityError(
                    "an unavailable outcome cannot carry a return"
                )
            return TargetLabel(None, status, None, outcome_id)
        if not isinstance(raw, str):
            raise EventDatasetIntegrityError("an available outcome requires a return")
        try:
            value = Decimal(raw)
        except InvalidOperation as error:
            raise EventDatasetIntegrityError("outcome return is not decimal") from error
        if not value.is_finite():
            raise EventDatasetIntegrityError("outcome return is not finite")
        return TargetLabel(value > self.target.threshold, status, raw, outcome_id)


__all__ = [
    "EVENT_TARGET_CONTRACT_VERSION",
    "BoundEventTarget",
    "ForwardReturnBinaryTarget",
    "TargetKind",
    "TargetLabel",
]
