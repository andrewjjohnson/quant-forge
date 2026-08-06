"""Immutable, engine-neutral strategy decision records."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveScalar,
    decimal_to_primitive,
)
from quantforge.data.models import MarketDataset
from quantforge.strategies.exceptions import (
    InvalidStrategyOutputError,
    InvalidTargetWeightError,
)


class PositionIntent(StrEnum):
    """Desired long-only position state; it is not an order or fill."""

    FLAT = "flat"
    LONG = "long"


class ExecutionTiming(StrEnum):
    """When a completed daily-bar signal first becomes executable."""

    NEXT_SESSION_AFTER_CLOSE = "next_session_after_close"


class ExecutionSessionStatus(StrEnum):
    """Whether a next session is calendar-resolved; neither value means filled."""

    PENDING = "pending"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class ParameterValue:
    """One stable primitive strategy parameter captured with a decision."""

    name: str
    value: PrimitiveScalar

    def __post_init__(self) -> None:
        if not self.name:
            raise InvalidStrategyOutputError("parameter names must not be empty")
        if isinstance(self.value, float) and not isfinite(self.value):
            raise InvalidStrategyOutputError(
                "floating-point strategy parameters must be finite"
            )


@dataclass(frozen=True, slots=True)
class IndicatorObservation:
    """An indicator value retained to make a strategy decision auditable."""

    name: str
    value: Decimal

    def __post_init__(self) -> None:
        if not self.name:
            raise InvalidStrategyOutputError(
                "indicator observation names must not be empty"
            )
        if not self.value.is_finite():
            raise InvalidStrategyOutputError(
                "decision indicator observations must be finite"
            )


@dataclass(frozen=True, slots=True)
class MarketDataReference:
    """Stable QF-3 dataset provenance retained without provider SDK objects."""

    dataset_id: str
    schema_version: str
    adjustment_mode: str
    calendar: str
    corporate_action_snapshot_id: str

    @classmethod
    def from_dataset(cls, dataset: MarketDataset) -> "MarketDataReference":
        metadata = dataset.metadata
        return cls(
            metadata.dataset_id,
            metadata.schema_version,
            metadata.adjustment_mode.value,
            metadata.calendar,
            metadata.corporate_action_snapshot_id,
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "adjustment_mode": self.adjustment_mode,
            "calendar": self.calendar,
            "corporate_action_snapshot_id": self.corporate_action_snapshot_id,
        }


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """A target-position intent known after one completed daily session."""

    canonical_symbol: str
    signal_session: date
    earliest_executable_session: date | None
    execution_timing: ExecutionTiming
    execution_session_status: ExecutionSessionStatus
    target_position: PositionIntent
    target_weight: Decimal
    strategy_id: str
    strategy_configuration_id: str
    strategy_parameters: tuple[ParameterValue, ...]
    reason: str | None
    indicator_values: tuple[IndicatorObservation, ...]

    def __post_init__(self) -> None:
        if not self.canonical_symbol or not self.strategy_id:
            raise InvalidStrategyOutputError(
                "canonical symbol and strategy identifier are required"
            )
        if not self.strategy_configuration_id:
            raise InvalidStrategyOutputError(
                "strategy configuration identity is required"
            )
        _validate_position_intent(self.target_position)
        if not self.target_weight.is_finite() or not Decimal(
            0
        ) <= self.target_weight <= Decimal(1):
            raise InvalidTargetWeightError("target weight must be between 0 and 1")
        if self.target_position is PositionIntent.FLAT and self.target_weight != 0:
            raise InvalidTargetWeightError("a flat target must have zero weight")
        if self.target_position is PositionIntent.LONG and self.target_weight <= 0:
            raise InvalidTargetWeightError("a long target must have positive weight")
        if self.earliest_executable_session is None:
            if self.execution_session_status is not ExecutionSessionStatus.UNRESOLVED:
                raise InvalidStrategyOutputError(
                    "an unresolved execution session must be marked unresolved"
                )
        elif (
            self.earliest_executable_session <= self.signal_session
            or self.execution_session_status is not ExecutionSessionStatus.PENDING
        ):
            raise InvalidStrategyOutputError(
                "a resolved execution session must be later than the signal and pending"
            )
        parameter_names = tuple(item.name for item in self.strategy_parameters)
        observation_names = tuple(item.name for item in self.indicator_values)
        if parameter_names != tuple(sorted(parameter_names)) or len(
            parameter_names
        ) != len(set(parameter_names)):
            raise InvalidStrategyOutputError(
                "strategy parameter snapshots must be sorted and unique"
            )
        if observation_names != tuple(sorted(observation_names)) or len(
            observation_names
        ) != len(set(observation_names)):
            raise InvalidStrategyOutputError(
                "indicator observations must be sorted and unique"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "canonical_symbol": self.canonical_symbol,
            "signal_session": self.signal_session.isoformat(),
            "earliest_executable_session": (
                None
                if self.earliest_executable_session is None
                else self.earliest_executable_session.isoformat()
            ),
            "execution_timing": self.execution_timing.value,
            "execution_session_status": self.execution_session_status.value,
            "target_position": self.target_position.value,
            "target_weight": decimal_to_primitive(self.target_weight),
            "strategy_id": self.strategy_id,
            "strategy_configuration_id": self.strategy_configuration_id,
            "strategy_parameters": {
                item.name: item.value for item in self.strategy_parameters
            },
            "reason": self.reason,
            "indicator_values": {
                item.name: decimal_to_primitive(item.value)
                for item in self.indicator_values
            },
        }


@dataclass(frozen=True, slots=True)
class StrategyOutput:
    """Ordered decisions supporting tabular and chronological consumption."""

    strategy_id: str
    strategy_configuration_id: str
    market_data: MarketDataReference
    decisions: tuple[StrategyDecision, ...]
    contract_version: str = "1"

    def __iter__(self) -> Iterator[StrategyDecision]:
        return iter(self.decisions)

    def __len__(self) -> int:
        return len(self.decisions)

    def to_rows(self) -> list[PrimitiveMapping]:
        """Return the decision table for vectorized-style consumers."""
        return [decision.to_primitive() for decision in self.decisions]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "contract_version": self.contract_version,
            "strategy_id": self.strategy_id,
            "strategy_configuration_id": self.strategy_configuration_id,
            "market_data": self.market_data.to_primitive(),
            "decisions": cast(list[Primitive], self.to_rows()),
        }


def _validate_position_intent(value: object) -> None:
    if not isinstance(value, PositionIntent):
        raise InvalidStrategyOutputError("target position is unsupported")
