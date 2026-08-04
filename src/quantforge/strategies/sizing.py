"""Engine-neutral position-sizing intent boundary."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Protocol

from quantforge.configuration import PrimitiveMapping, decimal_to_primitive
from quantforge.strategies.exceptions import InvalidTargetWeightError
from quantforge.strategies.models import PositionIntent


class SizingContextField(StrEnum):
    """Context a future sizing policy may require from an execution engine."""

    AVAILABLE_EQUITY = "available_equity"
    CURRENT_POSITION = "current_position"
    REFERENCE_PRICE = "reference_price"
    RISK_BUDGET = "risk_budget"


@dataclass(frozen=True, slots=True)
class SizingContext:
    """Optional future engine inputs; the reference policy needs none of them."""

    available_equity: Decimal | None = None
    current_position: Decimal | None = None
    reference_price: Decimal | None = None
    risk_budget: Decimal | None = None


@dataclass(frozen=True, slots=True)
class TargetWeightIntent:
    """Requested normalized allocation, not a share quantity or order."""

    target_position: PositionIntent
    target_weight: Decimal

    def __post_init__(self) -> None:
        _validate_position_intent(self.target_position)
        if not self.target_weight.is_finite() or not Decimal(
            0
        ) <= self.target_weight <= Decimal(1):
            raise InvalidTargetWeightError("target weight must be between 0 and 1")
        if self.target_position is PositionIntent.FLAT and self.target_weight != 0:
            raise InvalidTargetWeightError("a flat target must have zero weight")
        if self.target_position is PositionIntent.LONG and self.target_weight <= 0:
            raise InvalidTargetWeightError("a long target must have positive weight")


class PositionSizingPolicy(Protocol):
    """Convert position state into engine-neutral sizing intent."""

    @property
    def name(self) -> str: ...

    @property
    def required_context_fields(self) -> frozenset[SizingContextField]: ...

    def configuration(self) -> PrimitiveMapping: ...

    def size(
        self,
        target_position: PositionIntent,
        context: SizingContext | None = None,
    ) -> TargetWeightIntent: ...


@dataclass(frozen=True, slots=True)
class TargetWeightSizingPolicy:
    """Request zero allocation while flat and a configured weight while long."""

    target_long_weight: Decimal = Decimal(1)
    name = "long_only_target_weight"
    required_context_fields = frozenset[SizingContextField]()

    def __post_init__(self) -> None:
        try:
            weight = Decimal(str(self.target_long_weight))
        except InvalidOperation as error:
            raise InvalidTargetWeightError(
                "target long weight must be numeric"
            ) from error
        if not weight.is_finite() or not Decimal(0) < weight <= Decimal(1):
            raise InvalidTargetWeightError(
                "target long weight must be greater than 0 and at most 1"
            )
        object.__setattr__(self, "target_long_weight", weight)

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_type": "position_sizing",
            "component_name": self.name,
            "contract_version": "1",
            "parameters": {
                "target_long_weight": decimal_to_primitive(self.target_long_weight)
            },
            "required_context_fields": [],
        }

    def size(
        self,
        target_position: PositionIntent,
        context: SizingContext | None = None,
    ) -> TargetWeightIntent:
        del context
        if target_position is PositionIntent.FLAT:
            return TargetWeightIntent(target_position, Decimal(0))
        if target_position is PositionIntent.LONG:
            return TargetWeightIntent(target_position, self.target_long_weight)
        raise InvalidTargetWeightError(
            f"unsupported target position: {target_position!r}"
        )


def _validate_position_intent(value: object) -> None:
    if not isinstance(value, PositionIntent):
        raise InvalidTargetWeightError("target position is unsupported")
