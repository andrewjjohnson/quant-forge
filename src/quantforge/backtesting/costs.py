"""Typed deterministic commission and adverse-slippage models."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Protocol, cast

from quantforge.backtesting._arithmetic import arithmetic, decimal_from
from quantforge.backtesting.errors import InvalidBacktestConfigurationError
from quantforge.configuration import PrimitiveMapping, decimal_to_primitive


class OrderSide(StrEnum):
    """Supported long-only market-order sides."""

    BUY = "buy"
    SELL = "sell"


class CommissionModel(Protocol):
    """Calculate a separately auditable nonnegative fill commission."""

    @property
    def name(self) -> str: ...

    def calculate(self, quantity: int, fill_price: Decimal) -> Decimal: ...

    def configuration(self) -> PrimitiveMapping: ...


class SlippageModel(Protocol):
    """Transform a reference price in the adverse trade direction."""

    @property
    def name(self) -> str: ...

    def apply(self, reference_price: Decimal, side: OrderSide) -> Decimal: ...

    def configuration(self) -> PrimitiveMapping: ...


def _nonnegative_decimal(value: object, field_name: str) -> Decimal:
    try:
        converted = decimal_from(value, field_name)
    except ValueError as error:
        raise InvalidBacktestConfigurationError(str(error)) from error
    if converted < 0:
        raise InvalidBacktestConfigurationError(f"{field_name} must be nonnegative")
    return converted


def _validate_fill_inputs(quantity: object, fill_price: object) -> None:
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise InvalidBacktestConfigurationError(
            "commission quantity must be a positive integer"
        )
    if not isinstance(fill_price, Decimal) or not fill_price.is_finite():
        raise InvalidBacktestConfigurationError("fill price must be finite")
    if fill_price <= 0:
        raise InvalidBacktestConfigurationError("fill price must be positive")


@dataclass(frozen=True, slots=True)
class FixedCommission:
    """Charge one fixed amount for every filled order, including explicit zero."""

    amount: Decimal
    name = "fixed_per_fill"

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", _nonnegative_decimal(self.amount, "amount"))

    def calculate(self, quantity: int, fill_price: Decimal) -> Decimal:
        _validate_fill_inputs(quantity, fill_price)
        return self.amount

    def configuration(self) -> PrimitiveMapping:
        return {
            "model": self.name,
            "parameters": {"amount": decimal_to_primitive(self.amount)},
        }


@dataclass(frozen=True, slots=True)
class PerShareCommission:
    """Charge per share with an optional nonnegative minimum per fill."""

    amount_per_share: Decimal
    minimum: Decimal = Decimal(0)
    name = "per_share"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "amount_per_share",
            _nonnegative_decimal(self.amount_per_share, "amount_per_share"),
        )
        object.__setattr__(
            self, "minimum", _nonnegative_decimal(self.minimum, "minimum")
        )

    def calculate(self, quantity: int, fill_price: Decimal) -> Decimal:
        _validate_fill_inputs(quantity, fill_price)
        with arithmetic():
            return max(self.amount_per_share * Decimal(quantity), self.minimum)

    def configuration(self) -> PrimitiveMapping:
        return {
            "model": self.name,
            "parameters": {
                "amount_per_share": decimal_to_primitive(self.amount_per_share),
                "minimum": decimal_to_primitive(self.minimum),
            },
        }


@dataclass(frozen=True, slots=True)
class BasisPointCommission:
    """Charge basis points of final fill notional."""

    basis_points: Decimal
    name = "basis_points"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "basis_points",
            _nonnegative_decimal(self.basis_points, "basis_points"),
        )

    def calculate(self, quantity: int, fill_price: Decimal) -> Decimal:
        _validate_fill_inputs(quantity, fill_price)
        with arithmetic():
            return Decimal(quantity) * fill_price * self.basis_points / Decimal(10_000)

    def configuration(self) -> PrimitiveMapping:
        return {
            "model": self.name,
            "parameters": {"basis_points": decimal_to_primitive(self.basis_points)},
        }


@dataclass(frozen=True, slots=True)
class BasisPointSlippage:
    """Apply deterministic adverse-direction basis-point slippage."""

    basis_points: Decimal
    name = "adverse_basis_points"

    def __post_init__(self) -> None:
        basis_points = _nonnegative_decimal(self.basis_points, "basis_points")
        if basis_points >= Decimal(10_000):
            raise InvalidBacktestConfigurationError(
                "slippage basis points must be less than 10000"
            )
        object.__setattr__(self, "basis_points", basis_points)

    def apply(self, reference_price: Decimal, side: OrderSide) -> Decimal:
        reference_value = cast(object, reference_price)
        side_value = cast(object, side)
        if not isinstance(reference_value, Decimal) or not reference_value.is_finite():
            raise InvalidBacktestConfigurationError("reference price must be finite")
        if reference_value <= 0:
            raise InvalidBacktestConfigurationError("reference price must be positive")
        if not isinstance(side_value, OrderSide):
            raise InvalidBacktestConfigurationError("order side is unsupported")
        with arithmetic():
            rate = self.basis_points / Decimal(10_000)
            multiplier = (
                Decimal(1) + rate if side_value is OrderSide.BUY else Decimal(1) - rate
            )
            fill_price = reference_value * multiplier
        if not fill_price.is_finite() or fill_price <= 0:
            raise InvalidBacktestConfigurationError(
                "slippage produced an invalid fill price"
            )
        return fill_price

    def configuration(self) -> PrimitiveMapping:
        return {
            "model": self.name,
            "parameters": {"basis_points": decimal_to_primitive(self.basis_points)},
        }
