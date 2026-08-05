"""Deterministic evaluation boundary for execution-cost model callbacks."""

from decimal import Decimal
from typing import cast

from quantforge.backtesting._arithmetic import arithmetic
from quantforge.backtesting.costs import (
    CommissionModel,
    FeeModel,
    OrderSide,
    SlippageModel,
)
from quantforge.backtesting.errors import ExecutionError


def _operation(label: str, context: str) -> str:
    return f"{context} {label}" if context else label


def commission_amount(
    model: CommissionModel,
    quantity: int,
    fill_price: Decimal,
    *,
    context: str = "",
) -> Decimal:
    """Evaluate a commission callback under the serialized arithmetic policy."""
    try:
        with arithmetic():
            value = model.calculate(quantity, fill_price)
    except (ArithmeticError, ValueError) as error:
        raise ExecutionError(
            _operation("commission calculation failed", context)
        ) from error
    value_object = cast(object, value)
    if (
        not isinstance(value_object, Decimal)
        or not value_object.is_finite()
        or value_object < 0
    ):
        raise ExecutionError("commission model returned an invalid amount")
    return value_object


def fee_amount(
    model: FeeModel,
    side: OrderSide,
    quantity: int,
    fill_price: Decimal,
    *,
    context: str = "",
) -> Decimal:
    """Evaluate a transaction-fee callback under the arithmetic policy."""
    try:
        with arithmetic():
            value = model.calculate(side, quantity, fill_price)
    except (ArithmeticError, ValueError) as error:
        raise ExecutionError(
            _operation("transaction-fee calculation failed", context)
        ) from error
    value_object = cast(object, value)
    if (
        not isinstance(value_object, Decimal)
        or not value_object.is_finite()
        or value_object < 0
    ):
        raise ExecutionError("transaction-fee model returned an invalid amount")
    return value_object


def slipped_price(
    model: SlippageModel,
    reference_price: Decimal,
    side: OrderSide,
    *,
    context: str = "",
) -> Decimal:
    """Evaluate a slippage callback under the serialized arithmetic policy."""
    try:
        with arithmetic():
            value = model.apply(reference_price, side)
    except (ArithmeticError, ValueError) as error:
        raise ExecutionError(
            _operation("slippage calculation failed", context)
        ) from error
    value_object = cast(object, value)
    if (
        not isinstance(value_object, Decimal)
        or not value_object.is_finite()
        or value_object <= 0
    ):
        raise ExecutionError("slippage model returned an invalid fill price")
    if (side is OrderSide.BUY and value_object < reference_price) or (
        side is OrderSide.SELL and value_object > reference_price
    ):
        raise ExecutionError("slippage must be adverse or zero")
    return value_object


def affordable_quantity(
    cash_budget: Decimal,
    fill_price: Decimal,
    commission: CommissionModel,
    fees: FeeModel,
    *,
    context: str = "",
) -> int:
    """Find the maximum affordable quantity for verified monotonic costs."""
    with arithmetic():
        upper = int(cash_budget / fill_price)
    low = 0
    high = upper
    while low < high:
        candidate = (low + high + 1) // 2
        candidate_commission = commission_amount(
            commission,
            candidate,
            fill_price,
            context=context,
        )
        candidate_fees = fee_amount(
            fees,
            OrderSide.BUY,
            candidate,
            fill_price,
            context=context,
        )
        with arithmetic():
            affordable = (
                Decimal(candidate) * fill_price + candidate_commission + candidate_fees
                <= cash_budget
            )
        if affordable:
            low = candidate
        else:
            high = candidate - 1
    return low
