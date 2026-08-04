"""Deterministic decimal arithmetic shared by backtesting components."""

from contextlib import AbstractContextManager
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)

from quantforge.configuration import PrimitiveMapping

DECIMAL_PRECISION = 34
DECIMAL_ROUNDING = ROUND_HALF_EVEN
DECIMAL_EMIN = -999_999
DECIMAL_EMAX = 999_999
DECIMAL_TRAPS: tuple[type[DecimalException], ...] = (
    DivisionByZero,
    InvalidOperation,
    Overflow,
)


def arithmetic_configuration() -> PrimitiveMapping:
    """Serialize the complete arithmetic policy included in run identity."""
    return {
        "decimal_precision": DECIMAL_PRECISION,
        "rounding": DECIMAL_ROUNDING,
        "decimal_emin": DECIMAL_EMIN,
        "decimal_emax": DECIMAL_EMAX,
        "capitals": 1,
        "clamp": 0,
        "initial_flags": [],
        "traps": [signal.__name__ for signal in DECIMAL_TRAPS],
    }


def arithmetic_context() -> Context:
    """Return a fresh, complete decimal policy for reproducible calculations."""
    return Context(
        prec=DECIMAL_PRECISION,
        rounding=DECIMAL_ROUNDING,
        Emin=DECIMAL_EMIN,
        Emax=DECIMAL_EMAX,
        capitals=1,
        clamp=0,
        flags=[],
        traps=list(DECIMAL_TRAPS),
    )


def arithmetic() -> AbstractContextManager[Context]:
    """Activate the private backtesting arithmetic policy."""
    return localcontext(arithmetic_context())


def decimal_from(value: object, field_name: str) -> Decimal:
    """Convert a supported numeric input to a finite decimal."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        converted = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field_name} must be numeric") from error
    if not converted.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return converted


def decimal_sqrt(value: Decimal) -> Decimal:
    """Calculate a square root under the private policy."""
    with arithmetic():
        return value.sqrt()


def fractional_power(base: Decimal, exponent: Decimal) -> Decimal:
    """Calculate a positive-base fractional power under the private policy."""
    with arithmetic():
        return (base.ln() * exponent).exp()
