"""Private deterministic arithmetic policy for prediction labels and metrics."""

from collections.abc import Generator
from contextlib import contextmanager
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)

DECIMAL_PRECISION = 34
DECIMAL_ROUNDING = "ROUND_HALF_EVEN"
DECIMAL_EMIN = -999_999
DECIMAL_EMAX = 999_999
DECIMAL_CAPITALS = 1
DECIMAL_CLAMP = 0
DECIMAL_TRAPS: tuple[type[DecimalException], ...] = (
    DivisionByZero,
    InvalidOperation,
    Overflow,
)


@contextmanager
def arithmetic() -> Generator[None]:
    """Run calculations under the complete serialized analysis policy."""
    context = Context(
        prec=DECIMAL_PRECISION,
        rounding=ROUND_HALF_EVEN,
        Emin=DECIMAL_EMIN,
        Emax=DECIMAL_EMAX,
        capitals=DECIMAL_CAPITALS,
        clamp=DECIMAL_CLAMP,
        flags=[],
        traps=list(DECIMAL_TRAPS),
    )
    with localcontext(context):
        yield
