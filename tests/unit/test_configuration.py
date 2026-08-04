from decimal import ROUND_DOWN, ROUND_UP, Decimal, localcontext

import pytest

from quantforge.configuration import decimal_to_primitive


@pytest.mark.parametrize(
    ("decimal_value", "expected"),
    [
        (Decimal("12300.00"), "12300"),
        (Decimal("1E+3"), "1000"),
        (Decimal("0.0012300"), "0.00123"),
        (Decimal("-0.000"), "0"),
    ],
)
def test_decimal_primitive_is_canonical(decimal_value: Decimal, expected: str) -> None:
    assert decimal_to_primitive(decimal_value) == expected


def test_decimal_primitive_does_not_round_under_ambient_context() -> None:
    decimal_value = Decimal("0.123456789012345678901234567890")

    with localcontext() as low_precision:
        low_precision.prec = 8
        low_precision.rounding = ROUND_DOWN
        low_result = decimal_to_primitive(decimal_value)
    with localcontext() as high_precision:
        high_precision.prec = 50
        high_precision.rounding = ROUND_UP
        high_result = decimal_to_primitive(decimal_value)

    expected = "0.12345678901234567890123456789"
    assert low_result == high_result == expected
