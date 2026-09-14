"""Small strict readers for the existing canonical primitive contracts."""

from decimal import Decimal

from quantforge.backtesting._arithmetic import decimal_from
from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity


class OOSIntegrityError(ValueError):
    """Source evidence cannot support an OOS or pristine-holdout claim."""


def mapping(value: Primitive) -> PrimitiveMapping:
    if not isinstance(value, dict):
        raise OOSIntegrityError("expected a structured record")
    return value


def records(value: Primitive) -> list[PrimitiveMapping]:
    if not isinstance(value, list):
        raise OOSIntegrityError("expected a record collection")
    return [mapping(item) for item in value]


def texts(value: Primitive) -> list[str]:
    if not isinstance(value, list):
        raise OOSIntegrityError("expected a text collection")
    return [text(item) for item in value]


def text(value: Primitive) -> str:
    if not isinstance(value, str) or not value:
        raise OOSIntegrityError("expected nonempty text")
    return value


def number(value: Primitive) -> Decimal:
    return decimal_from(value, "OOS metric")


def verify_identity(record: PrimitiveMapping, key: str) -> None:
    if record.get(key) != configuration_identity(
        {name: value for name, value in record.items() if name != key}
    ):
        raise OOSIntegrityError(f"invalid {key}")
