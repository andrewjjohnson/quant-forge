"""Deterministic primitive configuration types and identity helpers."""

import hashlib
import json
from decimal import Decimal

type PrimitiveScalar = str | int | float | bool | None
type Primitive = PrimitiveScalar | list[Primitive] | dict[str, Primitive]
type PrimitiveMapping = dict[str, Primitive]


def decimal_to_primitive(value: Decimal) -> str:
    """Render a finite decimal exactly without representation-only zeros.

    Decimal formatting without an explicit format precision is independent of
    the active arithmetic context. Only fractional trailing zeros are removed;
    integer zeros remain part of the value.
    """
    if not value.is_finite():
        raise ValueError("configuration decimals must be finite")
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def configuration_identity(configuration: PrimitiveMapping) -> str:
    """Return a stable SHA-256 identity for a primitive configuration."""
    encoded = json.dumps(
        configuration,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
