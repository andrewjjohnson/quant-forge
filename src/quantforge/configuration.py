"""Deterministic primitive configuration types and identity helpers."""

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

type PrimitiveScalar = str | int | float | bool | None
type Primitive = PrimitiveScalar | list[Primitive] | dict[str, Primitive]
type PrimitiveMapping = dict[str, Primitive]


def _canonical_json(configuration: PrimitiveMapping) -> str:
    return json.dumps(
        configuration,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True, slots=True)
class PrimitiveMappingSnapshot:
    """Deeply immutable canonical snapshot of primitive configuration values."""

    canonical_json: str

    @classmethod
    def capture(cls, configuration: PrimitiveMapping) -> "PrimitiveMappingSnapshot":
        return cls(_canonical_json(configuration))

    def to_primitive(self) -> PrimitiveMapping:
        """Return a detached mutable representation of the immutable snapshot."""
        loaded = json.loads(self.canonical_json)
        if not isinstance(loaded, dict):
            raise TypeError("primitive mapping snapshot must decode to an object")
        return cast(PrimitiveMapping, loaded)


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
    encoded = _canonical_json(configuration).encode()
    return hashlib.sha256(encoded).hexdigest()
