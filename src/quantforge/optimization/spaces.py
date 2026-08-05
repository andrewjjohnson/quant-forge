"""Typed finite parameter spaces with deterministic normalization."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from enum import StrEnum
from math import prod
from typing import Protocol, cast

from quantforge.configuration import Primitive, PrimitiveMapping, decimal_to_primitive
from quantforge.optimization.errors import InvalidSearchSpaceError

type SearchValue = str | int | bool | Decimal


class ParameterValues(Protocol):
    """One finite, ordered, typed candidate set."""

    @property
    def kind(self) -> str: ...

    @property
    def values(self) -> tuple[SearchValue, ...]: ...

    def to_primitive(self) -> PrimitiveMapping: ...


def search_value_to_primitive(value: SearchValue) -> str | int | bool:
    """Serialize a normalized candidate without losing decimal precision."""
    if isinstance(value, Decimal):
        return decimal_to_primitive(value)
    return value


@dataclass(frozen=True, slots=True, init=False)
class IntegerValues:
    """Explicit true-integer candidates in caller-declared order."""

    _values: tuple[int, ...]
    kind = "integer"

    def __init__(self, values: Iterable[int]) -> None:
        normalized_values = cast(tuple[object, ...], tuple(values))
        if not normalized_values:
            raise InvalidSearchSpaceError("integer candidate values cannot be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in normalized_values
        ):
            raise InvalidSearchSpaceError(
                "integer candidates must be true integers, not booleans"
            )
        normalized = cast(tuple[int, ...], normalized_values)
        if len(set(normalized)) != len(normalized):
            raise InvalidSearchSpaceError(
                "duplicate integer candidates are not allowed"
            )
        object.__setattr__(self, "_values", normalized)

    @classmethod
    def inclusive_range(cls, start: int, stop: int, step: int = 1) -> "IntegerValues":
        """Expand a finite inclusive integer range."""
        range_values = cast(tuple[object, ...], (start, stop, step))
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in range_values
        ):
            raise InvalidSearchSpaceError("integer range values must be true integers")
        if step <= 0:
            raise InvalidSearchSpaceError("integer range step must be positive")
        if stop < start:
            raise InvalidSearchSpaceError("integer range stop must not precede start")
        return cls(range(start, stop + 1, step))

    @property
    def values(self) -> tuple[SearchValue, ...]:
        return self._values

    def to_primitive(self) -> PrimitiveMapping:
        return {"kind": self.kind, "values": list(self._values)}


def _decimal_candidate(value: Decimal | int | float | str) -> Decimal:
    if isinstance(value, bool):
        raise InvalidSearchSpaceError("boolean values are not floating candidates")
    try:
        normalized = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise InvalidSearchSpaceError("floating candidates must be numeric") from error
    if not normalized.is_finite():
        raise InvalidSearchSpaceError("floating candidates must be finite")
    return normalized


@dataclass(frozen=True, slots=True, init=False)
class FloatValues:
    """Decimal-normalized floating candidates in caller-declared order."""

    _values: tuple[Decimal, ...]
    kind = "float"

    def __init__(self, values: Iterable[Decimal | int | float | str]) -> None:
        normalized = tuple(_decimal_candidate(value) for value in values)
        if not normalized:
            raise InvalidSearchSpaceError("floating candidate values cannot be empty")
        rendered = tuple(decimal_to_primitive(value) for value in normalized)
        if len(set(rendered)) != len(rendered):
            raise InvalidSearchSpaceError(
                "duplicate normalized floating candidates are not allowed"
            )
        object.__setattr__(self, "_values", normalized)

    @classmethod
    def inclusive_range(
        cls,
        start: Decimal | int | float | str,
        stop: Decimal | int | float | str,
        step: Decimal | int | float | str,
    ) -> "FloatValues":
        """Expand an inclusive grid using exact Decimal addition."""
        normalized_start = _decimal_candidate(start)
        normalized_stop = _decimal_candidate(stop)
        normalized_step = _decimal_candidate(step)
        if normalized_step <= 0:
            raise InvalidSearchSpaceError("floating range step must be positive")
        if normalized_stop < normalized_start:
            raise InvalidSearchSpaceError("floating range stop must not precede start")
        values: list[Decimal] = []
        with localcontext() as context:
            context.prec = 34
            current = normalized_start
            while current <= normalized_stop:
                values.append(current)
                if current == normalized_stop:
                    break
                next_value = current + normalized_step
                if next_value <= current:
                    raise InvalidSearchSpaceError(
                        "floating range step is too small to advance under the "
                        "fixed decimal normalization context"
                    )
                current = next_value
        return cls(values)

    @property
    def values(self) -> tuple[SearchValue, ...]:
        return self._values

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "kind": self.kind,
            "values": [decimal_to_primitive(value) for value in self._values],
            "normalization": "decimal_exact",
        }


type CategoricalCandidate = str | int | StrEnum


@dataclass(frozen=True, slots=True, init=False)
class CategoricalValues:
    """Stable string or integer categorical candidates."""

    _values: tuple[str | int, ...]
    kind = "categorical"

    def __init__(self, values: Iterable[CategoricalCandidate]) -> None:
        normalized: list[str | int] = []
        for value in values:
            candidate = cast(
                object, value.value if isinstance(value, StrEnum) else value
            )
            if isinstance(candidate, bool) or not isinstance(candidate, (str, int)):
                raise InvalidSearchSpaceError(
                    "categorical candidates must serialize as strings or integers"
                )
            normalized.append(candidate)
        if not normalized:
            raise InvalidSearchSpaceError(
                "categorical candidate values cannot be empty"
            )
        typed_values = tuple(normalized)
        if len(set(typed_values)) != len(typed_values):
            raise InvalidSearchSpaceError(
                "duplicate categorical candidates are not allowed"
            )
        object.__setattr__(self, "_values", typed_values)

    @property
    def values(self) -> tuple[SearchValue, ...]:
        return self._values

    def to_primitive(self) -> PrimitiveMapping:
        return {"kind": self.kind, "values": cast(list[Primitive], list(self._values))}


@dataclass(frozen=True, slots=True, init=False)
class BooleanValues:
    """Explicit boolean candidates, kept distinct from integers."""

    _values: tuple[bool, ...]
    kind = "boolean"

    def __init__(self, values: Iterable[bool]) -> None:
        normalized_values = cast(tuple[object, ...], tuple(values))
        if not normalized_values:
            raise InvalidSearchSpaceError("boolean candidate values cannot be empty")
        if any(not isinstance(value, bool) for value in normalized_values):
            raise InvalidSearchSpaceError("boolean candidates must be true booleans")
        normalized = cast(tuple[bool, ...], normalized_values)
        if len(set(normalized)) != len(normalized):
            raise InvalidSearchSpaceError(
                "duplicate boolean candidates are not allowed"
            )
        object.__setattr__(self, "_values", normalized)

    @property
    def values(self) -> tuple[SearchValue, ...]:
        return self._values

    def to_primitive(self) -> PrimitiveMapping:
        return {"kind": self.kind, "values": list(self._values)}


@dataclass(frozen=True, slots=True, init=False)
class ParameterSearchSpace:
    """A finite search space stored independently of input dictionary order."""

    _parameters: tuple[tuple[str, ParameterValues], ...]

    def __init__(self, parameters: Mapping[str, ParameterValues]) -> None:
        if not parameters:
            raise InvalidSearchSpaceError("parameter search space cannot be empty")
        normalized: list[tuple[str, ParameterValues]] = []
        for name, values in parameters.items():
            name_value = cast(object, name)
            if not isinstance(name_value, str) or not name_value.strip():
                raise InvalidSearchSpaceError(
                    "searched parameter names must be nonempty"
                )
            if not isinstance(
                values, (IntegerValues, FloatValues, CategoricalValues, BooleanValues)
            ):
                raise InvalidSearchSpaceError(
                    f"unsupported value-space model for parameter {name!r}"
                )
            normalized.append((name, values))
        normalized.sort(key=lambda item: item[0])
        object.__setattr__(self, "_parameters", tuple(normalized))

    @property
    def names(self) -> frozenset[str]:
        return frozenset(name for name, _ in self._parameters)

    def space_for(self, name: str) -> ParameterValues:
        for parameter_name, values in self._parameters:
            if parameter_name == name:
                return values
        raise KeyError(name)

    def ordered_items(
        self, parameter_contract_order: tuple[str, ...]
    ) -> tuple[tuple[str, ParameterValues], ...]:
        missing = self.names.difference(parameter_contract_order)
        if missing:
            rendered = ", ".join(sorted(missing))
            raise InvalidSearchSpaceError(
                f"searched parameters are absent from the strategy contract: {rendered}"
            )
        return tuple(
            (name, self.space_for(name))
            for name in parameter_contract_order
            if name in self.names
        )

    def combination_count(self) -> int:
        return prod(len(values.values) for _, values in self._parameters)

    def count_expression(self, parameter_contract_order: tuple[str, ...]) -> str:
        counts = [
            len(values.values)
            for _, values in self.ordered_items(parameter_contract_order)
        ]
        return " x ".join(str(count) for count in counts) + f" = {prod(counts):,}"

    def to_primitive(
        self, parameter_contract_order: tuple[str, ...]
    ) -> PrimitiveMapping:
        return {
            "parameter_ordering": "strategy_parameter_contract_declaration_order",
            "parameters": [
                {"name": name, **values.to_primitive()}
                for name, values in self.ordered_items(parameter_contract_order)
            ],
        }
