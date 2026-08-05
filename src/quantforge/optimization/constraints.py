"""Deterministic, serializable parameter-combination constraints."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from operator import ge, gt, le, lt
from typing import Any, Protocol, cast

from quantforge.configuration import PrimitiveMapping
from quantforge.optimization.errors import InvalidStudyConfigurationError
from quantforge.optimization.spaces import SearchValue, search_value_to_primitive


class ComparisonOperator(StrEnum):
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "le"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "ge"
    EQUAL = "eq"
    NOT_EQUAL = "ne"


@dataclass(frozen=True, slots=True)
class ConstraintDecision:
    passed: bool
    code: str
    message: str


class ParameterConstraint(Protocol):
    @property
    def name(self) -> str: ...

    def validate(self, parameter_names: frozenset[str]) -> None: ...

    def evaluate(self, parameters: dict[str, SearchValue]) -> ConstraintDecision: ...

    def to_primitive(self) -> PrimitiveMapping: ...


def _compare(
    left: SearchValue, operator: ComparisonOperator, right: SearchValue
) -> bool:
    if operator is ComparisonOperator.EQUAL:
        return left == right
    if operator is ComparisonOperator.NOT_EQUAL:
        return left != right
    left_value = cast(Any, left)
    right_value = cast(Any, right)
    try:
        if operator is ComparisonOperator.LESS_THAN:
            return lt(left_value, right_value)
        if operator is ComparisonOperator.LESS_THAN_OR_EQUAL:
            return le(left_value, right_value)
        if operator is ComparisonOperator.GREATER_THAN:
            return gt(left_value, right_value)
        if operator is ComparisonOperator.GREATER_THAN_OR_EQUAL:
            return ge(left_value, right_value)
        raise InvalidStudyConfigurationError(
            "constraint comparison operator is unsupported"
        )
    except TypeError as error:
        raise InvalidStudyConfigurationError(
            "constraint operands are not deterministically comparable"
        ) from error


@dataclass(frozen=True, slots=True)
class ParameterComparison:
    """Compare one searched parameter with another parameter or a constant."""

    parameter: str
    operator: ComparisonOperator
    other_parameter: str | None = None
    constant: SearchValue | None = None
    name: str = "parameter_comparison"

    def __post_init__(self) -> None:
        if not self.parameter:
            raise InvalidStudyConfigurationError("constraint parameter is required")
        operator = cast(object, self.operator)
        if not isinstance(operator, ComparisonOperator):
            raise InvalidStudyConfigurationError(
                "constraint comparison operator must be a ComparisonOperator"
            )
        if (self.other_parameter is None) == (self.constant is None):
            raise InvalidStudyConfigurationError(
                "a parameter comparison requires exactly one right-hand operand"
            )

    def validate(self, parameter_names: frozenset[str]) -> None:
        missing = {self.parameter}.difference(parameter_names)
        if self.other_parameter is not None:
            missing.update({self.other_parameter}.difference(parameter_names))
        if missing:
            rendered = ", ".join(sorted(missing))
            raise InvalidStudyConfigurationError(
                f"constraint references unknown searched parameters: {rendered}"
            )

    def evaluate(self, parameters: dict[str, SearchValue]) -> ConstraintDecision:
        right = (
            parameters[self.other_parameter]
            if self.other_parameter is not None
            else self.constant
        )
        assert right is not None
        passed = _compare(parameters[self.parameter], self.operator, right)
        rendered_right = self.other_parameter or repr(search_value_to_primitive(right))
        expression = f"{self.parameter} {self.operator.value} {rendered_right}"
        return ConstraintDecision(
            passed,
            self.name,
            f"constraint passed: {expression}"
            if passed
            else f"constraint failed: {expression}",
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "type": "parameter_comparison",
            "name": self.name,
            "parameter": self.parameter,
            "operator": self.operator.value,
            "other_parameter": self.other_parameter,
            "constant": (
                None
                if self.constant is None
                else search_value_to_primitive(self.constant)
            ),
        }


class ParameterLessThan(ParameterComparison):
    """Declarative ``left < right`` constraint."""

    def __init__(self, left_parameter: str, right_parameter: str) -> None:
        super().__init__(
            left_parameter,
            ComparisonOperator.LESS_THAN,
            other_parameter=right_parameter,
            name="parameter_less_than",
        )


class ParameterAtMost(ParameterComparison):
    """Declarative ``parameter <= maximum`` constraint."""

    def __init__(self, parameter: str, maximum: SearchValue) -> None:
        super().__init__(
            parameter,
            ComparisonOperator.LESS_THAN_OR_EQUAL,
            constant=maximum,
            name="parameter_at_most",
        )


@dataclass(frozen=True, slots=True)
class CustomParameterConstraint:
    """Named, versioned deterministic predicate for uncommon parameter rules."""

    name: str
    version: str
    predicate: Callable[[dict[str, SearchValue]], bool]
    parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        predicate_name = getattr(self.predicate, "__name__", "")
        predicate_module = getattr(self.predicate, "__module__", "")
        predicate_qualname = getattr(self.predicate, "__qualname__", "")
        if not self.name.strip() or not self.version.strip():
            raise InvalidStudyConfigurationError(
                "custom constraints require nonempty names and versions"
            )
        if predicate_module == "__main__":
            raise InvalidStudyConfigurationError(
                "custom constraint predicates require a durable import module"
            )
        if (
            predicate_name == "<lambda>"
            or not predicate_name
            or not predicate_module
            or predicate_qualname != predicate_name
        ):
            raise InvalidStudyConfigurationError(
                "custom constraints require a named module-level predicate"
            )
        if not self.parameter_names:
            raise InvalidStudyConfigurationError(
                "custom constraints must declare the parameters they inspect"
            )

    def validate(self, parameter_names: frozenset[str]) -> None:
        missing = set(self.parameter_names).difference(parameter_names)
        if missing:
            rendered = ", ".join(sorted(missing))
            raise InvalidStudyConfigurationError(
                f"custom constraint references unknown parameters: {rendered}"
            )

    def evaluate(self, parameters: dict[str, SearchValue]) -> ConstraintDecision:
        passed_value = cast(object, self.predicate(parameters.copy()))
        if not isinstance(passed_value, bool):
            raise InvalidStudyConfigurationError(
                "custom constraint predicates must return a boolean"
            )
        passed = passed_value
        return ConstraintDecision(
            passed,
            self.name,
            f"custom constraint {self.name!r} {'passed' if passed else 'failed'}",
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "type": "custom_parameter_constraint",
            "name": self.name,
            "version": self.version,
            "predicate": (f"{self.predicate.__module__}.{self.predicate.__qualname__}"),
            "parameter_names": list(self.parameter_names),
        }
