"""Lazy deterministic Cartesian combination generation."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.optimization.constraints import ParameterConstraint
from quantforge.optimization.errors import InvalidStudyConfigurationError
from quantforge.optimization.factories import StrategyFactory
from quantforge.optimization.spaces import (
    ParameterSearchSpace,
    search_value_to_primitive,
)
from quantforge.strategies import InvalidStrategyParametersError


@dataclass(frozen=True, slots=True)
class ParameterCombination:
    """One factory-validated assignment in canonical Cartesian order."""

    index: int
    combination_id: str
    parameters_snapshot: PrimitiveMappingSnapshot
    strategy_parameters_snapshot: PrimitiveMappingSnapshot
    strategy_configuration_id: str
    coordinates: tuple[int, ...]

    @property
    def parameters(self) -> PrimitiveMapping:
        return self.parameters_snapshot.to_primitive()

    @property
    def strategy_parameters(self) -> PrimitiveMapping:
        return self.strategy_parameters_snapshot.to_primitive()


@dataclass(frozen=True, slots=True)
class CombinationExclusion:
    """A Cartesian assignment excluded before QF-5 execution."""

    index: int
    combination_id: str
    parameters_snapshot: PrimitiveMappingSnapshot
    coordinates: tuple[int, ...]
    reason_code: str
    reason: str

    @property
    def parameters(self) -> PrimitiveMapping:
        return self.parameters_snapshot.to_primitive()


type CombinationCandidate = ParameterCombination | CombinationExclusion


def _combination_id(
    strategy_factory: StrategyFactory, parameters: PrimitiveMapping
) -> str:
    return configuration_identity(
        {
            "component": "quantforge_parameter_combination",
            "combination_schema_version": "1",
            "strategy_name": strategy_factory.strategy_name,
            "strategy_version": strategy_factory.strategy_version,
            "strategy_factory": strategy_factory.configuration(),
            "parameters": parameters,
        }
    )


def _validate_factory(strategy_factory: StrategyFactory) -> None:
    parameter_order = strategy_factory.parameter_order
    if not parameter_order or len(set(parameter_order)) != len(parameter_order):
        raise InvalidStudyConfigurationError(
            "strategy factory parameter order must be nonempty and unique"
        )
    if (
        not strategy_factory.strategy_name.strip()
        or not strategy_factory.strategy_version.strip()
    ):
        raise InvalidStudyConfigurationError(
            "strategy factory identity and version must be explicit"
        )
    if not strategy_factory.required_parameter_names.issubset(parameter_order):
        raise InvalidStudyConfigurationError(
            "strategy factory required parameters must belong to its contract"
        )
    try:
        configuration_identity(strategy_factory.configuration())
    except (TypeError, ValueError) as error:
        raise InvalidStudyConfigurationError(
            "strategy factory configuration must be stably serializable"
        ) from error


def validate_combination_definition(
    strategy_factory: StrategyFactory,
    search_space: ParameterSearchSpace,
    constraints: Sequence[ParameterConstraint],
) -> None:
    """Fail before execution for malformed factories, names, and constraints."""
    _validate_factory(strategy_factory)
    search_space.ordered_items(strategy_factory.parameter_order)
    missing = strategy_factory.required_parameter_names.difference(search_space.names)
    if missing:
        rendered = ", ".join(sorted(missing))
        raise InvalidStudyConfigurationError(
            f"search space omits required strategy parameters: {rendered}"
        )
    for constraint in constraints:
        constraint.validate(search_space.names)
        try:
            configuration_identity(constraint.to_primitive())
        except (TypeError, ValueError) as error:
            raise InvalidStudyConfigurationError(
                "parameter constraints must be stably serializable"
            ) from error


def iter_combination_candidates(
    strategy_factory: StrategyFactory,
    search_space: ParameterSearchSpace,
    constraints: Sequence[ParameterConstraint],
) -> Iterator[CombinationCandidate]:
    """Yield every Cartesian assignment lazily in canonical contract order."""
    validate_combination_definition(strategy_factory, search_space, constraints)
    ordered_items = search_space.ordered_items(strategy_factory.parameter_order)
    candidate_axes = tuple(values.values for _, values in ordered_items)
    coordinate_axes = tuple(range(len(axis)) for axis in candidate_axes)
    seen_identifiers: set[str] = set()

    for index, coordinates in enumerate(product(*coordinate_axes)):
        normalized_values = tuple(
            candidate_axes[axis][coordinate]
            for axis, coordinate in enumerate(coordinates)
        )
        search_values = {
            name: value
            for (name, _), value in zip(ordered_items, normalized_values, strict=True)
        }
        primitive_parameters: PrimitiveMapping = {
            name: search_value_to_primitive(value)
            for name, value in search_values.items()
        }
        combination_id = _combination_id(strategy_factory, primitive_parameters)
        if combination_id in seen_identifiers:
            continue
        seen_identifiers.add(combination_id)
        snapshot = PrimitiveMappingSnapshot.capture(primitive_parameters)

        failed_decision = None
        for constraint in constraints:
            decision = constraint.evaluate(search_values.copy())
            if not decision.passed:
                failed_decision = decision
                break
        if failed_decision is not None:
            yield CombinationExclusion(
                index=index,
                combination_id=combination_id,
                parameters_snapshot=snapshot,
                coordinates=tuple(coordinates),
                reason_code=failed_decision.code,
                reason=failed_decision.message,
            )
            continue

        try:
            strategy = strategy_factory.build(primitive_parameters.copy())
        except InvalidStrategyParametersError as error:
            message = " ".join(str(error).split())[:500] or error.__class__.__name__
            yield CombinationExclusion(
                index=index,
                combination_id=combination_id,
                parameters_snapshot=snapshot,
                coordinates=tuple(coordinates),
                reason_code="invalid_strategy_parameters",
                reason=f"{error.__class__.__name__}: {message}",
            )
            continue
        if strategy.name != strategy_factory.strategy_name:
            raise InvalidStudyConfigurationError(
                "factory produced a strategy with an incompatible identity"
            )
        if strategy.implementation_version != strategy_factory.strategy_version:
            raise InvalidStudyConfigurationError(
                "factory produced a strategy with an incompatible version"
            )
        strategy_configuration = strategy.configuration()
        expected_configuration_id = configuration_identity(strategy_configuration)
        if strategy.configuration_id != expected_configuration_id:
            raise InvalidStudyConfigurationError(
                "factory strategy configuration identity is stale or invalid"
            )
        strategy_parameters_value = cast(
            object, strategy_configuration.get("parameters")
        )
        if not isinstance(strategy_parameters_value, dict):
            raise InvalidStudyConfigurationError(
                "strategy configuration must contain primitive parameters"
            )
        strategy_parameters = cast(PrimitiveMapping, strategy_parameters_value)
        strategy_parameters_snapshot = PrimitiveMappingSnapshot.capture(
            strategy_parameters
        )

        yield ParameterCombination(
            index=index,
            combination_id=combination_id,
            parameters_snapshot=snapshot,
            strategy_parameters_snapshot=strategy_parameters_snapshot,
            strategy_configuration_id=strategy.configuration_id,
            coordinates=tuple(coordinates),
        )
