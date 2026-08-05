from collections.abc import Callable
from decimal import Decimal
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.optimization import (
    BooleanValues,
    CategoricalValues,
    CombinationExclusion,
    CustomParameterConstraint,
    FloatValues,
    IntegerValues,
    InvalidSearchSpaceError,
    InvalidStudyConfigurationError,
    MovingAverageCrossoverFactory,
    ParameterAtMost,
    ParameterLessThan,
    ParameterSearchSpace,
    iter_combination_candidates,
)
from quantforge.optimization.spaces import SearchValue
from quantforge.strategies import Strategy


class _MismatchedIdentityFactory(MovingAverageCrossoverFactory):
    strategy_name = "mismatched_strategy_identity"


class _ExplodingFactory(MovingAverageCrossoverFactory):
    def build(self, parameters: PrimitiveMapping) -> Strategy:
        raise RuntimeError("broken factory implementation")


def _fast_window_is_two(parameters: dict[str, SearchValue]) -> bool:
    return parameters["fast_window"] == 2


def test_typed_value_spaces_normalize_and_expand_deterministically() -> None:
    assert IntegerValues.inclusive_range(1, 5, 2).values == (1, 3, 5)
    assert FloatValues([0.1, "0.20", Decimal("0.300")]).to_primitive() == {
        "kind": "float",
        "values": ["0.1", "0.2", "0.3"],
        "normalization": "decimal_exact",
    }
    assert FloatValues.inclusive_range("0.1", "0.3", "0.1").to_primitive()[
        "values"
    ] == ["0.1", "0.2", "0.3"]
    assert CategoricalValues(["close", 7]).values == ("close", 7)
    assert BooleanValues([True, False]).values == (True, False)


@pytest.mark.parametrize(
    "construction",
    [
        lambda: IntegerValues([]),
        lambda: IntegerValues(cast(list[int], [True])),
        lambda: IntegerValues([1, 1]),
        lambda: IntegerValues.inclusive_range(1, 3, 0),
        lambda: FloatValues([]),
        lambda: FloatValues(["NaN"]),
        lambda: FloatValues(["Infinity"]),
        lambda: FloatValues(["0.30", Decimal("0.3")]),
        lambda: FloatValues.inclusive_range("1e100", "2e100", "1"),
        lambda: CategoricalValues([]),
        lambda: CategoricalValues(cast(list[str | int], [True])),
        lambda: BooleanValues([]),
        lambda: BooleanValues(cast(list[bool], [0, 1])),
    ],
)
def test_invalid_or_duplicate_candidates_are_rejected(
    construction: Callable[[], object],
) -> None:
    with pytest.raises(InvalidSearchSpaceError):
        construction()


def test_parameter_order_uses_strategy_contract_not_dictionary_construction() -> None:
    first = ParameterSearchSpace(
        {
            "slow_window": IntegerValues([3, 5]),
            "fast_window": IntegerValues([2, 4]),
        }
    )
    second = ParameterSearchSpace(
        {
            "fast_window": IntegerValues([2, 4]),
            "slow_window": IntegerValues([3, 5]),
        }
    )
    factory = MovingAverageCrossoverFactory()

    assert first.to_primitive(factory.parameter_order) == second.to_primitive(
        factory.parameter_order
    )
    assert [name for name, _ in first.ordered_items(factory.parameter_order)] == [
        "fast_window",
        "slow_window",
    ]


def test_cartesian_generation_is_lazy_ordered_identified_and_constrained() -> None:
    factory = MovingAverageCrossoverFactory()
    search_space = ParameterSearchSpace(
        {
            "slow_window": IntegerValues([3, 5]),
            "fast_window": IntegerValues([2, 4]),
        }
    )
    generator = iter_combination_candidates(
        factory,
        search_space,
        (ParameterLessThan("fast_window", "slow_window"),),
    )
    assert iter(generator) is generator
    candidates = tuple(generator)

    assert search_space.combination_count() == 4
    assert [candidate.parameters for candidate in candidates] == [
        {"fast_window": 2, "slow_window": 3},
        {"fast_window": 2, "slow_window": 5},
        {"fast_window": 4, "slow_window": 3},
        {"fast_window": 4, "slow_window": 5},
    ]
    assert isinstance(candidates[2], CombinationExclusion)
    assert candidates[2].reason_code == "parameter_less_than"
    assert len({candidate.combination_id for candidate in candidates}) == 4

    independently_created = tuple(
        iter_combination_candidates(
            factory,
            ParameterSearchSpace(
                {
                    "fast_window": IntegerValues([2, 4]),
                    "slow_window": IntegerValues([3, 5]),
                }
            ),
            (ParameterLessThan("fast_window", "slow_window"),),
        )
    )
    assert [item.combination_id for item in candidates] == [
        item.combination_id for item in independently_created
    ]


def test_real_parameter_model_excludes_invalid_values_before_backtesting() -> None:
    candidates = tuple(
        iter_combination_candidates(
            MovingAverageCrossoverFactory(),
            ParameterSearchSpace(
                {
                    "fast_window": IntegerValues([0, 2]),
                    "slow_window": IntegerValues([3]),
                }
            ),
            (),
        )
    )
    assert isinstance(candidates[0], CombinationExclusion)
    assert candidates[0].reason_code == "invalid_strategy_parameters"
    assert not isinstance(candidates[1], CombinationExclusion)


def test_factory_contract_invariant_failures_abort_combination_generation() -> None:
    search_space = ParameterSearchSpace(
        {
            "fast_window": IntegerValues([2]),
            "slow_window": IntegerValues([3]),
        }
    )
    with pytest.raises(InvalidStudyConfigurationError, match="incompatible identity"):
        tuple(
            iter_combination_candidates(
                _MismatchedIdentityFactory(),
                search_space,
                (),
            )
        )
    with pytest.raises(RuntimeError, match="broken factory implementation"):
        tuple(iter_combination_candidates(_ExplodingFactory(), search_space, ()))


def test_custom_constraints_require_true_module_level_predicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constraint = CustomParameterConstraint(
        "fast_window_is_two",
        "1",
        _fast_window_is_two,
        ("fast_window",),
    )

    def nested_predicate(parameters: dict[str, SearchValue]) -> bool:
        return parameters["fast_window"] == 2

    assert constraint.to_primitive()["predicate"] == (f"{__name__}._fast_window_is_two")
    monkeypatch.setattr(_fast_window_is_two, "__module__", "__main__")
    with pytest.raises(InvalidStudyConfigurationError, match="durable import module"):
        CustomParameterConstraint(
            "entry_point",
            "1",
            _fast_window_is_two,
            ("fast_window",),
        )
    with pytest.raises(InvalidStudyConfigurationError, match="module-level"):
        CustomParameterConstraint(
            "nested",
            "1",
            nested_predicate,
            ("fast_window",),
        )


def test_constraints_fail_early_for_unknown_names_and_support_constants() -> None:
    factory = MovingAverageCrossoverFactory()
    search_space = ParameterSearchSpace(
        {
            "fast_window": IntegerValues([2]),
            "slow_window": IntegerValues([3]),
        }
    )
    constrained = tuple(
        iter_combination_candidates(
            factory,
            search_space,
            (ParameterAtMost("fast_window", 1),),
        )
    )
    assert isinstance(constrained[0], CombinationExclusion)

    with pytest.raises(InvalidStudyConfigurationError, match="unknown"):
        tuple(
            iter_combination_candidates(
                factory,
                search_space,
                (ParameterLessThan("frist_window", "slow_window"),),
            )
        )
