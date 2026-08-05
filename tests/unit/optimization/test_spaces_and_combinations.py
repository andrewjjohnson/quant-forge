from collections.abc import Callable
from decimal import Decimal
from typing import cast

import pytest

from quantforge.optimization import (
    BooleanValues,
    CategoricalValues,
    CombinationExclusion,
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
