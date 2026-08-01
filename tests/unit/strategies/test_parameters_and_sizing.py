import json
from decimal import Decimal
from typing import cast

import pytest

from quantforge.indicators import MarketField
from quantforge.strategies import (
    InvalidStrategyParametersError,
    InvalidTargetWeightError,
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
    PositionIntent,
    TargetWeightSizingPolicy,
)


def test_valid_parameters_are_immutable_and_stably_serialized() -> None:
    parameters = MovingAverageCrossoverParameters(
        2, 5, MarketField.CLOSE, Decimal("0.50")
    )

    assert parameters.to_primitive() == {
        "fast_window": 2,
        "slow_window": 5,
        "source_field": "close",
        "target_long_weight": "0.5",
    }
    assert json.dumps(parameters.to_primitive(), sort_keys=True)
    with pytest.raises(AttributeError):
        parameters.fast_window = 3  # type: ignore[misc]


@pytest.mark.parametrize(("fast", "slow"), [(0, 3), (2, 0), (3, 3), (4, 3)])
def test_rejects_invalid_window_relationships(fast: int, slow: int) -> None:
    with pytest.raises(InvalidStrategyParametersError):
        MovingAverageCrossoverParameters(fast, slow)


@pytest.mark.parametrize("weight", [Decimal("0"), Decimal("-0.1"), Decimal("1.1")])
def test_rejects_invalid_strategy_target_weight(weight: Decimal) -> None:
    with pytest.raises(InvalidStrategyParametersError, match="weight"):
        MovingAverageCrossoverParameters(2, 3, target_long_weight=weight)


def test_rejects_unknown_strategy_source_field() -> None:
    with pytest.raises(InvalidStrategyParametersError, match="source_field"):
        MovingAverageCrossoverParameters(
            2, 3, source_field=cast(MarketField, "adjusted_close")
        )


def test_semantically_equal_parameters_have_same_configuration_identity() -> None:
    first = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverParameters(2, 3, target_long_weight=Decimal("0.50"))
    )
    same = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverParameters(2, 3, target_long_weight=Decimal("0.5"))
    )
    different = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverParameters(2, 4, target_long_weight=Decimal("0.5"))
    )

    assert first.configuration_id == same.configuration_id
    assert first.configuration_id != different.configuration_id


def test_reference_sizing_policy_emits_only_normalized_target_intent() -> None:
    policy = TargetWeightSizingPolicy(Decimal("0.75"))

    assert policy.required_context_fields == frozenset()
    assert policy.size(PositionIntent.LONG).target_weight == Decimal("0.75")
    assert policy.size(PositionIntent.FLAT).target_weight == 0
    assert policy.configuration()["component_name"] == "long_only_target_weight"


def test_sizing_policy_rejects_invalid_long_weight() -> None:
    with pytest.raises(InvalidTargetWeightError):
        TargetWeightSizingPolicy(Decimal("1.01"))
