import json
from collections.abc import Callable
from decimal import Decimal
from typing import cast

import pytest

from quantforge.backtesting import (
    BacktestConfig,
    BasisPointCommission,
    BasisPointFees,
    BasisPointSlippage,
    CommissionModel,
    ExplicitZeroFees,
    FeeModel,
    FixedCommission,
    InvalidBacktestConfigurationError,
    OrderSide,
    PerShareCommission,
    SlippageModel,
)
from quantforge.configuration import PrimitiveMapping


class UnversionedCommission:
    name = "legacy_commission"

    def calculate(self, quantity: int, fill_price: Decimal) -> Decimal:
        del quantity, fill_price
        return Decimal(0)

    def configuration(self) -> PrimitiveMapping:
        return {"model": self.name, "parameters": {}}


def test_config_requires_positive_finite_capital_and_explicit_cost_models() -> None:
    with pytest.raises(InvalidBacktestConfigurationError, match="positive"):
        BacktestConfig(
            Decimal(0),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(1)),
        )
    with pytest.raises(InvalidBacktestConfigurationError, match="finite"):
        BacktestConfig(
            Decimal("NaN"),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(1)),
        )
    with pytest.raises(InvalidBacktestConfigurationError, match="commission"):
        BacktestConfig(
            Decimal(1),
            cast(CommissionModel, None),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(1)),
        )
    with pytest.raises(InvalidBacktestConfigurationError, match="fee"):
        BacktestConfig(
            Decimal(1),
            FixedCommission(Decimal(1)),
            cast(FeeModel, None),
            BasisPointSlippage(Decimal(1)),
        )
    with pytest.raises(InvalidBacktestConfigurationError, match="slippage"):
        BacktestConfig(
            Decimal(1),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            cast(SlippageModel, None),
        )
    with pytest.raises(InvalidBacktestConfigurationError, match="implementation"):
        BacktestConfig(
            Decimal(1),
            cast(CommissionModel, UnversionedCommission()),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(1)),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FixedCommission(Decimal("-0.01")),
        lambda: PerShareCommission(Decimal("-0.01")),
        lambda: PerShareCommission(Decimal("0.01"), Decimal("-1")),
        lambda: BasisPointCommission(Decimal("-1")),
        lambda: BasisPointFees(Decimal("-1")),
        lambda: BasisPointSlippage(Decimal("-1")),
    ],
)
def test_negative_cost_parameters_are_rejected(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(InvalidBacktestConfigurationError, match="nonnegative"):
        factory()


def test_zero_cost_models_are_explicit_and_serialize_stably() -> None:
    first = BacktestConfig(
        Decimal("100.00"),
        FixedCommission(Decimal("0.00")),
        ExplicitZeroFees(),
        BasisPointSlippage(Decimal("0.0")),
    )
    second = BacktestConfig(
        Decimal(100),
        FixedCommission(Decimal(0)),
        ExplicitZeroFees(),
        BasisPointSlippage(Decimal(0)),
    )

    assert first.to_primitive() == second.to_primitive()
    assert first.to_primitive()["commission"] == {
        "model": "fixed_per_fill",
        "implementation_version": "1",
        "parameters": {"amount": "0"},
    }
    assert first.to_primitive()["fees"] == {
        "model": "explicit_zero_fees",
        "implementation_version": "1",
        "parameters": {},
    }
    assert first.to_primitive()["slippage"] == {
        "model": "adverse_basis_points",
        "implementation_version": "1",
        "parameters": {"basis_points": "0"},
    }
    json.dumps(first.to_primitive(), allow_nan=False, sort_keys=True)
    arithmetic = cast(PrimitiveMapping, first.to_primitive()["arithmetic"])
    assert arithmetic["decimal_precision"] == 34
    assert arithmetic["rounding"] == "ROUND_HALF_EVEN"


def test_commissions_and_adverse_slippage_are_separate_and_deterministic() -> None:
    slippage = BasisPointSlippage(Decimal("25"))
    buy = slippage.apply(Decimal("100"), OrderSide.BUY)
    sell = slippage.apply(Decimal("100"), OrderSide.SELL)

    assert buy == Decimal("100.25")
    assert sell == Decimal("99.75")
    assert PerShareCommission(Decimal("0.01"), Decimal("1")).calculate(
        10, buy
    ) == Decimal(1)
    assert BasisPointCommission(Decimal("10")).calculate(10, buy) == Decimal("1.0025")
    assert ExplicitZeroFees().calculate(OrderSide.SELL, 10, sell) == Decimal(0)
    assert BasisPointFees(Decimal("2")).calculate(OrderSide.SELL, 10, sell) == Decimal(
        "0.1995"
    )
