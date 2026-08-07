from decimal import Decimal, localcontext

from quantforge.data import MarketDataset
from quantforge.prediction import (
    DerivedFeatureParameters,
    derive_completed_session_features,
)

from ..helpers import make_dataset


def _feature_dataset() -> MarketDataset:
    closes = (
        "100",
        "102",
        "101",
        "103",
        "99",
        "101",
        "100",
        "102",
        "98",
        "100",
        "99",
        "101",
        "100",
        "102",
        "101",
    )
    return make_dataset(
        closes,
        opens=tuple(str(int(value) - 1) for value in closes),
        highs=tuple(str(int(value) + 1) for value in closes),
        lows=tuple(str(int(value) - 1) for value in closes),
    )


def test_typed_derived_feature_definitions_are_aligned() -> None:
    dataset = _feature_dataset()
    parameters = DerivedFeatureParameters(atr_period=2, average_volume_period=3)

    rows = derive_completed_session_features(dataset, parameters)
    atr_row = rows[2]
    mature_row = rows[-1]

    assert atr_row.atr == Decimal("2.5")
    with localcontext() as context:
        context.prec = 34
        assert atr_row.atr_percentage_of_close == Decimal("2.5") / Decimal(101)
    assert atr_row.average_volume == Decimal(1000)
    assert atr_row.volume_ratio == Decimal(1)
    assert atr_row.candle_return == Decimal(101) / Decimal(100) - Decimal(1)
    assert mature_row.rsi is not None
    assert mature_row.previous_rsi is not None
    assert mature_row.adx is not None
    assert mature_row.previous_adx is not None
    assert mature_row.positive_di is not None
    assert mature_row.negative_di is not None
    with localcontext() as context:
        context.prec = 34
        assert mature_row.rsi_change == mature_row.rsi - mature_row.previous_rsi
        assert mature_row.adx_change == mature_row.adx - mature_row.previous_adx
        assert mature_row.di_spread == mature_row.positive_di - mature_row.negative_di
    assert mature_row.previous_di_spread is not None
    assert mature_row.signal_weekday == mature_row.signal_session.weekday()


def test_feature_derivation_does_not_mutate_input_or_read_appended_future() -> None:
    complete = _feature_dataset()
    cutoff = make_dataset(
        tuple(str(bar.close) for bar in complete.bars[:12]),
        opens=tuple(str(bar.open) for bar in complete.bars[:12]),
        highs=tuple(str(bar.high) for bar in complete.bars[:12]),
        lows=tuple(str(bar.low) for bar in complete.bars[:12]),
    )
    original_bars = complete.bars
    parameters = DerivedFeatureParameters(atr_period=2, average_volume_period=3)

    cutoff_rows = derive_completed_session_features(cutoff, parameters)
    complete_rows = derive_completed_session_features(complete, parameters)

    assert complete.bars == original_bars
    assert complete_rows[: len(cutoff_rows)] == cutoff_rows
