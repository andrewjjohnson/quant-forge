from decimal import Decimal

import pytest

from quantforge.prediction import (
    FOCUSED_REASONS,
    AlwaysUpParameters,
    AlwaysUpPredictionStrategy,
    FocusedGapPredictionParameters,
    FocusedGapPredictionStrategy,
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    PredictionDirection,
    RsiOversoldUpParameters,
    RsiOversoldUpPredictionStrategy,
    predict_focused_gap_direction,
)

from ..helpers import make_dataset

BASE_VALUES = {
    "previous_rsi": Decimal(50),
    "current_rsi": Decimal(50),
    "previous_positive_di": Decimal(40),
    "current_positive_di": Decimal(40),
    "previous_negative_di": Decimal(20),
    "current_negative_di": Decimal(20),
    "previous_adx": Decimal(10),
    "current_adx": Decimal(10),
}


def test_original_strategy_identity_and_parameters_are_preserved() -> None:
    strategy = OvernightGapPredictionStrategy(OvernightGapPredictionParameters())

    assert strategy.name == "overnight_gap_direction"
    assert strategy.implementation_version == "1"
    assert strategy.configuration_id == (
        "5101fb7bf4c1197285e07c0919253dfdb26c78dece3de29f2b4aa9becac83a67"
    )
    assert strategy.parameters == OvernightGapPredictionParameters()


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        (
            {"previous_adx": Decimal(50), "current_adx": Decimal(30)},
            (PredictionDirection.UP, "adx_entered_di_zone_plus_di_on_top"),
        ),
        (
            {"previous_rsi": Decimal(70), "current_rsi": Decimal(30)},
            (PredictionDirection.UP, "rsi_stabbed_di_zone_from_above"),
        ),
        (
            {"previous_rsi": Decimal(10), "current_rsi": Decimal(30)},
            (PredictionDirection.DOWN, "rsi_stabbed_di_zone_from_below"),
        ),
        (
            {"previous_rsi": Decimal(30), "current_rsi": Decimal(14)},
            (PredictionDirection.UP, "rsi_below_lower_threshold"),
        ),
    ],
)
def test_focused_strategy_contains_exactly_the_retained_predicates(
    updates: dict[str, Decimal], expected: tuple[PredictionDirection, str]
) -> None:
    result = predict_focused_gap_direction(
        FocusedGapPredictionParameters(), **{**BASE_VALUES, **updates}
    )

    assert result is not None
    assert result == expected
    assert result[1] in FOCUSED_REASONS


@pytest.mark.parametrize(
    "updates",
    [
        {"current_rsi": Decimal(86)},
        {
            "previous_positive_di": Decimal(20),
            "current_positive_di": Decimal(20),
            "previous_negative_di": Decimal(40),
            "current_negative_di": Decimal(40),
            "previous_adx": Decimal(50),
            "current_adx": Decimal(30),
        },
        {"current_rsi": Decimal(50)},
    ],
)
def test_focused_strategy_excludes_broad_bearish_and_middle_rules(
    updates: dict[str, Decimal],
) -> None:
    assert (
        predict_focused_gap_direction(
            FocusedGapPredictionParameters(), **{**BASE_VALUES, **updates}
        )
        is None
    )


def test_generated_focused_signals_never_use_an_excluded_reason() -> None:
    values = (
        100,
        103,
        106,
        101,
        102,
        107,
        110,
        111,
        116,
        118,
        117,
        115,
        116,
        114,
        111,
    )
    closes = tuple(str(value) for value in values)
    dataset = make_dataset(
        closes,
        opens=tuple(str(value - 1) for value in values),
        highs=tuple(str(value + 2) for value in values),
        lows=tuple(str(value - 2) for value in values),
    )

    signals = (
        FocusedGapPredictionStrategy(FocusedGapPredictionParameters())
        .generate(dataset)
        .signals
    )

    assert signals
    assert {signal.reason for signal in signals} <= set(FOCUSED_REASONS)


def test_simple_rsi_strategy_is_strictly_below_and_up_only() -> None:
    rising = tuple(str(value) for value in range(100, 115))
    falling = tuple(str(value) for value in range(115, 100, -1))

    equality_output = RsiOversoldUpPredictionStrategy(
        RsiOversoldUpParameters(lower_rsi=Decimal(100))
    ).generate(make_dataset(rising))
    oversold_output = RsiOversoldUpPredictionStrategy(
        RsiOversoldUpParameters()
    ).generate(make_dataset(falling))

    assert not equality_output.signals
    assert oversold_output.signals
    assert all(
        signal.direction is PredictionDirection.UP
        and signal.reason == "rsi_below_lower_threshold"
        for signal in oversold_output.signals
    )


def test_always_up_predicts_every_eligible_session_and_supports_inclusion() -> None:
    dataset = make_dataset(tuple("100" for _ in range(15)))
    default_output = AlwaysUpPredictionStrategy(AlwaysUpParameters()).generate(dataset)
    monday_only = AlwaysUpPredictionStrategy(
        AlwaysUpParameters(excluded_weekdays=(), included_weekdays=(0,))
    ).generate(dataset)

    assert len(default_output.signals) == sum(
        bar.session_date.weekday() != 4 for bar in dataset.bars
    )
    assert all(
        signal.direction is PredictionDirection.UP for signal in default_output.signals
    )
    assert monday_only.signals
    assert all(signal.signal_session.weekday() == 0 for signal in monday_only.signals)
