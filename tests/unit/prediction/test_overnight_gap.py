from decimal import Decimal

import pytest

from quantforge.prediction import (
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    PredictionDirection,
    predict_overnight_gap_direction,
)

from ..helpers import make_dataset

PARAMETERS = OvernightGapPredictionParameters()
BASE_VALUES = {
    "previous_rsi": Decimal(50),
    "current_rsi": Decimal(50),
    "previous_positive_di": Decimal(40),
    "current_positive_di": Decimal(40),
    "previous_negative_di": Decimal(20),
    "current_negative_di": Decimal(20),
    "previous_adx": Decimal(50),
    "current_adx": Decimal(10),
    "session_open": Decimal(100),
    "session_close": Decimal(101),
}


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        (
            {"previous_adx": Decimal(50), "current_adx": Decimal(30)},
            (
                PredictionDirection.UP,
                "adx_entered_di_zone_plus_di_on_top",
            ),
        ),
        (
            {
                "previous_positive_di": Decimal(20),
                "current_positive_di": Decimal(20),
                "previous_negative_di": Decimal(40),
                "current_negative_di": Decimal(40),
                "previous_adx": Decimal(50),
                "current_adx": Decimal(30),
            },
            (
                PredictionDirection.DOWN,
                "adx_entered_di_zone_minus_di_on_top",
            ),
        ),
        (
            {
                "previous_rsi": Decimal(70),
                "current_rsi": Decimal(30),
                "previous_adx": Decimal(10),
            },
            (PredictionDirection.UP, "rsi_stabbed_di_zone_from_above"),
        ),
        (
            {
                "previous_rsi": Decimal(10),
                "current_rsi": Decimal(30),
                "previous_adx": Decimal(10),
            },
            (PredictionDirection.DOWN, "rsi_stabbed_di_zone_from_below"),
        ),
        (
            {
                "previous_rsi": Decimal(30),
                "current_rsi": Decimal(14),
                "previous_adx": Decimal(10),
            },
            (PredictionDirection.UP, "rsi_below_lower_threshold"),
        ),
        (
            {"current_rsi": Decimal(86), "previous_adx": Decimal(10)},
            (PredictionDirection.DOWN, "rsi_above_upper_threshold"),
        ),
        (
            {
                "previous_rsi": Decimal(30),
                "current_rsi": Decimal(15),
                "previous_adx": Decimal(10),
            },
            (PredictionDirection.UP, "bullish_candle_in_middle_rsi_range"),
        ),
        (
            {
                "current_rsi": Decimal(85),
                "previous_adx": Decimal(10),
                "session_open": Decimal(101),
                "session_close": Decimal(100),
            },
            (PredictionDirection.DOWN, "bearish_candle_in_middle_rsi_range"),
        ),
    ],
)
def test_rule_priority_and_boundary_semantics(
    updates: dict[str, Decimal],
    expected: tuple[PredictionDirection, str],
) -> None:
    values = {**BASE_VALUES, **updates}

    assert predict_overnight_gap_direction(PARAMETERS, **values) == expected


def test_adx_veto_and_middle_range_doji_produce_no_prediction() -> None:
    assert (
        predict_overnight_gap_direction(
            PARAMETERS, **{**BASE_VALUES, "current_adx": Decimal(61)}
        )
        is None
    )
    assert (
        predict_overnight_gap_direction(
            PARAMETERS,
            **{
                **BASE_VALUES,
                "previous_adx": Decimal(10),
                "session_open": Decimal(100),
                "session_close": Decimal(100),
            },
        )
        is None
    )


def test_default_strategy_excludes_friday_signal_sessions() -> None:
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
    opens = tuple(str(int(value) - 1) for value in closes)
    highs = tuple(str(int(value) + 1) for value in closes)
    lows = tuple(str(int(value) - 2) for value in closes)
    dataset = make_dataset(closes, opens=opens, highs=highs, lows=lows)

    output = OvernightGapPredictionStrategy(PARAMETERS).generate(dataset)

    assert output.signals
    assert all(signal.signal_session.weekday() != 4 for signal in output.signals)


def test_appended_future_bars_do_not_change_historical_predictions() -> None:
    strategy = OvernightGapPredictionStrategy(PARAMETERS)
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
    opens = tuple(str(int(value) - 1) for value in closes)
    highs = tuple(str(int(value) + 1) for value in closes)
    lows = tuple(str(int(value) - 2) for value in closes)
    through_cutoff = make_dataset(
        closes[:13],
        opens=opens[:13],
        highs=highs[:13],
        lows=lows[:13],
    )
    with_future = make_dataset(closes, opens=opens, highs=highs, lows=lows)

    cutoff_output = strategy.generate(through_cutoff)
    future_output = strategy.generate(with_future)
    comparable_future = tuple(
        signal
        for signal in future_output.signals
        if signal.signal_session <= through_cutoff.bars[-1].session_date
    )

    assert comparable_future == cutoff_output.signals
