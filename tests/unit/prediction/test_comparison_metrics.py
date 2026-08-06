from decimal import Decimal

import pytest

from quantforge.prediction import InvalidPredictionOutputError, wilson_interval


def test_wilson_interval_matches_hand_checked_fixture() -> None:
    interval = wilson_interval(56, 100)

    assert interval.method == "wilson_score"
    assert interval.confidence_level == Decimal("0.95")
    assert interval.sample_count == 100
    assert interval.lower_bound is not None
    assert interval.upper_bound is not None
    assert interval.lower_bound.quantize(Decimal("0.000001")) == Decimal("0.462281")
    assert interval.upper_bound.quantize(Decimal("0.000001")) == Decimal("0.653280")


def test_wilson_interval_handles_zero_all_correct_and_all_incorrect() -> None:
    empty = wilson_interval(0, 0)
    incorrect = wilson_interval(0, 10)
    correct = wilson_interval(10, 10)

    assert empty.lower_bound is None
    assert empty.upper_bound is None
    assert incorrect.lower_bound == 0
    assert incorrect.upper_bound is not None
    assert incorrect.upper_bound < Decimal("0.28")
    assert correct.lower_bound is not None
    assert correct.lower_bound > Decimal("0.72")
    assert correct.upper_bound == 1


@pytest.mark.parametrize(("correct", "sample"), [(-1, 1), (2, 1), (0, -1)])
def test_wilson_interval_rejects_invalid_counts(correct: int, sample: int) -> None:
    with pytest.raises(InvalidPredictionOutputError, match="invalid Wilson"):
        wilson_interval(correct, sample)
