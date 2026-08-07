"""Transparent deterministic statistics for prediction comparison studies."""

from dataclasses import dataclass
from decimal import Decimal, DecimalException

from quantforge.prediction._arithmetic import arithmetic
from quantforge.prediction.comparison_models import (
    AccuracyInterval,
    ComparisonPrediction,
    MetricSummary,
    StreakStatistics,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.models import PredictionDirection

DEFAULT_CONFIDENCE_LEVEL = Decimal("0.95")
WILSON_Z_95 = Decimal("1.95996398454005423552")
POSITIVE_TAIL_THRESHOLDS = (
    Decimal("0.001"),
    Decimal("0.0025"),
    Decimal("0.005"),
    Decimal("0.01"),
)


@dataclass(frozen=True, slots=True)
class MetricObservation:
    direction: PredictionDirection
    raw_return: Decimal
    signed_return: Decimal
    absolute_gap: Decimal
    correct: bool


def strategy_observations(
    predictions: tuple[ComparisonPrediction, ...],
) -> tuple[MetricObservation, ...]:
    return tuple(
        MetricObservation(
            direction=item.prediction.direction,
            raw_return=item.prediction.overnight_gap_percentage,
            signed_return=item.prediction.signed_prediction_return,
            absolute_gap=item.prediction.gap_size_percentage,
            correct=item.prediction.correct,
        )
        for item in predictions
    )


def baseline_observations(
    predictions: tuple[ComparisonPrediction, ...],
) -> tuple[MetricObservation, ...]:
    return tuple(
        MetricObservation(
            direction=PredictionDirection.UP,
            raw_return=item.prediction.overnight_gap_percentage,
            signed_return=item.baseline_signed_return,
            absolute_gap=item.prediction.gap_size_percentage,
            correct=item.baseline_correct,
        )
        for item in predictions
    )


def summarize_predictions(
    predictions: tuple[ComparisonPrediction, ...],
    *,
    confidence_level: Decimal = DEFAULT_CONFIDENCE_LEVEL,
) -> MetricSummary:
    return summarize_observations(
        strategy_observations(predictions), confidence_level=confidence_level
    )


def summarize_observations(
    observations: tuple[MetricObservation, ...],
    *,
    confidence_level: Decimal = DEFAULT_CONFIDENCE_LEVEL,
) -> MetricSummary:
    count = len(observations)
    correct = tuple(item for item in observations if item.correct)
    incorrect = tuple(item for item in observations if not item.correct)
    raw_returns = tuple(item.raw_return for item in observations)
    signed_returns = tuple(item.signed_return for item in observations)
    absolute_gaps = tuple(item.absolute_gap for item in observations)
    correct_returns = tuple(item.signed_return for item in correct)
    incorrect_returns = tuple(item.signed_return for item in incorrect)
    try:
        with arithmetic():
            accuracy = _ratio(len(correct), count)
            return MetricSummary(
                prediction_count=count,
                up_prediction_count=sum(
                    item.direction is PredictionDirection.UP for item in observations
                ),
                down_prediction_count=sum(
                    item.direction is PredictionDirection.DOWN for item in observations
                ),
                correct_count=len(correct),
                incorrect_count=len(incorrect),
                neutral_outcome_count=sum(
                    item.raw_return == 0 for item in observations
                ),
                accuracy=accuracy,
                accuracy_interval=wilson_interval(
                    len(correct), count, confidence_level=confidence_level
                ),
                average_raw_overnight_return=_average(raw_returns),
                median_raw_overnight_return=_median(raw_returns),
                average_signed_prediction_return=_average(signed_returns),
                median_signed_prediction_return=_median(signed_returns),
                signed_prediction_return_standard_deviation=(
                    _population_standard_deviation(signed_returns)
                ),
                average_absolute_gap=_average(absolute_gaps),
                best_signed_prediction_return=(
                    None if not signed_returns else max(signed_returns)
                ),
                worst_signed_prediction_return=(
                    None if not signed_returns else min(signed_returns)
                ),
                probability_signed_greater_than_0_10_percent=_tail_probability(
                    signed_returns, POSITIVE_TAIL_THRESHOLDS[0], greater=True
                ),
                probability_signed_greater_than_0_25_percent=_tail_probability(
                    signed_returns, POSITIVE_TAIL_THRESHOLDS[1], greater=True
                ),
                probability_signed_greater_than_0_50_percent=_tail_probability(
                    signed_returns, POSITIVE_TAIL_THRESHOLDS[2], greater=True
                ),
                probability_signed_greater_than_1_00_percent=_tail_probability(
                    signed_returns, POSITIVE_TAIL_THRESHOLDS[3], greater=True
                ),
                probability_signed_less_than_negative_0_10_percent=(
                    _tail_probability(
                        signed_returns,
                        -POSITIVE_TAIL_THRESHOLDS[0],
                        greater=False,
                    )
                ),
                probability_signed_less_than_negative_0_25_percent=(
                    _tail_probability(
                        signed_returns,
                        -POSITIVE_TAIL_THRESHOLDS[1],
                        greater=False,
                    )
                ),
                probability_signed_less_than_negative_0_50_percent=(
                    _tail_probability(
                        signed_returns,
                        -POSITIVE_TAIL_THRESHOLDS[2],
                        greater=False,
                    )
                ),
                probability_signed_less_than_negative_1_00_percent=(
                    _tail_probability(
                        signed_returns,
                        -POSITIVE_TAIL_THRESHOLDS[3],
                        greater=False,
                    )
                ),
                average_signed_return_correct=_average(correct_returns),
                average_signed_return_incorrect=_average(incorrect_returns),
                median_signed_return_correct=_median(correct_returns),
                median_signed_return_incorrect=_median(incorrect_returns),
            )
    except DecimalException as error:
        raise InvalidPredictionOutputError(
            "comparison metric arithmetic failed"
        ) from error


def wilson_interval(
    correct_count: int,
    sample_count: int,
    *,
    confidence_level: Decimal = DEFAULT_CONFIDENCE_LEVEL,
) -> AccuracyInterval:
    """Return the deterministic 95% Wilson score interval for a proportion."""
    if correct_count < 0 or sample_count < 0 or correct_count > sample_count:
        raise InvalidPredictionOutputError("invalid Wilson interval counts")
    if confidence_level != DEFAULT_CONFIDENCE_LEVEL:
        raise InvalidPredictionOutputError(
            "only the documented 95% Wilson confidence level is supported"
        )
    if sample_count == 0:
        return AccuracyInterval(confidence_level, None, None, 0)
    with arithmetic():
        count = Decimal(sample_count)
        proportion = Decimal(correct_count) / count
        z_squared = WILSON_Z_95 * WILSON_Z_95
        denominator = Decimal(1) + z_squared / count
        center = (proportion + z_squared / (Decimal(2) * count)) / denominator
        radius = (
            WILSON_Z_95
            * (
                (
                    proportion * (Decimal(1) - proportion) / count
                    + z_squared / (Decimal(4) * count * count)
                ).sqrt()
            )
            / denominator
        )
        lower = max(Decimal(0), center - radius)
        upper = min(Decimal(1), center + radius)
    return AccuracyInterval(confidence_level, lower, upper, sample_count)


def calculate_streak_statistics(
    predictions: tuple[ComparisonPrediction, ...],
) -> StreakStatistics:
    ordered = tuple(
        sorted(
            predictions,
            key=lambda item: (
                item.prediction.signal_session,
                item.prediction.prediction_id,
            ),
        )
    )
    longest_correct = 0
    longest_incorrect = 0
    current_correct = 0
    current_incorrect = 0
    incorrect_lengths: list[int] = []
    cumulative = Decimal(0)
    peak = Decimal(0)
    maximum_decline = Decimal(0)
    with arithmetic():
        for item in ordered:
            if item.prediction.correct:
                if current_incorrect:
                    incorrect_lengths.append(current_incorrect)
                current_incorrect = 0
                current_correct += 1
                longest_correct = max(longest_correct, current_correct)
            else:
                current_correct = 0
                current_incorrect += 1
                longest_incorrect = max(longest_incorrect, current_incorrect)
            cumulative += item.prediction.signed_prediction_return
            peak = max(peak, cumulative)
            maximum_decline = max(maximum_decline, peak - cumulative)
    if current_incorrect:
        incorrect_lengths.append(current_incorrect)
    return StreakStatistics(
        longest_incorrect_streak=longest_incorrect,
        longest_correct_streak=longest_correct,
        maximum_cumulative_signed_return_decline=maximum_decline,
        incorrect_streaks_at_least_3=sum(length >= 3 for length in incorrect_lengths),
        incorrect_streaks_at_least_5=sum(length >= 5 for length in incorrect_lengths),
    )


def average_incremental_return(
    predictions: tuple[ComparisonPrediction, ...],
) -> Decimal | None:
    return _average(tuple(item.incremental_signed_return for item in predictions))


def median_incremental_return(
    predictions: tuple[ComparisonPrediction, ...],
) -> Decimal | None:
    return _median(tuple(item.incremental_signed_return for item in predictions))


def _average(values: tuple[Decimal, ...]) -> Decimal | None:
    if not values:
        return None
    with arithmetic():
        return sum(values, Decimal(0)) / Decimal(len(values))


def _median(values: tuple[Decimal, ...]) -> Decimal | None:
    if not values:
        return None
    ordered = tuple(sorted(values))
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    with arithmetic():
        return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def _population_standard_deviation(values: tuple[Decimal, ...]) -> Decimal | None:
    if not values:
        return None
    mean = _average(values)
    if mean is None:
        return None
    with arithmetic():
        variance = sum(((value - mean) ** 2 for value in values), Decimal(0)) / Decimal(
            len(values)
        )
        return variance.sqrt()


def _tail_probability(
    values: tuple[Decimal, ...], threshold: Decimal, *, greater: bool
) -> Decimal | None:
    if not values:
        return None
    matches = sum(
        value > threshold if greater else value < threshold for value in values
    )
    return Decimal(matches) / Decimal(len(values))


def _ratio(numerator: int, denominator: int) -> Decimal | None:
    if denominator == 0:
        return None
    with arithmetic():
        return Decimal(numerator) / Decimal(denominator)
