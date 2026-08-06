from datetime import date
from decimal import Decimal, localcontext

import pytest

from quantforge.data import MarketDataset
from quantforge.prediction import (
    ALL_REASONS,
    DEFAULT_RSI_THRESHOLDS,
    ComparisonPrediction,
    FeatureRangeBin,
    InvalidPredictionConfigurationError,
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    PredictionComparisonParameters,
    PredictionDirection,
    RsiOversoldUpParameters,
    StudyPeriod,
    run_prediction_analysis,
    run_prediction_comparison,
)

from ..helpers import make_dataset


def _comparison_dataset(*, bearish_candles: bool = False) -> MarketDataset:
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
    offset = 1 if bearish_candles else -1
    return make_dataset(
        closes,
        opens=tuple(str(int(value) + offset) for value in closes),
        highs=tuple(str(int(value) + 2) for value in closes),
        lows=tuple(str(int(value) - 2) for value in closes),
    )


def test_comparison_preserves_original_and_uses_stable_configuration_names() -> None:
    dataset = _comparison_dataset()
    baseline = run_prediction_analysis(
        dataset,
        OvernightGapPredictionStrategy(OvernightGapPredictionParameters()),
    )

    result = run_prediction_comparison(dataset)
    combined = tuple(
        item.prediction
        for item in result.predictions
        if item.configuration_name == "combined_original"
    )

    assert tuple(
        item.configuration_name for item in result.configuration_summaries
    ) == (
        "combined_original",
        "focused_rules",
        "rsi_oversold_up",
        "always_up",
    )
    assert combined == baseline.rows
    assert result.strategy_configurations[0].strategy_id == "overnight_gap_direction"
    assert tuple(item.threshold for item in result.threshold_summaries[:1]) == (
        Decimal(5),
    )


def test_all_configurations_share_the_same_outcome_label_for_each_session() -> None:
    result = run_prediction_comparison(_comparison_dataset())
    by_session: dict[date, set[tuple[date, Decimal, Decimal]]] = {}
    for item in result.predictions:
        by_session.setdefault(item.prediction.signal_session, set()).add(
            (
                item.prediction.outcome_session,
                item.prediction.next_open,
                item.prediction.overnight_gap_percentage,
            )
        )

    assert by_session
    assert all(len(labels) == 1 for labels in by_session.values())


def test_matched_baseline_is_honest_for_up_and_down_predictions() -> None:
    up_result = run_prediction_comparison(_comparison_dataset())
    up_rows = tuple(
        item
        for item in up_result.predictions
        if item.configuration_name == "combined_original"
        and item.prediction.direction is PredictionDirection.UP
    )
    assert up_rows
    assert all(item.incremental_signed_return == 0 for item in up_rows)
    assert all(item.baseline_correct == item.prediction.correct for item in up_rows)

    down_result = run_prediction_comparison(_comparison_dataset(bearish_candles=True))
    down_rows = tuple(
        item
        for item in down_result.predictions
        if item.configuration_name == "combined_original"
        and item.prediction.direction is PredictionDirection.DOWN
    )
    assert down_rows
    for item in down_rows:
        assert (
            item.prediction.signed_prediction_return
            == item.baseline_signed_return.copy_negate()
        )
        with localcontext() as context:
            context.prec = 34
            assert item.incremental_signed_return == (
                item.prediction.signed_prediction_return - item.baseline_signed_return
            )
        assert item.incremental_correctness == (
            int(item.prediction.correct) - int(item.baseline_correct)
        )


def test_threshold_sensitivity_is_exact_ordered_monotonic_and_nonmutating() -> None:
    parameters = PredictionComparisonParameters()
    result = run_prediction_comparison(_comparison_dataset(), parameters)
    full = tuple(
        item for item in result.threshold_summaries if item.segment_type == "full"
    )

    assert tuple(item.threshold for item in full) == DEFAULT_RSI_THRESHOLDS
    assert tuple(item.metrics.prediction_count for item in full) == tuple(
        sorted(item.metrics.prediction_count for item in full)
    )
    assert RsiOversoldUpParameters().lower_rsi == Decimal(15)
    assert {item.segment_name for item in result.threshold_summaries} >= {
        "development",
        "validation",
        "observed_2025",
    }


def test_period_and_weekday_summaries_are_explicit_and_nonoverlapping() -> None:
    result = run_prediction_comparison(_comparison_dataset())
    validation = tuple(
        item
        for item in result.period_summaries
        if item.configuration_name == "always_up"
        and item.reason == ALL_REASONS
        and item.period_name == "validation"
    )

    assert len(validation) == 1
    assert validation[0].metrics.prediction_count == result.eligible_session_count
    assert {item.weekday for item in result.weekday_summaries} == {0, 1, 2, 3}
    assert all(
        item.prediction.signal_session.weekday() != 4 for item in result.predictions
    )
    with pytest.raises(InvalidPredictionConfigurationError, match="must not overlap"):
        PredictionComparisonParameters(
            periods=(
                StudyPeriod("first", date(2024, 1, 1), date(2024, 7, 1), "exploratory"),
                StudyPeriod(
                    "second", date(2024, 7, 1), date(2024, 12, 31), "exploratory"
                ),
            )
        )


def test_feature_bins_have_half_open_boundaries_and_sample_flags() -> None:
    lower = FeatureRangeBin("lower", Decimal(0), Decimal(10))
    upper = FeatureRangeBin("upper", Decimal(10), Decimal(20), True)

    assert lower.contains(Decimal("9.999"))
    assert not lower.contains(Decimal(10))
    assert upper.contains(Decimal(10))
    assert upper.contains(Decimal(20))
    assert not upper.contains(Decimal("20.001"))
    with pytest.raises(InvalidPredictionConfigurationError, match="contiguous"):
        PredictionComparisonParameters(
            rsi_bins=(lower, FeatureRangeBin("gap", Decimal(11), None))
        )

    result = run_prediction_comparison(
        _comparison_dataset(), PredictionComparisonParameters(minimum_sample_size=999)
    )
    assert result.feature_bin_summaries
    assert any(
        item.metrics.prediction_count == 0 for item in result.feature_bin_summaries
    )
    assert all(not item.adequate_sample for item in result.feature_bin_summaries)


def test_outlier_ordering_is_stable_and_prediction_streaks_are_reported() -> None:
    result = run_prediction_comparison(_comparison_dataset())
    for configuration_name in (
        "combined_original",
        "focused_rules",
        "rsi_oversold_up",
        "always_up",
    ):
        best = tuple(
            item
            for item in result.best_outcomes
            if item.configuration_name == configuration_name
        )
        worst = tuple(
            item
            for item in result.worst_outcomes
            if item.configuration_name == configuration_name
        )
        assert best == tuple(
            sorted(
                best,
                key=lambda item: (
                    -item.prediction.signed_prediction_return,
                    item.prediction.signal_session,
                    item.prediction.prediction_id,
                ),
            )
        )
        assert worst == tuple(
            sorted(
                worst,
                key=lambda item: (
                    item.prediction.signed_prediction_return,
                    item.prediction.signal_session,
                    item.prediction.prediction_id,
                ),
            )
        )
    assert all(
        summary.streaks.statistic_label == "prediction_sequence_not_portfolio_drawdown"
        for summary in result.configuration_summaries
    )


def test_future_rows_do_not_change_prior_predictions_or_features() -> None:
    complete = _comparison_dataset()
    cutoff = make_dataset(
        tuple(str(bar.close) for bar in complete.bars[:13]),
        opens=tuple(str(bar.open) for bar in complete.bars[:13]),
        highs=tuple(str(bar.high) for bar in complete.bars[:13]),
        lows=tuple(str(bar.low) for bar in complete.bars[:13]),
    )

    cutoff_result = run_prediction_comparison(cutoff)
    complete_result = run_prediction_comparison(complete)
    cutoff_end = cutoff.bars[-1].session_date

    def causal_values(item: ComparisonPrediction) -> tuple[object, ...]:
        return (
            item.configuration_name,
            item.prediction.signal_session,
            item.prediction.outcome_session,
            item.prediction.direction,
            item.prediction.reason,
            item.prediction.overnight_gap_percentage,
            item.features,
        )

    expected = tuple(
        causal_values(item)
        for item in complete_result.predictions
        if item.prediction.signal_session < cutoff_end
    )
    actual = tuple(
        causal_values(item)
        for item in cutoff_result.predictions
        if item.prediction.signal_session < cutoff_end
    )

    assert actual == expected


def test_comparison_is_independent_of_ambient_decimal_precision() -> None:
    dataset = _comparison_dataset(bearish_candles=True)
    expected = run_prediction_comparison(dataset)

    with localcontext() as context:
        context.prec = 6
        actual = run_prediction_comparison(dataset)

    assert actual == expected
