from decimal import Decimal
from pathlib import Path

from quantforge.prediction import (
    AtrPercentageContext,
    OvernightGapPredictionParameters,
    OvernightGapSignalFeatureRule,
    PredictionStudy,
    SignalFeatureCandidate,
    TrendDistanceContext,
    VolumeRatioContext,
    build_signal_feature_dataset,
    forward_return_outcome,
)
from quantforge.prediction.feature_analysis import (
    WinnerDefinition,
    analyze_signal_features,
    default_overnight_gap_feature_bins,
    export_signal_feature_analysis,
)
from quantforge.prediction.feature_outcomes import ForwardReturnValues

from ..helpers import make_dataset

FEATURE_NAMES = (
    "feature_atr_percentage_of_close",
    "feature_trend_distance_percentage",
    "feature_volume_ratio",
)


def test_analysis_compares_three_features_and_retains_all_bins(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tuple(str(100 + (index % 5) - 2) for index in range(15)))
    rule = OvernightGapSignalFeatureRule(
        OvernightGapPredictionParameters(excluded_weekdays=(4,))
    )
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    feature_dataset = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(
            AtrPercentageContext(2),
            TrendDistanceContext(2),
            VolumeRatioContext(2),
        ),
        outcomes=(primary,),
        output_root=tmp_path / "features",
    )

    analysis = analyze_signal_features(
        feature_dataset,
        feature_names=FEATURE_NAMES,
        outcome_name="outcome_forward_return_1_raw_return",
        winner_definition=WinnerDefinition.DECIMAL_GREATER_THAN_ZERO,
        bins=default_overnight_gap_feature_bins(),
    )
    destination = export_signal_feature_analysis(analysis, tmp_path / "analysis")

    assert analysis.eligible_row_count == len(feature_dataset.rows) - 1
    assert analysis.winner_count + analysis.loser_count == analysis.eligible_row_count
    assert len(analysis.group_summaries) == 6
    assert len(analysis.bin_summaries) == 9
    assert {item.feature_name for item in analysis.group_summaries} == set(
        FEATURE_NAMES
    )
    assert all(
        item.sample_count == item.winner_count + item.loser_count
        for item in analysis.bin_summaries
    )
    assert destination.is_file()
    assert (
        export_signal_feature_analysis(analysis, tmp_path / "analysis") == destination
    )
    assert "exploratory" in str(analysis.to_primitive()["warning"])


def test_analysis_winner_definition_is_configurable(tmp_path: Path) -> None:
    dataset = make_dataset(tuple(str(100 + index % 3) for index in range(15)))
    rule = OvernightGapSignalFeatureRule(
        OvernightGapPredictionParameters(excluded_weekdays=())
    )
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    feature_dataset = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(
            AtrPercentageContext(2),
            TrendDistanceContext(2),
            VolumeRatioContext(2),
        ),
        outcomes=(primary,),
        output_root=tmp_path,
    )

    greater_than_zero = analyze_signal_features(
        feature_dataset,
        feature_names=FEATURE_NAMES,
        outcome_name="outcome_forward_return_1_raw_return",
        winner_definition=WinnerDefinition.DECIMAL_GREATER_THAN_ZERO,
        bins=default_overnight_gap_feature_bins(),
    )
    equals_zero = analyze_signal_features(
        feature_dataset,
        feature_names=FEATURE_NAMES,
        outcome_name="outcome_forward_return_1_raw_return",
        winner_definition=WinnerDefinition.VALUE_EQUALS,
        winner_value=str(Decimal(0)),
        bins=default_overnight_gap_feature_bins(),
    )

    assert greater_than_zero.analysis_id != equals_zero.analysis_id
    assert greater_than_zero.winner_count != equals_zero.winner_count
