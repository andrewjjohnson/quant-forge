from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from quantforge.prediction import (
    AtrPercentageContext,
    OvernightGapPredictionParameters,
    OvernightGapSignalFeatureRule,
    PredictionStudy,
    SignalFeatureCandidate,
    SignalFeatureDatasetError,
    SignalFeatureRow,
    TrendDistanceContext,
    VolumeRatioContext,
    build_signal_feature_dataset,
    forward_return_outcome,
    target_stop_outcome,
)
from quantforge.prediction.feature_analysis import (
    FeatureAnalysisBin,
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


def test_value_equals_compares_decimal_outcomes_numerically(tmp_path: Path) -> None:
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
        contextual_features=(AtrPercentageContext(2),),
        outcomes=(primary,),
        output_root=tmp_path,
    )
    outcome_name = "outcome_forward_return_1_raw_return"
    row_values = feature_dataset.rows[0].to_primitive()
    row_values[outcome_name] = "1.0"
    one_row_dataset = replace(
        feature_dataset,
        rows=(SignalFeatureRow.capture(row_values),),
    )
    feature_name = "feature_atr_percentage_of_close"
    bins = {feature_name: (FeatureAnalysisBin("all", None, None),)}

    integer_scale = analyze_signal_features(
        one_row_dataset,
        feature_names=(feature_name,),
        outcome_name=outcome_name,
        winner_definition=WinnerDefinition.VALUE_EQUALS,
        winner_value="1",
        bins=bins,
    )
    fractional_scale = analyze_signal_features(
        one_row_dataset,
        feature_names=(feature_name,),
        outcome_name=outcome_name,
        winner_definition=WinnerDefinition.VALUE_EQUALS,
        winner_value="1.00",
        bins=bins,
    )

    assert integer_scale.winner_count == fractional_scale.winner_count == 1
    assert integer_scale.analysis_id == fractional_scale.analysis_id


def test_value_equals_parses_boolean_winner_values_from_schema(tmp_path: Path) -> None:
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
        contextual_features=(AtrPercentageContext(2),),
        outcomes=(primary,),
        output_root=tmp_path,
    )
    outcome_name = "outcome_forward_return_1_raw_return"
    boolean_schema = replace(
        feature_dataset.schema,
        fields=tuple(
            replace(field, data_type="boolean", unit="flag")
            if field.name == outcome_name
            else field
            for field in feature_dataset.schema.fields
        ),
    )
    row_values = feature_dataset.rows[0].to_primitive()
    row_values[outcome_name] = True
    boolean_dataset = replace(
        feature_dataset,
        schema=boolean_schema,
        rows=(SignalFeatureRow.capture(row_values),),
    )
    feature_name = "feature_atr_percentage_of_close"
    bins = {feature_name: (FeatureAnalysisBin("all", None, None),)}

    true_analysis = analyze_signal_features(
        boolean_dataset,
        feature_names=(feature_name,),
        outcome_name=outcome_name,
        winner_definition=WinnerDefinition.VALUE_EQUALS,
        winner_value="true",
        bins=bins,
    )
    false_analysis = analyze_signal_features(
        boolean_dataset,
        feature_names=(feature_name,),
        outcome_name=outcome_name,
        winner_definition=WinnerDefinition.VALUE_EQUALS,
        winner_value="false",
        bins=bins,
    )

    assert true_analysis.winner_count == 1
    assert false_analysis.loser_count == 1
    with pytest.raises(SignalFeatureDatasetError, match="true or false"):
        analyze_signal_features(
            boolean_dataset,
            feature_names=(feature_name,),
            outcome_name=outcome_name,
            winner_definition=WinnerDefinition.VALUE_EQUALS,
            winner_value="True",
            bins=bins,
        )


@pytest.mark.parametrize(
    ("data_type", "row_value", "winner_value", "normalized", "invalid", "error"),
    [
        ("integer", 1, "01", "1", "not-an-integer", "integer winner_value"),
        (
            "date",
            "2026-08-08",
            "2026-08-08",
            "2026-08-08",
            "not-a-date",
            "canonical ISO date",
        ),
    ],
)
def test_value_equals_validates_integer_and_date_winner_values(
    tmp_path: Path,
    data_type: str,
    row_value: int | str,
    winner_value: str,
    normalized: str,
    invalid: str,
    error: str,
) -> None:
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
        contextual_features=(AtrPercentageContext(2),),
        outcomes=(primary,),
        output_root=tmp_path,
    )
    outcome_name = "outcome_forward_return_1_raw_return"
    typed_schema = replace(
        feature_dataset.schema,
        fields=tuple(
            replace(field, data_type=data_type, unit="fixture_unit")
            if field.name == outcome_name
            else field
            for field in feature_dataset.schema.fields
        ),
    )
    row_values = feature_dataset.rows[0].to_primitive()
    row_values[outcome_name] = row_value
    typed_dataset = replace(
        feature_dataset,
        schema=typed_schema,
        rows=(SignalFeatureRow.capture(row_values),),
    )
    feature_name = "feature_atr_percentage_of_close"
    bins = {feature_name: (FeatureAnalysisBin("all", None, None),)}

    analysis = analyze_signal_features(
        typed_dataset,
        feature_names=(feature_name,),
        outcome_name=outcome_name,
        winner_definition=WinnerDefinition.VALUE_EQUALS,
        winner_value=winner_value,
        bins=bins,
    )

    assert analysis.winner_count == 1
    assert analysis.configuration["winner_value"] == normalized
    with pytest.raises(SignalFeatureDatasetError, match=error):
        analyze_signal_features(
            typed_dataset,
            feature_names=(feature_name,),
            outcome_name=outcome_name,
            winner_definition=WinnerDefinition.VALUE_EQUALS,
            winner_value=invalid,
            bins=bins,
        )


def test_analysis_excludes_rows_with_an_unavailable_outcome(tmp_path: Path) -> None:
    dataset = make_dataset(tuple(str(100 + index % 3) for index in range(15)))
    rule = OvernightGapSignalFeatureRule(
        OvernightGapPredictionParameters(excluded_weekdays=())
    )
    primary = forward_return_outcome(1)
    target_stop = target_stop_outcome(2, Decimal("0.01"), Decimal("0.005"))
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    feature_dataset = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(AtrPercentageContext(2),),
        outcomes=(primary, target_stop),
        output_root=tmp_path,
    )
    feature_name = "feature_atr_percentage_of_close"

    analysis = analyze_signal_features(
        feature_dataset,
        feature_names=(feature_name,),
        outcome_name="outcome_target_stop_2_label",
        winner_definition=WinnerDefinition.VALUE_EQUALS,
        winner_value="target_first",
        bins={feature_name: (FeatureAnalysisBin("all", None, None),)},
    )
    available_rows = sum(
        row.to_primitive()["outcome_target_stop_2_available"] is True
        for row in feature_dataset.rows
    )

    assert available_rows < len(feature_dataset.rows)
    assert analysis.eligible_row_count == available_rows
    assert analysis.winner_count + analysis.loser_count == available_rows


def test_analysis_rejects_an_availability_flag_as_the_outcome(tmp_path: Path) -> None:
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
        contextual_features=(AtrPercentageContext(2),),
        outcomes=(primary,),
        output_root=tmp_path,
    )
    feature_name = "feature_atr_percentage_of_close"

    with pytest.raises(SignalFeatureDatasetError, match="availability fields"):
        analyze_signal_features(
            feature_dataset,
            feature_names=(feature_name,),
            outcome_name="outcome_forward_return_1_available",
            winner_definition=WinnerDefinition.VALUE_EQUALS,
            winner_value="False",
            bins={feature_name: (FeatureAnalysisBin("all", None, None),)},
        )


def test_analysis_enforces_feature_outcome_schema_categories(tmp_path: Path) -> None:
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
        contextual_features=(AtrPercentageContext(2),),
        outcomes=(primary,),
        output_root=tmp_path,
    )
    feature_name = "feature_atr_percentage_of_close"
    outcome_name = "outcome_forward_return_1_raw_return"
    all_values = (FeatureAnalysisBin("all", None, None),)

    with pytest.raises(SignalFeatureDatasetError, match="contemporaneous-feature"):
        analyze_signal_features(
            feature_dataset,
            feature_names=(outcome_name,),
            outcome_name=outcome_name,
            winner_definition=WinnerDefinition.DECIMAL_GREATER_THAN_ZERO,
            bins={outcome_name: all_values},
        )

    with pytest.raises(SignalFeatureDatasetError, match="future-outcome"):
        analyze_signal_features(
            feature_dataset,
            feature_names=(feature_name,),
            outcome_name=feature_name,
            winner_definition=WinnerDefinition.DECIMAL_GREATER_THAN_ZERO,
            bins={feature_name: all_values},
        )


def test_analysis_rejects_nonnumeric_feature_schema_types(tmp_path: Path) -> None:
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
        contextual_features=(AtrPercentageContext(2),),
        outcomes=(primary,),
        output_root=tmp_path,
    )
    feature_name = "feature_atr_percentage_of_close"
    nonnumeric_schema = replace(
        feature_dataset.schema,
        fields=tuple(
            replace(field, data_type="string") if field.name == feature_name else field
            for field in feature_dataset.schema.fields
        ),
    )

    with pytest.raises(SignalFeatureDatasetError, match="features must use numeric"):
        analyze_signal_features(
            replace(feature_dataset, schema=nonnumeric_schema),
            feature_names=(feature_name,),
            outcome_name="outcome_forward_return_1_raw_return",
            winner_definition=WinnerDefinition.DECIMAL_GREATER_THAN_ZERO,
            bins={feature_name: (FeatureAnalysisBin("all", None, None),)},
        )


def test_analysis_rejects_decimal_rule_for_nonnumeric_outcome(tmp_path: Path) -> None:
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
        contextual_features=(AtrPercentageContext(2),),
        outcomes=(primary,),
        output_root=tmp_path,
    )
    feature_name = "feature_atr_percentage_of_close"

    with pytest.raises(SignalFeatureDatasetError, match="requires a numeric outcome"):
        analyze_signal_features(
            feature_dataset,
            feature_names=(feature_name,),
            outcome_name="outcome_forward_return_1_outcome_session",
            winner_definition=WinnerDefinition.DECIMAL_GREATER_THAN_ZERO,
            bins={feature_name: (FeatureAnalysisBin("all", None, None),)},
        )


def test_analysis_rejects_value_equals_for_structured_outcome(tmp_path: Path) -> None:
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
        contextual_features=(AtrPercentageContext(2),),
        outcomes=(primary,),
        output_root=tmp_path,
    )
    feature_name = "feature_atr_percentage_of_close"
    outcome_name = "outcome_forward_return_1_raw_return"
    structured_schema = replace(
        feature_dataset.schema,
        fields=tuple(
            replace(field, data_type="object") if field.name == outcome_name else field
            for field in feature_dataset.schema.fields
        ),
    )

    with pytest.raises(SignalFeatureDatasetError, match="requires a scalar outcome"):
        analyze_signal_features(
            replace(feature_dataset, schema=structured_schema),
            feature_names=(feature_name,),
            outcome_name=outcome_name,
            winner_definition=WinnerDefinition.VALUE_EQUALS,
            winner_value="ignored",
            bins={feature_name: (FeatureAnalysisBin("all", None, None),)},
        )
