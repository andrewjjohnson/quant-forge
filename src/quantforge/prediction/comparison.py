"""Reproducible comparison study for overnight-gap prediction configurations."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.models import MarketDataset
from quantforge.prediction._arithmetic import arithmetic
from quantforge.prediction.comparison_metrics import (
    DEFAULT_CONFIDENCE_LEVEL,
    average_incremental_return,
    baseline_observations,
    calculate_streak_statistics,
    median_incremental_return,
    summarize_observations,
    summarize_predictions,
)
from quantforge.prediction.comparison_models import (
    AnnualSummary,
    BaselineComparisonSummary,
    ComparisonPrediction,
    ConfigurationSummary,
    FeatureBinSummary,
    NamedStrategyConfiguration,
    PeriodSummary,
    PredictionComparisonStudyResult,
    RuleSummary,
    ThresholdSensitivitySummary,
    WeekdaySummary,
)
from quantforge.prediction.errors import (
    InvalidPredictionConfigurationError,
    InvalidPredictionOutputError,
)
from quantforge.prediction.experimental_strategies import (
    AlwaysUpParameters,
    AlwaysUpPredictionStrategy,
    FocusedGapPredictionParameters,
    FocusedGapPredictionStrategy,
    RsiOversoldUpParameters,
    RsiOversoldUpPredictionStrategy,
)
from quantforge.prediction.features import (
    DerivedFeatureParameters,
    DerivedFeatureRow,
    derive_completed_session_features,
)
from quantforge.prediction.models import (
    PredictionAnalysisResult,
    PredictionMarketData,
    PredictionRow,
)
from quantforge.prediction.overnight_gap import (
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
)
from quantforge.prediction.runner import run_prediction_analysis

COMPARISON_ENGINE_VERSION = "3"
COMPARISON_SCHEMA_VERSION = "3"
ALL_REASONS = "__all_reasons__"
DEFAULT_RSI_THRESHOLDS = tuple(Decimal(value) for value in (5, 10, 15, 20, 25, 30))
WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


@dataclass(frozen=True, slots=True)
class StudyPeriod:
    name: str
    start: date
    end: date
    exploratory_label: str

    def __post_init__(self) -> None:
        if not self.name or not self.exploratory_label or self.start > self.end:
            raise InvalidPredictionConfigurationError(
                "study periods require a name, label, and ordered boundaries"
            )

    def contains(self, session: date) -> bool:
        return self.start <= session <= self.end

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "end": self.end.isoformat(),
            "exploratory_label": self.exploratory_label,
            "name": self.name,
            "start": self.start.isoformat(),
        }


DEFAULT_PERIODS = (
    StudyPeriod(
        "development",
        date(2020, 1, 1),
        date(2022, 12, 31),
        "exploratory_development",
    ),
    StudyPeriod(
        "validation",
        date(2023, 1, 1),
        date(2024, 12, 31),
        "exploratory_validation",
    ),
    StudyPeriod(
        "observed_2025",
        date(2025, 1, 1),
        date(2025, 12, 31),
        "observed_not_pristine_holdout",
    ),
)


@dataclass(frozen=True, slots=True)
class FeatureRangeBin:
    label: str
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    include_upper_bound: bool = False

    def __post_init__(self) -> None:
        if not self.label:
            raise InvalidPredictionConfigurationError("feature bins require labels")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound >= self.upper_bound
        ):
            raise InvalidPredictionConfigurationError(
                "feature bin lower bounds must precede upper bounds"
            )
        if self.include_upper_bound and self.upper_bound is None:
            raise InvalidPredictionConfigurationError(
                "an unbounded feature bin cannot include its upper bound"
            )

    def contains(self, value: Decimal) -> bool:
        lower_matches = self.lower_bound is None or value >= self.lower_bound
        upper_matches = self.upper_bound is None or (
            value <= self.upper_bound
            if self.include_upper_bound
            else value < self.upper_bound
        )
        return lower_matches and upper_matches

    @property
    def interval_convention(self) -> str:
        left = "(" if self.lower_bound is None else "["
        right = "]" if self.include_upper_bound else ")"
        lower = "-inf" if self.lower_bound is None else str(self.lower_bound)
        upper = "+inf" if self.upper_bound is None else str(self.upper_bound)
        return f"{left}{lower}, {upper}{right}"

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "include_upper_bound": self.include_upper_bound,
            "interval_convention": self.interval_convention,
            "label": self.label,
            "lower_bound": _optional_decimal(self.lower_bound),
            "upper_bound": _optional_decimal(self.upper_bound),
        }


def _bins(
    values: tuple[tuple[str, str | None, str | None, bool], ...],
) -> tuple[FeatureRangeBin, ...]:
    return tuple(
        FeatureRangeBin(
            label,
            None if lower is None else Decimal(lower),
            None if upper is None else Decimal(upper),
            include_upper,
        )
        for label, lower, upper, include_upper in values
    )


DEFAULT_RSI_BINS = _bins(
    (
        ("0_to_5", "0", "5", False),
        ("5_to_10", "5", "10", False),
        ("10_to_15", "10", "15", False),
        ("15_to_20", "15", "20", False),
        ("20_to_30", "20", "30", False),
        ("30_to_40", "30", "40", False),
        ("40_to_50", "40", "50", False),
        ("50_to_60", "50", "60", False),
        ("60_to_70", "60", "70", False),
        ("70_to_80", "70", "80", False),
        ("80_to_90", "80", "90", False),
        ("90_to_100", "90", "100", True),
    )
)
DEFAULT_ADX_BINS = _bins(
    (
        ("0_to_10", "0", "10", False),
        ("10_to_20", "10", "20", False),
        ("20_to_30", "20", "30", False),
        ("30_to_40", "30", "40", False),
        ("40_to_50", "40", "50", False),
        ("50_to_60", "50", "60", False),
        ("60_plus", "60", None, False),
    )
)
DEFAULT_DI_SPREAD_BINS = _bins(
    (
        ("strongly_negative", None, "-20", False),
        ("moderately_negative", "-20", "-10", False),
        ("near_zero", "-10", "10", False),
        ("moderately_positive", "10", "20", False),
        ("strongly_positive", "20", None, False),
    )
)
DEFAULT_ATR_PERCENTAGE_BINS = _bins(
    (
        ("below_0_50_percent", "0", "0.005", False),
        ("0_50_to_1_00_percent", "0.005", "0.01", False),
        ("1_00_to_1_50_percent", "0.01", "0.015", False),
        ("1_50_to_2_00_percent", "0.015", "0.02", False),
        ("2_00_to_3_00_percent", "0.02", "0.03", False),
        ("3_00_percent_plus", "0.03", None, False),
    )
)
DEFAULT_VOLUME_RATIO_BINS = _bins(
    (
        ("below_0_75", "0", "0.75", False),
        ("0_75_to_1_00", "0.75", "1", False),
        ("1_00_to_1_25", "1", "1.25", False),
        ("1_25_to_1_50", "1.25", "1.5", False),
        ("1_50_to_2_00", "1.5", "2", False),
        ("2_00_plus", "2", None, False),
    )
)


@dataclass(frozen=True, slots=True)
class PredictionComparisonParameters:
    """Stable controls for reports; none may alter the original rule logic."""

    excluded_weekdays: tuple[int, ...] = (4,)
    included_weekdays: tuple[int, ...] | None = None
    rsi_thresholds: tuple[Decimal, ...] = DEFAULT_RSI_THRESHOLDS
    periods: tuple[StudyPeriod, ...] = DEFAULT_PERIODS
    derived_features: DerivedFeatureParameters = field(
        default_factory=DerivedFeatureParameters
    )
    minimum_sample_size: int = 30
    confidence_level: Decimal = DEFAULT_CONFIDENCE_LEVEL
    outlier_count: int = 20
    rsi_bins: tuple[FeatureRangeBin, ...] = DEFAULT_RSI_BINS
    adx_bins: tuple[FeatureRangeBin, ...] = DEFAULT_ADX_BINS
    di_spread_bins: tuple[FeatureRangeBin, ...] = DEFAULT_DI_SPREAD_BINS
    atr_percentage_bins: tuple[FeatureRangeBin, ...] = DEFAULT_ATR_PERCENTAGE_BINS
    volume_ratio_bins: tuple[FeatureRangeBin, ...] = DEFAULT_VOLUME_RATIO_BINS

    def __post_init__(self) -> None:
        excluded = _validated_weekdays("excluded_weekdays", self.excluded_weekdays)
        included = (
            None
            if self.included_weekdays is None
            else _validated_weekdays("included_weekdays", self.included_weekdays)
        )
        if included is not None and set(excluded) & set(included):
            raise InvalidPredictionConfigurationError(
                "included and excluded weekdays must not overlap"
            )
        thresholds = tuple(
            _decimal(value, "rsi threshold") for value in self.rsi_thresholds
        )
        if (
            not thresholds
            or thresholds != tuple(sorted(thresholds))
            or len(thresholds) != len(set(thresholds))
            or any(not Decimal(0) <= value <= Decimal(100) for value in thresholds)
        ):
            raise InvalidPredictionConfigurationError(
                "RSI thresholds must be sorted unique values from 0 through 100"
            )
        if isinstance(self.minimum_sample_size, bool) or self.minimum_sample_size < 1:
            raise InvalidPredictionConfigurationError(
                "minimum_sample_size must be positive"
            )
        if isinstance(self.outlier_count, bool) or self.outlier_count < 1:
            raise InvalidPredictionConfigurationError("outlier_count must be positive")
        if (
            _decimal(self.confidence_level, "confidence_level")
            != DEFAULT_CONFIDENCE_LEVEL
        ):
            raise InvalidPredictionConfigurationError(
                "the comparison currently supports the documented 95% confidence level"
            )
        _validate_periods(self.periods)
        for feature_name, bins in self.feature_bins:
            _validate_feature_bins(feature_name, bins)
        object.__setattr__(self, "excluded_weekdays", excluded)
        object.__setattr__(self, "included_weekdays", included)
        object.__setattr__(self, "rsi_thresholds", thresholds)
        object.__setattr__(self, "confidence_level", DEFAULT_CONFIDENCE_LEVEL)

    @property
    def feature_bins(self) -> tuple[tuple[str, tuple[FeatureRangeBin, ...]], ...]:
        return (
            ("rsi", self.rsi_bins),
            ("adx", self.adx_bins),
            ("di_spread", self.di_spread_bins),
            ("atr_percentage_of_close", self.atr_percentage_bins),
            ("volume_ratio", self.volume_ratio_bins),
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "confidence_interval": {
                "confidence_level": decimal_to_primitive(self.confidence_level),
                "method": "wilson_score",
            },
            "derived_features": self.derived_features.to_primitive(),
            "excluded_weekdays": list(self.excluded_weekdays),
            "feature_bins": {
                feature_name: [item.to_primitive() for item in bins]
                for feature_name, bins in self.feature_bins
            },
            "included_weekdays": (
                None if self.included_weekdays is None else list(self.included_weekdays)
            ),
            "minimum_sample_size": self.minimum_sample_size,
            "neutral_gap_policy": "incorrect_and_reported_separately",
            "outlier_count_per_configuration": self.outlier_count,
            "periods": [period.to_primitive() for period in self.periods],
            "rsi_thresholds": [
                decimal_to_primitive(value) for value in self.rsi_thresholds
            ],
            "study_status": "exploratory_in_sample",
        }


@dataclass(frozen=True, slots=True)
class _NamedAnalysis:
    name: str
    result: PredictionAnalysisResult


def run_prediction_comparison(
    dataset: MarketDataset,
    parameters: PredictionComparisonParameters | None = None,
) -> PredictionComparisonStudyResult:
    """Generate every causal strategy first, then build exploratory comparisons."""
    if parameters is None:
        parameters = PredictionComparisonParameters()
    primary_analyses, threshold_analyses = _generate_analyses(dataset, parameters)
    feature_rows = derive_completed_session_features(
        dataset, parameters.derived_features
    )
    features_by_session = {row.signal_session: row for row in feature_rows}
    always_analysis = next(
        item for item in primary_analyses if item.name == "always_up"
    )
    always_rows = {
        row.signal_session: row
        for row in always_analysis.result.rows
        if _session_is_eligible(row.signal_session, parameters)
    }
    eligible_session_count = len(always_rows)
    primary_predictions = {
        item.name: _comparison_predictions(
            item, features_by_session, always_rows, parameters
        )
        for item in primary_analyses
    }
    threshold_predictions = {
        threshold: _comparison_predictions(
            analysis, features_by_session, always_rows, parameters
        )
        for threshold, analysis in threshold_analyses
    }
    ordered_predictions = tuple(
        prediction
        for name in (
            "combined_original",
            "focused_rules",
            "rsi_oversold_up",
            "always_up",
        )
        for prediction in primary_predictions[name]
    )
    strategy_configurations = _strategy_configurations(
        primary_analyses, threshold_analyses
    )
    study_configuration_snapshot = PrimitiveMappingSnapshot.capture(
        parameters.to_primitive()
    )
    study_id = configuration_identity(
        {
            "component": "quantforge_prediction_comparison_study",
            "engine_version": COMPARISON_ENGINE_VERSION,
            "market_data": PredictionMarketData.from_qf3(
                dataset.metadata
            ).to_primitive(),
            "result_schema_version": COMPARISON_SCHEMA_VERSION,
            "strategies": [item.to_primitive() for item in strategy_configurations],
            "study_configuration": study_configuration_snapshot.to_primitive(),
        }
    )
    configuration_summaries = _configuration_summaries(
        primary_analyses,
        primary_predictions,
        eligible_session_count,
        parameters,
    )
    rule_summaries = _rule_summaries(
        primary_predictions, eligible_session_count, parameters
    )
    weekday_summaries = _weekday_summaries(primary_predictions, parameters)
    annual_summaries = _annual_summaries(dataset, primary_predictions, parameters)
    period_summaries = _period_summaries(primary_predictions, parameters)
    threshold_summaries = _threshold_summaries(
        dataset,
        threshold_predictions,
        always_rows,
        parameters,
    )
    feature_bin_summaries = _feature_bin_summaries(
        primary_predictions,
        features_by_session,
        always_rows,
        parameters,
    )
    baseline_comparisons = _baseline_comparisons(primary_predictions, parameters)
    best_outcomes, worst_outcomes = _outcomes(
        primary_predictions, parameters.outlier_count
    )
    return PredictionComparisonStudyResult(
        study_id=study_id,
        engine_version=COMPARISON_ENGINE_VERSION,
        result_schema_version=COMPARISON_SCHEMA_VERSION,
        market_data=PredictionMarketData.from_qf3(dataset.metadata),
        study_configuration_snapshot=study_configuration_snapshot,
        strategy_configurations=strategy_configurations,
        eligible_session_count=eligible_session_count,
        predictions=ordered_predictions,
        configuration_summaries=configuration_summaries,
        rule_summaries=rule_summaries,
        weekday_summaries=weekday_summaries,
        annual_summaries=annual_summaries,
        period_summaries=period_summaries,
        threshold_summaries=threshold_summaries,
        feature_bin_summaries=feature_bin_summaries,
        baseline_comparisons=baseline_comparisons,
        best_outcomes=best_outcomes,
        worst_outcomes=worst_outcomes,
        limitations=(
            "all results are exploratory and in-sample descriptive evidence",
            "2025 has already been observed and is not a pristine holdout",
            "directional accuracy can reflect SPY's structural overnight UP bias",
            "matched-session always-UP comparisons are required for sparse rules",
            "underlying gaps do not model options spreads, implied volatility, "
            "strikes, theta, or fills",
            "cumulative signed-return decline is a prediction-study statistic, "
            "not portfolio drawdown",
            "raw ex-dividend price effects remain in observed gaps",
        ),
    )


def _generate_analyses(
    dataset: MarketDataset, parameters: PredictionComparisonParameters
) -> tuple[tuple[_NamedAnalysis, ...], tuple[tuple[Decimal, _NamedAnalysis], ...]]:
    excluded = parameters.excluded_weekdays
    included = parameters.included_weekdays
    original = OvernightGapPredictionStrategy(
        OvernightGapPredictionParameters(excluded_weekdays=excluded)
    )
    focused = FocusedGapPredictionStrategy(
        FocusedGapPredictionParameters(
            excluded_weekdays=excluded, included_weekdays=included
        )
    )
    rsi = RsiOversoldUpPredictionStrategy(
        RsiOversoldUpParameters(excluded_weekdays=excluded, included_weekdays=included)
    )
    always = AlwaysUpPredictionStrategy(
        AlwaysUpParameters(excluded_weekdays=excluded, included_weekdays=included)
    )
    primary = (
        _NamedAnalysis("combined_original", run_prediction_analysis(dataset, original)),
        _NamedAnalysis("focused_rules", run_prediction_analysis(dataset, focused)),
        _NamedAnalysis("rsi_oversold_up", run_prediction_analysis(dataset, rsi)),
        _NamedAnalysis("always_up", run_prediction_analysis(dataset, always)),
    )
    threshold = tuple(
        (
            value,
            _NamedAnalysis(
                f"rsi_threshold_{decimal_to_primitive(value)}",
                run_prediction_analysis(
                    dataset,
                    RsiOversoldUpPredictionStrategy(
                        RsiOversoldUpParameters(
                            lower_rsi=value,
                            excluded_weekdays=excluded,
                            included_weekdays=included,
                        )
                    ),
                ),
            ),
        )
        for value in parameters.rsi_thresholds
    )
    return primary, threshold


def _comparison_predictions(
    analysis: _NamedAnalysis,
    features_by_session: dict[date, DerivedFeatureRow],
    always_rows: dict[date, PredictionRow],
    parameters: PredictionComparisonParameters,
) -> tuple[ComparisonPrediction, ...]:
    predictions: list[ComparisonPrediction] = []
    for row in analysis.result.rows:
        if not _session_is_eligible(row.signal_session, parameters):
            continue
        baseline_value = always_rows.get(row.signal_session)
        if not isinstance(baseline_value, PredictionRow):
            raise InvalidPredictionOutputError(
                "matched-session always-UP outcome is missing"
            )
        if (
            baseline_value.outcome_session != row.outcome_session
            or baseline_value.overnight_gap_percentage != row.overnight_gap_percentage
            or baseline_value.next_open != row.next_open
        ):
            raise InvalidPredictionOutputError(
                "prediction configurations do not share identical outcome labels"
            )
        features = features_by_session.get(row.signal_session)
        if features is None:
            raise InvalidPredictionOutputError(
                "completed-session derived features are missing"
            )
        baseline_correct = row.overnight_gap_percentage > 0
        baseline_signed_return = row.overnight_gap_percentage
        with arithmetic():
            incremental_signed_return = (
                row.signed_prediction_return - baseline_signed_return
            )
        predictions.append(
            ComparisonPrediction(
                configuration_name=analysis.name,
                prediction=row,
                features=features,
                period_name=_period_name(row.signal_session, parameters.periods),
                baseline_correct=baseline_correct,
                baseline_signed_return=baseline_signed_return,
                incremental_correctness=(int(row.correct) - int(baseline_correct)),
                incremental_signed_return=incremental_signed_return,
            )
        )
    return tuple(predictions)


def _strategy_configurations(
    primary: tuple[_NamedAnalysis, ...],
    thresholds: tuple[tuple[Decimal, _NamedAnalysis], ...],
) -> tuple[NamedStrategyConfiguration, ...]:
    analyses = (*primary, *(analysis for _, analysis in thresholds))
    return tuple(
        NamedStrategyConfiguration(
            configuration_name=item.name,
            strategy_id=item.result.strategy_id,
            strategy_configuration_id=item.result.strategy_configuration_id,
            configuration_snapshot=item.result.strategy_configuration_snapshot,
        )
        for item in analyses
    )


def _configuration_summaries(
    analyses: tuple[_NamedAnalysis, ...],
    predictions: dict[str, tuple[ComparisonPrediction, ...]],
    eligible_count: int,
    parameters: PredictionComparisonParameters,
) -> tuple[ConfigurationSummary, ...]:
    return tuple(
        ConfigurationSummary(
            configuration_name=item.name,
            strategy_id=item.result.strategy_id,
            strategy_configuration_id=item.result.strategy_configuration_id,
            eligible_session_count=eligible_count,
            prediction_frequency=_frequency(
                len(predictions[item.name]), eligible_count
            ),
            metrics=summarize_predictions(
                predictions[item.name], confidence_level=parameters.confidence_level
            ),
            streaks=calculate_streak_statistics(predictions[item.name]),
        )
        for item in analyses
    )


def _rule_summaries(
    predictions: dict[str, tuple[ComparisonPrediction, ...]],
    eligible_count: int,
    parameters: PredictionComparisonParameters,
) -> tuple[RuleSummary, ...]:
    rows: list[RuleSummary] = []
    for configuration_name, values in predictions.items():
        for reason in sorted({item.prediction.reason for item in values}):
            selected = tuple(
                item for item in values if item.prediction.reason == reason
            )
            rows.append(
                RuleSummary(
                    configuration_name,
                    reason,
                    eligible_count,
                    _frequency(len(selected), eligible_count),
                    summarize_predictions(
                        selected, confidence_level=parameters.confidence_level
                    ),
                )
            )
    return tuple(rows)


def _weekday_summaries(
    predictions: dict[str, tuple[ComparisonPrediction, ...]],
    parameters: PredictionComparisonParameters,
) -> tuple[WeekdaySummary, ...]:
    rows: list[WeekdaySummary] = []
    weekdays = tuple(
        day
        for day in range(5)
        if day not in parameters.excluded_weekdays
        and (
            parameters.included_weekdays is None or day in parameters.included_weekdays
        )
    )
    for configuration_name, values in predictions.items():
        reasons = (ALL_REASONS, *sorted({item.prediction.reason for item in values}))
        for reason in reasons:
            reason_values = (
                values
                if reason == ALL_REASONS
                else tuple(item for item in values if item.prediction.reason == reason)
            )
            for weekday in weekdays:
                selected = tuple(
                    item
                    for item in reason_values
                    if item.prediction.signal_session.weekday() == weekday
                )
                rows.append(
                    WeekdaySummary(
                        configuration_name,
                        reason,
                        weekday,
                        WEEKDAY_NAMES[weekday],
                        summarize_predictions(
                            selected, confidence_level=parameters.confidence_level
                        ),
                    )
                )
    return tuple(rows)


def _annual_summaries(
    dataset: MarketDataset,
    predictions: dict[str, tuple[ComparisonPrediction, ...]],
    parameters: PredictionComparisonParameters,
) -> tuple[AnnualSummary, ...]:
    rows: list[AnnualSummary] = []
    years = range(
        dataset.metadata.actual_first_session.year,
        dataset.metadata.actual_last_session.year + 1,
    )
    for configuration_name, values in predictions.items():
        reasons = (ALL_REASONS, *sorted({item.prediction.reason for item in values}))
        for reason in reasons:
            reason_values = (
                values
                if reason == ALL_REASONS
                else tuple(item for item in values if item.prediction.reason == reason)
            )
            for year in years:
                selected = tuple(
                    item
                    for item in reason_values
                    if item.prediction.signal_session.year == year
                )
                rows.append(
                    AnnualSummary(
                        configuration_name,
                        reason,
                        year,
                        summarize_predictions(
                            selected, confidence_level=parameters.confidence_level
                        ),
                    )
                )
    return tuple(rows)


def _period_summaries(
    predictions: dict[str, tuple[ComparisonPrediction, ...]],
    parameters: PredictionComparisonParameters,
) -> tuple[PeriodSummary, ...]:
    rows: list[PeriodSummary] = []
    for configuration_name, values in predictions.items():
        reasons = (ALL_REASONS, *sorted({item.prediction.reason for item in values}))
        for reason in reasons:
            reason_values = (
                values
                if reason == ALL_REASONS
                else tuple(item for item in values if item.prediction.reason == reason)
            )
            for period in parameters.periods:
                selected = tuple(
                    item
                    for item in reason_values
                    if period.contains(item.prediction.signal_session)
                )
                rows.append(
                    PeriodSummary(
                        configuration_name,
                        reason,
                        period.name,
                        period.start.isoformat(),
                        period.end.isoformat(),
                        period.exploratory_label,
                        summarize_predictions(
                            selected, confidence_level=parameters.confidence_level
                        ),
                    )
                )
    return tuple(rows)


def _threshold_summaries(
    dataset: MarketDataset,
    threshold_predictions: dict[Decimal, tuple[ComparisonPrediction, ...]],
    always_rows: dict[date, PredictionRow],
    parameters: PredictionComparisonParameters,
) -> tuple[ThresholdSensitivitySummary, ...]:
    rows: list[ThresholdSensitivitySummary] = []
    years = range(
        dataset.metadata.actual_first_session.year,
        dataset.metadata.actual_last_session.year + 1,
    )
    for threshold in parameters.rsi_thresholds:
        values = threshold_predictions[threshold]
        assessment = _threshold_stability(values, parameters)
        segments: list[tuple[str, str, date | None, date | None]] = [
            ("full", "full_period", None, None)
        ]
        segments.extend(
            ("annual", str(year), date(year, 1, 1), date(year, 12, 31))
            for year in years
        )
        segments.extend(
            ("period", period.name, period.start, period.end)
            for period in parameters.periods
        )
        for segment_type, segment_name, start, end in segments:
            selected = tuple(
                item
                for item in values
                if start is None
                or start <= item.prediction.signal_session <= cast(date, end)
            )
            eligible_count = sum(
                start is None or start <= session <= cast(date, end)
                for session in always_rows
            )
            rows.append(
                ThresholdSensitivitySummary(
                    threshold=threshold,
                    segment_type=segment_type,
                    segment_name=segment_name,
                    segment_start=None if start is None else start.isoformat(),
                    segment_end=None if end is None else end.isoformat(),
                    eligible_session_count=eligible_count,
                    prediction_frequency=_frequency(len(selected), eligible_count),
                    adequate_sample=len(selected) >= parameters.minimum_sample_size,
                    stability_assessment=assessment,
                    metrics=summarize_predictions(
                        selected, confidence_level=parameters.confidence_level
                    ),
                )
            )
    return tuple(rows)


def _threshold_stability(
    predictions: tuple[ComparisonPrediction, ...],
    parameters: PredictionComparisonParameters,
) -> str:
    period_metrics = tuple(
        summarize_predictions(
            tuple(
                item
                for item in predictions
                if period.contains(item.prediction.signal_session)
            ),
            confidence_level=parameters.confidence_level,
        )
        for period in parameters.periods
    )
    if any(
        metric.prediction_count < parameters.minimum_sample_size
        for metric in period_metrics
    ):
        return "insufficient_period_sample"
    averages = tuple(
        metric.average_signed_prediction_return for metric in period_metrics
    )
    if all(value is not None and value > 0 for value in averages):
        return "consistent_across_periods"
    if (
        averages
        and averages[0] is not None
        and averages[0] > 0
        and any(value is not None and value <= 0 for value in averages[1:])
    ):
        return "positive_first_period_with_later_nonpositive_period"
    return "mixed_or_weak_across_periods"


def _feature_bin_summaries(
    predictions: dict[str, tuple[ComparisonPrediction, ...]],
    features_by_session: dict[date, DerivedFeatureRow],
    always_rows: dict[date, PredictionRow],
    parameters: PredictionComparisonParameters,
) -> tuple[FeatureBinSummary, ...]:
    rows: list[FeatureBinSummary] = []
    eligible_features = tuple(
        features_by_session[session]
        for session in sorted(always_rows)
        if session in features_by_session
    )
    for configuration_name, values in predictions.items():
        for feature_name, bins in parameters.feature_bins:
            for feature_bin in bins:
                observation_count = sum(
                    (value := _feature_value(feature, feature_name)) is not None
                    and feature_bin.contains(value)
                    for feature in eligible_features
                )
                selected = tuple(
                    item
                    for item in values
                    if (value := _feature_value(item.features, feature_name))
                    is not None
                    and feature_bin.contains(value)
                )
                rows.append(
                    FeatureBinSummary(
                        configuration_name=configuration_name,
                        feature_name=feature_name,
                        bin_label=feature_bin.label,
                        lower_bound=feature_bin.lower_bound,
                        upper_bound=feature_bin.upper_bound,
                        interval_convention=feature_bin.interval_convention,
                        observation_count=observation_count,
                        minimum_sample_size=parameters.minimum_sample_size,
                        adequate_sample=len(selected) >= parameters.minimum_sample_size,
                        metrics=summarize_predictions(
                            selected, confidence_level=parameters.confidence_level
                        ),
                    )
                )
    return tuple(rows)


def _baseline_comparisons(
    predictions: dict[str, tuple[ComparisonPrediction, ...]],
    parameters: PredictionComparisonParameters,
) -> tuple[BaselineComparisonSummary, ...]:
    rows: list[BaselineComparisonSummary] = []
    all_baseline_predictions = predictions["always_up"]
    all_baseline_metrics = summarize_predictions(
        all_baseline_predictions, confidence_level=parameters.confidence_level
    )
    for configuration_name, values in predictions.items():
        strategy_metrics = summarize_predictions(
            values, confidence_level=parameters.confidence_level
        )
        rows.append(
            BaselineComparisonSummary(
                configuration_name=configuration_name,
                comparison_scope="all_eligible_sessions",
                strategy_metrics=strategy_metrics,
                baseline_metrics=all_baseline_metrics,
                incremental_accuracy=_difference(
                    strategy_metrics.accuracy, all_baseline_metrics.accuracy
                ),
                average_incremental_signed_return=None,
                median_incremental_signed_return=None,
            )
        )
        matched_baseline_metrics = summarize_observations(
            baseline_observations(values),
            confidence_level=parameters.confidence_level,
        )
        rows.append(
            BaselineComparisonSummary(
                configuration_name=configuration_name,
                comparison_scope="matched_prediction_sessions",
                strategy_metrics=strategy_metrics,
                baseline_metrics=matched_baseline_metrics,
                incremental_accuracy=_difference(
                    strategy_metrics.accuracy, matched_baseline_metrics.accuracy
                ),
                average_incremental_signed_return=average_incremental_return(values),
                median_incremental_signed_return=median_incremental_return(values),
            )
        )
    return tuple(rows)


def _outcomes(
    predictions: dict[str, tuple[ComparisonPrediction, ...]], count: int
) -> tuple[tuple[ComparisonPrediction, ...], tuple[ComparisonPrediction, ...]]:
    best: list[ComparisonPrediction] = []
    worst: list[ComparisonPrediction] = []
    for configuration_name in (
        "combined_original",
        "focused_rules",
        "rsi_oversold_up",
        "always_up",
    ):
        values = predictions[configuration_name]
        best.extend(
            sorted(
                values,
                key=lambda item: (
                    -item.prediction.signed_prediction_return,
                    item.prediction.signal_session,
                    item.prediction.prediction_id,
                ),
            )[:count]
        )
        worst.extend(
            sorted(
                values,
                key=lambda item: (
                    item.prediction.signed_prediction_return,
                    item.prediction.signal_session,
                    item.prediction.prediction_id,
                ),
            )[:count]
        )
    return tuple(best), tuple(worst)


def _feature_value(row: DerivedFeatureRow, name: str) -> Decimal | None:
    value = getattr(row, name)
    if value is None or isinstance(value, Decimal):
        return value
    raise InvalidPredictionOutputError(f"unsupported binned feature: {name}")


def _period_name(session: date, periods: tuple[StudyPeriod, ...]) -> str | None:
    matches = tuple(period.name for period in periods if period.contains(session))
    if len(matches) > 1:
        raise InvalidPredictionOutputError("period labels overlap")
    return None if not matches else matches[0]


def _session_is_eligible(
    session: date, parameters: PredictionComparisonParameters
) -> bool:
    weekday = session.weekday()
    return weekday not in parameters.excluded_weekdays and (
        parameters.included_weekdays is None or weekday in parameters.included_weekdays
    )


def _validate_periods(periods: tuple[StudyPeriod, ...]) -> None:
    if not periods:
        raise InvalidPredictionConfigurationError(
            "at least one study period is required"
        )
    if (
        tuple(sorted(periods, key=lambda item: (item.start, item.end, item.name)))
        != periods
    ):
        raise InvalidPredictionConfigurationError(
            "study periods must be deterministically ordered"
        )
    if len({item.name for item in periods}) != len(periods):
        raise InvalidPredictionConfigurationError("study period names must be unique")
    for previous, current in pairwise(periods):
        if previous.end >= current.start:
            raise InvalidPredictionConfigurationError("study periods must not overlap")


def _validate_feature_bins(
    feature_name: str, bins: tuple[FeatureRangeBin, ...]
) -> None:
    if not bins:
        raise InvalidPredictionConfigurationError(
            f"{feature_name} requires at least one bin"
        )
    for previous, current in pairwise(bins):
        if previous.upper_bound != current.lower_bound:
            raise InvalidPredictionConfigurationError(
                f"{feature_name} bins must be contiguous and nonoverlapping"
            )
        if previous.include_upper_bound:
            raise InvalidPredictionConfigurationError(
                f"{feature_name} nonfinal bins must be half-open"
            )


def _validated_weekdays(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise InvalidPredictionConfigurationError(f"{name} must be a tuple")
    weekdays = cast(tuple[object, ...], value)
    if any(
        isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6
        for day in weekdays
    ):
        raise InvalidPredictionConfigurationError(
            f"{name} must contain Monday=0 through Sunday=6"
        )
    if len(weekdays) != len(set(weekdays)):
        raise InvalidPredictionConfigurationError(f"{name} must be unique")
    return tuple(sorted(cast(tuple[int, ...], weekdays)))


def _frequency(count: int, eligible_count: int) -> Decimal | None:
    if not eligible_count:
        return None
    with arithmetic():
        return Decimal(count) / Decimal(eligible_count)


def _difference(first: Decimal | None, second: Decimal | None) -> Decimal | None:
    if first is None or second is None:
        return None
    with arithmetic():
        return first - second


def _decimal(value: object, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise InvalidPredictionConfigurationError(f"{name} must be numeric") from error
    if not result.is_finite():
        raise InvalidPredictionConfigurationError(f"{name} must be finite")
    return result


def _optional_decimal(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_primitive(value)
