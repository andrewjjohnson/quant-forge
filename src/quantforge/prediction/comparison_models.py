"""Immutable records for multi-configuration overnight-gap comparisons."""

from dataclasses import dataclass
from decimal import Decimal
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    decimal_to_primitive,
)
from quantforge.prediction._arithmetic import arithmetic
from quantforge.prediction.features import DerivedFeatureRow
from quantforge.prediction.models import PredictionMarketData, PredictionRow

COMPARISON_PREDICTION_FIELDNAMES = (
    "configuration_name",
    "prediction_id",
    "dataset_id",
    "dataset_fingerprint",
    "symbol",
    "signal_session",
    "outcome_session",
    "signal_weekday",
    "signal_year",
    "period_name",
    "strategy_id",
    "strategy_implementation_version",
    "strategy_configuration_id",
    "strategy_parameters",
    "reason",
    "direction",
    "rsi",
    "previous_rsi",
    "rsi_change",
    "adx",
    "previous_adx",
    "adx_change",
    "positive_di",
    "negative_di",
    "di_spread",
    "previous_di_spread",
    "atr",
    "atr_percentage_of_close",
    "open",
    "close",
    "candle_return",
    "volume",
    "average_volume",
    "volume_ratio",
    "next_open",
    "raw_overnight_return",
    "absolute_gap",
    "signed_prediction_return",
    "correct",
    "neutral_outcome",
    "baseline_direction",
    "baseline_correct",
    "baseline_signed_return",
    "incremental_correctness",
    "incremental_signed_return",
)


@dataclass(frozen=True, slots=True)
class AccuracyInterval:
    confidence_level: Decimal
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    sample_count: int
    method: str = "wilson_score"

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "confidence_level": decimal_to_primitive(self.confidence_level),
            "lower_bound": _optional(self.lower_bound),
            "method": self.method,
            "sample_count": self.sample_count,
            "upper_bound": _optional(self.upper_bound),
        }


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Direction, magnitude, tail, and uncertainty statistics for predictions."""

    prediction_count: int
    up_prediction_count: int
    down_prediction_count: int
    correct_count: int
    incorrect_count: int
    neutral_outcome_count: int
    accuracy: Decimal | None
    accuracy_interval: AccuracyInterval
    average_raw_overnight_return: Decimal | None
    median_raw_overnight_return: Decimal | None
    average_signed_prediction_return: Decimal | None
    median_signed_prediction_return: Decimal | None
    signed_prediction_return_standard_deviation: Decimal | None
    average_absolute_gap: Decimal | None
    best_signed_prediction_return: Decimal | None
    worst_signed_prediction_return: Decimal | None
    probability_signed_greater_than_0_10_percent: Decimal | None
    probability_signed_greater_than_0_25_percent: Decimal | None
    probability_signed_greater_than_0_50_percent: Decimal | None
    probability_signed_greater_than_1_00_percent: Decimal | None
    probability_signed_less_than_negative_0_10_percent: Decimal | None
    probability_signed_less_than_negative_0_25_percent: Decimal | None
    probability_signed_less_than_negative_0_50_percent: Decimal | None
    probability_signed_less_than_negative_1_00_percent: Decimal | None
    average_signed_return_correct: Decimal | None
    average_signed_return_incorrect: Decimal | None
    median_signed_return_correct: Decimal | None
    median_signed_return_incorrect: Decimal | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "accuracy": _optional(self.accuracy),
            "accuracy_confidence_level": decimal_to_primitive(
                self.accuracy_interval.confidence_level
            ),
            "accuracy_interval_lower": _optional(self.accuracy_interval.lower_bound),
            "accuracy_interval_method": self.accuracy_interval.method,
            "accuracy_interval_upper": _optional(self.accuracy_interval.upper_bound),
            "average_absolute_gap": _optional(self.average_absolute_gap),
            "average_raw_overnight_return": _optional(
                self.average_raw_overnight_return
            ),
            "average_signed_prediction_return": _optional(
                self.average_signed_prediction_return
            ),
            "average_signed_return_correct": _optional(
                self.average_signed_return_correct
            ),
            "average_signed_return_incorrect": _optional(
                self.average_signed_return_incorrect
            ),
            "best_signed_prediction_return": _optional(
                self.best_signed_prediction_return
            ),
            "correct_count": self.correct_count,
            "down_prediction_count": self.down_prediction_count,
            "incorrect_count": self.incorrect_count,
            "median_raw_overnight_return": _optional(self.median_raw_overnight_return),
            "median_signed_prediction_return": _optional(
                self.median_signed_prediction_return
            ),
            "median_signed_return_correct": _optional(
                self.median_signed_return_correct
            ),
            "median_signed_return_incorrect": _optional(
                self.median_signed_return_incorrect
            ),
            "neutral_outcome_count": self.neutral_outcome_count,
            "prediction_count": self.prediction_count,
            "probability_signed_greater_than_0_10_percent": _optional(
                self.probability_signed_greater_than_0_10_percent
            ),
            "probability_signed_greater_than_0_25_percent": _optional(
                self.probability_signed_greater_than_0_25_percent
            ),
            "probability_signed_greater_than_0_50_percent": _optional(
                self.probability_signed_greater_than_0_50_percent
            ),
            "probability_signed_greater_than_1_00_percent": _optional(
                self.probability_signed_greater_than_1_00_percent
            ),
            "probability_signed_less_than_negative_0_10_percent": _optional(
                self.probability_signed_less_than_negative_0_10_percent
            ),
            "probability_signed_less_than_negative_0_25_percent": _optional(
                self.probability_signed_less_than_negative_0_25_percent
            ),
            "probability_signed_less_than_negative_0_50_percent": _optional(
                self.probability_signed_less_than_negative_0_50_percent
            ),
            "probability_signed_less_than_negative_1_00_percent": _optional(
                self.probability_signed_less_than_negative_1_00_percent
            ),
            "signed_prediction_return_standard_deviation": _optional(
                self.signed_prediction_return_standard_deviation
            ),
            "up_prediction_count": self.up_prediction_count,
            "worst_signed_prediction_return": _optional(
                self.worst_signed_prediction_return
            ),
        }


@dataclass(frozen=True, slots=True)
class ComparisonPrediction:
    """One labeled prediction plus causal features and matched always-UP outcome."""

    configuration_name: str
    prediction: PredictionRow
    features: DerivedFeatureRow
    period_name: str | None
    baseline_correct: bool
    baseline_signed_return: Decimal
    incremental_correctness: int
    incremental_signed_return: Decimal

    def to_primitive(self) -> PrimitiveMapping:
        prediction = self.prediction
        feature_values = self.features.to_primitive()
        return {
            "configuration_name": self.configuration_name,
            "prediction_id": prediction.prediction_id,
            "dataset_id": prediction.dataset_id,
            "dataset_fingerprint": prediction.dataset_fingerprint,
            "symbol": prediction.symbol,
            "signal_session": prediction.signal_session.isoformat(),
            "outcome_session": prediction.outcome_session.isoformat(),
            "signal_weekday": self.features.signal_weekday,
            "signal_year": prediction.signal_session.year,
            "period_name": self.period_name,
            "strategy_id": prediction.strategy_id,
            "strategy_implementation_version": (
                prediction.strategy_implementation_version
            ),
            "strategy_configuration_id": prediction.strategy_configuration_id,
            "strategy_parameters": {
                item.name: item.value for item in prediction.strategy_parameters
            },
            "reason": prediction.reason,
            "direction": prediction.direction.value,
            "rsi": feature_values["rsi"],
            "previous_rsi": feature_values["previous_rsi"],
            "rsi_change": feature_values["rsi_change"],
            "adx": feature_values["adx"],
            "previous_adx": feature_values["previous_adx"],
            "adx_change": feature_values["adx_change"],
            "positive_di": feature_values["positive_di"],
            "negative_di": feature_values["negative_di"],
            "di_spread": feature_values["di_spread"],
            "previous_di_spread": feature_values["previous_di_spread"],
            "atr": feature_values["atr"],
            "atr_percentage_of_close": feature_values["atr_percentage_of_close"],
            "open": feature_values["open"],
            "close": feature_values["close"],
            "candle_return": feature_values["candle_return"],
            "volume": feature_values["volume"],
            "average_volume": feature_values["average_volume"],
            "volume_ratio": feature_values["volume_ratio"],
            "next_open": decimal_to_primitive(prediction.next_open),
            "raw_overnight_return": decimal_to_primitive(
                prediction.overnight_gap_percentage
            ),
            "absolute_gap": decimal_to_primitive(prediction.gap_size_percentage),
            "signed_prediction_return": decimal_to_primitive(
                prediction.signed_prediction_return
            ),
            "correct": prediction.correct,
            "neutral_outcome": prediction.overnight_gap_percentage == 0,
            "baseline_direction": "up",
            "baseline_correct": self.baseline_correct,
            "baseline_signed_return": decimal_to_primitive(self.baseline_signed_return),
            "incremental_correctness": self.incremental_correctness,
            "incremental_signed_return": decimal_to_primitive(
                self.incremental_signed_return
            ),
        }


@dataclass(frozen=True, slots=True)
class StreakStatistics:
    longest_incorrect_streak: int
    longest_correct_streak: int
    maximum_cumulative_signed_return_decline: Decimal
    incorrect_streaks_at_least_3: int
    incorrect_streaks_at_least_5: int
    statistic_label: str = "prediction_sequence_not_portfolio_drawdown"

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "incorrect_streaks_at_least_3": self.incorrect_streaks_at_least_3,
            "incorrect_streaks_at_least_5": self.incorrect_streaks_at_least_5,
            "longest_correct_streak": self.longest_correct_streak,
            "longest_incorrect_streak": self.longest_incorrect_streak,
            "maximum_cumulative_signed_return_decline": decimal_to_primitive(
                self.maximum_cumulative_signed_return_decline
            ),
            "statistic_label": self.statistic_label,
        }


@dataclass(frozen=True, slots=True)
class ConfigurationSummary:
    configuration_name: str
    strategy_id: str
    strategy_configuration_id: str
    eligible_session_count: int
    prediction_frequency: Decimal | None
    metrics: MetricSummary
    streaks: StreakStatistics

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "configuration_name": self.configuration_name,
            "eligible_session_count": self.eligible_session_count,
            "prediction_frequency": _optional(self.prediction_frequency),
            "strategy_configuration_id": self.strategy_configuration_id,
            "strategy_id": self.strategy_id,
            **self.metrics.to_primitive(),
            **self.streaks.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class RuleSummary:
    configuration_name: str
    reason: str
    eligible_session_count: int
    prediction_frequency: Decimal | None
    metrics: MetricSummary

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "configuration_name": self.configuration_name,
            "eligible_session_count": self.eligible_session_count,
            "prediction_frequency": _optional(self.prediction_frequency),
            "reason": self.reason,
            **self.metrics.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class WeekdaySummary:
    configuration_name: str
    reason: str
    weekday: int
    weekday_name: str
    metrics: MetricSummary

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "configuration_name": self.configuration_name,
            "reason": self.reason,
            "weekday": self.weekday,
            "weekday_name": self.weekday_name,
            **self.metrics.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class AnnualSummary:
    configuration_name: str
    reason: str
    year: int
    metrics: MetricSummary

    def to_primitive(self) -> PrimitiveMapping:
        if not self.metrics.prediction_count:
            positive_percentage = None
        else:
            with arithmetic():
                positive_percentage = Decimal(self.metrics.correct_count) / Decimal(
                    self.metrics.prediction_count
                )
        return {
            "configuration_name": self.configuration_name,
            "positive_signed_return_percentage": _optional(positive_percentage),
            "reason": self.reason,
            "year": self.year,
            **self.metrics.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class PeriodSummary:
    configuration_name: str
    reason: str
    period_name: str
    period_start: str
    period_end: str
    exploratory_label: str
    metrics: MetricSummary

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "configuration_name": self.configuration_name,
            "exploratory_label": self.exploratory_label,
            "period_end": self.period_end,
            "period_name": self.period_name,
            "period_start": self.period_start,
            "reason": self.reason,
            **self.metrics.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class ThresholdSensitivitySummary:
    threshold: Decimal
    segment_type: str
    segment_name: str
    segment_start: str | None
    segment_end: str | None
    eligible_session_count: int
    prediction_frequency: Decimal | None
    adequate_sample: bool
    stability_assessment: str
    metrics: MetricSummary

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "adequate_sample": self.adequate_sample,
            "eligible_session_count": self.eligible_session_count,
            "prediction_frequency": _optional(self.prediction_frequency),
            "segment_end": self.segment_end,
            "segment_name": self.segment_name,
            "segment_start": self.segment_start,
            "segment_type": self.segment_type,
            "stability_assessment": self.stability_assessment,
            "threshold": decimal_to_primitive(self.threshold),
            **self.metrics.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class FeatureBinSummary:
    configuration_name: str
    feature_name: str
    bin_label: str
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    interval_convention: str
    observation_count: int
    minimum_sample_size: int
    adequate_sample: bool
    metrics: MetricSummary

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "adequate_sample": self.adequate_sample,
            "bin_label": self.bin_label,
            "configuration_name": self.configuration_name,
            "feature_name": self.feature_name,
            "interval_convention": self.interval_convention,
            "lower_bound": _optional(self.lower_bound),
            "minimum_sample_size": self.minimum_sample_size,
            "observation_count": self.observation_count,
            "upper_bound": _optional(self.upper_bound),
            **self.metrics.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class BaselineComparisonSummary:
    configuration_name: str
    comparison_scope: str
    strategy_metrics: MetricSummary
    baseline_metrics: MetricSummary
    incremental_accuracy: Decimal | None
    average_incremental_signed_return: Decimal | None
    median_incremental_signed_return: Decimal | None

    def to_primitive(self) -> PrimitiveMapping:
        result: PrimitiveMapping = {
            "average_incremental_signed_return": _optional(
                self.average_incremental_signed_return
            ),
            "baseline_configuration_name": "always_up",
            "comparison_scope": self.comparison_scope,
            "configuration_name": self.configuration_name,
            "incremental_accuracy": _optional(self.incremental_accuracy),
            "median_incremental_signed_return": _optional(
                self.median_incremental_signed_return
            ),
        }
        result.update(
            {
                f"strategy_{key}": value
                for key, value in self.strategy_metrics.to_primitive().items()
            }
        )
        result.update(
            {
                f"baseline_{key}": value
                for key, value in self.baseline_metrics.to_primitive().items()
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class NamedStrategyConfiguration:
    configuration_name: str
    strategy_id: str
    strategy_configuration_id: str
    configuration_snapshot: PrimitiveMappingSnapshot

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "configuration": self.configuration_snapshot.to_primitive(),
            "configuration_name": self.configuration_name,
            "strategy_configuration_id": self.strategy_configuration_id,
            "strategy_id": self.strategy_id,
        }


@dataclass(frozen=True, slots=True)
class PredictionComparisonStudyResult:
    study_id: str
    engine_version: str
    result_schema_version: str
    market_data: PredictionMarketData
    study_configuration_snapshot: PrimitiveMappingSnapshot
    strategy_configurations: tuple[NamedStrategyConfiguration, ...]
    eligible_session_count: int
    predictions: tuple[ComparisonPrediction, ...]
    configuration_summaries: tuple[ConfigurationSummary, ...]
    rule_summaries: tuple[RuleSummary, ...]
    weekday_summaries: tuple[WeekdaySummary, ...]
    annual_summaries: tuple[AnnualSummary, ...]
    period_summaries: tuple[PeriodSummary, ...]
    threshold_summaries: tuple[ThresholdSensitivitySummary, ...]
    feature_bin_summaries: tuple[FeatureBinSummary, ...]
    baseline_comparisons: tuple[BaselineComparisonSummary, ...]
    best_outcomes: tuple[ComparisonPrediction, ...]
    worst_outcomes: tuple[ComparisonPrediction, ...]
    limitations: tuple[str, ...]

    @property
    def study_configuration(self) -> PrimitiveMapping:
        return self.study_configuration_snapshot.to_primitive()

    def manifest_primitive(self) -> PrimitiveMapping:
        return {
            "artifacts": [
                "manifest.json",
                "configuration_summary.csv",
                "predictions.csv",
                "rule_summary.csv",
                "weekday_summary.csv",
                "annual_summary.csv",
                "period_summary.csv",
                "threshold_sensitivity.csv",
                "feature_bin_summary.csv",
                "baseline_comparison.csv",
                "best_outcomes.csv",
                "worst_outcomes.csv",
                "metrics.json",
            ],
            "eligible_session_count": self.eligible_session_count,
            "engine_version": self.engine_version,
            "exploratory": True,
            "feature_outcome_boundary": (
                "features are completed-session values; next-open outcomes are "
                "attached only after strategy generation"
            ),
            "limitations": list(self.limitations),
            "market_data": self.market_data.to_primitive(),
            "result_schema_version": self.result_schema_version,
            "strategy_configurations": [
                item.to_primitive() for item in self.strategy_configurations
            ],
            "study_configuration": self.study_configuration,
            "study_id": self.study_id,
        }

    def metrics_primitive(self) -> PrimitiveMapping:
        return {
            "baseline_comparisons": cast(
                list[Primitive],
                [item.to_primitive() for item in self.baseline_comparisons],
            ),
            "configuration_summaries": cast(
                list[Primitive],
                [item.to_primitive() for item in self.configuration_summaries],
            ),
            "exploratory": True,
            "study_id": self.study_id,
        }


def _optional(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_primitive(value)
