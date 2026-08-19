#!/usr/bin/env python3
"""Run the documented QF-33 SPY scanner entirely from the QF-30 cache fixture."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from export_spy_multi_timeframe_context import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_FIXTURE_PATH,
    ExampleDatasets,
    build_datasets,
    load_fixture,
)

from quantforge.data import (
    ContextCompletionPolicy,
    TimeframeBarSeries,
    build_multi_timeframe_context,
)
from quantforge.indicators import (
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    TALIB_INDICATOR_BACKEND,
    MarketField,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
)
from quantforge.prediction import (
    AlertDeduplicationPolicy,
    ConsolePredictionAlertSink,
    HistoricalPredictionStudyReference,
    JsonFileAlertDeduplicationStore,
    JsonFilePredictionAlertSink,
    PredictionContextRequirements,
    PredictionIndicatorRequirement,
    PredictionScanner,
    PredictionScannerError,
    PredictionScannerRuleBinding,
    PredictionScannerSnapshot,
    PredictionTimeframeRequirement,
    TechnicalCondition,
    TechnicalConditionOperand,
    TechnicalConditionOperator,
    TechnicalConfluenceParameters,
    TechnicalConfluencePredictionRule,
)
from quantforge.timeframes import Timeframe

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALERT_ROOT = REPOSITORY_ROOT / "reports" / "qf33-spy-alerts"
DEFAULT_STATE_ROOT = REPOSITORY_ROOT / "data" / "qf33-spy-alert-state"
HISTORICAL_STUDY_ID = "qf33_spy_fixture_parity_study_v1"


def _aware_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "as-of must be an ISO-8601 timestamp"
        ) from error
    if parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("as-of must include a UTC offset")
    return parsed.astimezone(UTC)


def _timeframes(
    datasets: ExampleDatasets,
) -> tuple[Timeframe, Timeframe, Timeframe, Timeframe]:
    return (
        datasets.source.request.timeframe,
        datasets.four_hour.request.timeframe,
        datasets.daily.metadata.target_timeframe,
        datasets.weekly.metadata.target_timeframe,
    )


def _series(datasets: ExampleDatasets) -> tuple[TimeframeBarSeries, ...]:
    return (
        TimeframeBarSeries.from_source_dataset(
            datasets.source,
            family=datasets.family,
            cache=datasets.cache,
        ),
        TimeframeBarSeries.from_aggregated_intraday_dataset(
            datasets.four_hour,
            family=datasets.family,
        ),
        TimeframeBarSeries.from_aggregated_session_dataset(
            datasets.daily,
            family=datasets.family,
        ),
        TimeframeBarSeries.from_aggregated_session_dataset(
            datasets.weekly,
            family=datasets.family,
        ),
    )


@dataclass(frozen=True, slots=True)
class CachedSpyScannerSource:
    """QF-33 source adapter that rebuilds contexts from cache-validated QF-30 data."""

    datasets: ExampleDatasets

    def prepare_context(
        self,
        requirements: PredictionContextRequirements,
        *,
        as_of: datetime,
        refresh: bool,
    ) -> PredictionScannerSnapshot:
        if refresh:
            raise PredictionScannerError(
                "the fixture source is cache-only; run the documented example in "
                "dry-run mode"
            )
        context = build_multi_timeframe_context(
            as_of=as_of,
            primary_timeframe=requirements.primary.timeframe,
            required_timeframes=requirements.context_timeframe_requirements(),
            series=_series(self.datasets),
            completion_policy=requirements.context_completion_policy,
        )
        return PredictionScannerSnapshot(
            context=context,
            prediction_dataset_id=self.datasets.source.metadata.dataset_id,
            symbol=self.datasets.source.request.symbol,
            adjustment_basis=self.datasets.source.request.adjustment_basis,
            source_mode=self.datasets.cache_status,
        )


def create_fixture_parity_rule(
    datasets: ExampleDatasets,
    completion_policy: ContextCompletionPolicy,
) -> TechnicalConfluencePredictionRule:
    """Create the exact fixed rule configuration referenced by this dry-run study."""
    primary, four_hour, daily, weekly = _timeframes(datasets)
    feed_scope = datasets.source.request.feed_scope

    def timeframe_requirement(
        timeframe: Timeframe,
    ) -> PredictionTimeframeRequirement:
        return PredictionTimeframeRequirement(
            timeframe,
            feed_scope,
            (
                PredictionIndicatorRequirement(
                    "trend",
                    SimpleMovingAverage(
                        SimpleMovingAverageParameters(2),
                        backend_id=TALIB_INDICATOR_BACKEND,
                    ),
                ),
            ),
            completion_policy,
        )

    requirements = PredictionContextRequirements(
        PredictionTimeframeRequirement(primary, feed_scope),
        (
            timeframe_requirement(four_hour),
            timeframe_requirement(daily),
            timeframe_requirement(weekly),
        ),
    )
    close = TechnicalConditionOperand.bar(MarketField.CLOSE, "price_per_share")
    trend = TechnicalConditionOperand.indicator(
        "trend", SIMPLE_MOVING_AVERAGE_OUTPUT, "price_per_share"
    )

    def condition(
        direction: str,
        timeframe_name: str,
        timeframe: Timeframe,
        operator: TechnicalConditionOperator,
    ) -> TechnicalCondition:
        return TechnicalCondition(
            f"{direction}_{timeframe_name}_close_vs_sma_2",
            timeframe_name,
            timeframe,
            close,
            operator,
            trend,
        )

    named_timeframes = (
        ("weekly", weekly),
        ("daily", daily),
        ("four_hour", four_hour),
    )
    up = tuple(
        condition(
            "up",
            name,
            timeframe,
            TechnicalConditionOperator.GREATER_THAN,
        )
        for name, timeframe in named_timeframes
    )
    down = tuple(
        condition(
            "down",
            name,
            timeframe,
            TechnicalConditionOperator.LESS_THAN,
        )
        for name, timeframe in named_timeframes
    )
    return TechnicalConfluencePredictionRule(
        TechnicalConfluenceParameters(up, down, "qf33_spy_fixture_parity_v1"),
        requirements,
    )


def _parser(default_as_of: datetime) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--alert-root", type=Path, default=DEFAULT_ALERT_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument(
        "--as-of",
        type=_aware_timestamp,
        default=default_as_of,
        help="timezone-aware decision timestamp (default: fixture midweek scenario)",
    )
    parser.add_argument(
        "--completion-policy",
        choices=tuple(item.value for item in ContextCompletionPolicy),
        default=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF.value,
    )
    parser.add_argument(
        "--deduplication-policy",
        choices=tuple(item.value for item in AlertDeduplicationPolicy),
        default=AlertDeduplicationPolicy.DECISION_BAR.value,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    fixture = load_fixture(DEFAULT_FIXTURE_PATH)
    default_scenario = next(
        item for item in fixture.decision_scenarios if item.name == "midweek"
    )
    arguments = _parser(default_scenario.as_of).parse_args(argv)
    fixture = load_fixture(arguments.fixture)
    datasets = build_datasets(fixture, arguments.cache_root)
    completion_policy = ContextCompletionPolicy(arguments.completion_policy)
    rule = create_fixture_parity_rule(datasets, completion_policy)
    historical_reference = HistoricalPredictionStudyReference.capture(
        study_id=HISTORICAL_STUDY_ID,
        rule=rule,
        summary={
            "scope": "deterministic SPY fixture parity validation",
            "validated": True,
            "profitability_claim": False,
        },
        sample_count=2,
    )
    scanner = PredictionScanner(
        source=CachedSpyScannerSource(datasets),
        bindings=(PredictionScannerRuleBinding(rule, historical_reference),),
        sinks=(
            ConsolePredictionAlertSink(),
            JsonFilePredictionAlertSink(arguments.alert_root),
        ),
        deduplication_store=JsonFileAlertDeduplicationStore(arguments.state_root),
        deduplication_policy=AlertDeduplicationPolicy(arguments.deduplication_policy),
    )
    result = scanner.scan(as_of=arguments.as_of, dry_run=True)
    summary = {
        "alerts_emitted": len(result.alerts),
        "as_of": result.as_of.isoformat(),
        "cache_status": datasets.cache_status,
        "completion_policy": completion_policy.value,
        "dry_run": result.dry_run,
        "historical_study_id": HISTORICAL_STUDY_ID,
        "rule_configuration_id": rule.configuration_id,
    }
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
