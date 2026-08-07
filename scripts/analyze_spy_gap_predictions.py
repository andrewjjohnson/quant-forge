#!/usr/bin/env python3
"""Compare QF-11 overnight-gap strategies on the fixed Tiingo SPY dataset."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path

from quantforge.data import (
    AdjustmentMode,
    MarketDataCache,
    MarketDataError,
    MarketDataService,
    MarketDataset,
    RequestError,
)
from quantforge.data.providers import TiingoProvider
from quantforge.prediction import (
    ALL_REASONS,
    PredictionAnalysisError,
    PredictionComparisonParameters,
    PredictionComparisonStudyResult,
    PredictionExportError,
    export_prediction_comparison,
    run_prediction_comparison,
    validate_prediction_comparison_export,
)
from quantforge.prediction.comparison_models import AnnualSummary, WeekdaySummary

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "SPY"
REQUESTED_START = date(2020, 1, 1)
REQUESTED_END = date(2025, 12, 31)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id", help="load this immutable dataset ID from --cache-root"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="retrieve a new immutable Tiingo snapshot and advance the cache index",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "market-data",
        help="QF-3 cache root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "predictions",
        help="root for immutable QF-11 comparison exports",
    )
    parsed = parser.parse_args(arguments)
    if parsed.refresh and parsed.dataset_id:
        parser.error("--refresh applies only to a Tiingo retrieval")
    return parsed


def load_dataset(arguments: argparse.Namespace) -> tuple[MarketDataset, str]:
    cache = MarketDataCache(arguments.cache_root)
    if arguments.dataset_id is not None:
        dataset = cache.load(arguments.dataset_id)
        source = "cached QF-3 dataset"
    else:
        api_key = os.environ.get("TIINGO_API_KEY")
        if not api_key:
            raise RequestError(
                "TIINGO_API_KEY is required; set it in the environment or pass "
                "--dataset-id for an existing cache entry"
            )
        dataset = MarketDataService(TiingoProvider(api_key), cache).get_daily_bars(
            SYMBOL,
            REQUESTED_START,
            REQUESTED_END,
            AdjustmentMode.UNADJUSTED,
            refresh=arguments.refresh,
        )
        source = "Tiingo End-of-Day"
    _validate_maintained_dataset(dataset)
    return dataset, source


def _validate_maintained_dataset(dataset: MarketDataset) -> None:
    metadata = dataset.metadata
    mismatches: list[str] = []
    if metadata.canonical_symbol != SYMBOL:
        mismatches.append(f"symbol={metadata.canonical_symbol}")
    if metadata.provider_name != TiingoProvider.name:
        mismatches.append(f"provider={metadata.provider_name}")
    if metadata.requested_start != REQUESTED_START:
        mismatches.append(f"requested_start={metadata.requested_start.isoformat()}")
    if metadata.requested_end != REQUESTED_END:
        mismatches.append(f"requested_end={metadata.requested_end.isoformat()}")
    if metadata.adjustment_mode is not AdjustmentMode.UNADJUSTED:
        mismatches.append(f"adjustment={metadata.adjustment_mode.value}")
    if mismatches:
        raise RequestError(
            "dataset does not match the maintained unadjusted Tiingo SPY request "
            f"from {REQUESTED_START.isoformat()} through {REQUESTED_END.isoformat()}: "
            + ", ".join(mismatches)
        )


def export_result(
    result: PredictionComparisonStudyResult, output_root: Path
) -> tuple[Path, str]:
    expected = output_root / result.study_id
    try:
        return export_prediction_comparison(result, output_root), "created"
    except PredictionExportError:
        if not expected.is_dir():
            raise
        validate_prediction_comparison_export(result, expected)
        return expected, "reused existing immutable export"


def build_summary(
    result: PredictionComparisonStudyResult,
    source: str,
    destination: Path,
    export_status: str,
) -> dict[str, object]:
    matched = {
        item.configuration_name: item
        for item in result.baseline_comparisons
        if item.comparison_scope == "matched_prediction_sessions"
    }
    configuration_rows: list[dict[str, object]] = []
    for item in result.configuration_summaries:
        comparison = matched[item.configuration_name]
        configuration_rows.append(
            {
                "accuracy": _decimal(item.metrics.accuracy),
                "average_signed_return": _decimal(
                    item.metrics.average_signed_prediction_return
                ),
                "configuration_name": item.configuration_name,
                "matched_always_up_accuracy": _decimal(
                    comparison.baseline_metrics.accuracy
                ),
                "matched_incremental_accuracy": _decimal(
                    comparison.incremental_accuracy
                ),
                "matched_incremental_signed_return": _decimal(
                    comparison.average_incremental_signed_return
                ),
                "prediction_count": item.metrics.prediction_count,
            }
        )
    annual = tuple(
        item
        for item in result.annual_summaries
        if item.reason == ALL_REASONS and item.metrics.prediction_count
    )
    weekdays = tuple(
        item
        for item in result.weekday_summaries
        if item.reason == ALL_REASONS and item.metrics.prediction_count
    )
    thresholds = tuple(
        item for item in result.threshold_summaries if item.segment_type == "full"
    )
    return {
        "configurations": configuration_rows,
        "dataset_id": result.market_data.dataset_id,
        "date_range": {
            "first": result.market_data.actual_first_session.isoformat(),
            "last": result.market_data.actual_last_session.isoformat(),
        },
        "export_status": export_status,
        "output": str(destination.resolve()),
        "source": source,
        "strongest_years": _extremes(annual, strongest=True),
        "weakest_years": _extremes(annual, strongest=False),
        "strongest_weekdays": _extremes(weekdays, strongest=True),
        "weakest_weekdays": _extremes(weekdays, strongest=False),
        "study_id": result.study_id,
        "threshold_sensitivity": [
            {
                "accuracy": _decimal(item.metrics.accuracy),
                "average_signed_return": _decimal(
                    item.metrics.average_signed_prediction_return
                ),
                "prediction_count": item.metrics.prediction_count,
                "stability_assessment": item.stability_assessment,
                "threshold": str(item.threshold),
            }
            for item in thresholds
        ],
        "warning": (
            "Exploratory completed-session prediction analysis only; this is not "
            "an executable close-fill backtest or evidence of options profitability."
        ),
    }


def _extremes(
    values: Sequence[AnnualSummary | WeekdaySummary], *, strongest: bool
) -> list[dict[str, str | None]]:
    grouped: dict[str, list[AnnualSummary | WeekdaySummary]] = {}
    for value in values:
        grouped.setdefault(value.configuration_name, []).append(value)
    result: list[dict[str, str | None]] = []
    for configuration_name, items in sorted(grouped.items()):
        eligible = tuple(
            item
            for item in items
            if item.metrics.average_signed_prediction_return is not None
        )
        if not eligible:
            continue
        selected = sorted(
            eligible,
            key=lambda item: (
                item.metrics.average_signed_prediction_return,
                item.year if isinstance(item, AnnualSummary) else item.weekday,
            ),
            reverse=strongest,
        )[0]
        result.append(
            {
                "average_signed_return": _decimal(
                    selected.metrics.average_signed_prediction_return
                ),
                "configuration_name": configuration_name,
                "group": (
                    str(selected.year)
                    if isinstance(selected, AnnualSummary)
                    else selected.weekday_name
                ),
            }
        )
    return result


def _decimal(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    dataset, source = load_dataset(parsed)
    result = run_prediction_comparison(dataset, PredictionComparisonParameters())
    destination, export_status = export_result(result, parsed.output_root)
    print(
        json.dumps(
            build_summary(result, source, destination, export_status),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MarketDataError as error:
        raise SystemExit(f"market-data error: {error}") from None
    except PredictionAnalysisError as error:
        raise SystemExit(f"prediction-analysis error: {error}") from None
