#!/usr/bin/env python3
"""Evaluate overnight-gap direction rules on Tiingo EOD SPY data.

The default source is the fixed 2020-01-01 through 2025-12-31 Tiingo request.
Set ``TIINGO_API_KEY`` or reuse an immutable cached dataset with ``--dataset-id``.
This is prediction analysis only: it creates no orders, fills, or portfolio P&L.
"""

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
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    PredictionAnalysisError,
    PredictionAnalysisResult,
    PredictionExportError,
    export_prediction_analysis,
    run_prediction_analysis,
    validate_prediction_analysis_export,
)

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
        help="root for immutable QF-11 structured exports",
    )
    parsed = parser.parse_args(arguments)
    if parsed.refresh and parsed.dataset_id:
        parser.error("--refresh applies only to a Tiingo retrieval")
    return parsed


def load_dataset(arguments: argparse.Namespace) -> tuple[MarketDataset, str]:
    """Load a prior immutable dataset or retrieve the fixed Tiingo request."""
    cache = MarketDataCache(arguments.cache_root)
    if arguments.dataset_id is not None:
        dataset = cache.load(arguments.dataset_id)
        source_label = "cached QF-3 dataset"
    else:
        api_key = os.environ.get("TIINGO_API_KEY")
        if not api_key:
            raise RequestError(
                "TIINGO_API_KEY is required; set it in the environment or pass "
                "--dataset-id for an existing cache entry"
            )
        service = MarketDataService(TiingoProvider(api_key), cache)
        dataset = service.get_daily_bars(
            SYMBOL,
            REQUESTED_START,
            REQUESTED_END,
            AdjustmentMode.UNADJUSTED,
            refresh=arguments.refresh,
        )
        source_label = "Tiingo End-of-Day"
    if dataset.metadata.canonical_symbol != SYMBOL:
        raise RequestError("the maintained baseline requires a SPY dataset")
    return dataset, source_label


def export_result(
    result: PredictionAnalysisResult, output_root: Path
) -> tuple[Path, str]:
    """Create an immutable export or verify and reuse its exact prior copy."""
    expected_path = output_root / result.analysis_id
    try:
        return export_prediction_analysis(result, output_root), "created"
    except PredictionExportError:
        if not expected_path.is_dir():
            raise
        validate_prediction_analysis_export(result, expected_path)
        return expected_path, "reused existing immutable export"


def _optional_decimal(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    dataset, source_label = load_dataset(parsed)
    result = run_prediction_analysis(
        dataset,
        OvernightGapPredictionStrategy(OvernightGapPredictionParameters()),
    )
    destination, export_status = export_result(result, parsed.output_root)
    print(
        json.dumps(
            {
                "accuracy": _optional_decimal(result.metrics.accuracy),
                "analysis_id": result.analysis_id,
                "average_gap_size_correct": _optional_decimal(
                    result.metrics.average_gap_size_correct
                ),
                "average_gap_size_incorrect": _optional_decimal(
                    result.metrics.average_gap_size_incorrect
                ),
                "correct_count": result.metrics.correct_count,
                "dataset_id": result.market_data.dataset_id,
                "export_status": export_status,
                "incorrect_count": result.metrics.incorrect_count,
                "output": str(destination.resolve()),
                "prediction_count": result.metrics.prediction_count,
                "provider": result.market_data.provider_name,
                "source": source_label,
            },
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
