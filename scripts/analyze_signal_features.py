#!/usr/bin/env python3
"""Build QF-7 SPY features and compare three contexts across outcome groups."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from quantforge.data import MarketDataCache
from quantforge.prediction import (
    OvernightGapPredictionParameters,
    OvernightGapSignalFeatureRule,
    PredictionStudy,
    SignalFeatureCandidate,
    build_signal_feature_dataset,
    default_overnight_gap_contextual_features,
    excursion_outcome,
    forward_return_outcome,
    target_stop_outcome,
)
from quantforge.prediction.feature_analysis import (
    WinnerDefinition,
    analyze_signal_features,
    default_overnight_gap_feature_bins,
    export_signal_feature_analysis,
)
from quantforge.prediction.feature_outcomes import ForwardReturnValues

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = (
    "feature_atr_percentage_of_close",
    "feature_trend_distance_percentage",
    "feature_volume_ratio",
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="load this immutable QF-3 dataset from --cache-root",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "market-data",
        help="QF-3 cache root",
    )
    parser.add_argument(
        "--feature-output-root",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "features",
        help="root for resumable QF-7 datasets",
    )
    parser.add_argument(
        "--analysis-output-root",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "feature-analysis",
        help="root for exploratory analysis JSON",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    dataset = MarketDataCache(parsed.cache_root).load(parsed.dataset_id)
    rule = OvernightGapSignalFeatureRule(OvernightGapPredictionParameters())
    primary = forward_return_outcome(1)
    prediction_study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(
        rule,
        primary.labeler,
        primary.evaluator,
        feature_configuration={
            "feature_schema_version": "1",
            "source": "qf7_signal_feature_snapshot",
        },
    )
    outcomes = (
        primary,
        forward_return_outcome(2),
        forward_return_outcome(5),
        forward_return_outcome(10),
        forward_return_outcome(20),
        excursion_outcome(5),
        target_stop_outcome(5, Decimal("0.01"), Decimal("0.005")),
    )
    feature_dataset = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=prediction_study,
        contextual_features=default_overnight_gap_contextual_features(),
        outcomes=outcomes,
        output_root=parsed.feature_output_root,
    )
    analysis = analyze_signal_features(
        feature_dataset,
        feature_names=FEATURE_NAMES,
        outcome_name="outcome_forward_return_5_raw_return",
        winner_definition=WinnerDefinition.DECIMAL_GREATER_THAN_ZERO,
        bins=default_overnight_gap_feature_bins(),
    )
    analysis_path = export_signal_feature_analysis(
        analysis, parsed.analysis_output_root
    )
    summary = analysis.to_primitive()
    summary["feature_dataset_path"] = str(
        (parsed.feature_output_root / feature_dataset.dataset_id).resolve()
    )
    summary["analysis_path"] = str(analysis_path.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
