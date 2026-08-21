#!/usr/bin/env python3
"""Render the fixed QF-33 SPY prediction as a standalone QF-34 report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__:
    from scripts.export_spy_multi_timeframe_context import (
        DEFAULT_CACHE_ROOT,
        DEFAULT_FIXTURE_PATH,
        build_datasets,
        load_fixture,
    )
    from scripts.scan_spy_predictions import (
        CachedSpyScannerSource,
        create_fixture_parity_rule,
        load_historical_study_reference,
    )
else:
    from export_spy_multi_timeframe_context import (
        DEFAULT_CACHE_ROOT,
        DEFAULT_FIXTURE_PATH,
        build_datasets,
        load_fixture,
    )
    from scan_spy_predictions import (
        CachedSpyScannerSource,
        create_fixture_parity_rule,
        load_historical_study_reference,
    )

from quantforge.data import ContextCompletionPolicy
from quantforge.prediction import build_prediction_rule_context
from quantforge.reporting import (
    FutureOutcomeRegion,
    StudyInspectionReportConfig,
    StudyInspectionSelection,
    build_study_inspection_report,
    export_study_inspection_report,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "reports" / "qf34-spy-study-inspection"


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--as-of",
        type=_aware_timestamp,
        default=None,
        help="timezone-aware decision timestamp (default: fixture midweek scenario)",
    )
    parser.add_argument(
        "--completion-policy",
        choices=tuple(item.value for item in ContextCompletionPolicy),
        default=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF.value,
    )
    parser.add_argument("--max-bars-per-panel", type=int, default=60)
    parser.add_argument(
        "--exclude-primary-timeframe",
        action="store_true",
        help="omit the optional 5-minute primary panel",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    fixture = load_fixture(arguments.fixture)
    as_of = arguments.as_of
    if as_of is None:
        try:
            as_of = next(
                item.as_of
                for item in fixture.decision_scenarios
                if item.name == "midweek"
            )
        except StopIteration:
            parser.error("--as-of is required when the fixture has no midweek scenario")
    completion_policy = ContextCompletionPolicy(arguments.completion_policy)
    datasets = build_datasets(fixture, arguments.cache_root)
    rule = create_fixture_parity_rule(datasets, completion_policy)
    historical_study = load_historical_study_reference(completion_policy)
    snapshot = CachedSpyScannerSource(datasets).prepare_context(
        rule.context_requirements,
        as_of=as_of,
        refresh=False,
    )
    context = build_prediction_rule_context(
        rule.context_requirements,
        snapshot.context,
        prediction_dataset_id=snapshot.prediction_dataset_id,
        symbol=snapshot.symbol,
        prediction_adjustment_basis=snapshot.adjustment_basis,
    )
    evaluation = rule.evaluate(context)
    decision_timestamp = context.latest_bar_for(
        context.requirements.primary.timeframe
    ).end_timestamp
    next_bar = next(
        (
            bar
            for bar in datasets.source.bars
            if bar.start_timestamp >= decision_timestamp
        ),
        None,
    )
    future = (
        None
        if next_bar is None
        else FutureOutcomeRegion(
            decision_timestamp,
            next_bar.end_timestamp,
            (
                "First post-decision fixture interval; explicitly excluded from "
                "the causal chart state"
            ),
        )
    )
    report = build_study_inspection_report(
        (
            StudyInspectionSelection(
                "fixed_spy_prediction",
                context,
                rule,
                evaluation,
                historical_study,
                datasets.family,
                future,
            ),
        ),
        config=StudyInspectionReportConfig(
            max_bars_per_panel=arguments.max_bars_per_panel,
            include_primary_timeframe=not arguments.exclude_primary_timeframe,
        ),
    )
    destination, status = export_study_inspection_report(report, arguments.output_root)
    print(
        json.dumps(
            {
                "as_of": context.as_of.isoformat(),
                "cache_status": datasets.cache_status,
                "completion_policy": completion_policy.value,
                "historical_study_id": historical_study.study_id,
                "prediction_direction": evaluation.outcome.value,
                "report_id": report.report_id,
                "report_path": str(destination / "report.html"),
                "status": status.value,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
