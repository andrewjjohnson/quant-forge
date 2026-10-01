"""QF-68 fixture: a real QF-39 study whose trial runs on development rows.

It reuses QF-67's synthetic QF-45 composition (validated canonical 1m cache,
QuantForge 2m/daily aggregation, the unchanged EMA rule, outcomes and QF-45
adapter), fixed to the 8/48 pair, with one change of plan shape: the fold
declares **no selection window**, so QF-39 executes and freezes its trial on
the development window and QF-67 emits development (training) rows. This is a
legitimate QF-8 fold shape; nothing is relabeled.

Windows (New York): development 2024-12-12..2024-12-20 (its first 2m
decision follows 50 completed daily warm-up bars), test 2024-12-23..2024-12-27
(with the 12:40 early-close trigger whose target overflows), final holdout
2024-12-30..31. Post-jump drifts alternate on development sessions so the
training labels hold both classes. Synthetic plumbing, never QF-45 evidence.
"""

from dataclasses import replace
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from quantforge.examples import spy_ema_plan
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.examples.spy_ema_plan import SmokePredictionEvaluator
from quantforge.validation import (
    PartitionRole,
    TimestampBoundary,
    ValidationFold,
    ValidationInterval,
    ValidationWindow,
)
from quantforge.walk_forward import WalkForwardStudy
from quantforge.walk_forward.models import FoldStatus
from tests.integration import event_dataset_fixtures
from tests.integration.event_dataset_fixtures import EventStudy, event_inputs
from tests.integration.rapid_scan_fixtures import at

DEVELOPMENT_DRIFT = {
    date(2024, 12, 12): Decimal("0.01"),
    date(2024, 12, 13): Decimal("-0.01"),
    date(2024, 12, 16): Decimal(0),
    date(2024, 12, 17): Decimal("0.01"),
    date(2024, 12, 18): Decimal("-0.01"),
    date(2024, 12, 19): Decimal("0.01"),
    date(2024, 12, 20): Decimal("-0.01"),
}
WINDOWS: dict[PartitionRole, tuple[datetime, datetime]] = {
    PartitionRole.DEVELOPMENT: (
        at("2024-12-12", time(9, 32)),
        at("2024-12-20", time(16)),
    ),
    PartitionRole.WALK_FORWARD_TEST: (
        at("2024-12-23", time(9, 32)),
        at("2024-12-27", time(16)),
    ),
    PartitionRole.FINAL_HOLDOUT: (
        at("2024-12-30", time(9, 32)),
        at("2024-12-31", time(16)),
    ),
}


def development_inputs(root: Path) -> SmokeInputs:
    """QF-67 event inputs with labeled post-trigger paths on development days."""
    with pytest.MonkeyPatch.context() as patch:
        for day, drift in DEVELOPMENT_DRIFT.items():
            patch.setitem(event_dataset_fixtures.DRIFT, day, drift)
        return event_inputs(root)


def run_development_study(root: Path, inputs: SmokeInputs) -> EventStudy:
    """Run QF-39 (fixed 8/48, schema-4 windows) on a fold without selection."""
    original = spy_ema_plan.validation_window

    def day(value: datetime) -> str:
        return value.astimezone(event_dataset_fixtures.NEW_YORK).date().isoformat()

    def window(name: str, role: Any, dates: tuple[str, str]) -> ValidationWindow:
        start, end = WINDOWS.get(role, WINDOWS[PartitionRole.DEVELOPMENT])
        return replace(
            original(name, role, dates),
            interval=ValidationInterval(
                TimestampBoundary(start), TimestampBoundary(end)
            ),
        )

    def fold(
        name: str,
        development: ValidationWindow,
        test: ValidationWindow,
        selection: ValidationWindow | None = None,
    ) -> ValidationFold:
        del selection  # This fold declares no selection window.
        return ValidationFold(name, development, test)

    development = WINDOWS[PartitionRole.DEVELOPMENT]
    test = WINDOWS[PartitionRole.WALK_FORWARD_TEST]
    holdout = WINDOWS[PartitionRole.FINAL_HOLDOUT]
    splits = (
        (day(development[0]), day(development[1])),
        (day(development[0]), day(development[1])),
        (day(test[0]), day(test[1])),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(spy_ema_plan, "SPLITS", (splits,))
        patch.setattr(spy_ema_plan, "HOLDOUT", (day(holdout[0]), day(holdout[1])))
        patch.setattr(spy_ema_plan, "validation_window", window)
        patch.setattr(spy_ema_plan, "ValidationFold", fold)
        config, base = spy_ema_plan.prepare_walk_forward(
            inputs, root / "grid", fixed=True
        )
    assert config.plan.folds[0].selection is None
    adapter = SmokePredictionEvaluator(
        dataset=base.dataset,
        series=base.series,
        primary_timeframe=base.primary_timeframe,
        study_factory=base.factory,
        analyzer=base.analyzer,
        indicator_backend=base.backend,
        grid_config=replace(base.grid_config, window_schema_version="4"),
    )
    study = WalkForwardStudy(config, adapter, root / "walk-forward")
    result = study.run()
    assert [item.status for item in result.folds] == [FoldStatus.COMPLETED]
    return EventStudy(inputs, config, adapter, study.study_path, root, "4")
