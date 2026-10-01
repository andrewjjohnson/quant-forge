"""QF-72 rapid-scan fixtures over the real QF-45 composition and calendar edges.

The inputs are the QF-63 synthetic QF-45 cache (validated canonical 1m source,
QuantForge 2m/daily aggregation, the actual EMA rule, plan and outcomes) over
64 XNYS sessions from 2024-10-01. Every session dips before 11:00 New York and
jumps at 11:00, so the unchanged rule emits one midday candidate per session.

Windows (fold 0), chosen so 61 2m and 50 daily warm-up bars precede them:

- development: 2024-12-19 09:32 through 2024-12-20 16:00 (normal sessions);
- selection: 2024-12-23 09:32 through 2024-12-26 12:00, crossing a daily close
  (the rolling daily start shift), the 13:00 early close on 2024-12-24 and the
  2024-12-25 holiday gap;
- test: 2024-12-27; reserved final holdout: 2024-12-30 through 2024-12-31.
"""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quantforge.data import TimeframeBarSeries
from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples import spy_ema_plan
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.oos import HoldoutLedger
from quantforge.rapid import RapidResearchSession, rapid_research_session
from quantforge.rapid.scope import WORKSPACE_HOLDOUT_LEDGER
from quantforge.validation import (
    PartitionRole,
    TimestampBoundary,
    ValidationInterval,
    ValidationWindow,
)
from quantforge.walk_forward import PredictionEvaluator, WalkForwardConfig
from quantforge.walk_forward.persistence import write_record
from tests.integration.prepared_feature_fixtures import smoke_inputs

NEW_YORK = ZoneInfo("America/New_York")
FIRST_SESSION = date(2024, 10, 1)
SESSION_COUNT = 64


def at(day: str, clock: time) -> datetime:
    return datetime.combine(date.fromisoformat(day), clock, NEW_YORK).astimezone(UTC)


WINDOWS: dict[PartitionRole, tuple[datetime, datetime]] = {
    PartitionRole.DEVELOPMENT: (
        at("2024-12-19", time(9, 32)),
        at("2024-12-20", time(16)),
    ),
    PartitionRole.SELECTION: (
        at("2024-12-23", time(9, 32)),
        at("2024-12-26", time(12)),
    ),
    PartitionRole.WALK_FORWARD_TEST: (
        at("2024-12-27", time(9, 32)),
        at("2024-12-27", time(16)),
    ),
    PartitionRole.FINAL_HOLDOUT: (
        at("2024-12-30", time(9, 32)),
        at("2024-12-31", time(16)),
    ),
}


def calendar_inputs(root: Path) -> SmokeInputs:
    # Build inside a QF-65 session only to authenticate the fixture cache once.
    with canonical_preparation():
        return smoke_inputs(root, SESSION_COUNT, first_session=FIRST_SESSION)


def windowed_plan(
    inputs: SmokeInputs, root: Path, *, fixed: bool = True
) -> tuple[WalkForwardConfig, PredictionEvaluator]:
    """The unchanged QF-45 plan wiring with explicit timestamp windows."""

    def day(value: datetime) -> str:
        return value.astimezone(NEW_YORK).date().isoformat()

    original = spy_ema_plan.validation_window

    def window(name: str, role: Any, dates: tuple[str, str]) -> ValidationWindow:
        start, end = WINDOWS[role]
        return replace(
            original(name, role, dates),
            interval=ValidationInterval(
                TimestampBoundary(start), TimestampBoundary(end)
            ),
        )

    splits = tuple(
        (day(WINDOWS[role][0]), day(WINDOWS[role][1]))
        for role in (
            PartitionRole.DEVELOPMENT,
            PartitionRole.SELECTION,
            PartitionRole.WALK_FORWARD_TEST,
        )
    )
    holdout = WINDOWS[PartitionRole.FINAL_HOLDOUT]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(spy_ema_plan, "SPLITS", (splits,))
        patch.setattr(spy_ema_plan, "HOLDOUT", (day(holdout[0]), day(holdout[1])))
        patch.setattr(spy_ema_plan, "validation_window", window)
        return spy_ema_plan.prepare_walk_forward(inputs, root, fixed=fixed)


def reserve_scope(
    ledger: HoldoutLedger, lineage: str, *, start: date, end: date, symbol: str = "SPY"
) -> None:
    """Record a foreign lineage's reserved holdout exactly as the ledger stores it."""
    write_record(
        ledger.root / "lineages" / lineage / "reservation.json",
        {
            "schema_version": "1",
            "lineage_id": lineage,
            "lineage": {"fixture": "qf72 foreign reservation"},
            "exposure_scope": {
                "date_basis": "exchange_session_labels_v1",
                "symbol": symbol,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        },
        immutable=True,
    )


@dataclass(frozen=True)
class RapidCase:
    inputs: SmokeInputs
    config: WalkForwardConfig
    adapter: PredictionEvaluator
    root: Path

    def workspace(self, name: str = "workspace") -> Path:
        """A research workspace whose permanent ledger exists (created once)."""
        workspace = self.root / name
        ledger = workspace / WORKSPACE_HOLDOUT_LEDGER
        if not ledger.exists():
            HoldoutLedger.create(ledger)
        return workspace

    def ledger(self, workspace: Path) -> HoldoutLedger:
        return HoldoutLedger(workspace / WORKSPACE_HOLDOUT_LEDGER)

    @contextmanager
    def session(
        self,
        role: PartitionRole = PartitionRole.SELECTION,
        *,
        workspace: Path | None = None,
        series: tuple[TimeframeBarSeries, ...] | None = None,
    ) -> Generator[RapidResearchSession]:
        with rapid_research_session(
            plan=self.config.plan,
            fold_index=0,
            role=role,
            dataset=self.inputs.dataset,
            series=(self.inputs.primary, self.inputs.daily)
            if series is None
            else series,
            workspace=self.workspace() if workspace is None else workspace,
        ) as session:
            yield session


def rapid_case(root: Path) -> RapidCase:
    inputs = calendar_inputs(root / "cache")
    config, adapter = windowed_plan(inputs, root / "authoritative")
    adapter.validate(config.plan)
    return RapidCase(inputs, config, adapter, root)
