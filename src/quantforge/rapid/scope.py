"""Research-scope and final-holdout guards for rapid scans (QF-72).

Rapid exploration may use only QF-8 development and selection windows. Test
windows are authoritative OOS (AGENTS.md forbids tuning on them) and the final
holdout is authoritative-only. Before any market value is read, the complete
data footprint (first warm-up bar through the last decision plus maximum
outcome reach) must avoid the plan's final holdout and every holdout exposure
scope, reserved or consumed, recorded for the symbol in the research
workspace's permanent ledger (``reports/holdout-ledger``). There is
deliberately no override, and no other ledger can be substituted.
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import cast

from quantforge.data import AggregatedSessionBar, IntradayBar
from quantforge.data.multi_timeframe import ArtifactBar
from quantforge.oos import HoldoutLedger, OOSIntegrityError
from quantforge.rapid.models import RapidHoldoutError, RapidScopeError
from quantforge.validation import (
    PartitionRole,
    TimestampBoundary,
    ValidationPlan,
    ValidationWindow,
)
from quantforge.walk_forward.models import WalkForwardPersistenceError

PERMITTED_RAPID_ROLES = (PartitionRole.DEVELOPMENT, PartitionRole.SELECTION)
# The research workspace's one permanent holdout authority (the QF-45 CLI's
# location). Rapid scans bind to it; they never accept another ledger object.
WORKSPACE_HOLDOUT_LEDGER = Path("reports", "holdout-ledger")


def workspace_holdout_ledger(workspace: Path) -> HoldoutLedger:
    """Open the workspace's permanent ledger read-only; never create one.

    Binding to the workspace location means a caller cannot substitute a fresh,
    empty ledger and thereby scan dates another study has reserved.
    """
    if not isinstance(cast(object, workspace), Path):
        raise RapidScopeError("rapid scans require the research workspace root path")
    try:
        return HoldoutLedger(workspace.resolve() / WORKSPACE_HOLDOUT_LEDGER)
    except (OOSIntegrityError, WalkForwardPersistenceError) as error:
        raise RapidScopeError(
            "rapid scans require the research workspace's permanent holdout ledger "
            f"at {WORKSPACE_HOLDOUT_LEDGER.as_posix()}; rapid scans never create one"
        ) from error


def exploratory_window(
    plan: ValidationPlan, fold_index: int, role: PartitionRole
) -> ValidationWindow:
    """Return a permitted non-holdout plan window, or refuse the request."""
    if not isinstance(cast(object, plan), ValidationPlan):
        raise RapidScopeError("rapid scans require a QF-8 validation plan")
    if not isinstance(cast(object, role), PartitionRole):
        raise RapidScopeError("rapid scan role must be a partition role")
    if role is PartitionRole.FINAL_HOLDOUT:
        raise RapidHoldoutError(
            "the reserved final holdout is authoritative-only; rapid scans never "
            "read it"
        )
    if role is PartitionRole.WALK_FORWARD_TEST:
        raise RapidScopeError(
            "walk-forward test windows are authoritative OOS; rapid exploration "
            "may not tune on them"
        )
    index = cast(object, fold_index)
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(plan.folds)
    ):
        raise RapidScopeError("rapid scan fold index is out of range")
    fold = plan.folds[index]
    window = fold.development if role is PartitionRole.DEVELOPMENT else fold.selection
    if window is None:
        raise RapidScopeError("this fold has no selection window")
    if window == plan.final_holdout.window or window.role not in PERMITTED_RAPID_ROLES:
        raise RapidHoldoutError("rapid scans cannot treat the holdout as exploration")
    if plan.prediction_membership is None or not isinstance(
        window.interval.start, TimestampBoundary
    ):
        raise RapidScopeError("rapid scans require QF-42 timestamp membership")
    return window


def bar_sessions(bar: ArtifactBar) -> tuple[date, date]:
    """First and last exchange session labels covered by one completed bar."""
    if isinstance(bar, IntradayBar):
        return bar.session_date, bar.session_date
    session_bar: AggregatedSessionBar = bar
    return session_bar.session_dates[0], session_bar.session_dates[-1]


@dataclass(frozen=True, slots=True)
class HoldoutIsolation:
    """The reserved intervals a rapid footprint must never touch."""

    symbol: str
    plan_holdout_start: datetime
    plan_holdout_end: datetime
    ledger_scopes: tuple[tuple[date, date], ...]

    @classmethod
    def capture(
        cls, plan: ValidationPlan, ledger: HoldoutLedger, *, symbol: str
    ) -> "HoldoutIsolation":
        if not isinstance(cast(object, ledger), HoldoutLedger):
            raise RapidScopeError(
                "rapid scans require the permanent holdout ledger to exclude every "
                "reserved holdout"
            )
        interval = plan.final_holdout.window.interval
        if not isinstance(interval.start, TimestampBoundary) or not isinstance(
            interval.end, TimestampBoundary
        ):
            raise RapidScopeError("rapid scans require a timestamp final holdout")
        scopes = tuple(
            (
                date.fromisoformat(cast(str, scope["start"])),
                date.fromisoformat(cast(str, scope["end"])),
            )
            for scope in ledger.exposure_scopes()
            if scope["symbol"] == symbol
        )
        return cls(symbol, interval.start.timestamp, interval.end.timestamp, scopes)

    def require_isolated(
        self,
        *,
        first_timestamp: datetime,
        last_timestamp: datetime,
        first_session: date,
        last_session: date,
    ) -> None:
        """Fail closed when the footprint intersects any reserved holdout."""
        if (
            first_timestamp <= self.plan_holdout_end
            and self.plan_holdout_start <= last_timestamp
        ):
            raise RapidHoldoutError(
                "rapid footprint overlaps the plan's reserved final holdout"
            )
        for start, end in self.ledger_scopes:
            if first_session <= end and start <= last_session:
                raise RapidHoldoutError(
                    f"rapid footprint overlaps a reserved {self.symbol} holdout "
                    f"({start.isoformat()} to {end.isoformat()}) in the ledger"
                )
