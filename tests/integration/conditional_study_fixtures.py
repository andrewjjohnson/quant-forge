"""QF-69 fixture: the QF-69 runner on the synthetic QF-45 composition, offline.

The inputs are QF-68's labeled synthetic composition (validated canonical 1m
cache, QuantForge 2m/daily aggregation, the unchanged EMA rule and outcomes):
every session triggers once, at the first 2m bar after an 11:00 jump, and the
post-jump drift fixes the 30-minute label. The QF-69 design is patched only in
its dates and plumbing floors; the plan has all three fold roles, so QF-39
runs its trial on selection and QF-69 adds the frozen candidate's development
evidence. Synthetic plumbing only, never research evidence.

Windows (New York):

- development 2024-12-12..19: six triggers, three positive and three negative;
- selection 2024-12-20..23: two triggers (negative, positive);
- walk-forward test 2024-12-24..27: the 13:00 early close's late trigger
  (unavailable label: its target overflows the session) and two negatives;
- reserved final holdout 2024-12-30..31.
"""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, time
from pathlib import Path
from typing import Any

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.examples import spy_ema_ml_study
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.examples.spy_ema_ml_runner import run_pre_holdout
from quantforge.ml.sources import WORKSPACE_HOLDOUT_LEDGER
from quantforge.oos import HoldoutLedger
from quantforge.validation import (
    PartitionRole,
    TimestampBoundary,
    ValidationInterval,
    ValidationWindow,
)
from tests.integration.event_dataset_fixtures import NEW_YORK
from tests.integration.rapid_scan_fixtures import at
from tests.unit.experiments.test_contracts import execution

WINDOWS: dict[PartitionRole, tuple[datetime, datetime]] = {
    PartitionRole.DEVELOPMENT: (
        at("2024-12-12", time(9, 32)),
        at("2024-12-19", time(16)),
    ),
    PartitionRole.SELECTION: (
        at("2024-12-20", time(9, 32)),
        at("2024-12-23", time(16)),
    ),
    PartitionRole.WALK_FORWARD_TEST: (
        at("2024-12-24", time(9, 32)),
        at("2024-12-27", time(16)),
    ),
    PartitionRole.FINAL_HOLDOUT: (
        at("2024-12-30", time(9, 32)),
        at("2024-12-31", time(16)),
    ),
}
_CONSTANTS = {
    "DEVELOPMENT": PartitionRole.DEVELOPMENT,
    "SELECTION": PartitionRole.SELECTION,
    "TEST": PartitionRole.WALK_FORWARD_TEST,
    "HOLDOUT": PartitionRole.FINAL_HOLDOUT,
}


def _day(value: datetime) -> str:
    return value.astimezone(NEW_YORK).date().isoformat()


@contextmanager
def fixture_design() -> Generator[None]:
    """The QF-69 design with fixture dates and small plumbing floors."""
    original = spy_ema_ml_study.validation_window

    def window(name: str, role: Any, dates: tuple[str, str]) -> ValidationWindow:
        start, end = WINDOWS[role]
        return replace(
            original(name, role, dates),
            interval=ValidationInterval(
                TimestampBoundary(start), TimestampBoundary(end)
            ),
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(spy_ema_ml_study, "validation_window", window)
        for name, role in _CONSTANTS.items():
            start, end = WINDOWS[role]
            patch.setattr(spy_ema_ml_study, name, (_day(start), _day(end)))
        patch.setattr(spy_ema_ml_study, "WARMUP", ("2024-10-01", "2024-12-11"))
        patch.setattr(spy_ema_ml_study, "MINIMUM_TRAINING_OBSERVATIONS", 4)
        patch.setattr(spy_ema_ml_study, "MINIMUM_TRAINING_CLASS_OBSERVATIONS", 2)
        patch.setattr(spy_ema_ml_study, "MINIMUM_EVALUATION_OBSERVATIONS", 1)
        patch.setattr(
            spy_ema_ml_study,
            "PLANNING_EVIDENCE",
            {"fixture": "synthetic composition; no planning scan"},
        )
        yield


@dataclass(frozen=True)
class StudyRun:
    inputs: SmokeInputs
    workspace: Path
    output: Path
    gate: PrimitiveMapping
    log: tuple[str, ...]

    @property
    def ledger(self) -> HoldoutLedger:
        return HoldoutLedger(self.workspace / WORKSPACE_HOLDOUT_LEDGER)


def run_fixture_study(root: Path, inputs: SmokeInputs) -> StudyRun:
    """A fresh workspace (permanent ledger created once) and one runner call."""
    workspace = root / "workspace"
    ledger_path = workspace / WORKSPACE_HOLDOUT_LEDGER
    if not ledger_path.exists():
        HoldoutLedger.create(ledger_path)
    output = workspace / "reports" / "qf69-conditional-ml"
    lines: list[str] = []
    with fixture_design():
        gate = run_pre_holdout(
            inputs,
            output,
            workspace=workspace,
            ledger=HoldoutLedger(ledger_path),
            execution=execution("qf69-fixture"),
            log=lines.append,
        )
    return StudyRun(inputs, workspace, output, gate, tuple(lines))
