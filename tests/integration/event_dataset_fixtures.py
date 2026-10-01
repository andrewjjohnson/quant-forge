"""QF-67 fixtures: real QF-39 studies of the QF-45 composition, offline.

The inputs follow QF-72's synthetic QF-45 cache (validated canonical 1m
source, QuantForge 2m/daily aggregation, the unchanged EMA rule, plan, outcomes
and QF-45 comparison adapter) over 64 XNYS sessions. Synthetic prices dip for
40 minutes and then jump, so every pair triggers once per session, at the first
2m bar ending after the jump. After the jump each session drifts up, stays
flat or drifts down, giving positive, zero (negative) and negative 30-minute
returns. On the 2024-12-24 13:00 early close the jump is at 12:40, so the
trigger's 30-minute target overflows the session (an unavailable label).
These are synthetic plumbing fixtures, never QF-45 research evidence.

Windows (fold 0): development 2024-12-19..20 (not executed: a selection window
exists), selection 2024-12-23 09:32 .. 2024-12-26 12:00 (2024-12-24 13:00 early
close, 2024-12-25 holiday), test 2024-12-27, final holdout 2024-12-30..31.
"""

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data import DatasetFamily, IntradayBar
from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples import spy_ema_plan
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.examples.spy_ema_ml_dataset import (
    ema_event_feature_schema,
    ema_forward_return_target,
)
from quantforge.examples.spy_ema_plan import SmokePredictionEvaluator
from quantforge.ml import EventDataset, build_event_dataset
from quantforge.ml.sources import WORKSPACE_HOLDOUT_LEDGER
from quantforge.oos import HoldoutLedger, OOSSource, load_oos_source
from quantforge.timeframes import resolve_exchange_session
from quantforge.walk_forward import (
    PredictionEvaluator,
    WalkForwardConfig,
    WalkForwardStudy,
)
from quantforge.walk_forward.models import FoldStatus
from tests.integration import test_intraday_prediction_provenance as source_fixture
from tests.integration.rapid_scan_fixtures import (
    FIRST_SESSION,
    SESSION_COUNT,
    windowed_plan,
)

NEW_YORK = ZoneInfo("America/New_York")
# Post-jump drift per minute (ratio of 0.01 price units), by session.
DRIFT = {
    date(2024, 12, 23): Decimal("0.01"),  # positive 30m return
    date(2024, 12, 26): Decimal(0),  # exactly zero: a negative label
    date(2024, 12, 27): Decimal("-0.01"),  # negative 30m return
    date(2024, 12, 30): Decimal("0.01"),
    date(2024, 12, 31): Decimal("-0.01"),
}
# Jump clocks; the early close's late jump overflows the 30m target.
JUMP = {date(2024, 12, 24): time(12, 40)}


@dataclass(frozen=True)
class EventStudy:
    inputs: SmokeInputs
    config: WalkForwardConfig
    adapter: PredictionEvaluator
    study_path: Path
    root: Path
    schema: str

    def workspace(self, name: str = "workspace") -> Path:
        """A research workspace whose permanent ledger exists (created once)."""
        workspace = self.root / name
        ledger = workspace / WORKSPACE_HOLDOUT_LEDGER
        if not ledger.exists():
            HoldoutLedger.create(ledger)
        return workspace

    def ledger(self, workspace: Path) -> HoldoutLedger:
        return HoldoutLedger(workspace / WORKSPACE_HOLDOUT_LEDGER)

    def source(self) -> OOSSource:
        return load_oos_source(self.config.plan, self.study_path)

    def combination(self, pair: str) -> str:
        return next(
            item.combination_id
            for item in self.adapter.universe.candidates
            if item.parameters.to_primitive() == {"ema_pair": pair}
        )

    @property
    def selected_pair(self) -> str:
        frozen = self.source().folds[0].selection
        assert frozen is not None
        candidate = cast(PrimitiveMapping, frozen.snapshot.to_primitive()["candidate"])
        return cast(str, cast(PrimitiveMapping, candidate["parameters"])["ema_pair"])

    def build(
        self,
        *,
        workspace: Path | None = None,
        pair: str | None = None,
        study_path: Path | None = None,
        **options: Any,
    ) -> EventDataset:
        return build_event_dataset(
            plan=self.config.plan,
            study_path=self.study_path if study_path is None else study_path,
            workspace=self.workspace() if workspace is None else workspace,
            combination_id=self.combination(pair or self.selected_pair),
            feature_schema=options.pop("feature_schema", ema_event_feature_schema()),
            target=options.pop("target", ema_forward_return_target()),
            **options,
        )


def run_event_study(
    root: Path, inputs: SmokeInputs, *, schema: str, fixed: bool = False
) -> EventStudy:
    """Run the unchanged QF-45 comparison (or fixed 8/48) with ``schema`` windows."""
    config, base = windowed_plan(inputs, root / f"grid-{schema}", fixed=fixed)
    adapter = SmokePredictionEvaluator(
        dataset=base.dataset,
        series=base.series,
        primary_timeframe=base.primary_timeframe,
        study_factory=base.factory,
        analyzer=base.analyzer,
        indicator_backend=base.backend,
        grid_config=replace(base.grid_config, window_schema_version=schema),
    )
    study = WalkForwardStudy(config, adapter, root / f"walk-forward-{schema}")
    result = study.run()
    assert [fold.status for fold in result.folds] == [FoldStatus.COMPLETED]
    return EventStudy(inputs, config, adapter, study.study_path, root, schema)


def event_inputs(root: Path, session_count: int = SESSION_COUNT) -> SmokeInputs:
    """Validated cache-backed QF-45 inputs with labeled post-trigger paths."""
    policy = spy_ema_plan.TWO_MINUTES.session_policy
    sessions: list[date] = []
    current = FIRST_SESSION
    while len(sessions) < session_count:
        try:
            resolve_exchange_session(current, policy)
        except ValueError:
            pass
        else:
            sessions.append(current)
        current += timedelta(days=1)
    families: list[DatasetFamily] = []

    def family(*args: Any, **kwargs: Any) -> DatasetFamily:
        result = DatasetFamily(*args, **kwargs)
        families.append(result)
        return result

    def bar(*args: Any, **kwargs: Any) -> IntradayBar:
        result = IntradayBar(*args, **kwargs)
        local = result.end_timestamp.astimezone(NEW_YORK)
        session = result.session_date
        jump = datetime.combine(session, JUMP.get(session, time(11)), NEW_YORK)
        level = Decimal(100 + sessions.index(session))
        if local >= jump:
            minutes = int((local - jump).total_seconds() // 60)
            level += 30 + DRIFT.get(session, Decimal(0)) * minutes
        elif local >= jump - timedelta(minutes=40):
            level -= 1
        return replace(
            result,
            open=level,
            close=level,
            high=level + Decimal("0.2"),
            low=level - Decimal("0.2"),
        )

    # Build inside a QF-65 session only to authenticate the fixture cache once.
    with canonical_preparation(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_fixture, "DatasetFamily", family)
        patch.setattr(source_fixture, "IntradayBar", bar)
        fixture = source_fixture.cached_fixture(
            root / "cache", provider="massive", session_dates=tuple(sessions)
        )
    return SmokeInputs(
        fixture.source, fixture.dataset, fixture.primary, fixture.daily, families[-1]
    )
