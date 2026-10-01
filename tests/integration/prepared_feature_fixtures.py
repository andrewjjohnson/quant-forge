"""QF-45-like offline inputs for QF-63 prepared-context tests.

This is the real QF-45 composition: validated canonical 1m cache, QuantForge 2m
and daily aggregation, the actual EMA smoke rule/factory and ``spy_ema_plan``'s
QF-8/QF-39 wiring with 61 two-minute and 50 daily warm-up bars. Only the prices
are synthetic by construction (never repaired market observations): each session
dips before 11:00 New York and jumps at 11:00, so the actual rule emits midday
candidates while daily closes rise above the daily EMA50.

The selection window runs from 15:30 on one session through 11:06 on the next.
It therefore contains candidate and no-candidate decisions, a completed daily
bar at the first close (QF-45's rolling daily start shift), and a session change.
"""

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quantforge.data import DatasetFamily, IntradayBar, TimeframeBarSeries
from quantforge.examples import spy_ema_plan
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.prediction import PredictionDecisionSchedule
from quantforge.prediction.study import (
    PredictionStudyDatasetSession,
    prepare_prediction_study_dataset,
)
from quantforge.timeframes import resolve_exchange_session
from quantforge.validation import (
    PartitionRole,
    TimestampBoundary,
    ValidationInterval,
    ValidationWindow,
)
from quantforge.walk_forward import PredictionEvaluator, WalkForwardConfig
from quantforge.walk_forward.partitions import PermittedPartition
from quantforge.walk_forward.prediction import (
    _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration import test_intraday_prediction_provenance as source_fixture

NEW_YORK = ZoneInfo("America/New_York")
SESSION_COUNT = 56
SELECTION_START = time(15, 30)
SELECTION_END = time(11, 6)


def smoke_inputs(
    root: Path,
    session_count: int = SESSION_COUNT,
    *,
    first_session: date = date(2025, 1, 2),
) -> SmokeInputs:
    """Validated cache-backed inputs; 51+ completed daily bars precede selection."""
    policy = spy_ema_plan.TWO_MINUTES.session_policy
    sessions: list[date] = []
    current = first_session
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
        clock = result.end_timestamp.astimezone(NEW_YORK).time()
        level = Decimal(100 + sessions.index(result.session_date))
        if clock >= time(11):
            level += 30
        elif clock >= time(10, 20):
            level -= 1
        return replace(
            result,
            open=level,
            close=level,
            high=level + Decimal("0.2"),
            low=level - Decimal("0.2"),
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_fixture, "DatasetFamily", family)
        patch.setattr(source_fixture, "IntradayBar", bar)
        fixture = source_fixture.cached_fixture(
            root, provider="massive", session_dates=tuple(sessions)
        )
    return SmokeInputs(
        fixture.source, fixture.dataset, fixture.primary, fixture.daily, families[-1]
    )


def _at(day: str, clock: time) -> datetime:
    return datetime.combine(date.fromisoformat(day), clock, NEW_YORK).astimezone(UTC)


def smoke_plan(
    inputs: SmokeInputs, root: Path, *, fixed: bool, primary_warm_up: int = 61
) -> tuple[WalkForwardConfig, PredictionEvaluator]:
    """The unchanged QF-45 plan with bounded windows around the last sessions.

    ``primary_warm_up`` > 61 only enlarges the declared 2m history (for the
    large-history performance fixture); rule, grid and outcomes are unchanged.
    """
    sessions = [bar.session_date.isoformat() for bar in inputs.dataset.bars]
    development, first, second, test, holdout = sessions[-5:]
    original = spy_ema_plan.validation_window

    def window(name: str, role: Any, dates: tuple[str, str]) -> ValidationWindow:
        result = original(name, role, dates)
        if role is PartitionRole.SELECTION:
            interval = ValidationInterval(
                TimestampBoundary(_at(dates[0], SELECTION_START)),
                TimestampBoundary(_at(dates[1], SELECTION_END)),
            )
        else:
            start = _at(dates[0], time(11))
            interval = ValidationInterval(
                TimestampBoundary(start),
                TimestampBoundary(start + timedelta(minutes=4)),
            )
        warm_up = tuple(
            replace(item, observations=primary_warm_up)
            if item.timeframe == spy_ema_plan.TWO_MINUTES
            else item
            for item in result.warm_up_by_timeframe
        )
        return replace(result, interval=interval, warm_up_by_timeframe=warm_up)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            spy_ema_plan,
            "SPLITS",
            (((development, development), (first, second), (test, test)),),
        )
        patch.setattr(spy_ema_plan, "HOLDOUT", (holdout, holdout))
        patch.setattr(spy_ema_plan, "validation_window", window)
        return spy_ema_plan.prepare_walk_forward(inputs, root, fixed=fixed)


@dataclass(frozen=True)
class SmokeScope:
    """One QF-39 selection partition with helpers for both provider paths."""

    inputs: SmokeInputs
    config: WalkForwardConfig
    adapter: PredictionEvaluator
    permitted: PermittedPartition
    schedule: PredictionDecisionSchedule

    def provider(
        self,
        *,
        reference: bool = False,
        series: tuple[TimeframeBarSeries, ...] | None = None,
    ) -> _PermittedContextProvider:
        provider = _PermittedContextProvider(
            self.config.plan,
            self.permitted,
            self.adapter.series if series is None else series,
            self.schedule,
        )
        if reference:
            # QF-59 indexed selection plus the unchanged QF-20/QF-28 reference.
            object.__setattr__(provider, "_prepared_features", None)
        else:
            assert provider._prepared_features is not None  # pyright: ignore[reportPrivateUsage]
        return provider

    def session(self) -> PredictionStudyDatasetSession:
        return prepare_prediction_study_dataset(self.permitted.dataset)

    def study(self, pair: str = "8/48") -> Any:
        return self.adapter.factory.build({"ema_pair": pair})

    def arguments(self, provider: _PermittedContextProvider) -> dict[str, Any]:
        return {
            "schedule": self.schedule,
            "context_provider": provider,
            "dataset_family_fingerprint": self.adapter.series[
                0
            ].dataset_reference.family_id,
            "context_environment": self.adapter._environment(  # pyright: ignore[reportPrivateUsage]
                self.config.plan, self.permitted
            ).to_primitive(),
        }


def smoke_scope(
    inputs: SmokeInputs,
    root: Path,
    *,
    fixed: bool = True,
    primary_warm_up: int = 61,
) -> SmokeScope:
    config, adapter = smoke_plan(
        inputs, root, fixed=fixed, primary_warm_up=primary_warm_up
    )
    adapter.validate(config.plan)
    permitted = adapter._partition(config, 0, test=False)  # pyright: ignore[reportPrivateUsage]
    schedule = adapter._schedule(permitted)  # pyright: ignore[reportPrivateUsage]
    return SmokeScope(inputs, config, adapter, permitted, schedule)
