"""Bounded synthetic acceptance of the actual QF-45 pre-holdout composition."""

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

from quantforge.data import DatasetFamily, IntradayBar
from quantforge.examples import spy_ema_plan, spy_ema_runner
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.experiments import read_manifest, verify_artifacts
from quantforge.oos import HoldoutLedger
from quantforge.prediction import PredictionContextRequirements
from quantforge.timeframes import resolve_exchange_session
from quantforge.validation import (
    TimestampBoundary,
    ValidationInterval,
    ValidationWindow,
)
from quantforge.walk_forward import PredictionEvaluator, WalkForwardConfig
from tests.integration import test_intraday_prediction_provenance as source_fixture
from tests.unit.experiments.test_contracts import execution


@pytest.fixture(scope="module")
def synthetic_inputs(tmp_path_factory: pytest.TempPathFactory) -> SmokeInputs:
    # 51 completed daily bars precede the first evaluation day. Prices are
    # synthetic by construction, never repaired historical market observations.
    sessions: list[date] = []
    current = date(2025, 1, 2)
    while len(sessions) < 55:
        try:
            resolve_exchange_session(current, spy_ema_plan.TWO_MINUTES.session_policy)
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
        minute = result.end_timestamp.astimezone(ZoneInfo("America/New_York")).time()
        level = Decimal(100 + sessions.index(result.session_date))
        if (minute.hour, minute.minute) >= (11, 0):
            level += 30
        elif (minute.hour, minute.minute) >= (10, 20):
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
            tmp_path_factory.mktemp("qf45-real-contracts"),
            provider="massive",
            session_dates=tuple(sessions),
        )
    return SmokeInputs(
        fixture.source, fixture.dataset, fixture.primary, fixture.daily, families[-1]
    )


def bounded_plan(
    inputs: SmokeInputs, root: Path, *, fixed: bool = False
) -> tuple[WalkForwardConfig, PredictionEvaluator]:
    sessions = [bar.session_date for bar in inputs.dataset.bars]
    chosen = [session.isoformat() for session in sessions[-4:]]
    original = spy_ema_plan.validation_window

    def window(name: str, role: Any, dates: tuple[str, str]) -> ValidationWindow:
        result = original(name, role, dates)
        session = resolve_exchange_session(
            date.fromisoformat(dates[0]), spy_ema_plan.TWO_MINUTES.session_policy
        )
        first = session.open_timestamp + timedelta(minutes=90)
        return replace(
            result,
            interval=ValidationInterval(
                TimestampBoundary(first),
                TimestampBoundary(first + timedelta(minutes=4)),
            ),
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            spy_ema_plan,
            "SPLITS",
            (((chosen[0], chosen[0]), (chosen[1], chosen[1]), (chosen[2], chosen[2])),),
        )
        patch.setattr(spy_ema_plan, "HOLDOUT", (chosen[3], chosen[3]))
        patch.setattr(spy_ema_plan, "validation_window", window)
        return spy_ema_plan.prepare_walk_forward(inputs, root, fixed=fixed)


def test_pre_holdout_pipeline_and_completed_resume(
    synthetic_inputs: SmokeInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quantforge.walk_forward.prediction import (
        _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
    )

    monkeypatch.setattr(spy_ema_runner, "prepare_walk_forward", bounded_plan)
    root = tmp_path / "run"
    ledger = HoldoutLedger.create(tmp_path / "ledger")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("pre-holdout command must never consume holdout")

    monkeypatch.setattr(HoldoutLedger, "consume", forbidden)
    calls: list[object] = []
    original = _PermittedContextProvider.get_context_at

    def tracked(
        self: Any, requirements: PredictionContextRequirements, *, as_of: Any
    ) -> Any:
        assert as_of.date() != synthetic_inputs.dataset.bars[-1].session_date
        calls.append(as_of)
        return original(self, requirements, as_of=as_of)

    monkeypatch.setattr(_PermittedContextProvider, "get_context_at", tracked)
    completed = spy_ema_runner.run_pre_holdout(
        synthetic_inputs, root, ledger, execution()
    )
    assert completed["holdout_state"] == "reserved_unconsumed"
    assert "PENDING" in cast(str, completed["manual_audit"])
    manifest = read_manifest(Path(cast(str, completed["manifest_path"])))
    verify_artifacts(manifest.artifacts, tmp_path).require_valid()
    html = Path(cast(str, completed["report_path"])).read_text()
    assert "reserved_unconsumed" in html
    assert list(root.glob("features/*/features.csv"))
    windows = sorted(root.rglob("prediction-window.jsonl"))
    assert len(windows) == 5  # fixed + exactly three selection trials + one OOS
    before = {p: p.read_bytes() for p in windows}
    calls.clear()
    replay = spy_ema_runner.run_pre_holdout(synthetic_inputs, root, ledger, execution())
    assert replay == completed
    assert {p: p.read_bytes() for p in windows} == before
    # QF-7 checks context identity on resume but never repeats full QF-42 windows.
    assert len(calls) == 5
    assert not list((ledger.root / "exposures").iterdir())
