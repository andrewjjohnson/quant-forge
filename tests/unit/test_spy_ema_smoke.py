"""Focused QF-45 hypothesis tests; generic label arithmetic belongs to QF-47/49."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from quantforge.data import build_multi_timeframe_context
from quantforge.examples.spy_ema import (
    DAILY,
    TWO_MINUTES,
    EmaParameters,
    EmaSmokeRule,
    EmaStudyFactory,
    backend_environment,
    configured_outcomes,
    grid_configuration,
)
from quantforge.examples.spy_ema_inputs import source_request
from quantforge.indicators import EXPONENTIAL_MOVING_AVERAGE_OUTPUT
from quantforge.prediction import PredictionDirection, PredictionRuleContext
from quantforge.prediction.context import build_prediction_rule_context
from quantforge.timeframes import IntradayInterval, Timeframe
from quantforge.validation import ResearchRuleProvenance
from tests.integration.test_intraday_prediction_provenance import (
    Fixture,
    cached_fixture,
)


@dataclass(frozen=True)
class Values:
    values: tuple[Decimal | None, ...]

    def values_for(self, field_name: str) -> tuple[Decimal | None, ...]:
        assert field_name == EXPONENTIAL_MOVING_AVERAGE_OUTPUT
        return self.values


def rule_context(
    *,
    clock: str = "15:00",
    previous_fast: str = "99",
    fast: str = "101",
    slow: str = "100",
    daily_close: str = "110",
    trend: str | None = "100",
) -> PredictionRuleContext:
    instant = datetime.fromisoformat(f"2025-06-02T{clock}:00+00:00")
    values = {
        "fast": Values((Decimal(previous_fast), Decimal(fast))),
        "slow": Values((Decimal(slow), Decimal(slow))),
        "trend": Values((None if trend is None else Decimal(trend),)),
    }

    def indicator_for(timeframe: Timeframe, alias: str) -> Values:
        return values[alias]

    def latest_bar_for(timeframe: Timeframe) -> SimpleNamespace:
        return SimpleNamespace(close=Decimal(daily_close))

    return cast(
        PredictionRuleContext,
        SimpleNamespace(
            as_of=instant,
            symbol="SPY",
            decision_session=instant.date(),
            prediction_dataset_id="unit-rule-context",
            indicator_for=indicator_for,
            latest_bar_for=latest_bar_for,
        ),
    )


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ("99", "101", True),
        ("100", "101", True),
        ("101", "102", False),
        ("101", "99", False),
        ("99", "100", False),
    ],
)
def test_only_transition_emits(previous: str, current: str, expected: bool) -> None:
    rule = EmaSmokeRule()
    output = rule.generate_with_context(
        rule_context(previous_fast=previous, fast=current)
    )
    assert bool(output.signals) is expected
    if expected:
        signal = output.signals[0]
        assert signal.direction is PredictionDirection.UP
        assert signal.decision_timestamp == datetime(2025, 6, 2, 15, tzinfo=UTC)
        assert signal.strategy_configuration_id == rule.configuration_id


@pytest.mark.parametrize(
    ("clock", "expected"),
    [
        ("14:58", False),
        ("15:00", True),
        ("17:58", True),
        ("18:00", True),
        ("18:02", False),
    ],
)
def test_midday_endpoints_are_inclusive(clock: str, expected: bool) -> None:
    assert (
        bool(EmaSmokeRule().generate_with_context(rule_context(clock=clock)).signals)
        is expected
    )


@pytest.mark.parametrize(
    ("close", "trend", "expected"),
    [
        ("110", "100", True),
        ("100", "100", False),
        ("90", "100", False),
        ("110", None, False),
    ],
)
def test_completed_daily_trend_filter(
    close: str, trend: str | None, expected: bool
) -> None:
    assert (
        bool(
            EmaSmokeRule()
            .generate_with_context(rule_context(daily_close=close, trend=trend))
            .signals
        )
        is expected
    )


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return cached_fixture(tmp_path_factory.mktemp("ema-canonical"), provider="massive")


def test_daily_context_does_not_expose_current_session(inputs: Fixture) -> None:
    rule = EmaSmokeRule()
    as_of = datetime(2024, 1, 3, 16, tzinfo=UTC)
    context = build_multi_timeframe_context(
        as_of=as_of,
        primary_timeframe=TWO_MINUTES,
        required_timeframes=rule.context_requirements.context_timeframe_requirements(),
        series=(inputs.primary, inputs.daily),
    )
    restricted = build_prediction_rule_context(
        rule.context_requirements,
        context,
        prediction_dataset_id=inputs.dataset.metadata.dataset_id,
        symbol="SPY",
        prediction_adjustment_basis=inputs.source.request.adjustment_basis,
    )
    assert restricted.latest_bar_for(DAILY).end_timestamp == datetime(
        2024, 1, 2, 21, tzinfo=UTC
    )
    assert all(bar.end_timestamp <= as_of for bar in restricted.bars_for(DAILY))
    assert restricted.indicator_for(DAILY, "trend").values_for(
        EXPONENTIAL_MOVING_AVERAGE_OUTPUT
    ) == (None,)
    assert not rule.generate_with_context(restricted).signals


def test_exact_configuration_backend_source_grid_and_identity(
    inputs: Fixture, tmp_path: Path
) -> None:
    rule = EmaSmokeRule()
    assert rule.parameters.to_primitive() == {"fast": 8, "slow": 48, "daily": 50}
    assert rule.warm_up_observations == 49
    assert (
        ResearchRuleProvenance.capture_prediction(rule).configuration.configuration_id
        == rule.configuration_id
    )
    assert rule.configuration_id == EmaSmokeRule().configuration_id
    assert rule.configuration_id != EmaSmokeRule(EmaParameters(10, 48)).configuration_id
    for requirement in rule.context_requirements.all_timeframes:
        for indicator in requirement.indicators:
            assert indicator.backend_identity is not None
            assert backend_environment().matches(indicator.backend_identity)
            assert indicator.backend_identity.backend_id == "talib_v1"
    request = source_request()
    assert isinstance(request.timeframe.interval, IntradayInterval)
    assert request.timeframe.interval.nominal_duration == timedelta(minutes=1)
    assert request.timeframe.session_policy.calendar_name == "XNYS"
    assert request.start_timestamp == datetime(2025, 1, 1, tzinfo=UTC)
    assert request.end_timestamp == datetime(2026, 1, 1, tzinfo=UTC)
    assert request.adjustment_basis.ohlc_basis == "raw_provider"
    assert isinstance(TWO_MINUTES.interval, IntradayInterval)
    assert TWO_MINUTES.interval.nominal_duration == timedelta(minutes=2)
    grid = grid_configuration(tmp_path)
    assert grid.window_schema_version == "2"
    assert grid.maximum_combinations == 3
    assert grid.search_space.to_primitive(("ema_pair",))["parameters"] == [
        {
            "name": "ema_pair",
            "kind": "categorical",
            "values": ["8/40", "8/48", "12/60"],
        },
    ]
    assert grid.search_space.to_primitive(("ema_pair",)) == grid_configuration(
        tmp_path / "elsewhere"
    ).search_space.to_primitive(("ema_pair",))
    outcomes = configured_outcomes(inputs.primary)
    assert len(outcomes) == 6
    assert [item.labeler.required_future_duration for item in outcomes] == [
        timedelta(minutes=m) for m in (12, 32, 62, 122, 62, 62)
    ]
    target = outcomes[-1].evaluator.configuration()
    assert "0.003" in str(target)
    assert "0.002" in str(target)
    factory = EmaStudyFactory(inputs.primary)
    assert (
        factory.build({"ema_pair": "8/48"}).strategy.configuration_id
        == factory.build({"ema_pair": "8/48"}).strategy.configuration_id
    )


def test_invalid_periods_fail() -> None:
    with pytest.raises(ValueError, match="fast < slow"):
        EmaParameters(48, 8)


def test_session_splits_are_fixed_and_holdout_is_later() -> None:
    from quantforge.examples.spy_ema_plan import HOLDOUT, SPLITS, validation_window
    from quantforge.validation import PartitionRole

    assert len(SPLITS) == 1
    assert HOLDOUT == ("2025-06-30", "2025-07-31")
    previous_test_end = ""
    for index, (development, selection, test) in enumerate(SPLITS):
        assert development[0] == "2025-03-19"
        assert development[1] < selection[0] <= selection[1] < test[0]
        assert previous_test_end < test[0] <= test[1] < HOLDOUT[0]
        window = validation_window(str(index), PartitionRole.WALK_FORWARD_TEST, test)
        assert window.warm_up_observations_for(TWO_MINUTES) == 61
        assert window.warm_up_observations_for(DAILY) == 50
        previous_test_end = test[1]


def test_empty_analysis_stays_unrankable_and_zero_is_not_positive() -> None:
    from quantforge.examples.spy_ema import EmaWindowAnalyzer

    analyzer = EmaWindowAnalyzer()
    empty = analyzer._analyze([])  # pyright: ignore[reportPrivateUsage]
    assert empty.prediction_count == 0
    assert "mean_30m_return" not in empty.metrics
    timestamp = datetime(2025, 5, 1, 15, tzinfo=UTC)
    result = analyzer._analyze(  # pyright: ignore[reportPrivateUsage]
        [(timestamp, Decimal("0")), (timestamp, Decimal("0.01"))]
    )
    assert result.metrics["positive_30m_count"] == 1
    assert result.metrics["positive_30m_fraction"] == "0.5"


def test_cli_has_no_holdout_consumption_action(tmp_path: Path) -> None:
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts/run_spy_ema_smoke.py"
    result = subprocess.run(
        [sys.executable, str(script), "--consume-holdout"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
    assert not list(tmp_path.iterdir())


def test_execution_provenance_is_retained_and_changed_code_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quantforge.examples.spy_ema_runner import execution_for_run
    from quantforge.experiments import CodeProvenance

    original = CodeProvenance(git_commit="original", git_dirty=False)

    def captured(repository: Path) -> CodeProvenance:
        return original

    monkeypatch.setattr("quantforge.experiments.capture_code_provenance", captured)
    first = execution_for_run(tmp_path, tmp_path)
    assert execution_for_run(tmp_path, tmp_path) == first
    recorded = (tmp_path / "execution.json").read_bytes()
    original = CodeProvenance(git_commit="different", git_dirty=False)
    with pytest.raises(ValueError, match="original clean code"):
        execution_for_run(tmp_path, tmp_path)
    assert (tmp_path / "execution.json").read_bytes() == recorded
