"""Native economics and independently reset OOS return-index examples."""

from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting._arithmetic import arithmetic
from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.indicators import Indicator
from quantforge.oos import (
    OOSIntegrityError,
    aggregate_backtest,
    aggregate_prediction,
    export_oos_aggregate,
    load_oos_aggregate,
    load_oos_source,
)
from quantforge.oos._records import mapping, number, records
from quantforge.optimization import IntegerValues, ParameterSearchSpace
from quantforge.validation import (
    DatasetProvenance,
    IndicatorComponent,
    IndicatorProvenance,
    ResearchRuleProvenance,
)
from quantforge.walk_forward import BacktestEvaluator, WalkForwardStudy
from tests.unit.helpers import SESSIONS, make_dataset
from tests.unit.walk_forward.fixtures import backtest_fixture

from .conftest import CompletedStudy


def active_backtest(root: Path) -> CompletedStudy:
    config, original = backtest_fixture(root)
    dataset = make_dataset(
        tuple(
            str(value)
            for value in (
                100,
                99,
                98,
                99,
                101,
                103,
                102,
                110,
                100,
                90,
                120,
                100,
                140,
                100,
                99,
            )
        )
    )
    grid = replace(
        original.grid_config,
        search_space=ParameterSearchSpace(
            {"fast_window": IntegerValues([1]), "slow_window": IntegerValues([2])}
        ),
    )
    evaluator = BacktestEvaluator(dataset, original.factory, grid)
    candidate = evaluator.universe.candidates[0]
    strategy = original.factory.build(candidate.parameters.to_primitive())
    timeframe = config.plan.environment.timeframes[0]
    indicators: tuple[Indicator, ...] = strategy.required_indicators
    environment = replace(
        config.plan.environment,
        dataset=DatasetProvenance.from_market_dataset(dataset),
        research_rule=ResearchRuleProvenance.capture_trading(strategy),
        indicators=tuple(
            IndicatorProvenance.capture(cast(IndicatorComponent, indicator), timeframe)
            for indicator in indicators
        ),
    )
    config = replace(config, plan=replace(config.plan, environment=environment))
    study = WalkForwardStudy(config, evaluator, root / "active")
    study.run()
    source = load_oos_source(config.plan, study.study_path)
    assert all(f.artifact for f in source.folds)
    return CompletedStudy(study, source, evaluator)


def test_real_backtest_equity_native_metrics_and_stitching(tmp_path: Path) -> None:
    fixture = active_backtest(tmp_path)
    result = aggregate_backtest(fixture.source)
    summary = result.summary.to_primitive()
    assert summary["trade_count"] == 2
    assert summary["winning_trades"] == 1
    assert summary["losing_trades"] == 1
    assert summary["win_rate"] == "0.5"
    assert summary["profitable_window_fraction"] == "0.5"
    assert number(summary["native_commissions"]) > 0
    assert number(summary["native_slippage_cost"]) > 0
    assert summary["oos_session_count"] == 6
    assert [r.to_primitive()["session"] for r in result.normalized_equity] == [
        s.isoformat() for s in SESSIONS[7:13]
    ]
    assert not {"accuracy", "prediction_count", "mfe", "mae"} & summary.keys()
    with arithmetic():
        base = benchmark_base = peak = Decimal(1)
        expected_drawdown = Decimal(0)
        gross_profit = gross_loss = Decimal(0)
        exposed = 0
        for fold, window in zip(
            fixture.source.folds, result.native_windows, strict=True
        ):
            assert fold.artifact is not None
            native = mapping(window.to_primitive()["result"])
            assert native == fold.artifact.snapshot.to_primitive()
            performance = mapping(mapping(native["manifest"])["performance"])
            gross_profit += number(performance["gross_profit"])
            gross_loss += number(performance["gross_loss"])
            equity = records(native["daily_equity"])
            assert equity[0]["cash"] == "100000"
            assert equity[0]["shares"] == 0
            for row in equity:
                current = base * number(row["total_equity"]) / Decimal(100000)
                peak = max(peak, current)
                expected_drawdown = min(expected_drawdown, current / peak - 1)
                exposed += row["exposed"] is True
            base *= number(performance["ending_equity"]) / Decimal(100000)
            benchmark_base *= number(
                records(native["benchmark_daily_equity"])[-1]["total_equity"]
            ) / Decimal(100000)
        assert number(summary["total_return"]) == base - 1
        assert number(summary["benchmark_total_return"]) == benchmark_base - 1
        assert number(summary["maximum_drawdown"]) == expected_drawdown
        assert number(summary["profit_factor"]) == gross_profit / abs(gross_loss)
        assert number(summary["exposure"]) == Decimal(exposed) / 6
    with localcontext() as context:
        context.prec = 4
        context.rounding = "ROUND_UP"
        assert aggregate_backtest(fixture.source) == result
    path = export_oos_aggregate(result, tmp_path / "exports")
    assert load_oos_aggregate(path) == PrimitiveMappingSnapshot.capture(
        result.to_primitive()
    )


def test_zero_trade_windows_and_family_separation(
    backtest_study: CompletedStudy,
) -> None:
    result = aggregate_backtest(backtest_study.source)
    summary = result.summary.to_primitive()
    assert summary["trade_count"] == 0
    assert summary["win_rate"] is None
    assert summary["profit_factor"] is None
    assert summary["total_return"] == "0"
    with pytest.raises(OOSIntegrityError, match="prediction study"):
        aggregate_prediction(backtest_study.source)
