from datetime import date
from decimal import Decimal
from pathlib import Path

from quantforge.backtesting import (
    BacktestConfig,
    BasisPointSlippage,
    ExplicitZeroFees,
    PerShareCommission,
)
from quantforge.optimization import (
    ExecutionConfig,
    FilePersistenceConfig,
    GridSearchConfig,
    GridSearchStudy,
    MaximumDrawdown,
    MetricName,
    MinimumTrades,
    MovingAverageCrossoverFactory,
    ParameterLessThan,
    ParameterSearchSpace,
    PositiveReturn,
    RankingConfig,
    StabilityConfig,
)
from quantforge.optimization.spaces import IntegerValues

from ..unit.helpers import make_dataset


def test_spy_style_grid_uses_real_qf4_qf5_resumes_and_exports(tmp_path: Path) -> None:
    sessions = tuple(
        date(2024, 7, day)
        for day in (
            1,
            2,
            3,
            5,
            8,
            9,
            10,
            11,
            12,
            15,
            16,
            17,
            18,
            19,
            22,
            23,
            24,
            25,
            26,
            29,
        )
    )
    closes = tuple(
        str(value)
        for value in (
            100,
            98,
            96,
            98,
            102,
            108,
            115,
            120,
            125,
            123,
            121,
            119,
            117,
            115,
            113,
            111,
            109,
            107,
            105,
            103,
        )
    )
    dataset = make_dataset(
        closes,
        sessions=sessions,
        dataset_id="immutable-synthetic-spy-grid",
    )
    study = GridSearchStudy(
        dataset,
        MovingAverageCrossoverFactory(),
        GridSearchConfig(
            label="documented SPY moving-average grid",
            search_space=ParameterSearchSpace(
                {
                    "fast_window": IntegerValues([2, 3]),
                    "slow_window": IntegerValues([3, 4, 5]),
                }
            ),
            parameter_constraints=(ParameterLessThan("fast_window", "slow_window"),),
            backtest=BacktestConfig(
                initial_capital=Decimal("100000"),
                commission=PerShareCommission(
                    amount_per_share=Decimal("0.005"),
                    minimum=Decimal("1"),
                ),
                fees=ExplicitZeroFees(),
                slippage=BasisPointSlippage(Decimal("5")),
            ),
            execution=ExecutionConfig(),
            ranking=RankingConfig(
                objective=MetricName.SHARPE_RATIO,
                hard_constraints=(
                    MinimumTrades(1),
                    MaximumDrawdown(Decimal("0.30")),
                    PositiveReturn(),
                ),
            ),
            stability=StabilityConfig(minimum_eligible_neighbors=1),
            persistence=FilePersistenceConfig(tmp_path),
        ),
    )

    first = study.run()
    resumed = study.resume()

    assert len(first.successful_trials) == 5
    assert len(first.excluded_trials) == 1
    assert len(first.rankings) == 4
    assert first.rankings == resumed.rankings
    assert [trial.qf5_run_id for trial in first.successful_trials] == [
        trial.qf5_run_id for trial in resumed.successful_trials
    ]
    assert first.stability
    assert (study.study_path / "summary.json").is_file()
    assert (study.study_path / "trials.csv").is_file()
    assert all(trial.artifact_location for trial in first.successful_trials)
