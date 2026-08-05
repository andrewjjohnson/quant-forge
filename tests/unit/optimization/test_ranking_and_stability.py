from decimal import Decimal, localcontext
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.optimization import (
    InvalidStudyConfigurationError,
    MaximumDrawdown,
    MetricName,
    MetricTieBreaker,
    MinimumTrades,
    MovingAverageCrossoverFactory,
    ParameterCombination,
    ParameterLessThan,
    ParameterSearchSpace,
    PositiveReturn,
    RankingConfig,
    RankingDirection,
    StabilityClassification,
    StabilityConfig,
    TrialRecord,
    TrialStatus,
    analyze_stability,
    iter_combination_candidates,
    rank_trials,
)
from quantforge.optimization.spaces import IntegerValues


def _trial(
    index: int,
    combination_id: str,
    parameters: PrimitiveMapping,
    *,
    total_return: str | None,
    trade_count: int = 1,
    maximum_drawdown: str = "-0.1",
    status: TrialStatus = TrialStatus.SUCCEEDED,
) -> TrialRecord:
    metrics: PrimitiveMapping = {
        "total_return": total_return,
        "trade_count": trade_count,
        "maximum_drawdown": maximum_drawdown,
        "annualized_volatility": (
            None if total_return is None else str(Decimal(total_return) / Decimal(2))
        ),
    }
    return TrialRecord(
        study_id="study",
        trial_id=f"trial-{index}",
        combination_id=combination_id,
        combination_index=index,
        status=status,
        parameters_snapshot=PrimitiveMappingSnapshot.capture(parameters),
        strategy_parameters_snapshot=PrimitiveMappingSnapshot.capture(parameters),
        strategy_name="synthetic",
        strategy_version="1",
        strategy_configuration_id=f"strategy-{index}",
        dataset_snapshot=PrimitiveMappingSnapshot.capture({"dataset_id": "dataset"}),
        backtest_configuration_snapshot=PrimitiveMappingSnapshot.capture(
            {"initial_capital": "100000"}
        ),
        metrics_snapshot=PrimitiveMappingSnapshot.capture(metrics),
        qf5_run_id=f"run-{index}",
    )


def test_ranking_enforces_constraints_undefined_metrics_and_stable_ties() -> None:
    trials = (
        _trial(3, "combo-c", {"x": 3}, total_return="0.20", trade_count=0),
        _trial(2, "combo-b", {"x": 2}, total_return=None),
        _trial(1, "combo-a", {"x": 1}, total_return="0.10"),
        _trial(4, "combo-d", {"x": 4}, total_return="0.10"),
        _trial(
            5,
            "combo-e",
            {"x": 5},
            total_return="1",
            status=TrialStatus.FAILED,
        ),
    )
    config = RankingConfig(
        MetricName.TOTAL_RETURN,
        hard_constraints=(
            MinimumTrades(1),
            MaximumDrawdown(Decimal("0.25")),
            PositiveReturn(),
        ),
    )

    forward = rank_trials(trials, config)
    reverse = rank_trials(tuple(reversed(trials)), config)

    assert [item.combination_id for item in forward.rankings] == [
        "combo-a",
        "combo-d",
    ]
    assert forward.rankings == reverse.rankings
    assert {item.combination_id for item in forward.ineligible_trials} == {
        "combo-b",
        "combo-c",
    }
    assert all(item.combination_id != "combo-e" for item in forward.rankings)


def test_minimize_ranking_uses_configured_direction() -> None:
    trials = (
        _trial(1, "a", {"x": 1}, total_return="0.20"),
        _trial(2, "b", {"x": 2}, total_return="0.10"),
    )
    outcome = rank_trials(
        trials,
        RankingConfig(
            MetricName.ANNUALIZED_VOLATILITY,
            direction=RankingDirection.MINIMIZE,
        ),
    )
    assert [item.combination_id for item in outcome.rankings] == ["b", "a"]


@pytest.mark.parametrize(
    ("tie_breaker", "expected_message"),
    [
        (
            MetricTieBreaker(
                cast(MetricName, "maximum_drawdown"),
                RankingDirection.MAXIMIZE,
            ),
            "tie-breaker metric",
        ),
        (
            MetricTieBreaker(
                MetricName.MAXIMUM_DRAWDOWN,
                cast(RankingDirection, "maximize"),
            ),
            "tie-breaker direction",
        ),
    ],
)
def test_ranking_rejects_raw_tie_breaker_metric_or_direction(
    tie_breaker: MetricTieBreaker,
    expected_message: str,
) -> None:
    with pytest.raises(InvalidStudyConfigurationError, match=expected_message):
        RankingConfig(
            MetricName.TOTAL_RETURN,
            tie_breakers=(tie_breaker,),
        )


def _surface_trials(
    *, spike: bool
) -> tuple[
    tuple[TrialRecord, ...],
    tuple[ParameterCombination, ...],
    ParameterSearchSpace,
]:
    search_space = ParameterSearchSpace(
        {
            "fast_window": IntegerValues([1, 50, 100]),
            "slow_window": IntegerValues([200, 300, 400]),
        }
    )
    all_candidates = tuple(
        iter_combination_candidates(
            MovingAverageCrossoverFactory(),
            search_space,
            (ParameterLessThan("fast_window", "slow_window"),),
        )
    )
    candidates = tuple(
        item for item in all_candidates if isinstance(item, ParameterCombination)
    )
    records: list[TrialRecord] = []
    for candidate in candidates:
        is_center = candidate.coordinates == (1, 1)
        if spike:
            value = "10" if is_center else "1"
            ineligible_neighbor = candidate.coordinates in ((0, 1), (2, 1))
            trade_count = 0 if ineligible_neighbor else 1
        else:
            distance = sum(abs(position - 1) for position in candidate.coordinates)
            value = str(Decimal("10") - Decimal(distance) / Decimal("10"))
            trade_count = 1
        records.append(
            _trial(
                candidate.index,
                candidate.combination_id,
                candidate.parameters,
                total_return=value,
                trade_count=trade_count,
            )
        )
    return tuple(records), candidates, search_space


def test_stability_finds_broad_plateau_using_grid_adjacency() -> None:
    trials, candidates, search_space = _surface_trials(spike=False)
    ranking_config = RankingConfig(
        MetricName.TOTAL_RETURN,
        hard_constraints=(MinimumTrades(1), PositiveReturn()),
    )
    ranking = rank_trials(trials, ranking_config)
    outcome = analyze_stability(
        trials,
        candidates,
        ranking,
        search_space,
        MovingAverageCrossoverFactory(),
        ranking_config,
        StabilityConfig(
            minimum_eligible_neighbors=2,
            stable_maximum_relative_dispersion=Decimal("0.10"),
            robust_recommendation_top_fraction=Decimal("0.50"),
        ),
    )
    with localcontext() as context:
        context.prec = 4
        different_ambient_context = analyze_stability(
            trials,
            candidates,
            ranking,
            search_space,
            MovingAverageCrossoverFactory(),
            ranking_config,
            StabilityConfig(
                minimum_eligible_neighbors=2,
                stable_maximum_relative_dispersion=Decimal("0.10"),
                robust_recommendation_top_fraction=Decimal("0.50"),
            ),
        )
    center = next(item for item in outcome.summaries if item.objective_rank == 1)

    assert outcome == different_ambient_context
    assert center.valid_neighbor_count == 4
    assert center.successful_eligible_neighbor_count == 4
    assert center.classification is StabilityClassification.STABLE
    assert not center.is_isolated_peak
    assert outcome.recommended_robust_trial_id == center.trial_id
    assert outcome.parameter_summaries[1].parameter_value == 50


def test_stability_identifies_isolated_spike_without_zeroing_failed_neighbors() -> None:
    trials, candidates, search_space = _surface_trials(spike=True)
    ranking_config = RankingConfig(
        MetricName.TOTAL_RETURN,
        hard_constraints=(MinimumTrades(1), PositiveReturn()),
    )
    ranking = rank_trials(trials, ranking_config)
    outcome = analyze_stability(
        trials,
        candidates,
        ranking,
        search_space,
        MovingAverageCrossoverFactory(),
        ranking_config,
        StabilityConfig(
            minimum_eligible_neighbors=2,
            isolated_peak_top_fraction=Decimal("0.20"),
            isolated_peak_relative_drop=Decimal("0.50"),
            isolated_peak_maximum_constraint_pass_fraction=Decimal("0.50"),
        ),
    )
    spike = next(item for item in outcome.summaries if item.objective_rank == 1)

    assert spike.valid_neighbor_count == 4
    assert spike.successful_eligible_neighbor_count == 2
    assert spike.neighbor_objective_values == (Decimal(1), Decimal(1))
    assert spike.constraint_pass_fraction == Decimal("0.5")
    assert spike.is_isolated_peak
    assert spike.isolation_reason is not None


def test_zero_top_fractions_select_no_trials() -> None:
    ranking_config = RankingConfig(
        MetricName.TOTAL_RETURN,
        hard_constraints=(MinimumTrades(1), PositiveReturn()),
    )
    spike_trials, spike_candidates, search_space = _surface_trials(spike=True)
    spike_outcome = analyze_stability(
        spike_trials,
        spike_candidates,
        rank_trials(spike_trials, ranking_config),
        search_space,
        MovingAverageCrossoverFactory(),
        ranking_config,
        StabilityConfig(
            minimum_eligible_neighbors=2,
            isolated_peak_top_fraction=Decimal(0),
            isolated_peak_relative_drop=Decimal("0.50"),
            isolated_peak_maximum_constraint_pass_fraction=Decimal("0.50"),
        ),
    )
    plateau_trials, plateau_candidates, search_space = _surface_trials(spike=False)
    plateau_outcome = analyze_stability(
        plateau_trials,
        plateau_candidates,
        rank_trials(plateau_trials, ranking_config),
        search_space,
        MovingAverageCrossoverFactory(),
        ranking_config,
        StabilityConfig(
            minimum_eligible_neighbors=2,
            stable_maximum_relative_dispersion=Decimal("0.10"),
            robust_recommendation_top_fraction=Decimal(0),
        ),
    )

    assert not any(item.is_isolated_peak for item in spike_outcome.summaries)
    assert plateau_outcome.recommended_robust_trial_id is None
