"""Finite-grid neighborhood statistics and isolated-peak detection."""

from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, Decimal, localcontext

from quantforge.optimization.combinations import (
    CombinationCandidate,
    CombinationExclusion,
    ParameterCombination,
)
from quantforge.optimization.factories import StrategyFactory
from quantforge.optimization.models import (
    ParameterSummary,
    RankingConfig,
    RankingDirection,
    StabilityClassification,
    StabilityConfig,
    StabilitySummary,
    TrialRecord,
    TrialStatus,
)
from quantforge.optimization.ranking import RankingOutcome
from quantforge.optimization.spaces import (
    ParameterSearchSpace,
    search_value_to_primitive,
)


@dataclass(frozen=True, slots=True)
class StabilityOutcome:
    summaries: tuple[StabilitySummary, ...]
    parameter_summaries: tuple[ParameterSummary, ...]
    best_stability_trial_id: str | None
    recommended_robust_trial_id: str | None


def _mean(values: tuple[Decimal, ...]) -> Decimal | None:
    if not values:
        return None
    with localcontext() as context:
        context.prec = 34
        return sum(values, Decimal(0)) / Decimal(len(values))


def _median(values: tuple[Decimal, ...]) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    with localcontext() as context:
        context.prec = 34
        return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _standard_deviation(values: tuple[Decimal, ...]) -> Decimal | None:
    mean = _mean(values)
    if mean is None or len(values) < 2:
        return None
    with localcontext() as context:
        context.prec = 34
        variance = sum((value - mean) ** 2 for value in values) / Decimal(len(values))
        return variance.sqrt()


def _ratio(numerator: Decimal | int, denominator: Decimal | int) -> Decimal:
    with localcontext() as context:
        context.prec = 34
        return Decimal(numerator) / Decimal(denominator)


def _ceiling_fraction(count: int, fraction: Decimal) -> int:
    if count == 0 or fraction == 0:
        return 0
    with localcontext() as context:
        context.prec = 34
        value = (Decimal(count) * fraction).to_integral_value(rounding=ROUND_CEILING)
    return max(1, int(value))


def _potential_neighbors(
    coordinates: tuple[int, ...],
    axis_kinds: tuple[str, ...],
    axis_lengths: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    neighbors: list[tuple[int, ...]] = []
    for axis, (kind, length) in enumerate(zip(axis_kinds, axis_lengths, strict=True)):
        if kind in ("integer", "float"):
            alternate_positions = tuple(
                position
                for position in (coordinates[axis] - 1, coordinates[axis] + 1)
                if 0 <= position < length
            )
        else:
            alternate_positions = tuple(
                position for position in range(length) if position != coordinates[axis]
            )
        for position in alternate_positions:
            neighbor = list(coordinates)
            neighbor[axis] = position
            neighbors.append(tuple(neighbor))
    return tuple(neighbors)


def _is_boundary(
    coordinates: tuple[int, ...],
    axis_kinds: tuple[str, ...],
    axis_lengths: tuple[int, ...],
) -> bool:
    numeric_axes = tuple(
        axis for axis, kind in enumerate(axis_kinds) if kind in ("integer", "float")
    )
    return any(
        coordinates[axis] in (0, axis_lengths[axis] - 1) for axis in numeric_axes
    )


def _directional_difference(
    center: Decimal, neighbor: Decimal, direction: RankingDirection
) -> Decimal:
    with localcontext() as context:
        context.prec = 34
        return (
            center - neighbor
            if direction is RankingDirection.MAXIMIZE
            else neighbor - center
        )


def _relative_dispersion(
    mean: Decimal | None, standard_deviation: Decimal | None
) -> Decimal | None:
    if mean is None:
        return None
    if standard_deviation is None or standard_deviation == 0:
        return Decimal(0)
    if mean == 0:
        return None
    with localcontext() as context:
        context.prec = 34
        return standard_deviation / abs(mean)


def _classify(
    eligible_neighbors: int,
    pass_fraction: Decimal,
    relative_dispersion: Decimal | None,
    config: StabilityConfig,
) -> StabilityClassification:
    if eligible_neighbors < config.minimum_eligible_neighbors:
        return StabilityClassification.INSUFFICIENT_NEIGHBORS
    if (
        pass_fraction >= config.stable_constraint_pass_fraction
        and relative_dispersion is not None
        and relative_dispersion <= config.stable_maximum_relative_dispersion
    ):
        return StabilityClassification.STABLE
    if pass_fraction >= Decimal("0.5"):
        return StabilityClassification.MIXED
    return StabilityClassification.FRAGILE


def _stability_score(
    pass_fraction: Decimal, relative_dispersion: Decimal | None
) -> Decimal:
    if relative_dispersion is None:
        return Decimal(0)
    with localcontext() as context:
        context.prec = 34
        return pass_fraction / (Decimal(1) + relative_dispersion)


def _is_isolated(
    *,
    objective_rank: int,
    eligible_count: int,
    eligible_neighbor_count: int,
    pass_fraction: Decimal,
    difference: Decimal | None,
    relative_difference: Decimal | None,
    is_boundary: bool,
    config: StabilityConfig,
) -> tuple[bool, str | None]:
    top_count = _ceiling_fraction(eligible_count, config.isolated_peak_top_fraction)
    if objective_rank > top_count:
        return False, None
    if eligible_neighbor_count < config.minimum_eligible_neighbors:
        return False, "insufficient eligible neighbors for isolated-peak classification"
    if difference is None or difference < config.isolated_peak_absolute_drop:
        return False, None
    if (
        relative_difference is None
        or relative_difference < config.isolated_peak_relative_drop
    ):
        return False, None
    if pass_fraction > config.isolated_peak_maximum_constraint_pass_fraction:
        return False, None
    boundary_note = " on a search-space boundary" if is_boundary else ""
    return (
        True,
        "high-ranked center exceeds its eligible-neighbor median by the configured "
        "absolute and relative drops while neighbor constraint pass rate is low"
        f"{boundary_note}",
    )


def _parameter_summaries(
    trials: tuple[TrialRecord, ...],
    ranking: RankingOutcome,
    search_space: ParameterSearchSpace,
    strategy_factory: StrategyFactory,
    direction: RankingDirection,
) -> tuple[ParameterSummary, ...]:
    successful = tuple(
        trial for trial in trials if trial.status is TrialStatus.SUCCEEDED
    )
    objective_by_trial = {
        ranked.trial_id: ranked.objective_value for ranked in ranking.rankings
    }
    summaries: list[ParameterSummary] = []
    for name, values in search_space.ordered_items(strategy_factory.parameter_order):
        for candidate in values.values:
            primitive_candidate = search_value_to_primitive(candidate)
            matching = tuple(
                trial
                for trial in successful
                if trial.parameters.get(name) == primitive_candidate
            )
            eligible_values = tuple(
                objective_by_trial[trial.trial_id]
                for trial in matching
                if trial.trial_id in objective_by_trial
            )
            pass_fraction = (
                Decimal(0)
                if not matching
                else _ratio(len(eligible_values), len(matching))
            )
            best = (
                None
                if not eligible_values
                else (
                    max(eligible_values)
                    if direction is RankingDirection.MAXIMIZE
                    else min(eligible_values)
                )
            )
            summaries.append(
                ParameterSummary(
                    parameter_name=name,
                    parameter_value=primitive_candidate,
                    successful_count=len(matching),
                    eligible_count=len(eligible_values),
                    constraint_pass_fraction=pass_fraction,
                    mean_objective=_mean(eligible_values),
                    median_objective=_median(eligible_values),
                    best_objective=best,
                )
            )
    return tuple(summaries)


def analyze_stability(
    trials: tuple[TrialRecord, ...],
    candidates: tuple[CombinationCandidate, ...],
    ranking: RankingOutcome,
    search_space: ParameterSearchSpace,
    strategy_factory: StrategyFactory,
    ranking_config: RankingConfig,
    stability_config: StabilityConfig,
) -> StabilityOutcome:
    """Analyze immediate finite-grid neighbors without inventing failed values."""
    candidate_by_coordinates = {
        candidate.coordinates: candidate for candidate in candidates
    }
    valid_by_id = {
        candidate.combination_id: candidate
        for candidate in candidates
        if isinstance(candidate, ParameterCombination)
    }
    ranked_by_combination = {item.combination_id: item for item in ranking.rankings}
    ordered_items = search_space.ordered_items(strategy_factory.parameter_order)
    axis_kinds = tuple(values.kind for _, values in ordered_items)
    axis_lengths = tuple(len(values.values) for _, values in ordered_items)
    summaries: list[StabilitySummary] = []

    for ranked in ranking.rankings:
        center = valid_by_id[ranked.combination_id]
        potential = _potential_neighbors(center.coordinates, axis_kinds, axis_lengths)
        neighboring_candidates = tuple(
            candidate_by_coordinates[coordinates]
            for coordinates in potential
            if coordinates in candidate_by_coordinates
        )
        valid_neighbors = tuple(
            candidate
            for candidate in neighboring_candidates
            if isinstance(candidate, ParameterCombination)
        )
        excluded_neighbor_count = sum(
            isinstance(candidate, CombinationExclusion)
            for candidate in neighboring_candidates
        )
        eligible_neighbor_values = tuple(
            ranked_by_combination[candidate.combination_id].objective_value
            for candidate in valid_neighbors
            if candidate.combination_id in ranked_by_combination
        )
        eligible_count = len(eligible_neighbor_values)
        pass_fraction = (
            Decimal(0)
            if not valid_neighbors
            else _ratio(eligible_count, len(valid_neighbors))
        )
        mean = _mean(eligible_neighbor_values)
        median = _median(eligible_neighbor_values)
        standard_deviation = _standard_deviation(eligible_neighbor_values)
        worst = (
            None
            if not eligible_neighbor_values
            else (
                min(eligible_neighbor_values)
                if ranking_config.direction is RankingDirection.MAXIMIZE
                else max(eligible_neighbor_values)
            )
        )
        difference = (
            None
            if median is None
            else _directional_difference(
                ranked.objective_value, median, ranking_config.direction
            )
        )
        relative_difference = (
            None
            if difference is None or ranked.objective_value == 0
            else _ratio(difference, abs(ranked.objective_value))
        )
        relative_dispersion = _relative_dispersion(mean, standard_deviation)
        classification = _classify(
            eligible_count,
            pass_fraction,
            relative_dispersion,
            stability_config,
        )
        boundary = _is_boundary(center.coordinates, axis_kinds, axis_lengths)
        isolated, reason = _is_isolated(
            objective_rank=ranked.rank,
            eligible_count=len(ranking.rankings),
            eligible_neighbor_count=eligible_count,
            pass_fraction=pass_fraction,
            difference=difference,
            relative_difference=relative_difference,
            is_boundary=boundary,
            config=stability_config,
        )
        summaries.append(
            StabilitySummary(
                trial_id=ranked.trial_id,
                combination_id=ranked.combination_id,
                objective_rank=ranked.rank,
                objective_value=ranked.objective_value,
                valid_neighbor_count=len(valid_neighbors),
                excluded_neighbor_count=excluded_neighbor_count,
                successful_eligible_neighbor_count=eligible_count,
                neighbor_objective_values=eligible_neighbor_values,
                mean_neighbor_objective=mean,
                median_neighbor_objective=median,
                worst_neighbor_objective=worst,
                objective_standard_deviation=standard_deviation,
                constraint_pass_fraction=pass_fraction,
                center_to_neighbor_difference=difference,
                relative_center_to_neighbor_difference=relative_difference,
                is_boundary=boundary,
                stability_score=_stability_score(pass_fraction, relative_dispersion),
                classification=classification,
                is_isolated_peak=isolated,
                isolation_reason=reason,
            )
        )

    summaries.sort(
        key=lambda item: (
            -item.stability_score,
            item.objective_rank,
            item.combination_id,
        )
    )
    ranked_stability = tuple(
        replace(item, stability_rank=index)
        for index, item in enumerate(summaries, start=1)
    )
    best_stability = None if not ranked_stability else ranked_stability[0].trial_id
    robust_top_count = _ceiling_fraction(
        len(ranking.rankings),
        stability_config.robust_recommendation_top_fraction,
    )
    robust_candidates = tuple(
        item
        for item in ranked_stability
        if item.objective_rank <= robust_top_count
        and item.classification is StabilityClassification.STABLE
        and not item.is_isolated_peak
    )
    recommended = None if not robust_candidates else robust_candidates[0].trial_id
    parameter_summaries = _parameter_summaries(
        trials,
        ranking,
        search_space,
        strategy_factory,
        ranking_config.direction,
    )
    return StabilityOutcome(
        ranked_stability,
        parameter_summaries,
        best_stability,
        recommended,
    )
