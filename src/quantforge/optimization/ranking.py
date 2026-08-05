"""Deterministic QF-5 metric eligibility and ranking."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from quantforge.optimization.errors import RankingError
from quantforge.optimization.models import (
    IneligibleTrial,
    MetricName,
    RankedTrial,
    RankingConfig,
    RankingDirection,
    ThresholdOperator,
    TrialRecord,
    TrialStatus,
)


@dataclass(frozen=True, slots=True)
class RankingOutcome:
    rankings: tuple[RankedTrial, ...]
    ineligible_trials: tuple[IneligibleTrial, ...]
    warnings: tuple[str, ...]


def metric_value(trial: TrialRecord, metric: MetricName) -> Decimal | None:
    """Read a QF-5 metric without recalculating or replacing undefined values."""
    metrics = trial.metrics
    if metrics is None:
        return None
    raw = metrics.get(metric.value)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise RankingError(
            f"trial {trial.trial_id} has a nonnumeric {metric.value} metric"
        )
    try:
        value = Decimal(str(raw))
    except InvalidOperation as error:
        raise RankingError(
            f"trial {trial.trial_id} has an invalid {metric.value} metric"
        ) from error
    if not value.is_finite():
        raise RankingError(
            f"trial {trial.trial_id} has a nonfinite {metric.value} metric"
        )
    return value


def _passes_threshold(
    actual: Decimal, operator: ThresholdOperator, threshold: Decimal
) -> bool:
    if operator is ThresholdOperator.GREATER_THAN:
        return actual > threshold
    if operator is ThresholdOperator.GREATER_THAN_OR_EQUAL:
        return actual >= threshold
    if operator is ThresholdOperator.LESS_THAN:
        return actual < threshold
    if operator is ThresholdOperator.LESS_THAN_OR_EQUAL:
        return actual <= threshold
    if operator is ThresholdOperator.EQUAL:
        return actual == threshold
    raise RankingError("ranking threshold operator is unsupported")


def _eligibility_reasons(trial: TrialRecord, config: RankingConfig) -> tuple[str, ...]:
    reasons: list[str] = []
    if metric_value(trial, config.objective) is None:
        reasons.append(f"objective metric {config.objective.value} is undefined")
    for constraint in config.hard_constraints:
        actual = metric_value(trial, constraint.metric)
        if actual is None:
            reasons.append(f"required metric {constraint.metric.value} is undefined")
            continue
        threshold = Decimal(str(constraint.threshold))
        if not _passes_threshold(actual, constraint.operator, threshold):
            reasons.append(
                f"{constraint.metric.value}={actual} does not satisfy "
                f"{constraint.operator.value} {threshold}"
            )
    return tuple(reasons)


def _directional(value: Decimal, direction: RankingDirection) -> Decimal:
    if direction is RankingDirection.MAXIMIZE:
        return -value
    if direction is RankingDirection.MINIMIZE:
        return value
    raise RankingError("ranking direction is unsupported")


def _ranking_key(trial: TrialRecord, config: RankingConfig) -> tuple[object, ...]:
    objective = metric_value(trial, config.objective)
    assert objective is not None
    values: list[object] = [_directional(objective, config.direction)]
    for tie_breaker in config.tie_breakers:
        value = metric_value(trial, tie_breaker.metric)
        values.append(value is None)
        values.append(
            Decimal(0) if value is None else _directional(value, tie_breaker.direction)
        )
    values.append(trial.combination_id)
    return tuple(values)


def rank_trials(
    trials: tuple[TrialRecord, ...], config: RankingConfig
) -> RankingOutcome:
    """Apply hard constraints, then rank eligible successes canonically."""
    successful = tuple(
        trial for trial in trials if trial.status is TrialStatus.SUCCEEDED
    )
    warnings: list[str] = []
    forced_reason: str | None = None
    if (
        config.minimum_successful_trials is not None
        and len(successful) < config.minimum_successful_trials
    ):
        forced_reason = (
            f"study has {len(successful)} successful trials; ranking requires at least "
            f"{config.minimum_successful_trials}"
        )
        warnings.append(forced_reason)

    eligible: list[TrialRecord] = []
    ineligible: list[IneligibleTrial] = []
    for trial in sorted(successful, key=lambda item: item.combination_index):
        reasons = _eligibility_reasons(trial, config)
        if forced_reason is not None:
            reasons = (*reasons, forced_reason)
        if reasons:
            ineligible.append(
                IneligibleTrial(trial.trial_id, trial.combination_id, reasons)
            )
        else:
            eligible.append(trial)

    eligible.sort(key=lambda trial: _ranking_key(trial, config))
    rankings_list: list[RankedTrial] = []
    for index, trial in enumerate(eligible, start=1):
        objective = metric_value(trial, config.objective)
        assert objective is not None
        rankings_list.append(
            RankedTrial(
                rank=index,
                trial_id=trial.trial_id,
                combination_id=trial.combination_id,
                objective_metric=config.objective,
                objective_value=objective,
            )
        )
    rankings = tuple(rankings_list)
    if not rankings:
        warnings.append("no successful trial satisfied ranking eligibility")
    return RankingOutcome(rankings, tuple(ineligible), tuple(warnings))
