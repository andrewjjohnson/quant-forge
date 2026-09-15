"""Reconcile stored grid summaries without ranking or calculating stability."""

from decimal import Decimal, InvalidOperation
from typing import cast

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._json import ManifestError, mapping, text


def validate_prediction_summary(
    configuration: PrimitiveMapping,
    trials: list[PrimitiveMapping],
    summary: PrimitiveMapping,
) -> None:
    """Bind QF-32 selections to existing analyses without rerunning selection."""
    if summary.get("schema_version") != configuration.get("schema_version"):
        raise ManifestError("prediction summary schema differs from study")
    eligible = _records(summary.get("rankings"))
    ineligible = _records(summary.get("ineligible_trials"))
    stable = _records(summary.get("stability"))
    by_id = {text(trial.get("trial_id")): trial for trial in trials}
    eligible_by_id = _trial_references(eligible, by_id)
    ineligible_by_id = _trial_references(ineligible, by_id)
    stable_by_id = _trial_references(stable, by_id)
    successful = {
        trial_id
        for trial_id, trial in by_id.items()
        if trial.get("status") == "succeeded"
    }
    count = mapping(summary.get("counts")).get("eligible")
    if (
        eligible_by_id.keys() & ineligible_by_id.keys()
        or eligible_by_id.keys() | ineligible_by_id.keys() != successful
        or stable_by_id.keys() != eligible_by_id.keys()
        or type(count) is not int
        or count != len(eligible)
    ):
        raise ManifestError(
            "prediction summary trial coverage or counts are inconsistent"
        )
    ranking_config = mapping(configuration.get("ranking"))
    objective = text(ranking_config.get("objective_metric"))
    direction = ranking_config.get("direction")
    if direction not in {"maximize", "minimize"}:
        raise ManifestError("prediction summary ranking direction is invalid")
    previous: tuple[Decimal, str] | None = None
    for rank, record in enumerate(eligible, 1):
        trial = by_id[text(record.get("trial_id"))]
        metrics = mapping(mapping(trial.get("analysis")).get("metrics"))
        objective_value = _number(record.get("objective_value"))
        combination_id = text(record.get("combination_id"))
        if (
            type(record.get("rank")) is not int
            or record["rank"] != rank
            or record.get("objective_metric") != objective
            or objective_value != _number(metrics.get(objective))
        ):
            raise ManifestError(
                "prediction ranking differs from trial metrics or ranks"
            )
        if previous is not None and (
            (
                objective_value > previous[0]
                if direction == "maximize"
                else objective_value < previous[0]
            )
            or (objective_value == previous[0] and combination_id < previous[1])
        ):
            raise ManifestError(
                "prediction ranking order contradicts recorded objectives"
            )
        previous = objective_value, combination_id
    for record in ineligible:
        reasons = record.get("reasons")
        if (
            not isinstance(reasons, list)
            or not reasons
            or any(
                not isinstance(reason, str) or not reason.strip() for reason in reasons
            )
        ):
            raise ManifestError("prediction summary is missing ineligibility reasons")
    for rank, record in enumerate(stable, 1):
        ranked = eligible_by_id[text(record.get("trial_id"))]
        if (
            type(record.get("objective_rank")) is not int
            or record["objective_rank"] != rank
            or record["objective_rank"] != ranked.get("rank")
            or _number(record.get("objective_value"))
            != _number(ranked.get("objective_value"))
        ):
            raise ManifestError("prediction stability differs from recorded ranking")


def _records(value: object) -> list[PrimitiveMapping]:
    if not isinstance(value, list):
        raise ManifestError("optimization summary records must be arrays")
    return [mapping(item) for item in cast(list[object], value)]


def _number(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ManifestError("optimization summary objective must be numeric")
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ManifestError("optimization summary objective must be numeric") from error
    if not number.is_finite():
        raise ManifestError("optimization summary objective must be finite")
    return number


def _trial_references(
    records: list[PrimitiveMapping], trials: dict[str, PrimitiveMapping]
) -> dict[str, PrimitiveMapping]:
    references: dict[str, PrimitiveMapping] = {}
    for record in records:
        trial_id = text(record.get("trial_id"))
        trial = trials.get(trial_id)
        if (
            trial_id in references
            or trial is None
            or trial.get("status") != "succeeded"
            or record.get("combination_id") != trial.get("combination_id")
        ):
            raise ManifestError(
                "optimization summary has incompatible trial references"
            )
        references[trial_id] = record
    return references


def validate_optimization_summaries(
    configuration: PrimitiveMapping,
    trials: list[PrimitiveMapping],
    summary: PrimitiveMapping | None,
    derived: dict[str, PrimitiveMapping],
) -> None:
    """Compare configurations, membership, ranks and saved summary projections.

    Objective values are compared with existing trial metrics. Eligibility,
    ranking order, neighbor statistics and recommendation rules are not rerun.
    """
    ranking = mapping(derived.get("ranking.json"))
    stability = mapping(derived.get("stability.json"))
    for name, document in (
        ("ranking_configuration", ranking),
        ("stability_configuration", stability),
    ):
        if document.get("configuration") != configuration.get(name) or (
            summary is not None and summary.get(name) != configuration.get(name)
        ):
            raise ManifestError("optimization summary configuration differs from study")
    eligible = _records(ranking.get("eligible_rankings"))
    ineligible = _records(ranking.get("ineligible_trials"))
    stable = _records(stability.get("summaries"))
    by_id = {text(trial.get("trial_id")): trial for trial in trials}
    eligible_by_id = _trial_references(eligible, by_id)
    ineligible_by_id = _trial_references(ineligible, by_id)
    stable_by_id = _trial_references(stable, by_id)
    successful = {
        trial_id
        for trial_id, trial in by_id.items()
        if trial.get("status") == "succeeded"
    }
    if (
        eligible_by_id.keys() & ineligible_by_id.keys()
        or eligible_by_id.keys() | ineligible_by_id.keys() != successful
        or stable_by_id.keys() != eligible_by_id.keys()
    ):
        raise ManifestError("optimization summary trial coverage is inconsistent")
    objective = text(
        mapping(configuration.get("ranking_configuration")).get("objective")
    )
    for rank, record in enumerate(eligible, 1):
        trial = by_id[text(record.get("trial_id"))]
        if (
            type(record.get("rank")) is not int
            or record["rank"] != rank
            or record.get("objective_metric") != objective
            or _number(record.get("objective_value"))
            != _number(mapping(trial.get("metrics")).get(objective))
        ):
            raise ManifestError(
                "optimization ranking differs from recorded trial metrics or ranks"
            )
    for record in ineligible:
        reasons = record.get("reasons")
        if (
            not isinstance(reasons, list)
            or not reasons
            or any(
                not isinstance(reason, str) or not reason.strip() for reason in reasons
            )
        ):
            raise ManifestError("optimization summary is missing ineligibility reasons")
    for rank, record in enumerate(stable, 1):
        objective_record = eligible_by_id[text(record.get("trial_id"))]
        if (
            type(record.get("stability_rank")) is not int
            or record["stability_rank"] != rank
            or type(record.get("objective_rank")) is not int
            or record.get("objective_rank") != objective_record.get("rank")
            or record.get("objective_value") != objective_record.get("objective_value")
            or type(record.get("is_isolated_peak")) is not bool
        ):
            raise ManifestError("optimization stability differs from recorded ranking")
    if summary is None:
        return
    expected_counts = {
        "eligible": len(eligible),
        "ineligible_successful": len(ineligible),
        "isolated_peaks": sum(record["is_isolated_peak"] is True for record in stable),
    }
    counts = mapping(summary.get("counts"))
    if any(
        type(counts.get(key)) is not int or counts[key] != count
        for key, count in expected_counts.items()
    ):
        raise ManifestError("optimization summary ranking counts are inconsistent")
    if (
        summary.get("top_objective_trials") != eligible[:10]
        or summary.get("top_stability_trials") != stable[:10]
        or summary.get("best_objective_trial_id")
        != (eligible[0]["trial_id"] if eligible else None)
        or summary.get("best_stability_trial_id")
        != (stable[0]["trial_id"] if stable else None)
    ):
        raise ManifestError("optimization selected trials differ from summary")
    recommended = summary.get("recommended_robust_trial_id")
    if recommended is not None:
        record = stable_by_id.get(text(recommended))
        if (
            record is None
            or record.get("classification") != "stable"
            or record.get("is_isolated_peak") is not False
        ):
            raise ManifestError(
                "optimization recommendation has incompatible trial reference"
            )
