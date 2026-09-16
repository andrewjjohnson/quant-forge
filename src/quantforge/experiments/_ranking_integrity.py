"""Reconcile stored grid summaries without ranking or calculating stability."""

from dataclasses import fields
from decimal import ROUND_CEILING, Decimal, InvalidOperation, localcontext
from typing import cast

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._json import ManifestError, mapping, text


def validate_prediction_summary(
    configuration: PrimitiveMapping,
    trials: list[PrimitiveMapping],
    summary: PrimitiveMapping,
) -> None:
    """Bind QF-32 selections to existing analyses without rerunning selection."""
    from quantforge.experiments._stability_integrity import (
        validate_prediction_stability,
    )
    from quantforge.prediction.grid import PredictionGridCacheStatistics

    if summary.get("schema_version") != configuration.get("schema_version"):
        raise ManifestError("prediction summary schema differs from study")
    counters = summary.get("cache_statistics")
    if (
        not isinstance(counters, dict)
        or set(counters)
        != {field.name for field in fields(PredictionGridCacheStatistics)}
        or any(type(value) is not int or value < 0 for value in counters.values())
    ):
        raise ManifestError("prediction summary cache statistics are invalid")
    if set(summary) != {
        "study_id",
        "schema_version",
        "counts",
        "rankings",
        "ineligible_trials",
        "stability",
        "cache_statistics",
        "warnings",
        "limitations",
    }:
        raise ManifestError("prediction summary fields differ from producer schema")
    for field in ("warnings", "limitations"):
        disclosures = summary[field]
        if not isinstance(disclosures, list) or any(
            not isinstance(disclosure, str) for disclosure in disclosures
        ):
            raise ManifestError(f"prediction summary {field} must be a string array")
    eligible = _records(summary.get("rankings"))
    ineligible = _records(summary.get("ineligible_trials"))
    stable = _records(summary.get("stability"))
    for record in stable:
        validate_prediction_stability(record)
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


def _optimization_order_key(
    trial: PrimitiveMapping, configuration: PrimitiveMapping
) -> tuple[tuple[tuple[bool, Decimal], ...], str]:
    """Compare saved QF-6 metrics using the producer's ordered tie breakers."""
    if configuration.get("final_tie_breaker") != "combination_id_ascending":
        raise ManifestError("optimization final tie breaker is unsupported")
    criteria: list[PrimitiveMapping] = [
        {
            "metric": configuration.get("objective"),
            "direction": configuration.get("direction"),
        },
        *_records(configuration.get("tie_breakers")),
    ]
    metrics = mapping(trial.get("metrics"))
    values: list[tuple[bool, Decimal]] = []
    for criterion in criteria:
        direction = criterion.get("direction")
        if direction not in ("maximize", "minimize"):
            raise ManifestError("optimization ranking direction is invalid")
        raw = metrics.get(text(criterion.get("metric")))
        number = Decimal(0) if raw is None else _number(raw)
        # Undefined tie-break metrics sort last in either direction, as in QF-6.
        values.append((raw is None, -number if direction == "maximize" else number))
    return tuple(values), text(trial.get("combination_id"))


def validate_optimization_summaries(
    configuration: PrimitiveMapping,
    trials: list[PrimitiveMapping],
    summary: PrimitiveMapping | None,
    derived: dict[str, PrimitiveMapping],
) -> None:
    """Compare configurations, membership, ranks and saved summary projections.

    Ordering and the recommendation are checked against existing metrics and
    stability records. Eligibility and neighbor statistics are not recalculated.
    """
    from quantforge.experiments._stability_integrity import (
        validate_optimization_stability,
    )

    if summary is not None and summary.get("study_schema_version") != configuration.get(
        "study_schema_version"
    ):
        raise ManifestError("optimization summary schema differs from study")
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
    for record in stable:
        validate_optimization_stability(record)
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
    ranking_configuration = mapping(configuration.get("ranking_configuration"))
    objective = text(ranking_configuration.get("objective"))
    previous_objective: tuple[tuple[tuple[bool, Decimal], ...], str] | None = None
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
        order_key = _optimization_order_key(trial, ranking_configuration)
        if previous_objective is not None and order_key < previous_objective:
            raise ManifestError("optimization ranking order contradicts saved metrics")
        previous_objective = order_key
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
    previous_stability: tuple[Decimal, int, str] | None = None
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
        stability_key = (
            -_number(record.get("stability_score")),
            cast(int, record["objective_rank"]),
            text(record.get("combination_id")),
        )
        if previous_stability is not None and stability_key < previous_stability:
            raise ManifestError("optimization stability order contradicts saved scores")
        previous_stability = stability_key
    if summary is None:
        return
    distribution = mapping(summary.get("objective_distribution"))
    objective_values = [_number(record["objective_value"]) for record in eligible]
    if (
        set(distribution) != {"count", "minimum", "maximum"}
        or type(distribution.get("count")) is not int
        or distribution["count"] != len(objective_values)
        or any(
            (
                _number(distribution.get(key)) != extremum(objective_values)
                if objective_values
                else distribution.get(key) is not None
            )
            for key, extremum in (("minimum", min), ("maximum", max))
        )
    ):
        raise ManifestError("optimization objective distribution differs from rankings")
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
    fraction = _number(
        mapping(configuration.get("stability_configuration")).get(
            "robust_recommendation_top_fraction"
        )
    )
    if not 0 <= fraction <= 1:
        raise ManifestError("optimization recommendation fraction is outside [0, 1]")
    # Match QF-6's ceiling cutoff and precision, including a zero-sized shortlist.
    cutoff = 0
    if eligible and fraction != 0:
        with localcontext() as context:
            context.prec = 34
            cutoff = max(
                1,
                int(
                    (Decimal(len(eligible)) * fraction).to_integral_value(
                        rounding=ROUND_CEILING
                    )
                ),
            )
    expected_recommendation = next(
        (
            record["trial_id"]
            for record in stable
            if cast(int, record["objective_rank"]) <= cutoff
            and record.get("classification") == "stable"
            and record["is_isolated_peak"] is False
        ),
        None,
    )
    if summary.get("recommended_robust_trial_id") != expected_recommendation:
        raise ManifestError(
            "optimization recommendation differs from saved qualifying trials"
        )
