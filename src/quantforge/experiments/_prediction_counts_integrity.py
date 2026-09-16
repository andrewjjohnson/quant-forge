"""Reconcile sample metadata with captured rows without calculating statistics."""

from collections import Counter

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._aggregate_schema import records
from quantforge.experiments._json import ManifestError, mapping, text


def _same_counts(actual: PrimitiveMapping, expected: PrimitiveMapping) -> None:
    if configuration_identity(
        {key: actual.get(key) for key in expected}
    ) != configuration_identity(expected):
        raise ManifestError("OOS aggregate counts differ from captured observations")


def validate_prediction_summary_counts(
    summary: PrimitiveMapping,
    observations: list[PrimitiveMapping],
    scheduled_decisions: int,
    fields: PrimitiveMapping,
) -> None:
    """Count stored eligibility, labels and metric availability, never outcomes."""
    eligible = [item for item in observations if item["eligible"] is True]
    evaluations = [
        {}
        if item["row"] is None
        else mapping(mapping(mapping(item["row"])["evaluation"])["values"])
        for item in eligible
    ]
    correct_field = text(fields["correct"])
    correct_count = sum(
        type(values.get(correct_field)) is bool for values in evaluations
    )
    directions = Counter(
        text(
            mapping(mapping(mapping(item["signal"])["prediction"])["values"])[
                "direction"
            ]
        )
        for item in eligible
    )
    _same_counts(
        summary,
        {
            "prediction_count": len(eligible),
            "generated_signal_count": len(observations),
            "excluded_signal_count": len(observations) - len(eligible),
            "scheduled_decisions": scheduled_decisions,
            "labeled_prediction_count": sum(
                item["row"] is not None for item in eligible
            ),
            "accuracy_sample_count": correct_count,
            "accuracy_unavailable_count": len(eligible) - correct_count,
            "direction_distribution": {
                name: directions[name] for name in ("up", "down")
            },
        },
    )
    _same_counts(mapping(summary["accuracy_interval"]), {"sample_count": correct_count})
    baseline_field = fields["baseline_correct"]
    paired_count = (
        sum(
            type(values.get(correct_field)) is bool
            and type(values.get(text(baseline_field))) is bool
            for values in evaluations
        )
        if baseline_field
        else 0
    )
    _same_counts(
        mapping(summary["matched_baseline"]),
        {"name": fields["baseline_name"], "sample_count": paired_count},
    )
    for name in ("signed_outcome", "mfe", "mae"):
        count = sum(
            values.get(text(fields[name])) is not None for values in evaluations
        )
        _same_counts(
            mapping(summary[name]),
            {"sample_count": count, "unavailable_count": len(eligible) - count},
        )
    event_field = text(fields["event"])
    events = Counter(
        text(values[event_field])
        for values in evaluations
        if values.get(event_field) is not None and values.get("available") is not False
    )
    event_count = sum(events.values())
    _same_counts(
        mapping(summary["event_outcomes"]),
        {
            "sample_count": event_count,
            "unavailable_count": len(eligible) - event_count,
            "counts": dict(events),
        },
    )


def prediction_window_observations(
    window: PrimitiveMapping, fold_id: str, selection_id: str, window_result_id: str
) -> list[PrimitiveMapping]:
    """Project existing signals/rows and their references, without summarizing."""
    observations: list[PrimitiveMapping] = []
    for decision in records(window.get("decisions")):
        rows = {
            configuration_identity(
                {"prediction": row.get("prediction"), "features": row.get("features")}
            ): row
            for row in records(mapping(decision.get("prediction_study")).get("rows"))
        }
        for signal in records(decision.get("generated_signals")):
            values = mapping(mapping(signal.get("prediction")).get("values"))
            observations.append(
                {
                    "fold_id": fold_id,
                    "selection_id": selection_id,
                    "window_result_id": window_result_id,
                    "decision_timestamp": decision.get("decision_timestamp"),
                    "context_id": decision.get("context_id"),
                    "prediction_study_id": decision.get("prediction_study_id"),
                    "signal": signal,
                    "eligible": values.get("direction") in ("up", "down")
                    and values.get("disposition") in (None, "accepted"),
                    "row": rows.get(configuration_identity(signal)),
                }
            )
    return observations
