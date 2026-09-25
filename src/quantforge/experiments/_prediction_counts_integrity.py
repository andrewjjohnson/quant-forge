"""Reconcile sample metadata with captured rows without calculating statistics."""

from collections import Counter
from collections.abc import Iterable

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
    observations: Iterable[PrimitiveMapping],
    scheduled_decisions: int,
    fields: PrimitiveMapping,
) -> None:
    """Count stored eligibility/availability in one pass, never calculate metrics."""
    generated = eligible = labeled = correct_count = paired_count = 0
    directions: Counter[str] = Counter()
    numeric: Counter[str] = Counter()
    events: Counter[str] = Counter()
    correct_field = text(fields["correct"])
    baseline_field = fields["baseline_correct"]
    event_field = text(fields["event"])
    for item in observations:
        generated += 1
        if item["eligible"] is not True:
            continue
        eligible += 1
        labeled += item["row"] is not None
        directions[
            text(
                mapping(mapping(mapping(item["signal"])["prediction"])["values"])[
                    "direction"
                ]
            )
        ] += 1
        values = (
            {}
            if item["row"] is None
            else mapping(mapping(mapping(item["row"])["evaluation"])["values"])
        )
        correct_count += type(values.get(correct_field)) is bool
        if baseline_field:
            paired_count += (
                type(values.get(correct_field)) is bool
                and type(values.get(text(baseline_field))) is bool
            )
        for name in ("signed_outcome", "mfe", "mae"):
            numeric[name] += values.get(text(fields[name])) is not None
        if values.get(event_field) is not None and values.get("available") is not False:
            events[text(values[event_field])] += 1
    _same_counts(
        summary,
        {
            "prediction_count": eligible,
            "generated_signal_count": generated,
            "excluded_signal_count": generated - eligible,
            "scheduled_decisions": scheduled_decisions,
            "labeled_prediction_count": labeled,
            "accuracy_sample_count": correct_count,
            "accuracy_unavailable_count": eligible - correct_count,
            "direction_distribution": {
                name: directions[name] for name in ("up", "down")
            },
        },
    )
    _same_counts(mapping(summary["accuracy_interval"]), {"sample_count": correct_count})
    _same_counts(
        mapping(summary["matched_baseline"]),
        {"name": fields["baseline_name"], "sample_count": paired_count},
    )
    for name in ("signed_outcome", "mfe", "mae"):
        _same_counts(
            mapping(summary[name]),
            {
                "sample_count": numeric[name],
                "unavailable_count": eligible - numeric[name],
            },
        )
    event_count = sum(events.values())
    _same_counts(
        mapping(summary["event_outcomes"]),
        {
            "sample_count": event_count,
            "unavailable_count": eligible - event_count,
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
