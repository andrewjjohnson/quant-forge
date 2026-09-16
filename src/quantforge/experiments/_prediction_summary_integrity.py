"""Check complete QF-40 prediction summary records without calculating metrics."""

from dataclasses import fields

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._aggregate_schema import (
    counter,
    decimal_field,
    record,
    records,
    strings,
    validate_completeness,
)
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.oos.prediction import PredictionMetricFields
from quantforge.prediction.comparison_models import AccuracyInterval

_COUNTS = {
    "prediction_count",
    "generated_signal_count",
    "excluded_signal_count",
    "scheduled_decisions",
    "labeled_prediction_count",
    "accuracy_sample_count",
    "accuracy_unavailable_count",
}
_BASE_FIELDS = _COUNTS | {
    "prediction_frequency",
    "frequency_denominator",
    "direction_distribution",
    "accuracy",
    "accuracy_interval",
    "matched_baseline",
    "signed_outcome",
    "mfe",
    "mae",
    "event_outcomes",
}
_AGGREGATE_FIELDS = {
    "completeness",
    "windows",
    "window_accuracy_consistency",
    "window_signed_outcome_consistency",
    "metric_fields",
    "warnings",
}


def _availability(value: object, count: int, unavailable: int = 0) -> None:
    expected = (
        "unavailable" if count == 0 else "partial" if unavailable else "available"
    )
    if value != expected:
        raise ManifestError("OOS aggregate metric availability is inconsistent")


def _metric(value: object) -> None:
    metric = record(
        value,
        {
            "status",
            "sample_count",
            "unavailable_count",
            "mean",
            "median",
            "minimum",
            "maximum",
        },
        "metric summary",
    )
    count = counter(metric["sample_count"])
    unavailable = counter(metric["unavailable_count"])
    _availability(metric["status"], count, unavailable)
    for field in ("mean", "median", "minimum", "maximum"):
        if count == 0 and metric[field] is not None:
            raise ManifestError("OOS aggregate empty metric must be unavailable")
        decimal_field(metric[field], nullable=count == 0)


def _baseline(value: object) -> None:
    baseline = record(
        value,
        {"name", "sample_count", "status", "accuracy", "accuracy_difference"},
        "matched baseline",
    )
    count = counter(baseline["sample_count"])
    _availability(baseline["status"], count)
    if baseline["name"] is not None:
        text(baseline["name"])
    for field, minimum in (("accuracy", 0), ("accuracy_difference", -1)):
        if count == 0 and baseline[field] is not None:
            raise ManifestError("OOS aggregate empty baseline must be unavailable")
        decimal_field(baseline[field], nullable=count == 0, minimum=minimum, maximum=1)


def _events(value: object) -> None:
    events = record(
        value,
        {"status", "sample_count", "unavailable_count", "counts", "rates"},
        "event outcomes",
    )
    count = counter(events["sample_count"])
    counter(events["unavailable_count"])
    _availability(events["status"], count)
    counts, rates = mapping(events["counts"]), mapping(events["rates"])
    if counts.keys() != rates.keys() or bool(counts) != bool(count):
        raise ManifestError("OOS aggregate event records are inconsistent")
    for name, event_count in counts.items():
        text(name)
        if counter(event_count) == 0:
            raise ManifestError("OOS aggregate event count must be positive")
        decimal_field(rates[name], minimum=0, maximum=1)


def _prediction_summary(value: object, *, aggregate: bool) -> PrimitiveMapping:
    summary = record(
        value,
        _BASE_FIELDS | (_AGGREGATE_FIELDS if aggregate else set()),
        "prediction summary",
    )
    for field in _COUNTS:
        counter(summary[field])
    decimal_field(summary["prediction_frequency"], nullable=True, minimum=0)
    decimal_field(summary["accuracy"], nullable=True, minimum=0, maximum=1)
    if summary["frequency_denominator"] != "scheduled_decisions_in_completed_windows":
        raise ManifestError("OOS aggregate prediction frequency denominator is invalid")
    distribution = record(
        summary["direction_distribution"], {"up", "down"}, "direction distribution"
    )
    for count in distribution.values():
        counter(count)
    interval = record(
        summary["accuracy_interval"],
        {field.name for field in fields(AccuracyInterval)},
        "accuracy interval",
    )
    count = counter(interval["sample_count"])
    if interval["confidence_level"] != "0.95" or interval["method"] != "wilson_score":
        raise ManifestError("OOS aggregate accuracy interval contract is unsupported")
    for field in ("lower_bound", "upper_bound"):
        if count == 0 and interval[field] is not None:
            raise ManifestError(
                "OOS aggregate empty accuracy interval must be unavailable"
            )
        decimal_field(interval[field], nullable=count == 0, minimum=0, maximum=1)
    _baseline(summary["matched_baseline"])
    for field in ("signed_outcome", "mfe", "mae"):
        _metric(summary[field])
    _events(summary["event_outcomes"])
    return summary


def validate_prediction_aggregate_summary(value: object) -> None:
    summary = _prediction_summary(value, aggregate=True)
    metric_fields = record(
        summary["metric_fields"],
        {field.name for field in fields(PredictionMetricFields)},
        "metric fields",
    )
    for name, field in metric_fields.items():
        if name not in ("baseline_name", "baseline_correct") or field is not None:
            text(field)
    if (metric_fields["baseline_name"] is None) != (
        metric_fields["baseline_correct"] is None
    ):
        raise ManifestError("OOS aggregate baseline fields must be paired")
    validate_completeness(summary["completeness"])
    for field in ("window_accuracy_consistency", "window_signed_outcome_consistency"):
        _metric(summary[field])
    strings(summary["warnings"])
    for window in records(summary["windows"]):
        record(window, {"fold_id", "status", "summary"}, "prediction window summary")
        # Existing fold-membership checks establish status and null availability.
        if window["summary"] is not None:
            _prediction_summary(window["summary"], aggregate=False)
