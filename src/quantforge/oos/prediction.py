"""Reduce stored prediction values only; never execute rules or outcome labels."""

from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal
from statistics import median

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.oos._records import OOSIntegrityError, mapping, number, records, text
from quantforge.oos.common import completeness, configuration_stability, provenance
from quantforge.oos.models import (
    MetricSummary,
    OOSSource,
    PredictionOOSAggregate,
    optional_decimal,
)
from quantforge.prediction._arithmetic import arithmetic
from quantforge.prediction.comparison_metrics import wilson_interval
from quantforge.validation import ResearchStudyType
from quantforge.walk_forward.models import PredictionOOSArtifact


@dataclass(frozen=True, slots=True)
class PredictionMetricFields:
    """Explicit fields in persisted evaluation.values; absent fields stay absent.

    Defaults project QF-11 gap and QF-7 excursion/target-stop schemas. A custom
    evaluator can provide another field binding without a research callback.
    Baselines must already be evaluated on the same prediction row.
    """

    correct: str = "direction_correct"
    signed_outcome: str = "signed_prediction_return"
    mfe: str = "mfe_percentage"
    mae: str = "mae_percentage"
    event: str = "label"
    baseline_correct: str | None = None
    baseline_name: str | None = None

    def __post_init__(self) -> None:
        if (self.baseline_correct is None) != (self.baseline_name is None):
            raise OOSIntegrityError(
                "matched baseline requires both name and stored field"
            )
        if any(value == "" for value in asdict(self).values()):
            raise OOSIntegrityError("metric field names cannot be empty")

    def to_primitive(self) -> PrimitiveMapping:
        return dict(asdict(self))


def _metric(values: list[Primitive], total: int) -> MetricSummary:
    defined = tuple(number(value) for value in values if value is not None)
    with arithmetic():
        return MetricSummary(
            len(defined),
            total - len(defined),
            sum(defined, Decimal(0)) / len(defined) if defined else None,
            median(defined) if defined else None,
            min(defined) if defined else None,
            max(defined) if defined else None,
        )


def _booleans(values: list[Primitive]) -> list[bool]:
    if any(value is not None and type(value) is not bool for value in values):
        raise OOSIntegrityError("accuracy fields must be boolean or unavailable")
    return [value for value in values if isinstance(value, bool)]


def summarize_prediction_observations(
    observations: tuple[PrimitiveMappingSnapshot, ...],
    scheduled_decisions: int,
    fields: PredictionMetricFields = PredictionMetricFields(),
) -> PrimitiveMapping:
    """Pure adapter for existing row values; descriptive Wilson, no iid claim."""
    rows = [item.to_primitive() for item in observations]
    eligible = [item for item in rows if item["eligible"] is True]
    evaluations = [
        {}
        if item["row"] is None
        else mapping(mapping(mapping(item["row"])["evaluation"])["values"])
        for item in eligible
    ]
    correct_values = [values.get(fields.correct) for values in evaluations]
    correct = _booleans(correct_values)
    baseline_pairs = (
        [
            (values.get(fields.correct), values.get(fields.baseline_correct))
            for values in evaluations
        ]
        if fields.baseline_correct
        else []
    )
    _booleans([pair[1] for pair in baseline_pairs])
    paired = [
        (a, b) for a, b in baseline_pairs if isinstance(a, bool) and isinstance(b, bool)
    ]
    events: Counter[str] = Counter()
    for values in evaluations:
        if (
            values.get(fields.event) is not None
            and values.get("available") is not False
        ):
            events[text(values[fields.event])] += 1
    directions = Counter(
        text(
            mapping(mapping(mapping(item["signal"])["prediction"])["values"])[
                "direction"
            ]
        )
        for item in eligible
    )
    with arithmetic():
        baseline_accuracy = (
            Decimal(sum(b for _, b in paired)) / len(paired) if paired else None
        )
        difference = (
            Decimal(sum(a - b for a, b in paired)) / len(paired) if paired else None
        )
        event_count = sum(events.values())
        return {
            "prediction_count": len(eligible),
            "generated_signal_count": len(rows),
            "excluded_signal_count": len(rows) - len(eligible),
            "scheduled_decisions": scheduled_decisions,
            "prediction_frequency": optional_decimal(
                Decimal(len(eligible)) / scheduled_decisions
                if scheduled_decisions
                else None
            ),
            "frequency_denominator": "scheduled_decisions_in_completed_windows",
            "direction_distribution": {
                "down": directions["down"],
                "up": directions["up"],
            },
            "labeled_prediction_count": sum(
                item["row"] is not None for item in eligible
            ),
            "accuracy": optional_decimal(
                Decimal(sum(correct)) / len(correct) if correct else None
            ),
            "accuracy_sample_count": len(correct),
            "accuracy_unavailable_count": len(eligible) - len(correct),
            "accuracy_interval": wilson_interval(
                sum(correct), len(correct)
            ).to_primitive(),
            "matched_baseline": {
                "name": fields.baseline_name,
                "sample_count": len(paired),
                "status": "available" if paired else "unavailable",
                "accuracy": optional_decimal(baseline_accuracy),
                "accuracy_difference": optional_decimal(difference),
            },
            "signed_outcome": _metric(
                [v.get(fields.signed_outcome) for v in evaluations], len(eligible)
            ).to_primitive(),
            "mfe": _metric(
                [v.get(fields.mfe) for v in evaluations], len(eligible)
            ).to_primitive(),
            "mae": _metric(
                [v.get(fields.mae) for v in evaluations], len(eligible)
            ).to_primitive(),
            "event_outcomes": {
                "status": "available" if event_count else "unavailable",
                "sample_count": event_count,
                "unavailable_count": len(eligible) - event_count,
                "counts": dict(sorted(events.items())),
                "rates": {
                    event: optional_decimal(Decimal(count) / event_count)
                    for event, count in sorted(events.items())
                },
            },
        }


def prediction_observations(
    artifact: PredictionOOSArtifact, fold_id: str
) -> tuple[PrimitiveMappingSnapshot, ...]:
    observations: list[PrimitiveMappingSnapshot] = []
    for decision in records(artifact.snapshot.to_primitive()["decisions"]):
        study = mapping(decision["prediction_study"])
        rows = {
            configuration_identity(
                {"prediction": r["prediction"], "features": r["features"]}
            ): r
            for r in records(study["rows"])
        }
        for signal in records(decision["generated_signals"]):
            values = mapping(mapping(signal["prediction"])["values"])
            eligible = values.get("direction") in {"up", "down"} and values.get(
                "disposition"
            ) in {None, "accepted"}
            observations.append(
                PrimitiveMappingSnapshot.capture(
                    {
                        "fold_id": fold_id,
                        "selection_id": artifact.selection_id,
                        "window_result_id": artifact.window_result_id,
                        "decision_timestamp": decision["decision_timestamp"],
                        "context_id": decision["context_id"],
                        "prediction_study_id": decision["prediction_study_id"],
                        "signal": signal,
                        "eligible": eligible,
                        "row": rows.get(configuration_identity(signal)),
                    }
                )
            )
    return tuple(observations)


def aggregate_prediction(
    source: OOSSource, fields: PredictionMetricFields = PredictionMetricFields()
) -> PredictionOOSAggregate:
    if source.plan.environment.study_type is not ResearchStudyType.PREDICTION:
        raise OOSIntegrityError("prediction aggregation requires a prediction study")
    source_provenance = provenance(source)
    observations: list[PrimitiveMappingSnapshot] = []
    windows: list[PrimitiveMapping] = []
    schedules = 0
    semantics: set[str] = set()
    for fold in source.folds:
        if fold.artifact is None:
            windows.append(
                {"fold_id": fold.fold_id, "status": fold.status.value, "summary": None}
            )
            continue
        if not isinstance(fold.artifact, PredictionOOSArtifact):
            raise OOSIntegrityError("wrong OOS artifact family")
        manifest = mapping(fold.artifact.snapshot.to_primitive()["manifest"])
        config = mapping(manifest["configuration"])
        semantics.add(
            configuration_identity(
                {key: config[key] for key in ("outcome_labeler", "evaluator")}
            )
        )
        captured = prediction_observations(fold.artifact, fold.fold_id)
        scheduled = len(records(fold.artifact.snapshot.to_primitive()["decisions"]))
        schedules += scheduled
        observations.extend(captured)
        windows.append(
            {
                "fold_id": fold.fold_id,
                "status": fold.status.value,
                "summary": summarize_prediction_observations(
                    captured, scheduled, fields
                ),
            }
        )
    if len(semantics) > 1:
        raise OOSIntegrityError(
            "cannot pool different outcome/evaluator semantics; summarize "
            "separate studies"
        )
    summary = summarize_prediction_observations(tuple(observations), schedules, fields)
    with arithmetic():
        accuracies = [
            mapping(w["summary"])["accuracy"]
            for w in windows
            if w["summary"] is not None
        ]
        signed_means = [
            mapping(mapping(w["summary"])["signed_outcome"])["mean"]
            for w in windows
            if w["summary"] is not None
        ]
        summary.update(
            {
                "completeness": completeness(source),
                "windows": [item for item in windows],
                "window_accuracy_consistency": _metric(
                    accuracies, len(windows)
                ).to_primitive(),
                "window_signed_outcome_consistency": _metric(
                    signed_means, len(windows)
                ).to_primitive(),
                "metric_fields": fields.to_primitive(),
                "warnings": [
                    "Wilson interval is descriptive; repeated/overlapping "
                    "outcomes are not independent",
                    "insufficient OOS sample size",
                ]
                if summary["prediction_count"] == 0
                else [
                    "Wilson interval is descriptive; repeated/overlapping "
                    "outcomes are not independent"
                ],
            }
        )
    return PredictionOOSAggregate(
        PrimitiveMappingSnapshot.capture(summary),
        configuration_stability(source),
        tuple(observations),
        source_provenance,
    )
