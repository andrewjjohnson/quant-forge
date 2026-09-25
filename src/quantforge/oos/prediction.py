"""Reduce stored prediction values only; never execute rules or outcome labels."""

from collections import Counter
from collections.abc import Iterable, Iterator
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
from quantforge.prediction.window_reader import PredictionWindowReader
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


class PredictionObservationAccumulator:
    """Counters plus only numeric samples needed by the existing exact medians."""

    def __init__(self, fields: PredictionMetricFields) -> None:
        self.fields = fields
        self.generated = self.eligible = self.labeled = 0
        self.correct_count = self.correct_sum = 0
        self.paired_count = self.baseline_sum = self.difference_sum = 0
        self.events: Counter[str] = Counter()
        self.directions: Counter[str] = Counter()
        self.numeric: dict[str, list[Primitive]] = {
            name: [] for name in ("signed_outcome", "mfe", "mae")
        }

    def update(self, item: PrimitiveMapping) -> None:
        self.generated += 1
        if item["eligible"] is not True:
            return
        self.eligible += 1
        self.labeled += item["row"] is not None
        self.directions[
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
        correct = values.get(self.fields.correct)
        _booleans([correct])
        if isinstance(correct, bool):
            self.correct_count += 1
            self.correct_sum += correct
        if self.fields.baseline_correct:
            baseline = values.get(self.fields.baseline_correct)
            _booleans([baseline])
            if isinstance(correct, bool) and isinstance(baseline, bool):
                self.paired_count += 1
                self.baseline_sum += baseline
                self.difference_sum += correct - baseline
        for name, samples in self.numeric.items():
            value = values.get(getattr(self.fields, name))
            if value is not None:
                number(value)  # Fail closed even before final reduction.
                samples.append(value)
        event = values.get(self.fields.event)
        if event is not None and values.get("available") is not False:
            self.events[text(event)] += 1

    def summary(self, scheduled_decisions: int) -> PrimitiveMapping:
        with arithmetic():
            event_count = sum(self.events.values())
            return {
                "prediction_count": self.eligible,
                "generated_signal_count": self.generated,
                "excluded_signal_count": self.generated - self.eligible,
                "scheduled_decisions": scheduled_decisions,
                "prediction_frequency": optional_decimal(
                    Decimal(self.eligible) / scheduled_decisions
                    if scheduled_decisions
                    else None
                ),
                "frequency_denominator": "scheduled_decisions_in_completed_windows",
                "direction_distribution": {
                    name: self.directions[name] for name in ("down", "up")
                },
                "labeled_prediction_count": self.labeled,
                "accuracy": optional_decimal(
                    Decimal(self.correct_sum) / self.correct_count
                    if self.correct_count
                    else None
                ),
                "accuracy_sample_count": self.correct_count,
                "accuracy_unavailable_count": self.eligible - self.correct_count,
                "accuracy_interval": wilson_interval(
                    self.correct_sum, self.correct_count
                ).to_primitive(),
                "matched_baseline": {
                    "name": self.fields.baseline_name,
                    "sample_count": self.paired_count,
                    "status": "available" if self.paired_count else "unavailable",
                    "accuracy": optional_decimal(
                        Decimal(self.baseline_sum) / self.paired_count
                        if self.paired_count
                        else None
                    ),
                    "accuracy_difference": optional_decimal(
                        Decimal(self.difference_sum) / self.paired_count
                        if self.paired_count
                        else None
                    ),
                },
                **{
                    name: _metric(samples, self.eligible).to_primitive()
                    for name, samples in self.numeric.items()
                },
                "event_outcomes": {
                    "status": "available" if event_count else "unavailable",
                    "sample_count": event_count,
                    "unavailable_count": self.eligible - event_count,
                    "counts": dict(sorted(self.events.items())),
                    "rates": {
                        event: optional_decimal(Decimal(count) / event_count)
                        for event, count in sorted(self.events.items())
                    },
                },
            }


def summarize_prediction_observations(
    observations: Iterable[PrimitiveMappingSnapshot],
    scheduled_decisions: int,
    fields: PredictionMetricFields = PredictionMetricFields(),
) -> PrimitiveMapping:
    """Reduce stored values without retaining observation graphs."""
    accumulator = PredictionObservationAccumulator(fields)
    for observation in observations:
        accumulator.update(observation.to_primitive())
    return accumulator.summary(scheduled_decisions)


def source_prediction_reader(source: OOSSource, index: int) -> PredictionWindowReader:
    """Get the validated source reader, keeping legacy constructed sources usable."""
    artifact = source.folds[index].artifact
    if not isinstance(artifact, PredictionOOSArtifact):
        raise OOSIntegrityError("wrong OOS artifact family")
    reader = source.prediction_windows[index] if source.prediction_windows else None
    if reader is None:
        reader = PredictionWindowReader.from_snapshot(artifact.snapshot.to_primitive())
    if reader.header()["window_result_id"] != artifact.window_result_id:
        raise OOSIntegrityError("source window reader differs from captured artifact")
    return reader


def prediction_window_source(
    reader: PredictionWindowReader, artifact: PredictionOOSArtifact, fold_id: str
) -> PrimitiveMappingSnapshot:
    return PrimitiveMappingSnapshot.capture(
        {
            "fold_id": fold_id,
            "selection_id": artifact.selection_id,
            "schema_version": reader.schema_version,
            "window_id": reader.header()["window_id"],
            "window_result_id": artifact.window_result_id,
            "shared_evidence_id": reader.evidence.evidence_id,
            "schedule_id": reader.evidence.schedule.schedule_id,
            "decision_count": reader.decision_count,
        }
    )


def iter_prediction_observations(
    reader: PredictionWindowReader, artifact: PredictionOOSArtifact, fold_id: str
) -> Iterator[PrimitiveMappingSnapshot]:
    for compact in reader.iterate_decisions():
        decision = compact.to_primitive()
        study = mapping(decision["prediction_study"])
        rows = {
            configuration_identity(
                {"prediction": r["prediction"], "features": r["features"]}
            ): r
            for r in records(study["rows"])
        }
        for signal in records(decision["generated_signals"]):
            values = mapping(mapping(signal["prediction"])["values"])
            yield PrimitiveMappingSnapshot.capture(
                {
                    "fold_id": fold_id,
                    "selection_id": artifact.selection_id,
                    "window_result_id": artifact.window_result_id,
                    "decision_timestamp": decision["decision_timestamp"],
                    "context_id": decision["context_id"],
                    "prediction_study_id": decision["prediction_study_id"],
                    "signal": signal,
                    "eligible": values.get("direction") in {"up", "down"}
                    and values.get("disposition") in {None, "accepted"},
                    "row": rows.get(configuration_identity(signal)),
                }
            )


def prediction_observations(
    artifact: PredictionOOSArtifact, fold_id: str
) -> tuple[PrimitiveMappingSnapshot, ...]:
    return tuple(
        iter_prediction_observations(
            PredictionWindowReader.from_snapshot(artifact.snapshot.to_primitive()),
            artifact,
            fold_id,
        )
    )


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
    compact = any(
        reader is not None and reader.schema_version == "2"
        for reader in source.prediction_windows
    )
    sources: list[PrimitiveMappingSnapshot] = []
    pooled = PredictionObservationAccumulator(fields)
    for index, fold in enumerate(source.folds):
        if fold.artifact is None:
            windows.append(
                {"fold_id": fold.fold_id, "status": fold.status.value, "summary": None}
            )
            continue
        if not isinstance(fold.artifact, PredictionOOSArtifact):
            raise OOSIntegrityError("wrong OOS artifact family")
        reader = source_prediction_reader(source, index)
        manifest = reader.manifest()
        config = mapping(manifest["configuration"])
        semantics.add(
            configuration_identity(
                {key: config[key] for key in ("outcome_labeler", "evaluator")}
            )
        )
        accumulator = PredictionObservationAccumulator(fields)
        for observation in iter_prediction_observations(
            reader, fold.artifact, fold.fold_id
        ):
            record = observation.to_primitive()
            accumulator.update(record)
            pooled.update(record)
            if not compact:
                observations.append(observation)
        scheduled = reader.decision_count
        schedules += scheduled
        sources.append(prediction_window_source(reader, fold.artifact, fold.fold_id))
        windows.append(
            {
                "fold_id": fold.fold_id,
                "status": fold.status.value,
                "summary": accumulator.summary(scheduled),
            }
        )
    if len(semantics) > 1:
        raise OOSIntegrityError(
            "cannot pool different outcome/evaluator semantics; summarize "
            "separate studies"
        )
    summary = pooled.summary(schedules)
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
        tuple(sources) if compact else None,
    )
