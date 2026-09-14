"""Neutral fold completeness, provenance, stability and immutable persistence."""

from collections import Counter
from pathlib import Path

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.oos._records import OOSIntegrityError, mapping, records, text
from quantforge.oos.models import (
    BacktestOOSAggregate,
    ConfigurationStabilitySummary,
    OOSSource,
    PredictionOOSAggregate,
)
from quantforge.oos.source import validation_lineage
from quantforge.walk_forward.models import FoldStatus
from quantforge.walk_forward.persistence import read_record, write_record


def configuration_stability(source: OOSSource) -> ConfigurationStabilitySummary:
    windows: list[PrimitiveMappingSnapshot] = []
    counts: Counter[str] = Counter()
    parameter_changes: Counter[str] = Counter()
    previous: PrimitiveMapping | None = None
    transitions = changes = repeats = 0
    for fold in source.folds:
        snapshot = (
            None if fold.selection is None else fold.selection.snapshot.to_primitive()
        )
        candidate = None if snapshot is None else mapping(snapshot["candidate"])
        if candidate is not None:
            candidate_id = text(candidate["combination_id"])
            counts[candidate_id] += 1
            params = mapping(candidate["parameters"])
            for name in params:
                parameter_changes[name] += 0
            if previous is not None:
                transitions += 1
                changed = previous != candidate
                changes += changed
                repeats += not changed
                old_params = mapping(previous["parameters"])
                for name in old_params.keys() | params.keys():
                    parameter_changes[name] += (
                        name not in old_params
                        or name not in params
                        or old_params[name] != params[name]
                    )
        evidence = None if snapshot is None else mapping(snapshot["selection_evidence"])
        neighborhoods = (
            []
            if evidence is None or snapshot is None
            else [
                item
                for item in records(evidence["stability"])
                if item["trial_id"] == snapshot["selected_trial_id"]
            ]
        )
        windows.append(
            PrimitiveMappingSnapshot.capture(
                {
                    "fold_id": fold.fold_id,
                    "status": fold.status.value,
                    "selection_id": None
                    if fold.selection is None
                    else fold.selection.selection_id,
                    "candidate": candidate,
                    "neighborhood_evidence": [item for item in neighborhoods],
                }
            )
        )
        # A missing selection interrupts adjacency; never bridge a failed gap.
        previous = candidate
    return ConfigurationStabilitySummary(
        tuple(windows),
        transitions,
        changes,
        repeats,
        PrimitiveMappingSnapshot.capture(dict(sorted(counts.items()))),
        PrimitiveMappingSnapshot.capture(dict(sorted(parameter_changes.items()))),
    )


def provenance(source: OOSSource) -> PrimitiveMappingSnapshot:
    definition = source.definition.to_primitive()
    if (
        configuration_identity(definition) != source.study_id
        or mapping(definition["configuration"])["plan"] != source.plan.to_manifest()
        or validation_lineage(definition) != source.lineage
    ):
        raise OOSIntegrityError("source study, plan or lineage identity differs")
    ids = [fold.fold_id for fold in source.folds]
    if ids != [fold.fold_id for fold in source.plan.folds] or len(ids) != len(set(ids)):
        raise OOSIntegrityError(
            "source folds must be unique and in validation-plan order"
        )
    if len(source.references) != len(source.folds):
        raise OOSIntegrityError("source references are incomplete")
    for fold, reference in zip(source.folds, source.references, strict=True):
        record = reference.to_primitive()
        if (
            record["fold_id"] != fold.fold_id
            or (
                record["status"] != fold.status.value
                and not (
                    record["status"] == "missing" and fold.status is FoldStatus.PENDING
                )
            )
            or record["selection"]
            != (None if fold.selection is None else fold.selection.to_primitive())
            or record["artifact_sha256"]
            != (
                None
                if fold.artifact is None
                else configuration_identity(fold.artifact.to_primitive())
            )
            or (fold.artifact is not None and fold.status is not FoldStatus.COMPLETED)
        ):
            raise OOSIntegrityError(
                "fold differs from verified QF-39 source references"
            )
    return PrimitiveMappingSnapshot.capture(
        {
            "study_id": source.study_id,
            "plan_id": source.plan.plan_id,
            "lineage_id": source.lineage_id,
            "lineage": source.lineage.to_primitive(),
            "study_definition": source.definition.to_primitive(),
            "folds": [item.to_primitive() for item in source.references],
            "holdout_state": "consult_holdout_ledger; aggregation_does_not_consume",
            "eligible_role": "walk_forward_test",
        }
    )


def completeness(source: OOSSource) -> PrimitiveMapping:
    completed = sum(fold.status is FoldStatus.COMPLETED for fold in source.folds)
    return {
        "expected_windows": len(source.plan.folds),
        "completed_windows": completed,
        "complete": completed == len(source.plan.folds),
        "missing_windows": [
            r.to_primitive()["fold_id"]
            for r in source.references
            if r.to_primitive()["status"] == "missing"
        ],
        "failed_windows": [
            fold.fold_id for fold in source.folds if fold.status is FoldStatus.FAILED
        ],
        "incomplete_windows": [
            fold.fold_id
            for fold in source.folds
            if fold.status is not FoldStatus.COMPLETED
        ],
        "interpretation": "all_planned_windows"
        if completed == len(source.plan.folds)
        else "partial_observed_windows_only; failures_are_not_zero_returns",
    }


def export_oos_aggregate(
    result: PredictionOOSAggregate | BacktestOOSAggregate, output_root: Path
) -> Path:
    """Content-addressed QF-40 JSON; exact retries verify instead of overwrite."""
    path = output_root / f"{result.aggregate_id}.json"
    write_record(path, result.to_primitive(), immutable=True)
    return path


def load_oos_aggregate(path: Path) -> PrimitiveMappingSnapshot:
    """Verify and detach a machine-readable summary for QF-9/QF-41 consumers."""
    payload = read_record(path)
    if (
        path.stem != configuration_identity(payload)
        or payload.get("schema_version") != "1"
        or payload.get("kind")
        not in {"prediction_oos_aggregate", "backtest_oos_aggregate"}
    ):
        raise OOSIntegrityError("incompatible aggregate identity/schema")
    return PrimitiveMappingSnapshot.capture(payload)
