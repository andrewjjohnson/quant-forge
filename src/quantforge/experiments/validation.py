"""Index QF-8/39/40 evidence, consulting the original holdout authority."""

from pathlib import Path

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments._backtest_artifacts import index_backtest_files
from quantforge.experiments._holdout_integrity import validate_holdout_artifact
from quantforge.experiments._json import ManifestError, mapping, snapshot, text
from quantforge.experiments.adapters import StudyArtifacts
from quantforge.experiments.artifacts import (
    ArtifactEntry,
    ArtifactIndex,
    ArtifactRelationship,
    ArtifactType,
    RelationshipType,
    index_artifact,
)
from quantforge.experiments.models import StudyProvenance, StudyType
from quantforge.experiments.persistence import read_producer_record
from quantforge.oos.common import provenance as source_provenance
from quantforge.oos.holdout import HoldoutLedger
from quantforge.oos.models import OOSSource
from quantforge.walk_forward.models import BacktestOOSArtifact, FoldStatus


def _validate_captured_backtest_export(export: Path, fingerprint: str) -> None:
    """Bind existing QF-5 tables to the QF-39/40 captured sidecar fingerprint."""
    from quantforge.backtesting.export import validate_backtest_result_artifact

    validate_backtest_result_artifact(export)
    try:
        # The producer hashes the original sidecar text, not parsed/reformatted JSON.
        integrity = (export / "integrity.json").read_text()
    except (OSError, UnicodeError) as error:
        raise ManifestError("cannot read captured backtest export integrity") from error
    if configuration_identity({"integrity": integrity}) != fingerprint:
        raise ManifestError("backtest export differs from captured fingerprint")


def inspect_validation(
    source: OOSSource,
    study_path: Path,
    *,
    artifact_root: Path,
    ledger: HoldoutLedger | None = None,
    aggregate_path: Path | None = None,
    study_type: StudyType = StudyType.WALK_FORWARD,
) -> StudyArtifacts:
    """Read already-produced snapshots; never aggregate, select, or evaluate.

    `source` comes from QF-40's existing `load_oos_source`. An absent ledger is
    explicitly unknown, never unconsumed. A supplied ledger must already contain
    this reservation; errors propagate rather than restoring pristine state.
    """
    if study_type not in {
        StudyType.WALK_FORWARD,
        StudyType.OOS_VALIDATION,
        StudyType.HOLDOUT_VALIDATION,
    }:
        raise ManifestError("expected a validation study type")
    if study_type is StudyType.OOS_VALIDATION and aggregate_path is None:
        raise ManifestError("OOS validation requires its existing aggregate")
    if study_type is StudyType.HOLDOUT_VALIDATION and ledger is None:
        raise ManifestError("holdout validation requires the authoritative ledger")
    root, study_path = artifact_root.resolve(), study_path.resolve()
    # Existing QF-40 snapshot consistency checks; this function does not
    # calculate stability, partition membership, or aggregate observations.
    source_provenance(source)
    definition, base = read_producer_record(study_path / "manifest.json")
    if (
        definition != source.definition.to_primitive()
        or configuration_identity(definition) != source.study_id
    ):
        raise ManifestError("walk-forward source identity mismatch")
    plan = mapping(mapping(definition["configuration"])["plan"])
    if plan != source.plan.to_manifest():
        raise ManifestError("validation plan differs from study")
    entries: list[ArtifactEntry] = []
    edges: list[ArtifactRelationship] = []

    def add(
        path: Path,
        category: ArtifactType,
        logical_id: str,
        *,
        location: str = "",
        bindings: PrimitiveMapping | None = None,
    ) -> ArtifactEntry:
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ManifestError("validation artifact is outside artifact root")
        entry = index_artifact(
            root,
            path=path.relative_to(root).as_posix(),
            artifact_type=category,
            schema_version="1",
            producer_study_id=source.study_id,
            producer_artifact_id=logical_id,
            json_pointer=location,
            bindings=bindings,
        )
        entries.append(entry)
        return entry

    def link(
        origin: ArtifactEntry, relationship: RelationshipType, target: ArtifactEntry
    ) -> None:
        edges.append(
            ArtifactRelationship(origin.artifact_id, relationship, target.artifact_id)
        )

    config_entry = add(
        study_path / "manifest.json",
        ArtifactType.CONFIGURATION,
        source.study_id,
        location=base,
    )
    plan_entry = add(
        study_path / "manifest.json",
        ArtifactType.VALIDATION_PLAN,
        source.plan.plan_id,
        location=base + "/configuration/plan",
        bindings={base + "/configuration/plan/plan_id": source.plan.plan_id},
    )
    link(config_entry, RelationshipType.CONFIGURED_BY, plan_entry)
    environment = mapping(plan["environment"])
    for name in ("dataset", "prediction_dataset"):
        if environment.get(name) is not None:
            dataset_entry = add(
                study_path / "manifest.json",
                ArtifactType.SOURCE_DATASET,
                name,
                location=base + "/configuration/plan/environment/" + name,
            )
            link(plan_entry, RelationshipType.DERIVED_FROM, dataset_entry)

    window_entries: list[ArtifactEntry] = []
    fold_references: list[Primitive] = []
    for fold, reference in zip(source.folds, source.references, strict=True):
        captured_state = reference.to_primitive()["state"]
        if fold.status is FoldStatus.COMPLETED and fold.artifact is None:
            raise ManifestError("completed fold is missing its OOS artifact")
        fold_root = study_path / "folds" / fold.fold_id
        record: PrimitiveMapping = {
            "fold_id": fold.fold_id,
            "status": fold.status.value,
            "selection_id": None
            if fold.selection is None
            else fold.selection.selection_id,
            "result_id": None,
        }
        state: PrimitiveMapping = {}
        state_path = fold_root / "state.json"
        if state_path.is_file():
            state, state_base = read_producer_record(state_path)
            if (
                state != captured_state
                or state.get("status") != fold.status.value
                or state.get("fold_id") != fold.fold_id
                or state.get("study_id") != source.study_id
            ):
                raise ManifestError("fold state differs from captured source")
            add(
                state_path,
                ArtifactType.FOLD_STATE,
                fold.fold_id + "/state",
                location=state_base,
            )
        elif (
            captured_state is not None
            or fold.status is not FoldStatus.PENDING
            or fold.selection is not None
            or fold.artifact is not None
        ):
            raise ManifestError("missing fold state")
        selected = None
        if fold.selection is not None:
            selection, selected_base = read_producer_record(
                fold_root / "selection.json"
            )
            if selection != fold.selection.to_primitive():
                raise ManifestError("frozen selection differs from captured source")
            selected = add(
                fold_root / "selection.json",
                ArtifactType.FROZEN_SELECTION,
                fold.selection.selection_id,
                location=selected_base,
                bindings={selected_base + "/selection_id": fold.selection.selection_id},
            )
            link(selected, RelationshipType.CONFIGURED_BY, plan_entry)
        if fold.artifact is not None:
            if fold.status is not FoldStatus.COMPLETED or selected is None:
                raise ManifestError("only completed selected fold artifacts are OOS")
            artifact, artifact_base = read_producer_record(fold_root / "oos.json")
            if artifact != fold.artifact.to_primitive() or state.get(
                "artifact_id"
            ) != configuration_identity(artifact):
                raise ManifestError("fold artifact differs from captured source")
            record["result_id"] = artifact["result_id"]
            window = add(
                fold_root / "oos.json",
                ArtifactType.WALK_FORWARD_WINDOW,
                fold.fold_id,
                location=artifact_base,
                bindings={
                    artifact_base + "/selection_id": fold.selection.selection_id
                    if fold.selection
                    else None,
                    artifact_base + "/result_id": artifact["result_id"],
                },
            )
            window_entries.append(window)
            link(window, RelationshipType.SELECTED_BY, selected)
            if isinstance(fold.artifact, BacktestOOSArtifact):
                export = fold_root / "test" / fold.artifact.export_location
                _validate_captured_backtest_export(
                    export, fold.artifact.export_fingerprint
                )
                backtest, _ = read_producer_record(export / "manifest.json")
                for entry in index_backtest_files(root, export, backtest, fold.fold_id):
                    entries.append(entry)
                    link(entry, RelationshipType.DERIVED_FROM, window)
        fold_references.append(record)

    configuration: PrimitiveMapping = {
        "study_definition": definition,
        "validation_lineage_id": source.lineage_id,
    }
    observations: PrimitiveMapping = {
        "plan_id": source.plan.plan_id,
        "walk_forward_study_id": source.study_id,
        "lineage_id": source.lineage_id,
        "folds": fold_references,
        "aggregate_id": None,
        "holdout": {
            "state": None,
            "authority": "quantforge.oos.HoldoutLedger",
            "reason": "ledger_not_supplied",
        },
    }
    if aggregate_path is not None:
        aggregate, aggregate_base = read_producer_record(aggregate_path)
        provenance = mapping(aggregate["provenance"])
        if (
            aggregate.get("schema_version") != "1"
            or aggregate.get("kind")
            not in {"prediction_oos_aggregate", "backtest_oos_aggregate"}
            or aggregate_path.stem != configuration_identity(aggregate)
            or provenance.get("study_id") != source.study_id
            or provenance.get("plan_id") != source.plan.plan_id
            or provenance.get("lineage_id") != source.lineage_id
            or provenance.get("study_definition") != definition
            or provenance.get("folds")
            != [item.to_primitive() for item in source.references]
        ):
            raise ManifestError("OOS aggregate has incompatible source provenance")
        aggregate_id = aggregate_path.stem
        observations["aggregate_id"] = aggregate_id
        configuration["aggregate_kind"] = aggregate["kind"]
        configuration["metric_fields"] = mapping(aggregate["summary"]).get(
            "metric_fields"
        )
        aggregate_entry = add(
            aggregate_path,
            ArtifactType.OOS_AGGREGATE,
            aggregate_id,
            location=aggregate_base,
        )
        stability = add(
            aggregate_path,
            ArtifactType.CONFIGURATION_STABILITY,
            aggregate_id + "/stability",
            location=aggregate_base + "/stability",
        )
        link(stability, RelationshipType.DESCRIBES, aggregate_entry)
        link(aggregate_entry, RelationshipType.VALIDATES, plan_entry)
        for window in window_entries:
            link(aggregate_entry, RelationshipType.AGGREGATES, window)

    if ledger is not None:
        # This existing method owns exposure auditing, lineage verification, and
        # the one-way state. Never infer state from QF-8 reservation metadata.
        current = ledger.state(source)
        reservation_path = (
            ledger.root / "lineages" / source.lineage_id / "reservation.json"
        )
        reservation, reservation_base = read_producer_record(reservation_path)
        if reservation != current.reservation.to_primitive():
            raise ManifestError("holdout reservation changed during indexing")
        reserved = add(
            reservation_path,
            ArtifactType.HOLDOUT_RESERVATION,
            source.lineage_id + "/reservation",
            location=reservation_base,
        )
        link(reserved, RelationshipType.CONFIGURED_BY, plan_entry)
        holdout: PrimitiveMapping = {
            "state": current.state.value,
            "authority": "quantforge.oos.HoldoutLedger",
            "lineage_id": source.lineage_id,
            "reservation_artifact_id": reserved.artifact_id,
            "consumption_artifact_id": None,
            "result_artifact_id": None,
        }
        if current.consumption is not None:
            consumed_path = ledger.root / "exposures" / f"{source.lineage_id}.json"
            consumed, consumed_base = read_producer_record(consumed_path)
            if consumed != current.consumption.to_primitive():
                raise ManifestError("holdout consumption changed during indexing")
            marker = add(
                consumed_path,
                ArtifactType.HOLDOUT_CONSUMPTION,
                source.lineage_id + "/consumption",
                location=consumed_base,
            )
            link(marker, RelationshipType.CONSUMES, reserved)
            holdout["consumption_artifact_id"] = marker.artifact_id
            holdout["consumption_run_id"] = consumed["consumption_run_id"]
            holdout["consumed_at"] = consumed["consumed_at"]
            holdout["request_id"] = consumed["request_id"]
            if current.result_reference is not None:
                reference = current.result_reference.to_primitive()
                result_path = ledger.root / text(reference["path"])
                result, result_base = read_producer_record(result_path)
                if configuration_identity(result) != reference["sha256"]:
                    raise ManifestError("holdout result changed during indexing")
                validate_holdout_artifact(source, consumed, result)
                result_entry = add(
                    result_path,
                    ArtifactType.HOLDOUT_RESULT,
                    text(reference["result_id"]),
                    location=result_base,
                )
                link(result_entry, RelationshipType.DERIVED_FROM, marker)
                holdout["result_artifact_id"] = result_entry.artifact_id
                artifact = mapping(result["artifact"])
                if artifact["kind"] == "backtest":
                    from quantforge.experiments.artifacts import local_path

                    export_root = result_path.parent / "evaluation"
                    export = local_path(export_root, text(artifact["export_location"]))
                    _validate_captured_backtest_export(
                        export, text(artifact.get("export_fingerprint"))
                    )
                    backtest, _ = read_producer_record(export / "manifest.json")
                    if backtest != mapping(
                        mapping(artifact.get("result")).get("manifest")
                    ):
                        raise ManifestError(
                            "holdout backtest snapshot differs from its export"
                        )
                    for entry in index_backtest_files(
                        root, export, backtest, source.lineage_id + "/holdout"
                    ):
                        entries.append(entry)
                        link(entry, RelationshipType.DERIVED_FROM, result_entry)
        observations["holdout"] = holdout
        # A consumption racing with this read cannot publish a stale pristine
        # observation. Later consumers still query the ledger for current state.
        if ledger.state(source) != current:
            raise ManifestError("holdout state changed during indexing; retry")
    return StudyArtifacts(
        StudyProvenance(
            study_type, source.study_id, snapshot(configuration), snapshot(observations)
        ),
        ArtifactIndex(tuple(entries), tuple(edges)),
    )
