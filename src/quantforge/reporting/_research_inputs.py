"""Read indexed evidence only, delegating integrity and holdout state to owners."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import TYPE_CHECKING

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import (
    ArtifactFormat,
    ArtifactIndex,
    ArtifactType,
    ExperimentManifest,
    ManifestError,
    RelationshipType,
    verify_artifacts,
)
from quantforge.experiments._json import parse_json, pointer, safe_metadata, snapshot
from quantforge.experiments._producer_snapshot import ProducerReadSet
from quantforge.experiments.artifacts import local_path
from quantforge.reporting.research_models import (
    ReportArtifact,
    ResearchReportConfig,
    ResearchReportError,
)

if TYPE_CHECKING:
    from quantforge.oos import HoldoutLedger, OOSSource

RAW_RECORD_KEYS = frozenset({"rows", "observations", "decisions", "preview"})
RESULT_HISTORY_KEYS = RAW_RECORD_KEYS | frozenset(
    {
        "signals",
        "orders",
        "fills",
        "positions",
        "completed_trades",
        "open_trades",
        "daily_equity",
        "dividend_cashflows",
        "split_adjustments",
        "benchmark_daily_equity",
        "benchmark_dividend_cashflows",
        "benchmark_split_adjustments",
        "native_windows",
    }
)


def as_mapping(value: Primitive) -> PrimitiveMapping:
    return value if isinstance(value, dict) else {}


def artifact_value(artifact: ReportArtifact) -> Primitive:
    return (
        None if artifact.content is None else artifact.content.to_primitive()["value"]
    )


def result_metadata(value: Primitive) -> Primitive:
    """Retain result manifests and summaries without embedded raw histories."""
    if not isinstance(value, dict):
        return value
    return {
        key: result_metadata(child)
        if key in {"prediction_study", "prediction_window", "result", "artifact"}
        else child
        for key, child in value.items()
        if not (key in RESULT_HISTORY_KEYS and isinstance(child, list))
    }


def read_artifacts(
    index: ArtifactIndex, root: Path, config: ResearchReportConfig
) -> tuple[ReportArtifact, ...]:
    integrity = verify_artifacts(index, root)
    issues = {issue.artifact_id: issue.code for issue in integrity.issues}
    result: list[ReportArtifact] = []
    for entry in sorted(index.entries, key=lambda item: (item.path, item.json_pointer)):
        status = issues.get(entry.artifact_id)
        if status or entry.artifact_id in integrity.absent_optional:
            result.append(ReportArtifact(entry, status or "unavailable_optional"))
            continue
        content = None
        if entry.file_format is ArtifactFormat.JSONL:
            # QF-9 binds the header to the immutable file. Aggregate display
            # consumes these small records, never scans compact decisions.
            content = snapshot(
                {
                    "value": entry.bindings.to_primitive().get("/header")
                    if entry.json_pointer == "/header"
                    else entry.metadata.to_primitive()
                }
            )
        raw_records = entry.file_format is ArtifactFormat.JSON and (
            any(part in RAW_RECORD_KEYS for part in entry.json_pointer.split("/")[1:])
            or (
                entry.artifact_type is ArtifactType.FEATURE_DATASET
                and entry.producer_artifact_id.startswith("rows/")
            )
        )
        # QF-9 already verified these links. Raw observations are neither warning
        # metadata nor display summaries, so do not retain their full payloads.
        if not raw_records and entry.file_format in {
            ArtifactFormat.JSON,
            ArtifactFormat.CSV,
        }:
            reads = ProducerReadSet()
            try:
                path = local_path(root, entry.path)
                raw = path.read_bytes()
                reads.expect(path, raw)
                # Bind the actual buffer before parsing, including replacement races.
                reads.verify(ArtifactIndex((entry,)), root)
                value: Primitive
                if entry.file_format is ArtifactFormat.JSON:
                    document = parse_json(raw)
                    value = pointer(document, entry.json_pointer)
                    if entry.artifact_type in {
                        ArtifactType.PREDICTION_RESULT,
                        ArtifactType.BACKTEST_RESULT,
                        ArtifactType.WALK_FORWARD_WINDOW,
                        ArtifactType.OOS_AGGREGATE,
                        ArtifactType.HOLDOUT_RESULT,
                    }:
                        value = result_metadata(value)
                else:
                    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
                    fieldnames = reader.fieldnames
                    if (
                        not fieldnames
                        or any(not name for name in fieldnames)
                        or len(set(fieldnames)) != len(fieldnames)
                    ):
                        raise ManifestError("invalid CSV header")
                    rows: list[Primitive] = []
                    truncated = False
                    for row in reader:
                        if len(rows) == config.maximum_preview_rows:
                            truncated = True
                            break
                        if None in row or any(item is None for item in row.values()):
                            raise ManifestError("invalid CSV record")
                        record: PrimitiveMapping = {
                            key: None if item == "" else item
                            for key, item in row.items()
                        }
                        rows.append(record)
                    value = {"preview": rows, "preview_truncated": truncated}
                safe_metadata(value)
                content = snapshot({"value": value})
            except (OSError, UnicodeError, csv.Error, ManifestError):
                # Never include external exception text or unsafe artifact contents.
                result.append(ReportArtifact(entry, "invalid_or_changed_artifact"))
                continue
        result.append(ReportArtifact(entry, "verified", content))
    return tuple(result)


def validation_observations(manifest: ExperimentManifest) -> PrimitiveMapping:
    observed = manifest.provenance.observations.to_primitive()
    return as_mapping(observed["validation"]) if "validation" in observed else observed


def lineage_artifacts(
    manifest: ExperimentManifest, artifacts: tuple[ReportArtifact, ...]
) -> tuple[ReportArtifact, ...]:
    """Select owned evidence and explicit connections in the validated QF-9 index."""
    entries = {item.artifact_id: item for item in manifest.artifacts.entries}
    owners = {manifest.provenance.producer_study_id}
    validation_id = validation_observations(manifest).get("walk_forward_study_id")
    if isinstance(validation_id, str) and any(
        edge.relationship is RelationshipType.VALIDATES
        and entries[edge.source_id].producer_study_id == validation_id
        and entries[edge.source_id].artifact_type
        in {ArtifactType.WALK_FORWARD_WINDOW, ArtifactType.HOLDOUT_RESULT}
        and entries[edge.target_id].producer_study_id in owners
        and entries[edge.target_id].artifact_type is ArtifactType.CONFIGURATION
        for edge in manifest.artifacts.relationships
    ):
        # QF-9 attachments validate the captured primary result. Their owner also
        # supplies fold states that do not have individual relationship edges.
        owners.add(validation_id)
    neighbors: dict[str, set[str]] = {artifact_id: set() for artifact_id in entries}
    for edge in manifest.artifacts.relationships:
        neighbors[edge.source_id].add(edge.target_id)
        neighbors[edge.target_id].add(edge.source_id)
    included = {
        artifact_id
        for artifact_id, entry in entries.items()
        if entry.producer_study_id in owners
    }
    pending = list(included)
    while pending:
        for neighbor in neighbors[pending.pop()]:
            if neighbor not in included:
                included.add(neighbor)
                pending.append(neighbor)
    return tuple(item for item in artifacts if item.entry.artifact_id in included)


def holdout_snapshot(
    manifest: ExperimentManifest,
    artifacts: tuple[ReportArtifact, ...],
    source: OOSSource | None,
    ledger: HoldoutLedger | None,
) -> PrimitiveMapping:
    """Never promote a saved reservation to a current pristine assertion."""
    observed = validation_observations(manifest)
    historical = as_mapping(observed.get("holdout"))
    consumed = historical.get("state") == "consumed" or any(
        item.entry.artifact_type
        in {ArtifactType.HOLDOUT_CONSUMPTION, ArtifactType.HOLDOUT_RESULT}
        for item in lineage_artifacts(manifest, artifacts)
    )
    if (source is None) != (ledger is None):
        raise ResearchReportError("holdout source and authoritative ledger are paired")
    if source is None or ledger is None:
        return {
            "state": "consumed" if consumed else "unavailable",
            "authority": "QF-40 HoldoutLedger",
            "authority_checked": False,
            "explanation": (
                "Consumption evidence retained; current ledger was not supplied."
                if consumed
                else "Current ledger was not supplied. A saved reservation does not "
                "establish current unseen status."
            ),
            "result": None,
        }
    configuration = manifest.provenance.configuration.to_primitive()
    validation = as_mapping(configuration.get("validation", configuration))
    if (
        observed.get("walk_forward_study_id") != source.study_id
        or observed.get("lineage_id") != source.lineage_id
        or observed.get("plan_id") != source.plan.plan_id
        or validation.get("study_definition") != source.definition.to_primitive()
    ):
        raise ResearchReportError("holdout authority does not match manifest lineage")
    # This is the only reporting call into QF-40. No reserve/consume/evaluate path.
    current = ledger.state(source).to_primitive()
    if consumed and current["state"] != "consumed":
        raise ResearchReportError("holdout authority contradicts consumed evidence")
    reference = as_mapping(current["result_reference"])
    return {
        "state": current["state"],
        "authority": "QF-40 HoldoutLedger",
        "authority_checked": True,
        "lineage_id": source.lineage_id,
        "reservation": current["reservation"],
        "consumption": current["consumption"],
        # Results are displayed only from QF-9 indexed, verified artifacts below.
        "result_status": current["result_status"],
        "result_reference": {
            key: reference.get(key) for key in ("path", "sha256", "result_id")
        }
        if reference
        else None,
    }
