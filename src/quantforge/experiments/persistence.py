"""Strict v1 reader and atomic, no-clobber, content-addressed manifest writer."""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import (
    ManifestError,
    mapping,
    read_json,
    snapshot,
    text,
)
from quantforge.experiments.artifacts import (
    ArtifactEntry,
    ArtifactFormat,
    ArtifactIndex,
    ArtifactRelationship,
    ArtifactType,
    RelationshipType,
    verify_artifacts,
)
from quantforge.experiments.models import (
    MANIFEST_SCHEMA_VERSION,
    CodeProvenance,
    ExecutionProvenance,
    ExperimentManifest,
    StudyProvenance,
    StudyType,
)


def write_manifest(
    manifest: ExperimentManifest, output_root: Path, *, artifact_root: Path
) -> Path:
    """Verify references, then publish once; exact retries compare bytes."""
    verify_artifacts(manifest.artifacts, artifact_root).require_valid()
    content = manifest.serialize()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"{manifest.manifest_id}.json"
    temporary: Path | None = None
    try:
        descriptor, filename = tempfile.mkstemp(prefix=".manifest-", dir=output_root)
        temporary = Path(filename)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != content:
                raise ManifestError(
                    "immutable manifest differs; refusing overwrite"
                ) from None
        directory = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination
    except OSError as error:
        raise ManifestError("failed to persist immutable manifest") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _records(value: object) -> list[PrimitiveMapping]:
    if not isinstance(value, list):
        raise ManifestError("expected metadata records")
    return [mapping(item) for item in cast(list[object], value)]


def read_manifest(
    path: Path, *, artifact_root: Path | None = None
) -> ExperimentManifest:
    """Read v1 only; no historical field or backend is guessed or migrated.

    Existing producer manifests use different schemas and enter via adapters.
    A future QF-9 migration must be explicit before another version is accepted.
    """
    payload = read_json(path)
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError("unsupported experiment manifest schema version")
    try:
        index = mapping(payload["artifact_index"])
        entries: list[ArtifactEntry] = []
        for item in _records(index["entries"]):
            entry = ArtifactEntry(
                ArtifactType(text(item["artifact_type"])),
                text(item["schema_version"]),
                text(item["path"]),
                ArtifactFormat(text(item["file_format"])),
                text(item["producer_study_id"]),
                text(item["producer_artifact_id"]),
                None if item["sha256"] is None else text(item["sha256"]),
                None
                if item["producer_run_id"] is None
                else text(item["producer_run_id"]),
                cast(str, item["json_pointer"]),
                snapshot(mapping(item["metadata"])),
                snapshot(mapping(item["bindings"])),
                cast(bool, item["required"]),
            )
            if {"artifact_id": entry.artifact_id, **entry.to_primitive()} != item:
                raise ManifestError("incompatible artifact identity or metadata")
            entries.append(entry)
        edges = tuple(
            ArtifactRelationship(
                text(item["source_id"]),
                RelationshipType(text(item["relationship"])),
                text(item["target_id"]),
            )
            for item in _records(index["relationships"])
        )
        provenance = mapping(payload["provenance"])
        execution = mapping(payload["execution"])
        code = mapping(execution["code"])
        code_record = CodeProvenance(
            cast(str | None, code["quantforge_version"]),
            cast(str | None, code["git_commit"]),
            cast(bool | None, code["git_dirty"]),
            cast(str | None, code["dependency_lock_sha256"]),
            cast(str | None, code["python_version"]),
            snapshot(mapping(code["dependencies"])),
        )
        result = ExperimentManifest(
            StudyProvenance(
                StudyType(text(provenance["study_type"])),
                text(provenance["producer_study_id"]),
                snapshot(mapping(provenance["configuration"])),
                snapshot(mapping(provenance["observations"])),
            ),
            ExecutionProvenance(
                text(execution["run_id"]),
                datetime.fromisoformat(text(execution["created_at"])),
                code_record,
                None
                if execution["execution_started_at"] is None
                else datetime.fromisoformat(text(execution["execution_started_at"])),
                snapshot(mapping(execution["random_seeds"])),
            ),
            ArtifactIndex(tuple(entries), edges),
        )
        if (
            payload != {"manifest_id": result.manifest_id, **result.to_primitive()}
            or path.stem != result.manifest_id
            or path.read_bytes() != result.serialize()
        ):
            raise ManifestError(
                "manifest identity, canonical bytes, or schema mismatch"
            )
        if artifact_root is not None:
            verify_artifacts(result.artifacts, artifact_root).require_valid()
        return result
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError("invalid or incompatible experiment manifest") from error


def read_producer_record(path: Path) -> tuple[PrimitiveMapping, str]:
    """Unwrap existing QF-39/QF-40 hash envelopes without execution callbacks."""
    document = read_json(path)
    if set(document) == {"payload", "fingerprint"}:
        payload = mapping(document["payload"])
        if configuration_identity(payload) != document["fingerprint"]:
            raise ManifestError("producer artifact fingerprint mismatch")
        return payload, "/payload"
    return document, ""
