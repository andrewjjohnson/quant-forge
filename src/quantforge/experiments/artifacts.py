"""Typed local artifact references and byte-level integrity checks."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.experiments._json import (
    EMPTY_SNAPSHOT,
    ManifestError,
    digest,
    pointer,
    read_json,
    snapshot,
    text,
)


class ArtifactType(StrEnum):
    CONFIGURATION = "configuration"
    SOURCE_DATASET = "source_dataset"
    DATA_QUALITY = "data_quality"
    PREDICTION_RESULT = "prediction_result"
    FEATURE_DATASET = "feature_dataset"
    FEATURE_SCHEMA = "feature_schema"
    OUTCOME_LABELS = "outcome_labels"
    TRIAL_RESULT = "trial_result"
    PARAMETER_SUMMARY = "parameter_summary"
    BACKTEST_RESULT = "backtest_result"
    VALIDATION_PLAN = "validation_plan"
    FOLD_STATE = "fold_state"
    FROZEN_SELECTION = "frozen_selection"
    WALK_FORWARD_WINDOW = "walk_forward_window"
    OOS_AGGREGATE = "oos_aggregate"
    CONFIGURATION_STABILITY = "configuration_stability"
    HOLDOUT_RESERVATION = "holdout_reservation"
    HOLDOUT_CONSUMPTION = "holdout_consumption"
    HOLDOUT_RESULT = "holdout_result"
    INSPECTION = "inspection"
    CHART = "chart"
    REPORT = "report"


class ArtifactFormat(StrEnum):
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    HTML = "html"
    SVG = "svg"
    PNG = "png"
    BINARY = "binary"


class RelationshipType(StrEnum):
    DERIVED_FROM = "derived_from"
    CONFIGURED_BY = "configured_by"
    USES_FEATURES = "uses_features"
    SELECTED_BY = "selected_by"
    AGGREGATES = "aggregates"
    VALIDATES = "validates"
    VISUALIZES = "visualizes"
    DESCRIBES = "describes"
    CONSUMES = "consumes"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _relative_path(location: str) -> PurePosixPath:
    relative = PurePosixPath(location)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or str(relative) != location
        or location == "."
    ):
        raise ManifestError("artifact path must be normalized and relative to its root")
    return relative


def local_path(root: Path, location: str) -> Path:
    _relative_path(location)
    path = root / location
    if not path.resolve().is_relative_to(root.resolve()):
        raise ManifestError("artifact path escapes its root")
    return path


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    artifact_type: ArtifactType
    schema_version: str
    path: str
    file_format: ArtifactFormat
    producer_study_id: str
    producer_artifact_id: str
    sha256: str | None
    producer_run_id: str | None = None
    json_pointer: str = ""
    metadata: PrimitiveMappingSnapshot = EMPTY_SNAPSHOT
    # Exact stored metadata at JSON pointers, checked without research validation.
    bindings: PrimitiveMappingSnapshot = EMPTY_SNAPSHOT
    required: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            cast(object, self.artifact_type), ArtifactType
        ) or not isinstance(cast(object, self.file_format), ArtifactFormat):
            raise ManifestError("artifact categories and formats must be typed")
        for item in (
            self.schema_version,
            self.path,
            self.producer_study_id,
            self.producer_artifact_id,
        ):
            text(item)
        if type(self.required) is not bool or (self.required and self.sha256 is None):
            raise ManifestError("required artifacts need a content hash")
        if self.sha256 is not None:
            digest(self.sha256)
        if (
            self.json_pointer or self.bindings.to_primitive()
        ) and self.file_format is not ArtifactFormat.JSON:
            raise ManifestError("only JSON artifacts support metadata bindings")
        _relative_path(self.path)
        if self.producer_run_id is not None:
            text(self.producer_run_id)
        if not isinstance(cast(object, self.json_pointer), str):
            raise ManifestError("JSON pointer must be a string")
        snapshot(self.to_primitive())

    @property
    def artifact_id(self) -> str:
        return configuration_identity(self.to_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "artifact_type": self.artifact_type.value,
            "schema_version": self.schema_version,
            "path": self.path,
            "file_format": self.file_format.value,
            "producer_study_id": self.producer_study_id,
            "producer_artifact_id": self.producer_artifact_id,
            "producer_run_id": self.producer_run_id,
            "sha256": self.sha256,
            "json_pointer": self.json_pointer,
            "metadata": self.metadata.to_primitive(),
            "bindings": self.bindings.to_primitive(),
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class ArtifactRelationship:
    source_id: str
    relationship: RelationshipType
    target_id: str

    def to_primitive(self) -> PrimitiveMapping:
        if not isinstance(cast(object, self.relationship), RelationshipType):
            raise ManifestError("relationship must be typed")
        return {
            "source_id": digest(self.source_id),
            "relationship": self.relationship.value,
            "target_id": digest(self.target_id),
        }


@dataclass(frozen=True, slots=True)
class ArtifactIndex:
    entries: tuple[ArtifactEntry, ...]
    relationships: tuple[ArtifactRelationship, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.entries), tuple) or not isinstance(
            cast(object, self.relationships), tuple
        ):
            raise ManifestError("artifact index collections must be immutable")
        ids = [item.artifact_id for item in self.entries]
        logical = [
            (
                item.producer_study_id,
                item.producer_run_id,
                item.producer_artifact_id,
            )
            for item in self.entries
        ]
        locations = [(item.path, item.json_pointer) for item in self.entries]
        if (
            len(set(ids)) != len(ids)
            or len(set(logical)) != len(logical)
            or len(set(locations)) != len(locations)
        ):
            raise ManifestError("duplicate or inconsistent artifact identity")
        edges = [
            configuration_identity(item.to_primitive()) for item in self.relationships
        ]
        if len(set(edges)) != len(edges):
            raise ManifestError("duplicate artifact relationship")
        for edge in self.relationships:
            if (
                edge.source_id not in ids
                or edge.target_id not in ids
                or edge.source_id == edge.target_id
            ):
                raise ManifestError("artifact relationship has invalid references")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": "1",
            "entries": [
                {"artifact_id": item.artifact_id, **item.to_primitive()}
                for item in sorted(self.entries, key=lambda item: item.artifact_id)
            ],
            "relationships": cast(
                list[Primitive],
                sorted(
                    [item.to_primitive() for item in self.relationships],
                    key=configuration_identity,
                ),
            ),
        }

    @property
    def index_id(self) -> str:
        return configuration_identity(self.to_primitive())


@dataclass(frozen=True, slots=True)
class IntegrityIssue:
    artifact_id: str
    code: str


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    issues: tuple[IntegrityIssue, ...]
    absent_optional: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.issues

    def require_valid(self) -> None:
        if not self.valid:
            raise ManifestError(
                "artifact integrity verification failed: "
                + ", ".join(sorted({item.code for item in self.issues}))
            )


def verify_artifacts(index: ArtifactIndex, root: Path) -> IntegrityReport:
    issues: list[IntegrityIssue] = []
    absent: list[str] = []
    for entry in index.entries:
        code = None
        try:
            path = local_path(root, entry.path)
            if not path.is_file():
                if entry.required or entry.sha256 is not None:
                    code = "missing_artifact"
                else:
                    absent.append(entry.artifact_id)
            elif entry.sha256 != file_sha256(path):
                code = "content_hash_mismatch"
            elif entry.file_format is ArtifactFormat.JSON:
                document = read_json(path)
                pointer(document, entry.json_pointer)
                if any(
                    pointer(document, key) != value
                    for key, value in entry.bindings.to_primitive().items()
                ):
                    code = "incompatible_metadata"
        except (OSError, ManifestError):
            code = "invalid_artifact"
        if code:
            issues.append(IntegrityIssue(entry.artifact_id, code))
    return IntegrityReport(tuple(issues), tuple(sorted(absent)))


def index_artifact(
    root: Path,
    *,
    path: str,
    artifact_type: ArtifactType,
    schema_version: str,
    producer_study_id: str,
    producer_artifact_id: str,
    file_format: ArtifactFormat | None = None,
    producer_run_id: str | None = None,
    json_pointer: str = "",
    metadata: PrimitiveMapping | None = None,
    bindings: PrimitiveMapping | None = None,
    required: bool = True,
) -> ArtifactEntry:
    resolved = local_path(root, path)
    try:
        format_ = file_format or ArtifactFormat(resolved.suffix.lstrip("."))
        entry = ArtifactEntry(
            artifact_type,
            schema_version,
            path,
            format_,
            producer_study_id,
            producer_artifact_id,
            file_sha256(resolved) if resolved.is_file() else None,
            producer_run_id,
            json_pointer,
            snapshot(metadata or {}),
            snapshot(bindings or {}),
            required,
        )
        verify_artifacts(ArtifactIndex((entry,)), root).require_valid()
        return entry
    except OSError as error:
        raise ManifestError("cannot hash artifact") from error
