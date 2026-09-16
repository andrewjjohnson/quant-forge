"""Versioned immutable experiment identity, provenance, and execution records."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.experiments._json import (
    EMPTY_SNAPSHOT,
    ManifestError,
    digest,
    snapshot,
    text,
)
from quantforge.experiments.artifacts import ArtifactIndex

MANIFEST_SCHEMA_VERSION = "1"


class StudyType(StrEnum):
    PREDICTION = "prediction"
    PREDICTION_WINDOW = "prediction_window"
    FEATURE_DATASET = "feature_dataset"
    PARAMETER_STUDY = "parameter_study"
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    WALK_FORWARD = "walk_forward"
    OOS_VALIDATION = "oos_validation"
    HOLDOUT_VALIDATION = "holdout_validation"


@dataclass(frozen=True, slots=True)
class CodeProvenance:
    """Execution environment supplied by its owner; None means unavailable.

    Never substitute the manifest writer's runtime for a historical execution.
    Backend versions are retained separately in the producing study snapshot.
    """

    quantforge_version: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    dependency_lock_sha256: str | None = None
    python_version: str | None = None
    dependencies: PrimitiveMappingSnapshot = EMPTY_SNAPSHOT

    def __post_init__(self) -> None:
        for item in (self.quantforge_version, self.git_commit, self.python_version):
            if item is not None:
                text(item)
        if self.git_dirty is not None and type(self.git_dirty) is not bool:
            raise ManifestError("git dirty state must be boolean or unavailable")
        if self.dependency_lock_sha256 is not None:
            digest(self.dependency_lock_sha256)
        for name, version in self.dependencies.to_primitive().items():
            text(name)
            if version is not None:
                text(version)
        snapshot(self.to_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        result: PrimitiveMapping = {
            "quantforge_version": self.quantforge_version,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "python_version": self.python_version,
            "dependencies": self.dependencies.to_primitive(),
        }
        snapshot(result)
        return result


@dataclass(frozen=True, slots=True)
class ExecutionProvenance:
    run_id: str
    created_at: datetime
    code: CodeProvenance = CodeProvenance()
    execution_started_at: datetime | None = None
    random_seeds: PrimitiveMappingSnapshot = EMPTY_SNAPSHOT

    def __post_init__(self) -> None:
        text(self.run_id)
        for timestamp in (self.created_at, self.execution_started_at):
            if timestamp is not None and (
                timestamp.tzinfo is None or timestamp.utcoffset() is None
            ):
                raise ManifestError("execution timestamps must be timezone-aware")
        snapshot(self.to_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "execution_started_at": None
            if self.execution_started_at is None
            else self.execution_started_at.astimezone(UTC).isoformat(),
            "code": self.code.to_primitive(),
            "random_seeds": self.random_seeds.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class StudyProvenance:
    study_type: StudyType
    producer_study_id: str
    configuration: PrimitiveMappingSnapshot
    # References/status/counts: these do not define scientific configuration.
    observations: PrimitiveMappingSnapshot = EMPTY_SNAPSHOT

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.study_type), StudyType):
            raise ManifestError("study type must be typed")
        text(self.producer_study_id)
        snapshot(self.configuration.to_primitive())
        snapshot(self.observations.to_primitive())

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "study_type": self.study_type.value,
            "producer_study_id": self.producer_study_id,
            "configuration": self.configuration.to_primitive(),
            "observations": self.observations.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    provenance: StudyProvenance
    execution: ExecutionProvenance
    artifacts: ArtifactIndex

    @property
    def study_id(self) -> str:
        """Bind the existing identity plus supplemental material provenance.

        This is a namespaced experiment identity, not a replacement for any
        producer identity algorithm. Result bytes, paths, timestamps, fold
        outcomes, and current holdout state are deliberately excluded.
        """
        return configuration_identity(
            {
                "component": "quantforge_experiment_study",
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "study_type": self.provenance.study_type.value,
                "producer_study_id": self.provenance.producer_study_id,
                "configuration": self.provenance.configuration.to_primitive(),
                "code": self.execution.code.to_primitive(),
                "random_seeds": self.execution.random_seeds.to_primitive(),
            }
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "component": "quantforge_experiment_manifest",
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "study_id": self.study_id,
            "provenance": self.provenance.to_primitive(),
            "execution": self.execution.to_primitive(),
            "artifact_index_id": self.artifacts.index_id,
            "artifact_index": self.artifacts.to_primitive(),
        }

    @property
    def manifest_id(self) -> str:
        return configuration_identity(self.to_primitive())

    def serialize(self) -> bytes:
        return (
            snapshot(
                {"manifest_id": self.manifest_id, **self.to_primitive()}
            ).canonical_json
            + "\n"
        ).encode()
