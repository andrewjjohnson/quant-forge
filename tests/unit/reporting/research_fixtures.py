"""Native offline producer exports plus deliberately tiny indexed metadata fixtures."""

from pathlib import Path

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.experiments import (
    ArtifactIndex,
    ArtifactType,
    StudyArtifacts,
    StudyProvenance,
    StudyType,
    create_manifest,
    index_artifact,
    inspect_study,
    write_manifest,
)
from tests.unit.experiments.test_contracts import execution, write_json


def native_manifest(root: Path, kind: StudyType, source: Path) -> Path:
    return write_manifest(
        create_manifest(inspect_study(kind, source, artifact_root=root), execution()),
        root / "experiments",
        artifact_root=root,
    )


def metadata_manifest(
    root: Path,
    *,
    kind: StudyType = StudyType.PREDICTION,
    configuration: PrimitiveMapping | None = None,
    observations: PrimitiveMapping | None = None,
    payload: PrimitiveMapping | None = None,
    category: ArtifactType = ArtifactType.PREDICTION_RESULT,
) -> Path:
    write_json(root / "result.json", payload or {})
    artifact = index_artifact(
        root,
        path="result.json",
        artifact_type=category,
        schema_version="1",
        producer_study_id="fixture-study",
        producer_artifact_id="result",
    )
    study = StudyArtifacts(
        StudyProvenance(
            kind,
            "fixture-study",
            PrimitiveMappingSnapshot.capture(configuration or {}),
            PrimitiveMappingSnapshot.capture(observations or {}),
        ),
        ArtifactIndex((artifact,)),
    )
    return write_manifest(
        create_manifest(study, execution()), root / "experiments", artifact_root=root
    )
