from pathlib import Path

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import (
    ArtifactEntry,
    ArtifactRelationship,
    ArtifactType,
    RelationshipType,
    StudyArtifacts,
    StudyType,
    create_manifest,
    index_artifact,
    read_manifest,
    write_manifest,
)
from quantforge.reporting import (
    ResearchReportConfig,
    build_research_report,
    export_research_report,
)
from tests.unit.experiments.test_contracts import write_json
from tests.unit.reporting.research_fixtures import metadata_manifest
from tests.unit.reporting.test_research_reports import codes, values


@pytest.mark.parametrize(
    ("kind", "category", "title", "field"),
    [
        (
            StudyType.PREDICTION,
            ArtifactType.PREDICTION_RESULT,
            "Prediction metrics and comparisons",
            "summary",
        ),
        (
            StudyType.FEATURE_DATASET,
            ArtifactType.FEATURE_DATASET,
            "Published feature / outcome distributions",
            "summary",
        ),
        (
            StudyType.PARAMETER_STUDY,
            ArtifactType.PARAMETER_SUMMARY,
            "Ranked configurations — published order",
            "rankings",
        ),
        (
            StudyType.OPTIMIZATION,
            ArtifactType.PARAMETER_SUMMARY,
            "Ranked configurations — published order",
            "eligible_rankings",
        ),
    ],
)
@pytest.mark.parametrize("linked", [False, True])
def test_other_producer_requires_explicit_lineage_for_sections_and_warnings(
    tmp_path: Path,
    kind: StudyType,
    category: ArtifactType,
    title: str,
    field: str,
    linked: bool,
) -> None:
    primary: PrimitiveMapping = {field: {"label": "PRIMARY-EVIDENCE"}}
    foreign: PrimitiveMapping = {
        field: {"label": "FOREIGN-EVIDENCE", "prediction_count": 1}
    }
    path = metadata_manifest(tmp_path, kind=kind, category=category, payload=primary)
    manifest = read_manifest(path)
    write_json(tmp_path / "foreign.json", foreign)
    entry = index_artifact(
        tmp_path,
        path="foreign.json",
        artifact_type=category,
        schema_version="1",
        producer_study_id="other-study",
        producer_artifact_id="result",
    )
    relationships = (
        (
            ArtifactRelationship(
                entry.artifact_id,
                RelationshipType.DERIVED_FROM,
                manifest.artifacts.entries[0].artifact_id,
            ),
        )
        if linked
        else ()
    )
    path = write_manifest(
        create_manifest(
            StudyArtifacts(manifest.provenance, manifest.artifacts),
            manifest.execution,
            additional_artifacts=(entry,),
            relationships=relationships,
        ),
        tmp_path / "experiments",
        artifact_root=tmp_path,
    )
    report = build_research_report(
        path, artifact_root=tmp_path, config=ResearchReportConfig(minimum_sample_size=5)
    )
    assert primary in values(report, title)
    assert (foreign in values(report, title)) is linked
    assert ("LOW_SAMPLE_SIZE" in codes(report)) is linked
    html = export_research_report(report, tmp_path / "reports").read_text()
    assert ("FOREIGN-EVIDENCE" in html) is linked
    # The complete index retains exact evidence links without attributing results.
    assert entry.artifact_id in html
    assert "./../foreign.json" in html


@pytest.mark.parametrize(
    "category",
    [
        ArtifactType.OOS_AGGREGATE,
        ArtifactType.WALK_FORWARD_WINDOW,
        ArtifactType.HOLDOUT_RESULT,
    ],
)
@pytest.mark.parametrize("linked", [False, True])
def test_unrelated_validation_cannot_change_primary_research_state(
    tmp_path: Path, category: ArtifactType, linked: bool
) -> None:
    path = metadata_manifest(tmp_path)
    manifest = read_manifest(path)
    write_json(tmp_path / "foreign.json", {"summary": {"label": "FOREIGN-OOS"}})
    entry = index_artifact(
        tmp_path,
        path="foreign.json",
        artifact_type=category,
        schema_version="1",
        producer_study_id="other-study",
        producer_artifact_id="result",
    )
    relationships = (
        (
            ArtifactRelationship(
                entry.artifact_id,
                RelationshipType.VALIDATES,
                manifest.artifacts.entries[0].artifact_id,
            ),
        )
        if linked
        else ()
    )
    path = write_manifest(
        create_manifest(
            StudyArtifacts(manifest.provenance, manifest.artifacts),
            manifest.execution,
            additional_artifacts=(entry,),
            relationships=relationships,
        ),
        tmp_path / "experiments",
        artifact_root=tmp_path,
    )
    report = build_research_report(path, artifact_root=tmp_path)
    assert ("IN_SAMPLE_ONLY" in codes(report)) is (not linked)
    consumed = linked and category is ArtifactType.HOLDOUT_RESULT
    assert ("HOLDOUT_ALREADY_CONSUMED" in codes(report)) is consumed
    assert report.header.to_primitive()["holdout_state"] == (
        "consumed" if consumed else "unavailable"
    )


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_invalid_oos_aggregate_warns_even_with_verified_windows(
    tmp_path: Path, damage: str
) -> None:
    path = metadata_manifest(
        tmp_path,
        kind=StudyType.OOS_VALIDATION,
        category=ArtifactType.WALK_FORWARD_WINDOW,
        observations={"folds": [{"fold_id": "one", "status": "completed"}]},
    )
    manifest = read_manifest(path)
    aggregate_path = tmp_path / "aggregate.json"
    write_json(aggregate_path, {"summary": {"complete": True}})
    entry = index_artifact(
        tmp_path,
        path="aggregate.json",
        artifact_type=ArtifactType.OOS_AGGREGATE,
        schema_version="1",
        producer_study_id="fixture-study",
        producer_artifact_id="aggregate",
    )
    path = write_manifest(
        create_manifest(
            StudyArtifacts(manifest.provenance, manifest.artifacts),
            manifest.execution,
            additional_artifacts=(entry,),
        ),
        tmp_path / "experiments",
        artifact_root=tmp_path,
    )
    if damage == "missing":
        aggregate_path.unlink()
    else:
        write_json(aggregate_path, {"summary": {"complete": False}})
    report = build_research_report(path, artifact_root=tmp_path)
    assert "ARTIFACT_INTEGRITY" in codes(report)
    assert "IN_SAMPLE_ONLY" not in codes(report)
    assert values(report, "Walk-forward OOS summary") == [None]
    assert any(
        warning.code == "FAILED_OR_MISSING_OOS_WINDOWS"
        and warning.source == entry.artifact_id
        for warning in report.warnings
    )


def test_lineage_follows_relationship_paths_without_importing_unlinked_peers(
    tmp_path: Path,
) -> None:
    path = metadata_manifest(tmp_path)
    manifest = read_manifest(path)
    entries: list[ArtifactEntry] = []
    for name in ("bridge", "linked", "cycle", "unlinked"):
        write_json(tmp_path / f"{name}.json", {"summary": {"label": name}})
        entries.append(
            index_artifact(
                tmp_path,
                path=f"{name}.json",
                artifact_type=ArtifactType.PREDICTION_RESULT,
                schema_version="1",
                producer_study_id="other-study",
                producer_artifact_id=name,
            )
        )
    bridge, linked, cycle, unlinked = entries
    relationships = tuple(
        ArtifactRelationship(
            origin.artifact_id, RelationshipType.DERIVED_FROM, target.artifact_id
        )
        for origin, target in (
            (manifest.artifacts.entries[0], bridge),
            (linked, bridge),
            (cycle, linked),
            (bridge, cycle),
        )
    )
    path = write_manifest(
        create_manifest(
            StudyArtifacts(manifest.provenance, manifest.artifacts),
            manifest.execution,
            additional_artifacts=tuple(entries),
            relationships=relationships,
        ),
        tmp_path / "experiments",
        artifact_root=tmp_path,
    )
    report = build_research_report(path, artifact_root=tmp_path)
    assert values(report, "Prediction metrics and comparisons") == [
        {"summary": {"label": name}} for name in ("bridge", "cycle", "linked")
    ]
    assert unlinked.artifact_id in {item.entry.artifact_id for item in report.artifacts}
