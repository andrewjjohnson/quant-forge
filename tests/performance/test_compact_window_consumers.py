"""5,760 decisions stay normalized through QF-40 reduction, QF-9 and QF-41."""

import socket
from datetime import timedelta
from pathlib import Path

import pytest

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.experiments import (
    ArtifactRelationship,
    ArtifactType,
    RelationshipType,
    StudyType,
    create_manifest,
    index_artifact,
    inspect_study,
    verify_artifacts,
    write_manifest,
)
from quantforge.oos.prediction import (
    PredictionMetricFields,
    PredictionObservationAccumulator,
    iter_prediction_observations,
)
from quantforge.prediction import PredictionContextFailurePolicy, PredictionWindowResult
from quantforge.prediction.window_compact import (
    CompactPredictionWindowDecision,
    CompactPredictionWindowResult,
)
from quantforge.prediction.window_compact_validation import (
    PredictionWindowDecisionValidator,
)
from quantforge.prediction.window_encoding import StudyIdentity, canonical, mapping
from quantforge.prediction.window_incremental import IncrementalPredictionWindowWriter
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.reporting import build_research_report, export_research_report
from quantforge.walk_forward.models import PredictionOOSArtifact
from tests.performance.test_compact_prediction_window_shape import (
    DECISIONS,
    MARKER,
    SHARED_PAYLOAD_BYTES,
)
from tests.unit.experiments.test_contracts import execution
from tests.unit.prediction.test_compact_prediction_window import rehash_decision
from tests.unit.prediction.test_multi_timeframe_study import (
    FixtureParameters,
    _requirements,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_window import (
    START,
    WindowProvider,
    run_window,
    schedule,
)


def test_5760_decisions_downstream_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = WindowProvider()
    provider.series = ()
    original = run_window(
        provider,
        requirements=_requirements(failure_policy=PredictionContextFailurePolicy.SKIP),
        decision_schedule=schedule(START, START),
    )
    broad = schedule(START, START + timedelta(days=130))
    exact = schedule(START, broad.decision_timestamps[DECISIONS - 1])
    identity = original.identity_snapshot.to_primitive()
    identity["schedule"] = exact.to_primitive()
    mapping(identity["market_data"])["synthetic_scale_evidence"] = MARKER + "x" * (
        SHARED_PAYLOAD_BYTES - len(MARKER)
    )
    validator = PredictionWindowDecisionValidator(
        expected_identity=PrimitiveMappingSnapshot.capture(identity),
        schedule=exact,
        outcome_sessions=(),
        strategy_parameters=FixtureParameters().to_primitive(),
    )
    template = (
        CompactPredictionWindowResult.from_window(original).decisions[0].to_primitive()
    )
    study_manifest = mapping(mapping(template["prediction_study"])["manifest"])
    study_id = StudyIdentity(identity).for_context(
        mapping(study_manifest["prediction_context"])
    )
    template["prediction_study_id"] = study_id
    study_manifest["study_id"] = study_id
    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "prediction-window.jsonl", validator=validator
    )
    for index, timestamp in enumerate(exact.decision_timestamps):
        record = {
            **template,
            "sequence": index,
            "decision_timestamp": timestamp.isoformat(),
            "shared_evidence_id": writer.evidence.evidence_id,
        }
        rehash_decision(record)
        writer.append(
            CompactPredictionWindowDecision(PrimitiveMappingSnapshot.capture(record))
        )
    reader = writer.finalize()
    assert reader.decision_count == 5760

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("compact consumer expanded or materialized legacy results")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(PredictionWindowResult, "to_primitive", forbidden)
    monkeypatch.setattr(CompactPredictionWindowDecision, "from_embedded", forbidden)
    artifact = PredictionOOSArtifact(
        "fixture-selection",
        str(reader.header()["window_result_id"]),
        PrimitiveMappingSnapshot.capture(
            {"schema_version": "2", "path": reader.path.name, "header": reader.header()}
        ),
    )
    accumulator = PredictionObservationAccumulator(PredictionMetricFields())
    for observation in iter_prediction_observations(reader, artifact, "fixture-test"):
        accumulator.update(observation.to_primitive())
    summary = accumulator.summary(reader.decision_count)
    assert summary["scheduled_decisions"] == 5760
    assert summary["prediction_count"] == 0  # Explicitly skipped, never fake zeros.
    assert summary["accuracy"] is None
    assert accumulator.numeric == {"signed_outcome": [], "mfe": [], "mae": []}

    import quantforge.prediction.window_compact_validation as validation

    original_validate = validation.validate_prediction_provenance
    shared_verifications = 0

    def tracked(*args: object, **kwargs: object) -> object:
        nonlocal shared_verifications
        shared_verifications += 1
        return original_validate(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(validation, "validate_prediction_provenance", tracked)
    inspected = inspect_study(
        StudyType.PREDICTION_WINDOW, reader.path, artifact_root=tmp_path
    )
    assert shared_verifications == 1
    verify_artifacts(inspected.index, tmp_path).require_valid()
    assert len(inspected.index.entries) == 3
    assert len(inspected.index.relationships) == 2
    summary_path = tmp_path / "published-summary.json"
    summary_path.write_bytes(canonical(summary))
    summary_entry = index_artifact(
        tmp_path,
        path=summary_path.name,
        artifact_type=ArtifactType.PREDICTION_RESULT,
        schema_version="1",
        producer_study_id=inspected.provenance.producer_study_id,
        producer_artifact_id="qf40-reduction",
    )
    manifest = create_manifest(
        inspected,
        execution(),
        additional_artifacts=(summary_entry,),
        relationships=(
            ArtifactRelationship(
                summary_entry.artifact_id,
                RelationshipType.DERIVED_FROM,
                inspected.index.entries[0].artifact_id,
            ),
        ),
    )
    assert MARKER not in manifest.serialize().decode()
    manifest_path = write_manifest(
        manifest, tmp_path / "manifests", artifact_root=tmp_path
    )
    monkeypatch.setattr(PredictionWindowReader, "iterate_decisions", forbidden)
    report = build_research_report(manifest_path, artifact_root=tmp_path)
    html = export_research_report(report, tmp_path / "reports").read_text()
    assert any(
        s.title == "Prediction metrics and comparisons"
        and s.content.to_primitive()["value"] is not None
        for s in report.sections
    )
    assert "5760" in html
    assert MARKER not in html
    print(
        {
            "decisions": reader.decision_count,
            "artifact_bytes": reader.path.stat().st_size,
            "shared_scientific_verifications": shared_verifications,
            "retained_legacy_decisions": 0,
        }
    )
