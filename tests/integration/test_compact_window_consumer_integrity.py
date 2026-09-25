"""Fail-closed downstream compact artifact, lineage, and reference checks."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.experiments import (
    ArtifactIndex,
    ManifestError,
    StudyType,
    create_manifest,
    inspect_study,
    inspect_validation,
    verify_artifacts,
)
from quantforge.experiments._aggregate_integrity import validate_aggregate_folds
from quantforge.experiments._producer_snapshot import ProducerReadSet
from quantforge.oos import aggregate_prediction, load_oos_source
from quantforge.prediction.window_compact import (
    CompactPredictionWindowDecision,
    CompactPredictionWindowResult,
    PredictionWindowEvidence,
)
from quantforge.prediction.window_encoding import canonical, decode, mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.reporting._research_sections import primary_phase
from quantforge.reporting.research_models import ReportPhase
from quantforge.walk_forward import WalkForwardConfig, WalkForwardStudy
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_contracts import execution
from tests.unit.prediction.test_compact_prediction_window import rehash_decision
from tests.unit.walk_forward.test_incremental_prediction import compact_adapter
from tests.unit.walk_forward.timestamp_fixtures import timestamp_fixture


@dataclass(frozen=True)
class CompactProducer:
    path: Path
    config: WalkForwardConfig


@pytest.fixture(scope="module")
def producer(tmp_path_factory: pytest.TempPathFactory) -> CompactProducer:
    root = tmp_path_factory.mktemp("compact-consumer-integrity")
    config, original = timestamp_fixture(root)
    study = WalkForwardStudy(config, compact_adapter(original), root / "studies")
    result = study.run()
    assert all(f.artifact is not None for f in result.folds)
    return CompactProducer(study.study_path, config)


def copy_producer(producer: CompactProducer, root: Path) -> tuple[Path, Path]:
    study = root / producer.path.name
    shutil.copytree(producer.path, study)
    return study, study / "folds" / producer.config.plan.folds[
        0
    ].fold_id / "test" / "prediction-window.jsonl"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_shared",
        "altered_shared",
        "bad_evidence_id",
        "modified_decision",
        "reordered",
        "duplicate",
        "count",
        "truncated",
        "unknown_version",
        "corrupt_version",
        "staging",
        "missing_file",
    ],
)
def test_corruption_rejected_by_oos_and_qf9(
    producer: CompactProducer, tmp_path: Path, mutation: str
) -> None:
    study, path = copy_producer(producer, tmp_path)
    source = load_oos_source(producer.config.plan, study)
    lines = path.read_bytes().splitlines(keepends=True)
    records = [decode(line) for line in lines]
    if mutation == "missing_shared":
        del lines[1]
    elif mutation == "altered_shared":
        identity = mapping(records[1]["window_identity"])
        mapping(identity["market_data"])["provider"] = "foreign"
        lines[1] = canonical(records[1]) + b"\n"
    elif mutation == "bad_evidence_id":
        records[2]["shared_evidence_id"] = "0" * 64
        rehash_decision(records[2])
        lines[2] = canonical(records[2]) + b"\n"
    elif mutation == "modified_decision":
        records[2]["status"] = "skipped"
        lines[2] = canonical(records[2]) + b"\n"
    elif mutation == "reordered":
        lines[2], lines[3] = lines[3], lines[2]
    elif mutation == "duplicate":
        lines.append(lines[-1])
    elif mutation == "count":
        records[0]["decision_count"] = 999
        lines[0] = canonical(records[0]) + b"\n"
    elif mutation == "truncated":
        lines[-1] = lines[-1][:-7]
    elif mutation in {"unknown_version", "corrupt_version"}:
        records[0]["schema_version"] = "999" if mutation == "unknown_version" else 2
        lines[0] = canonical(records[0]) + b"\n"
    elif mutation == "staging":
        lines = lines[1:]
    if mutation == "missing_file":
        path.unlink()
    else:
        path.write_bytes(b"".join(lines))
    with pytest.raises((ValueError, OSError)):
        load_oos_source(producer.config.plan, study)
    with pytest.raises((ValueError, OSError)):
        inspect_validation(source, study, artifact_root=tmp_path)
    with pytest.raises((ValueError, OSError)):
        inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "mutation", ["fold", "validation", "ancestry", "selection", "path"]
)
def test_rehashed_foreign_scope_and_reference_rejected(
    producer: CompactProducer, tmp_path: Path, mutation: str
) -> None:
    study, path = copy_producer(producer, tmp_path)
    reader = PredictionWindowReader.open(path)
    identity = reader.evidence.identity_snapshot.to_primitive()
    if mutation in {"fold", "validation"}:
        environment = mapping(mapping(identity["context_environment"])["configuration"])
        if mutation == "validation":
            environment["plan_id"] = "0" * 64
        else:
            mapping(mapping(environment["partition"])["window"])["window_id"] = "0" * 64
    elif mutation == "ancestry":
        mapping(identity["market_data"])["dataset_id"] = "foreign"
    evidence = PredictionWindowEvidence(PrimitiveMappingSnapshot.capture(identity))
    decisions: list[CompactPredictionWindowDecision] = []
    for compact in reader.iterate_decisions():
        record = compact.to_primitive()
        record["shared_evidence_id"] = evidence.evidence_id
        rehash_decision(record)
        decisions.append(
            CompactPredictionWindowDecision(PrimitiveMappingSnapshot.capture(record))
        )
    changed = CompactPredictionWindowResult(evidence, tuple(decisions))
    path.write_bytes(changed.serialize())
    fold_root = path.parent.parent
    artifact = read_record(fold_root / "oos.json")
    artifact["result_id"] = changed.window_result_id
    reference = mapping(artifact["result"])
    reference["header"] = changed.header_snapshot.to_primitive()
    if mutation == "selection":
        artifact["selection_id"] = "0" * 64
    elif mutation == "path":
        reference["path"] = "../test/prediction-window.jsonl"
    write_record(fold_root / "oos.json", artifact)
    state = read_record(fold_root / "state.json")
    state["artifact_id"] = configuration_identity(artifact)
    write_record(fold_root / "state.json", state)
    with pytest.raises(ValueError, match=r"partition|dataset|configuration|reference"):
        load_oos_source(producer.config.plan, study)


def test_version_aware_reference_rejects_bad_or_embedded_versions(
    producer: CompactProducer, tmp_path: Path
) -> None:
    _, path = copy_producer(producer, tmp_path)
    header = PredictionWindowReader.open(path).header()
    for version in ("3", 2, None):
        reference: PrimitiveMapping = {
            "schema_version": version,
            "path": path.name,
            "header": header,
        }
        with pytest.raises(ValueError, match="reference"):
            PredictionWindowReader.from_reference(reference, root=path.parent)


@pytest.mark.parametrize(
    "field",
    [
        "fold_id",
        "selection_id",
        "schema_version",
        "window_result_id",
        "shared_evidence_id",
        "schedule_id",
        "decision_count",
    ],
)
def test_compact_aggregate_references_are_authoritative(
    producer: CompactProducer, tmp_path: Path, field: str
) -> None:
    study, _ = copy_producer(producer, tmp_path)
    source = load_oos_source(producer.config.plan, study)
    aggregate = aggregate_prediction(source).to_primitive()
    references = aggregate["window_sources"]
    assert isinstance(references, list)
    mapping(references[0])[field] = 999 if field == "decision_count" else "foreign"
    with pytest.raises(ManifestError, match="window sources"):
        validate_aggregate_folds(source, aggregate)


def test_compact_file_replaced_during_indexing_is_rejected(
    producer: CompactProducer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, path = copy_producer(producer, tmp_path)
    verify = ProducerReadSet.verify

    def replace_before_verification(
        reads: ProducerReadSet, index: ArtifactIndex, root: Path
    ) -> None:
        with path.open("ab") as output:
            output.write(b"\n")
        verify(reads, index, root)

    monkeypatch.setattr(ProducerReadSet, "verify", replace_before_verification)
    with pytest.raises(ManifestError, match="changed during indexing"):
        inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)


@pytest.mark.parametrize("mismatch", ["fold", "role", "validation_lineage"])
def test_content_entry_reuse_does_not_authorize_a_validation_role(
    producer: CompactProducer, tmp_path: Path, mismatch: str
) -> None:
    study, path = copy_producer(producer, tmp_path)
    source = load_oos_source(producer.config.plan, study)
    primary = inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)
    validation = inspect_validation(source, study, artifact_root=tmp_path)
    assert all(entry in validation.index.entries for entry in primary.index.entries)
    attached = create_manifest(primary, execution(), validation=validation)
    assert (
        attached.serialize()
        == create_manifest(primary, execution(), validation=validation).serialize()
    )
    assert all(
        attached.artifacts.entries.count(entry) == 1 for entry in primary.index.entries
    )
    assert primary_phase(attached) is ReportPhase.OOS
    # Previously attached content has no remembered validation role. A fresh
    # standalone view needs authoritative relationships to claim OOS or holdout.
    assert primary_phase(create_manifest(primary, execution())) is ReportPhase.IN_SAMPLE

    fold_root = path.parent.parent
    selection = read_record(fold_root / "selection.json")
    if mismatch == "fold":
        selection["fold_id"] = source.folds[1].fold_id
    elif mismatch == "validation_lineage":
        selection["plan_id"] = "foreign-validation-plan"
    else:
        window = mapping(mapping(mapping(selection["membership"])["test"])["window"])
        window["role"] = "final_holdout"
    selection["selection_id"] = configuration_identity(
        {key: value for key, value in selection.items() if key != "selection_id"}
    )
    write_record(fold_root / "selection.json", selection)
    artifact = read_record(fold_root / "oos.json")
    artifact["selection_id"] = selection["selection_id"]
    write_record(fold_root / "oos.json", artifact)
    state = read_record(fold_root / "state.json")
    state["selection_id"] = selection["selection_id"]
    state["artifact_id"] = configuration_identity(artifact)
    write_record(fold_root / "state.json", state)

    # The physical window and all three canonical entries are still valid and
    # identical. Only their proposed external fold/partition relationship changed.
    assert (
        inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path).index
        == primary.index
    )
    verify_artifacts(primary.index, tmp_path).require_valid()
    with pytest.raises(ValueError, match=r"lineage|window/role"):
        load_oos_source(producer.config.plan, study)
    with pytest.raises(ManifestError, match="differs from captured source"):
        inspect_validation(source, study, artifact_root=tmp_path)
    assert not verify_artifacts(attached.artifacts, tmp_path).valid
