from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.experiments import (
    ArtifactIndex,
    ArtifactType,
    ManifestError,
    RelationshipType,
    StudyArtifacts,
    StudyType,
    create_manifest,
    inspect_study,
    inspect_validation,
    verify_artifacts,
)
from quantforge.oos import (
    HoldoutEvaluation,
    HoldoutLedger,
    aggregate_backtest,
    aggregate_prediction,
    export_oos_aggregate,
)
from quantforge.walk_forward.models import BacktestOOSArtifact
from quantforge.walk_forward.persistence import read_record
from tests.unit.backtesting.test_runner import configured_result
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import execution, write_json
from tests.unit.oos.conftest import complete_study
from tests.unit.prediction.test_prediction_window import run_window


def attached_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prediction: bool,
    holdout: bool,
    aggregate: bool = False,
) -> tuple[StudyArtifacts, StudyArtifacts]:
    completed = complete_study(tmp_path, prediction=prediction)
    source = completed.source
    ledger = None
    if holdout:
        ledger = HoldoutLedger.create(tmp_path / "ledger")
        ledger.reserve(source)
        consumed = ledger.consume(
            HoldoutEvaluation.prepare(
                source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
            ),
            run_id="attached-holdout",
        )
        assert consumed.result_reference is not None
        path = ledger.root / cast(str, consumed.result_reference.to_primitive()["path"])
        record = cast(PrimitiveMapping, read_record(path)["artifact"])
        payload = cast(PrimitiveMapping, record["result"])
        export = (
            path.parent / "evaluation" / cast(str, record.get("export_location", ""))
        )
    else:
        fold = source.folds[0]
        artifact = fold.artifact
        assert artifact is not None
        payload = artifact.snapshot.to_primitive()
        export = (
            completed.study.study_path
            / "folds"
            / fold.fold_id
            / "test"
            / (
                artifact.export_location
                if isinstance(artifact, BacktestOOSArtifact)
                else ""
            )
        )
    if prediction:
        path = tmp_path / "primary-window.json"
        write_json(path, payload)
    else:
        path = export
    aggregate_path = None
    if aggregate:
        aggregated = (aggregate_prediction if prediction else aggregate_backtest)(
            source
        )
        aggregate_path = export_oos_aggregate(aggregated, tmp_path / "aggregate")
    block_research(monkeypatch)
    primary = inspect_study(
        StudyType.PREDICTION_WINDOW if prediction else StudyType.BACKTEST,
        path,
        artifact_root=tmp_path,
    )
    validation = inspect_validation(
        source,
        completed.study.study_path,
        artifact_root=tmp_path,
        ledger=ledger,
        aggregate_path=aggregate_path,
        study_type=StudyType.HOLDOUT_VALIDATION
        if holdout
        else StudyType.OOS_VALIDATION
        if aggregate
        else StudyType.WALK_FORWARD,
    )
    return primary, validation


@pytest.mark.parametrize("prediction", [False, True], ids=["backtest", "prediction"])
@pytest.mark.parametrize("scope", ["fold", "aggregate", "holdout"])
def test_attachment_requires_and_links_a_captured_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction: bool, scope: str
) -> None:
    holdout = scope == "holdout"
    primary, validation = attached_fixture(
        tmp_path,
        monkeypatch,
        prediction=prediction,
        holdout=holdout,
        aggregate=scope == "aggregate",
    )
    manifest = create_manifest(primary, execution(), validation=validation)
    assert verify_artifacts(manifest.artifacts, tmp_path).valid
    config = next(
        entry
        for entry in primary.index.entries
        if entry.artifact_type is ArtifactType.CONFIGURATION
    )
    captured_type = (
        ArtifactType.HOLDOUT_RESULT if holdout else ArtifactType.WALK_FORWARD_WINDOW
    )
    captured = {
        entry.artifact_id
        for entry in validation.index.entries
        if entry.artifact_type is captured_type
    }
    assert any(
        edge.source_id in captured
        and edge.relationship is RelationshipType.VALIDATES
        and edge.target_id == config.artifact_id
        for edge in manifest.artifacts.relationships
    )
    assert manifest.study_id != create_manifest(primary, execution()).study_id


@pytest.mark.parametrize("prediction", [False, True], ids=["backtest", "prediction"])
def test_unrelated_result_cannot_attach_valid_validation_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction: bool
) -> None:
    foreign = (
        run_window().to_primitive()
        if prediction
        else configured_result().manifest_primitive()
    )
    path = tmp_path / "unrelated.json"
    write_json(path, foreign)
    _, validation = attached_fixture(
        tmp_path, monkeypatch, prediction=prediction, holdout=False
    )
    primary = inspect_study(
        StudyType.PREDICTION_WINDOW if prediction else StudyType.BACKTEST,
        path,
        artifact_root=tmp_path,
    )
    with pytest.raises(ManifestError, match="captured result"):
        create_manifest(primary, execution(), validation=validation)


@pytest.mark.parametrize(
    "unsupported",
    [
        StudyType.FEATURE_DATASET,
        StudyType.PARAMETER_STUDY,
        StudyType.OPTIMIZATION,
        StudyType.WALK_FORWARD,
        StudyType.OOS_VALIDATION,
        StudyType.HOLDOUT_VALIDATION,
    ],
)
def test_validation_attachment_rejects_unsupported_primary_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsupported: StudyType
) -> None:
    primary, validation = attached_fixture(
        tmp_path, monkeypatch, prediction=True, holdout=False
    )
    primary = replace(
        primary, provenance=replace(primary.provenance, study_type=unsupported)
    )
    with pytest.raises(ManifestError, match="primary study"):
        create_manifest(primary, execution(), validation=validation)


@pytest.mark.parametrize(
    "field", ["sha256", "schema_version", "producer_run_id", "bindings"]
)
def test_shared_backtest_files_require_matching_content_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    primary, validation = attached_fixture(
        tmp_path, monkeypatch, prediction=False, holdout=False
    )
    config = next(
        entry
        for entry in primary.index.entries
        if entry.artifact_type is ArtifactType.CONFIGURATION
    )
    original = next(
        entry
        for entry in validation.index.entries
        if entry.path == config.path and entry.json_pointer == config.json_pointer
    )
    if field == "bindings":
        changed = replace(
            original, bindings=PrimitiveMappingSnapshot.capture({"/foreign": "binding"})
        )
    elif field == "sha256":
        changed = replace(original, sha256="0" * 64)
    elif field == "schema_version":
        changed = replace(original, schema_version="foreign-version")
    else:
        changed = replace(original, producer_run_id="foreign-run")
    validation = replace(
        validation,
        index=ArtifactIndex(
            tuple(
                changed if entry == original else entry
                for entry in validation.index.entries
            ),
            tuple(
                replace(
                    edge,
                    source_id=changed.artifact_id
                    if edge.source_id == original.artifact_id
                    else edge.source_id,
                    target_id=changed.artifact_id
                    if edge.target_id == original.artifact_id
                    else edge.target_id,
                )
                for edge in validation.index.relationships
            ),
        ),
    )
    with pytest.raises(ManifestError, match="incompatible shared artifact"):
        create_manifest(primary, execution(), validation=validation)
