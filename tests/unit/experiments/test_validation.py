from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.experiments import (
    ArtifactType,
    ManifestError,
    RelationshipType,
    StudyType,
    create_manifest,
    inspect_study,
    inspect_validation,
    read_manifest,
    verify_artifacts,
    write_manifest,
)
from quantforge.oos import (
    HoldoutEvaluation,
    HoldoutLedger,
    aggregate_backtest,
    aggregate_prediction,
    export_oos_aggregate,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import execution, write_json
from tests.unit.oos.conftest import complete_study


@pytest.mark.parametrize("prediction", [True, False], ids=["prediction", "backtest"])
def test_real_validation_manifests_are_observational_and_retain_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction: bool
) -> None:
    completed = complete_study(tmp_path, prediction=prediction)
    source = completed.source
    aggregate = (
        aggregate_prediction(source) if prediction else aggregate_backtest(source)
    )
    aggregate_path = export_oos_aggregate(aggregate, tmp_path / "oos")
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    block_research(monkeypatch)
    bundle = inspect_validation(
        source,
        completed.study.study_path,
        artifact_root=tmp_path,
        ledger=ledger,
        aggregate_path=aggregate_path,
        study_type=StudyType.OOS_VALIDATION,
    )
    manifest = create_manifest(bundle, execution())
    observations = bundle.provenance.observations.to_primitive()
    assert observations["plan_id"] == source.plan.plan_id
    assert observations["lineage_id"] == source.lineage_id
    assert observations["aggregate_id"] == aggregate.aggregate_id
    assert (
        cast(PrimitiveMapping, observations["holdout"])["state"]
        == "reserved_unconsumed"
    )
    config = bundle.provenance.configuration.to_primitive()
    assert config["study_definition"] == source.definition.to_primitive()
    assert b"native_v1" in manifest.serialize()
    assert "purge_policy" in source.plan.to_manifest()
    types = [entry.artifact_type for entry in bundle.index.entries]
    assert types.count(ArtifactType.WALK_FORWARD_WINDOW) == 2
    assert types.count(ArtifactType.FROZEN_SELECTION) == 2
    assert types.count(ArtifactType.CONFIGURATION_STABILITY) == 1
    assert (
        sum(
            edge.relationship is RelationshipType.AGGREGATES
            for edge in bundle.index.relationships
        )
        == 2
    )
    assert (
        sum(
            edge.relationship is RelationshipType.SELECTED_BY
            for edge in bundle.index.relationships
        )
        == 2
    )
    if prediction:
        assert ArtifactType.BACKTEST_RESULT not in types
    else:
        assert ArtifactType.BACKTEST_RESULT in types
    path = write_manifest(manifest, tmp_path / "experiments", artifact_root=tmp_path)
    assert (
        read_manifest(path, artifact_root=tmp_path).serialize() == manifest.serialize()
    )
    assert ledger.state(source).state.value == "reserved_unconsumed"


@pytest.mark.parametrize("prediction", [True, False], ids=["prediction", "backtest"])
def test_consumed_holdout_stays_consumed_on_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction: bool
) -> None:
    completed = complete_study(tmp_path, prediction=prediction)
    source = completed.source
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    before = create_manifest(
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        ),
        execution(),
    )
    evaluation = HoldoutEvaluation.prepare(
        source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
    )
    consumed = ledger.consume(evaluation, run_id="actual-holdout-execution")
    assert consumed.state.value == "consumed"
    block_research(monkeypatch)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("indexing attempted to consume or evaluate a holdout")

    monkeypatch.setattr(HoldoutLedger, "consume", forbidden)
    monkeypatch.setattr(HoldoutLedger, "reserve", forbidden)
    monkeypatch.setattr(HoldoutEvaluation, "prepare", forbidden)
    monkeypatch.setattr(HoldoutEvaluation, "_evaluate", forbidden)
    after = create_manifest(
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        ),
        execution(),
    )
    regenerated = create_manifest(
        inspect_validation(
            source,
            completed.study.study_path,
            artifact_root=tmp_path,
            ledger=HoldoutLedger(tmp_path / "ledger"),
        ),
        execution("later-indexing-execution"),
    )
    assert before.study_id == after.study_id == regenerated.study_id
    assert before.manifest_id != after.manifest_id
    holdout = cast(
        PrimitiveMapping, after.provenance.observations.to_primitive()["holdout"]
    )
    assert holdout["state"] == "consumed"
    assert holdout["consumption_run_id"] == "actual-holdout-execution"
    assert holdout["consumption_artifact_id"] is not None
    assert holdout["result_artifact_id"] is not None
    assert regenerated.provenance.observations.to_primitive()["holdout"] == holdout
    assert verify_artifacts(after.artifacts, tmp_path).valid
    assert ledger.state(source) == consumed


def test_holdout_failure_is_still_consumed_without_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = complete_study(tmp_path, prediction=False)
    source = completed.source
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    evaluation = HoldoutEvaluation.prepare(
        source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("deliberate interrupted evaluation")

    monkeypatch.setattr(HoldoutEvaluation, "_evaluate", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        ledger.consume(evaluation, run_id="failed-execution")
    block_research(monkeypatch)
    bundle = inspect_validation(
        source,
        completed.study.study_path,
        artifact_root=tmp_path,
        ledger=ledger,
        study_type=StudyType.HOLDOUT_VALIDATION,
    )
    holdout = cast(
        PrimitiveMapping, bundle.provenance.observations.to_primitive()["holdout"]
    )
    assert holdout["state"] == "consumed"
    assert holdout["result_artifact_id"] is None
    assert holdout["consumption_artifact_id"] is not None


def test_absent_ledger_never_claims_reserved_is_pristine(tmp_path: Path) -> None:
    completed = complete_study(tmp_path, prediction=False)
    bundle = inspect_validation(
        completed.source, completed.study.study_path, artifact_root=tmp_path
    )
    holdout = cast(
        PrimitiveMapping, bundle.provenance.observations.to_primitive()["holdout"]
    )
    assert holdout["state"] is None
    assert "pristine" not in holdout
    with pytest.raises(ManifestError, match="ledger"):
        inspect_validation(
            completed.source,
            completed.study.study_path,
            artifact_root=tmp_path,
            study_type=StudyType.HOLDOUT_VALIDATION,
        )


def test_validation_rejects_mismatched_source_and_tampered_fold(tmp_path: Path) -> None:
    from quantforge.oos import OOSIntegrityError

    completed = complete_study(tmp_path, prediction=False)
    changed = replace(
        completed.source, definition=PrimitiveMappingSnapshot.capture({"changed": True})
    )
    with pytest.raises(OOSIntegrityError, match="identity differs"):
        inspect_validation(changed, completed.study.study_path, artifact_root=tmp_path)
    path = (
        completed.study.study_path
        / "folds"
        / completed.source.folds[0].fold_id
        / "oos.json"
    )
    path.write_text('{"payload":{},"fingerprint":"incorrect"}')
    with pytest.raises(ManifestError, match="fingerprint"):
        inspect_validation(
            completed.source, completed.study.study_path, artifact_root=tmp_path
        )


def test_qf42_snapshot_is_indexed_as_a_prediction_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = complete_study(tmp_path, prediction=True)
    artifact = completed.source.folds[0].artifact
    assert artifact is not None
    path = tmp_path / "prediction-window.json"
    write_json(path, artifact.snapshot.to_primitive())
    block_research(monkeypatch)
    bundle = inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)
    manifest = create_manifest(bundle, execution())
    assert b"schedule" in manifest.serialize()
    assert b"prediction_context" in manifest.serialize()
    assert verify_artifacts(bundle.index, tmp_path).valid
    validation = inspect_validation(
        completed.source, completed.study.study_path, artifact_root=tmp_path
    )
    linked = create_manifest(bundle, execution(), validation=validation)
    assert linked.study_id != manifest.study_id
    assert (
        linked.provenance.configuration.to_primitive()["validation"]
        == validation.provenance.configuration.to_primitive()
    )
    assert verify_artifacts(linked.artifacts, tmp_path).valid
