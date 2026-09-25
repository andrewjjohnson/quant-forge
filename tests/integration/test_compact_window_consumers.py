"""QF-57 downstream readers preserve compact production and research semantics."""

from pathlib import Path

import pytest

from quantforge.experiments import (
    ArtifactType,
    RelationshipType,
    StudyType,
    create_manifest,
    inspect_study,
    inspect_validation,
    verify_artifacts,
    write_manifest,
)
from quantforge.oos import (
    HoldoutEvaluation,
    HoldoutLedger,
    aggregate_prediction,
    export_oos_aggregate,
    load_oos_aggregate,
    load_oos_source,
)
from quantforge.reporting import build_research_report, export_research_report
from quantforge.reporting.research_models import ReportPhase
from quantforge.walk_forward import WalkForwardStudy
from tests.unit.experiments.test_contracts import execution
from tests.unit.walk_forward.test_incremental_prediction import compact_adapter
from tests.unit.walk_forward.timestamp_fixtures import timestamp_fixture


def test_compact_oos_and_holdout(tmp_path: Path) -> None:
    config, original = timestamp_fixture(tmp_path)
    legacy = WalkForwardStudy(config, original, tmp_path / "legacy")
    legacy.run()
    adapter = compact_adapter(original)
    study = WalkForwardStudy(config, adapter, tmp_path / "compact")
    study.run()
    source = load_oos_source(config.plan, study.study_path)
    result = aggregate_prediction(source)
    previous = aggregate_prediction(load_oos_source(config.plan, legacy.study_path))
    assert result.summary == previous.summary
    assert result.observations == ()
    assert result.window_sources
    old_source = load_oos_source(config.plan, legacy.study_path)
    for current, old in zip(source.folds, old_source.folds, strict=True):
        assert current.selection is not None
        assert old.selection is not None
        assert (
            current.selection.to_primitive()["candidate"]
            == old.selection.to_primitive()["candidate"]
        )
        assert (
            current.selection.to_primitive()["membership"]
            == old.selection.to_primitive()["membership"]
        )
    path = export_oos_aggregate(result, tmp_path / "oos")
    assert load_oos_aggregate(path).to_primitive() == result.to_primitive()
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    assert ledger.reserve(source).state == "reserved_unconsumed"

    def report(expected_state: str) -> Path:
        artifacts = inspect_validation(
            source,
            study.study_path,
            artifact_root=tmp_path,
            ledger=ledger,
            aggregate_path=path,
        )
        verify_artifacts(artifacts.index, tmp_path).require_valid()
        manifest = create_manifest(artifacts, execution())
        manifest_path = write_manifest(
            manifest, tmp_path / "manifests", artifact_root=tmp_path
        )
        rendered = build_research_report(
            manifest_path,
            artifact_root=tmp_path,
            holdout_source=source,
            holdout_ledger=ledger,
        )
        assert rendered.header.to_primitive()["holdout_state"] == expected_state
        exported = export_research_report(rendered, tmp_path / "reports")
        assert expected_state in exported.read_text()
        if expected_state == "reserved_unconsumed":
            old_aggregate_path = export_oos_aggregate(previous, tmp_path / "legacy-oos")
            old_ledger = HoldoutLedger.create(tmp_path / "legacy-ledger")
            old_ledger.reserve(old_source)
            old_artifacts = inspect_validation(
                old_source,
                legacy.study_path,
                artifact_root=tmp_path,
                ledger=old_ledger,
                aggregate_path=old_aggregate_path,
            )
            old_manifest = write_manifest(
                create_manifest(old_artifacts, execution()),
                tmp_path / "legacy-manifests",
                artifact_root=tmp_path,
            )
            old_report = build_research_report(
                old_manifest,
                artifact_root=tmp_path,
                holdout_source=old_source,
                holdout_ledger=old_ledger,
            )
            assert next(
                s.content
                for s in rendered.sections
                if s.title == "Walk-forward OOS summary"
            ) == next(
                s.content
                for s in old_report.sections
                if s.title == "Walk-forward OOS summary"
            )
            assert [w.code for w in rendered.warnings] == [
                w.code for w in old_report.warnings
            ]
        reader = source.prediction_windows[0]
        assert reader is not None
        primary_path = (
            ledger.root
            / "lineages"
            / source.lineage_id
            / "evaluation"
            / "prediction-window.jsonl"
            if expected_state == "consumed"
            else reader.path
        )
        primary = inspect_study(
            StudyType.PREDICTION_WINDOW, primary_path, artifact_root=tmp_path
        )
        attached = create_manifest(primary, execution(), validation=artifacts)
        verify_artifacts(attached.artifacts, tmp_path).require_valid()
        assert len(
            {(e.path, e.json_pointer) for e in attached.artifacts.entries}
        ) == len(attached.artifacts.entries)
        assert any(
            e.relationship is RelationshipType.VALIDATES
            for e in attached.artifacts.relationships
        )
        attached_path = write_manifest(
            attached, tmp_path / "attached", artifact_root=tmp_path
        )
        attached_report = build_research_report(
            attached_path,
            artifact_root=tmp_path,
            holdout_source=source,
            holdout_ledger=ledger,
        )
        assert next(
            s for s in attached_report.sections if s.title == "Prediction counts"
        ).phase is (
            ReportPhase.HOLDOUT if expected_state == "consumed" else ReportPhase.OOS
        )
        assert any(
            e.artifact_type is ArtifactType.SOURCE_DATASET and e.schema_version == "2"
            for e in attached.artifacts.entries
        )
        return manifest_path

    reserved_manifest = report("reserved_unconsumed")
    evaluation = HoldoutEvaluation.prepare(
        source, adapter, selection_fold_id=source.folds[-1].fold_id
    )
    consumed = ledger.consume(evaluation, run_id="compact-first")
    assert consumed.state == "consumed"
    assert ledger.consume(evaluation, run_id="compact-reuse") == consumed
    assert ledger.result(evaluation).to_primitive()["state"] == "consumed"
    report("consumed")
    stale = build_research_report(
        reserved_manifest,
        artifact_root=tmp_path,
        holdout_source=source,
        holdout_ledger=ledger,
    )
    assert stale.header.to_primitive()["holdout_state"] == "consumed"
    assert ledger.state(source) == consumed
    for reader in source.prediction_windows:
        assert reader is not None
        standalone = inspect_study(
            StudyType.PREDICTION_WINDOW, reader.path, artifact_root=tmp_path
        )
        verify_artifacts(standalone.index, tmp_path).require_valid()
    parameter_manifests = list(
        study.study_path.glob("folds/*/selection/**/manifest.json")
    )
    assert parameter_manifests
    for manifest_path in parameter_manifests:
        parameter = inspect_study(
            StudyType.PARAMETER_STUDY, manifest_path.parent, artifact_root=tmp_path
        )
        parameter_manifest = write_manifest(
            create_manifest(parameter, execution()),
            tmp_path / "parameter-manifests",
            artifact_root=tmp_path,
        )
        parameter_report = build_research_report(
            parameter_manifest, artifact_root=tmp_path
        )
        assert any(
            s.title == "Ranked configurations — published order"
            for s in parameter_report.sections
        )
    print(
        {
            "candidate_count": result.summary.to_primitive()["prediction_count"],
            "v1_v2_summary_equivalent": True,
        }
    )
    holdout_path = (
        ledger.root
        / "lineages"
        / source.lineage_id
        / "evaluation"
        / "prediction-window.jsonl"
    )
    holdout_path.write_bytes(
        holdout_path.read_bytes().replace(
            b'"record_type":"shared_evidence"', b'"record_type":"wrong_evidence"'
        )
    )
    with pytest.raises(ValueError, match="evidence"):
        ledger.result(evaluation)
    with pytest.raises(ValueError, match="evidence"):
        inspect_validation(
            source, study.study_path, artifact_root=tmp_path, ledger=ledger
        )
    assert ledger.state(source) == consumed
