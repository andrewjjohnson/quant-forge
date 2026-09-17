from pathlib import Path

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import (
    ArtifactType,
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
from tests.unit.reporting.test_research_reports import codes


@pytest.mark.parametrize(
    ("kind", "category"),
    [
        (StudyType.PREDICTION, ArtifactType.PREDICTION_RESULT),
        (StudyType.FEATURE_DATASET, ArtifactType.FEATURE_DATASET),
    ],
)
@pytest.mark.parametrize(
    "pointer",
    [
        "/rows",
        "/decisions",
        "/observations",
        "/preview",
        "/nested/rows",
        "/rows/0",
        "/decisions/0",
    ],
)
def test_raw_json_collections_are_link_only_and_cannot_supply_warnings(
    tmp_path: Path, kind: StudyType, category: ArtifactType, pointer: str
) -> None:
    path = metadata_manifest(tmp_path, kind=kind)
    manifest = read_manifest(path)
    rows: list[Primitive] = [
        {
            "prediction_count": 1,
            "coverage_complete": False,
            "classification": "fragile",
            "label": f"RAW-ROW-{index}",
        }
        for index in range(100)
    ]
    source = tmp_path / "raw.json"
    write_json(
        source,
        {
            "rows": rows,
            "decisions": rows,
            "observations": rows,
            "preview": rows,
            "nested": {"rows": rows},
        },
    )
    original = source.read_bytes()
    entry = index_artifact(
        tmp_path,
        path="raw.json",
        artifact_type=category,
        schema_version="1",
        producer_study_id="fixture-study",
        producer_artifact_id="result-rows",
        json_pointer=pointer,
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
    report = build_research_report(
        path,
        artifact_root=tmp_path,
        config=ResearchReportConfig(minimum_sample_size=5, maximum_preview_rows=2),
    )
    artifact = next(item for item in report.artifacts if item.entry == entry)
    assert artifact.status == "verified"
    assert artifact.content is None
    assert not {
        "LOW_SAMPLE_SIZE",
        "INCOMPLETE_DATA_COVERAGE",
        "PARAMETER_INSTABILITY",
    } & codes(report)
    html = export_research_report(report, tmp_path / "reports").read_text()
    assert "RAW-ROW-" not in html
    assert "./../raw.json" in html
    assert entry.sha256 is not None
    assert entry.sha256 in html
    assert source.read_bytes() == original


def test_native_feature_checkpoint_is_link_only(tmp_path: Path) -> None:
    path = metadata_manifest(tmp_path, kind=StudyType.FEATURE_DATASET)
    manifest = read_manifest(path)
    write_json(tmp_path / "row.json", {"features": {"coverage_complete": False}})
    entry = index_artifact(
        tmp_path,
        path="row.json",
        artifact_type=ArtifactType.FEATURE_DATASET,
        schema_version="1",
        producer_study_id="fixture-study",
        producer_artifact_id="rows/checkpoint-id",
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
    report = build_research_report(path, artifact_root=tmp_path)
    artifact = next(item for item in report.artifacts if item.entry == entry)
    assert artifact.status == "verified"
    assert artifact.content is None
    assert "INCOMPLETE_DATA_COVERAGE" not in codes(report)


@pytest.mark.parametrize(
    ("container", "collection"),
    [("prediction_study", "rows"), ("prediction_window", "decisions")],
)
@pytest.mark.parametrize("select_container", [False, True])
def test_nested_prediction_observations_are_not_retained_with_trial_metadata(
    tmp_path: Path, container: str, collection: str, select_container: bool
) -> None:
    path = metadata_manifest(tmp_path, kind=StudyType.PARAMETER_STUDY)
    manifest = read_manifest(path)
    analysis: PrimitiveMapping = {"prediction_count": 17, "metric": "0.314159"}
    native: PrimitiveMapping = {
        "manifest": {
            "study_id": "source-study",
            "record_counts": {"predictions": 17},
            "strategy_parameters": {"rows": [1, 2], "decisions": ["custom-setting"]},
        },
        collection: [
            {"label": f"RAW-TRIAL-OBSERVATION-{index}"} for index in range(100)
        ],
    }
    payload: PrimitiveMapping = {
        "trial_id": "trial-1",
        "analysis": analysis,
        container: native,
    }
    source = tmp_path / "trial.json"
    write_json(source, payload)
    original = source.read_bytes()
    entry = index_artifact(
        tmp_path,
        path="trial.json",
        artifact_type=ArtifactType.PREDICTION_RESULT,
        schema_version="1",
        producer_study_id="fixture-study",
        producer_artifact_id="trial-1/result",
        json_pointer=f"/{container}" if select_container else "",
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
    report = build_research_report(path, artifact_root=tmp_path)
    artifact = next(item for item in report.artifacts if item.entry == entry)
    assert artifact.status == "verified"
    assert artifact.content is not None
    retained_native: PrimitiveMapping = {"manifest": native["manifest"]}
    assert artifact.content.to_primitive()["value"] == (
        retained_native
        if select_container
        else {"trial_id": "trial-1", "analysis": analysis, container: retained_native}
    )
    html = export_research_report(report, tmp_path / "reports").read_text()
    assert "./../trial.json" in html
    assert "RAW-TRIAL-OBSERVATION-" not in html
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "category", [ArtifactType.WALK_FORWARD_WINDOW, ArtifactType.HOLDOUT_RESULT]
)
@pytest.mark.parametrize("prediction", [True, False])
def test_validation_result_wrappers_retain_metadata_without_raw_histories(
    tmp_path: Path, category: ArtifactType, prediction: bool
) -> None:
    raw_fields = (
        ("decisions",)
        if prediction
        else (
            "signals",
            "orders",
            "fills",
            "positions",
            "completed_trades",
            "open_trades",
            "daily_equity",
            "dividend_cashflows",
            "split_adjustments",
            "benchmark_daily_equity",
            "benchmark_dividend_cashflows",
            "benchmark_split_adjustments",
        )
    )
    native: PrimitiveMapping = {
        "manifest": {
            "record_counts": {"signals": 100},
            "performance": {"trade_count": 20},
            "strategy": {"parameters": {"orders": ["custom-setting"]}},
        },
    }
    for key in raw_fields:
        native[key] = [{"label": "RAW-HISTORY"} for _ in range(100)]
    fold: PrimitiveMapping = {
        "kind": "prediction" if prediction else "backtest",
        "result_id": "result-1",
        "result": native,
    }
    payload: PrimitiveMapping = (
        {"artifact": fold, "summary": {"prediction_count": 20}}
        if category is ArtifactType.HOLDOUT_RESULT
        else fold
    )
    path = metadata_manifest(
        tmp_path, kind=StudyType.OOS_VALIDATION, category=category, payload=payload
    )
    source = tmp_path / "result.json"
    original = source.read_bytes()
    report = build_research_report(path, artifact_root=tmp_path)
    artifact = report.artifacts[0]
    assert artifact.content is not None
    retained_fold: PrimitiveMapping = {
        **fold,
        "result": {"manifest": native["manifest"]},
    }
    assert artifact.content.to_primitive()["value"] == (
        {"artifact": retained_fold, "summary": {"prediction_count": 20}}
        if category is ArtifactType.HOLDOUT_RESULT
        else retained_fold
    )
    html = export_research_report(report, tmp_path / "reports").read_text()
    assert "RAW-HISTORY" not in html
    assert "./../result.json" in html
    assert source.read_bytes() == original
