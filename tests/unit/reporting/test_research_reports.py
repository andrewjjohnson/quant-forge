import json
import shutil
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import export_backtest_result
from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import (
    ArtifactType,
    StudyType,
    create_manifest,
    index_artifact,
    inspect_study,
    read_manifest,
    write_manifest,
)
from quantforge.optimization import GridSearchStudy, MovingAverageCrossoverFactory
from quantforge.prediction import (
    AlwaysUpParameters,
    AlwaysUpPredictionStrategy,
    create_overnight_gap_prediction_study,
    run_prediction_study,
)
from quantforge.reporting import (
    ReportPhase,
    ResearchReport,
    ResearchReportConfig,
    ResearchReportError,
    build_research_report,
    export_research_report,
)
from tests.unit.backtesting.test_runner import configured_result
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import execution, write_json
from tests.unit.helpers import make_dataset
from tests.unit.oos.conftest import complete_study
from tests.unit.optimization.test_study import (
    _dataset,  # pyright: ignore[reportPrivateUsage]
    _study_config,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _build,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.reporting.research_fixtures import metadata_manifest, native_manifest


def values(report: ResearchReport, title: str) -> list[Primitive]:
    return [
        section.content.to_primitive()["value"]
        for section in report.sections
        if section.title == title
    ]


def codes(report: ResearchReport) -> set[str]:
    return {warning.code for warning in report.warnings}


def test_native_prediction_standalone_is_deterministic_and_has_no_trading_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = run_prediction_study(
        make_dataset(("100", "101", "102", "103")),
        create_overnight_gap_prediction_study(
            AlwaysUpPredictionStrategy(AlwaysUpParameters())
        ),
    )
    write_json(tmp_path / "prediction.json", result.to_primitive())
    path = native_manifest(tmp_path, StudyType.PREDICTION, tmp_path / "prediction.json")
    original = path.read_bytes()
    block_research(monkeypatch)
    first = build_research_report(path, artifact_root=tmp_path)
    second = build_research_report(path, artifact_root=tmp_path)
    assert first == second
    html_path = export_research_report(first, tmp_path / "reports")
    assert export_research_report(second, tmp_path / "reports") == html_path
    html = html_path.read_text()
    assert html.startswith("<!doctype html>")
    assert "<script" not in html
    assert "<iframe" not in html
    assert f"./../experiments/{path.name}" in html
    assert first.header.to_primitive()["producer_study_id"] == result.study_id
    assert read_manifest(path).execution.run_id in html
    assert result.study_id in html
    assert values(first, "Prediction counts") == [
        result.manifest_primitive()["record_counts"]
    ]
    assert values(first, "Prediction metrics and comparisons") == [None]
    assert "Not applicable to prediction studies" in html
    assert "Unavailable" in html
    assert not values(first, "Backtest performance and drawdown")
    assert "MISSING_TRANSACTION_COST_ASSUMPTIONS" not in codes(first)
    assert "IN_SAMPLE_ONLY" in codes(first)
    assert "synthetic" in html
    assert "SPY" in html
    assert path.read_bytes() == original


def test_native_backtest_retains_exact_performance_benchmark_and_costs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = configured_result()
    source = export_backtest_result(result, tmp_path / "backtests")
    path = native_manifest(tmp_path, StudyType.BACKTEST, source)
    block_research(monkeypatch)
    report = build_research_report(
        path,
        artifact_root=tmp_path,
        config=ResearchReportConfig(maximum_preview_rows=2),
    )
    native = result.manifest_primitive()
    assert values(report, "Backtest performance and drawdown") == [
        native["performance"]
    ]
    benchmark = cast(PrimitiveMapping, values(report, "Benchmark comparison")[0])
    assert (
        benchmark["performance"]
        == cast(PrimitiveMapping, native["benchmark"])["performance"]
    )
    equity = cast(PrimitiveMapping, values(report, "Equity and returns")[0])
    assert equity["preview_truncated"] is True
    assert len(cast(list[Primitive], equity["preview"])) == 2
    assert values(report, "Published monthly / period returns") == [None]
    assert "MISSING_TRANSACTION_COST_ASSUMPTIONS" not in codes(report)
    html = export_research_report(report, tmp_path / "reports").read_text()
    for text in (
        "fixed_per_fill",
        "adverse_basis_points",
        "buy_and_hold",
        "next_session_after_close",
        "maximum drawdown",
    ):
        assert text in html
    assert "IN-SAMPLE" in html
    assert "Prediction counts" not in html


def test_existing_summary_fields_are_copied_without_calculating_missing_metrics(
    tmp_path: Path,
) -> None:
    summary: PrimitiveMapping = {
        "prediction_count": 17,
        "accuracy": "0.731",
        "accuracy_interval": {"lower_bound": "0.51", "upper_bound": "0.9"},
        "direction_distribution": {"up": 13, "down": 4},
        "signed_outcome": {"mean": "0.023", "median": None},
        "mfe": {"mean": "0.04"},
        "mae": {"mean": "-0.01"},
        "event_outcomes": {"rates": {"target_before_stop": "0.6"}},
        "matched_baseline_comparisons": [{"accuracy": "0.52", "baseline": "always_up"}],
        "weekday_comparisons": [{"weekday": "Monday", "count": 4}],
        "period_comparisons": [{"period": "2024", "count": 17}],
    }
    path = metadata_manifest(tmp_path, payload={"summary": summary})
    report = build_research_report(path, artifact_root=tmp_path)
    assert values(report, "Prediction metrics and comparisons") == [
        {"summary": summary}
    ]
    html = export_research_report(report, tmp_path / "reports").read_text()
    assert "0.731" in html
    assert "Unavailable" in html


def test_qf29_feature_schema_dispositions_and_backend_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = _build(tmp_path / "features")
    path = native_manifest(
        tmp_path, StudyType.FEATURE_DATASET, tmp_path / "features" / result.dataset_id
    )
    block_research(monkeypatch)
    report = build_research_report(path, artifact_root=tmp_path)
    assert values(report, "Feature and outcome schema") == [
        result.schema.to_primitive()
    ]
    assert values(report, "Candidate dispositions and row count") == [
        result.summary.to_primitive()
    ]
    assert values(report, "Published feature / outcome distributions") == [None]
    html = export_research_report(report, tmp_path / "reports").read_text()
    for text in (
        "accepted count",
        "rejected count",
        "blocked count",
        "overlapping count",
        "native_v1",
        "features.parquet",
        "outcome",
        "family",
    ):
        assert text in html
    assert not values(report, "Backtest performance and drawdown")


def test_qf32_published_rankings_are_never_reranked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = complete_study(tmp_path, prediction=True)
    source = next(
        (
            completed.study.study_path
            / "folds"
            / completed.source.folds[0].fold_id
            / "selection"
        ).iterdir()
    )
    path = native_manifest(tmp_path, StudyType.PARAMETER_STUDY, source)
    summary = cast(PrimitiveMapping, json.loads((source / "summary.json").read_bytes()))
    block_research(monkeypatch)
    report = build_research_report(path, artifact_root=tmp_path)
    ranks = cast(
        PrimitiveMapping, values(report, "Ranked configurations — published order")[0]
    )
    assert ranks["rankings"] == summary["rankings"]
    assert values(report, "Parameter neighborhoods and stability") == [
        {"stability": summary["stability"]}
    ]
    assert "IN_SAMPLE_ONLY" in codes(report)
    assert all(
        section.phase is ReportPhase.IN_SAMPLE
        for section in report.sections
        if section.title == "Ranked configurations — published order"
    )
    html = export_research_report(report, tmp_path / "reports").read_text()
    assert "not a validated winner" in html
    assert "minimum prediction count" in html


def test_qf6_optimization_preserves_native_ranking_and_execution_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = GridSearchStudy(
        dataset=_dataset(),
        strategy_factory=MovingAverageCrossoverFactory(),
        config=_study_config(tmp_path / "optimization"),
    )
    result = grid.run()
    grid.export(result)
    path = native_manifest(tmp_path, StudyType.OPTIMIZATION, grid.study_path)
    block_research(monkeypatch)
    report = build_research_report(
        path, artifact_root=tmp_path, config=ResearchReportConfig(high_trial_count=2)
    )
    rankings = cast(
        PrimitiveMapping, json.loads((grid.study_path / "ranking.json").read_bytes())
    )
    assert any(
        cast(PrimitiveMapping, item).get("eligible_rankings")
        == rankings["eligible_rankings"]
        for item in values(report, "Ranked configurations — published order")
    )
    assert "HIGH_TRIAL_COUNT" in codes(report)
    assert "MISSING_TRANSACTION_COST_ASSUMPTIONS" not in codes(report)
    html = export_research_report(report, tmp_path / "reports").read_text()
    assert "commission" in html
    assert "benchmark_equity.csv" in html
    assert "recommended robust trial id" in html


def test_existing_qf34_html_is_linked_without_inlining_or_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).parents[3]
    inspection = next(
        (repository / "examples/spy_multi_timeframe/study_inspection_reports").iterdir()
    )
    shutil.copyfile(inspection / "report.html", tmp_path / "inspection.html")
    write_json(
        tmp_path / "prediction.json",
        run_prediction_study(
            make_dataset(("100", "101", "102")),
            create_overnight_gap_prediction_study(
                AlwaysUpPredictionStrategy(AlwaysUpParameters())
            ),
        ).to_primitive(),
    )
    bundle = inspect_study(
        StudyType.PREDICTION, tmp_path / "prediction.json", artifact_root=tmp_path
    )
    chart = index_artifact(
        tmp_path,
        path="inspection.html",
        artifact_type=ArtifactType.INSPECTION,
        schema_version="1",
        producer_study_id=bundle.provenance.producer_study_id,
        producer_artifact_id="inspection",
    )
    path = write_manifest(
        create_manifest(bundle, execution(), additional_artifacts=(chart,)),
        tmp_path / "experiments",
        artifact_root=tmp_path,
    )
    block_research(monkeypatch)
    report = build_research_report(path, artifact_root=tmp_path)
    html = export_research_report(report, tmp_path / "reports").read_text()
    assert "./../inspection.html" in html
    assert chart.sha256 in html if chart.sha256 else False
    assert "<svg" not in html
    assert "<iframe" not in html


def test_export_refuses_clobber_and_outside_root(tmp_path: Path) -> None:
    path = metadata_manifest(tmp_path)
    report = build_research_report(path, artifact_root=tmp_path)
    exported = export_research_report(report, tmp_path / "reports")
    exported.write_text("changed")
    with pytest.raises(ResearchReportError, match="refusing overwrite"):
        export_research_report(report, tmp_path / "reports")
    assert exported.read_text() == "changed"
    with pytest.raises(ResearchReportError, match="inside artifact root"):
        export_research_report(report, tmp_path.parent)
    with pytest.raises(ResearchReportError, match="manifest must be inside"):
        build_research_report(path, artifact_root=tmp_path / "other")


def test_backtest_tables_render_when_export_directory_is_artifact_root(
    tmp_path: Path,
) -> None:
    source = export_backtest_result(configured_result(), tmp_path / "backtests")
    path = native_manifest(source, StudyType.BACKTEST, source)
    report = build_research_report(path, artifact_root=source)
    assert values(report, "Equity and returns")[0] is not None
    assert values(report, "Benchmark equity")[0] is not None


def test_original_ranking_order_is_retained_even_if_objectives_would_sort_differently(
    tmp_path: Path,
) -> None:
    ranks: list[Primitive] = [
        {"rank": 1, "objective_value": "0.1", "trial_id": "first"},
        {"rank": 2, "objective_value": "0.9", "trial_id": "second"},
    ]
    path = metadata_manifest(
        tmp_path,
        kind=StudyType.PARAMETER_STUDY,
        payload={"rankings": ranks},
        category=ArtifactType.PARAMETER_SUMMARY,
    )
    report = build_research_report(path, artifact_root=tmp_path)
    assert values(report, "Ranked configurations — published order") == [
        {"rankings": ranks}
    ]
