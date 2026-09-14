import ast
import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import (
    BacktestConfig,
    BasisPointSlippage,
    DividendPolicy,
    ExplicitZeroFees,
    FixedCommission,
    export_backtest_result,
    run_backtest,
)
from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments import (
    ArtifactType,
    ManifestError,
    RelationshipType,
    StudyType,
    create_manifest,
    inspect_study,
    read_manifest,
    verify_artifacts,
    write_manifest,
)
from quantforge.optimization import GridSearchStudy, MovingAverageCrossoverFactory
from quantforge.prediction import (
    AlwaysUpParameters,
    AlwaysUpPredictionStrategy,
    create_overnight_gap_prediction_study,
    run_prediction_study,
)
from quantforge.strategies import (
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
)
from tests.unit.backtesting.test_runner import PRICES, configured_result
from tests.unit.experiments.test_contracts import execution, write_json
from tests.unit.helpers import make_dataset
from tests.unit.optimization.test_study import (
    _dataset,  # pyright: ignore[reportPrivateUsage]
    _study_config,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _build,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_grid import (
    _grid,  # pyright: ignore[reportPrivateUsage]
)


def block_research(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("manifest generation attempted research computation")

    for module_name, function in (
        ("quantforge.prediction.study", "run_prediction_study"),
        ("quantforge.prediction.study", "run_prediction_study_in_session"),
        ("quantforge.prediction.window", "run_prediction_window"),
        ("quantforge.prediction.feature_dataset", "build_signal_feature_dataset"),
        ("quantforge.backtesting.runner", "run_backtest"),
        ("quantforge.oos.prediction", "aggregate_prediction"),
        ("quantforge.oos.prediction", "summarize_prediction_observations"),
        ("quantforge.oos.backtest", "aggregate_backtest"),
        ("quantforge.oos.common", "configuration_stability"),
        ("quantforge.reporting.study_inspection", "build_study_inspection_report"),
        ("quantforge.reporting.study_inspection", "export_study_inspection_report"),
    ):
        monkeypatch.setattr(f"{module_name}.{function}", forbidden)
    for class_name in (
        "quantforge.optimization.study.GridSearchStudy",
        "quantforge.prediction.grid.PredictionGridStudy",
        "quantforge.walk_forward.study.WalkForwardStudy",
    ):
        for method in ("run", "resume"):
            monkeypatch.setattr(f"{class_name}.{method}", forbidden)
    for backend in ("native.NativeIndicatorBackend", "talib.TalibIndicatorBackend"):
        monkeypatch.setattr(
            f"quantforge.indicators.backends.{backend}.compute", forbidden
        )
    import talib

    for function in cast(list[str], talib.get_functions()):
        monkeypatch.setattr(talib, function, forbidden)


def test_prediction_example_reads_existing_snapshot_without_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(("100", "101", "102", "103", "104"))
    result = run_prediction_study(
        dataset,
        create_overnight_gap_prediction_study(
            AlwaysUpPredictionStrategy(AlwaysUpParameters())
        ),
    )
    path = tmp_path / "prediction.json"
    write_json(path, result.to_primitive())
    block_research(monkeypatch)
    bundle = inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    manifest = create_manifest(bundle, execution())
    assert manifest.provenance.producer_study_id == result.study_id
    provenance = manifest.provenance.configuration.to_primitive()
    assert provenance["configuration"] == result.configuration.to_primitive()
    assert provenance["market_data"] == result.market_data.to_primitive()
    assert "backtest_configuration" not in provenance
    assert {entry.artifact_type for entry in bundle.index.entries} == {
        ArtifactType.CONFIGURATION,
        ArtifactType.PREDICTION_RESULT,
        ArtifactType.SOURCE_DATASET,
    }
    exported = write_manifest(
        manifest, tmp_path / "experiments", artifact_root=tmp_path
    )
    assert (
        read_manifest(exported, artifact_root=tmp_path).serialize()
        == manifest.serialize()
    )


def test_backtest_example_keeps_execution_and_native_backend_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = configured_result()
    exported = export_backtest_result(result, tmp_path / "backtests")
    block_research(monkeypatch)
    bundle = inspect_study(StudyType.BACKTEST, exported, artifact_root=tmp_path)
    manifest = create_manifest(bundle, execution())
    config = manifest.provenance.configuration.to_primitive()
    assert manifest.provenance.producer_study_id == result.run_id
    assert config["backtest_configuration"] == result.backtest_configuration
    assert config["strategy"] == result.manifest_primitive()["strategy"]
    assert config["benchmark_configuration"] == result.benchmark.configuration
    assert "arithmetic" in json.dumps(config)
    assert "talib_v1" not in json.dumps(config)
    assert len(bundle.index.entries) == 14
    assert verify_artifacts(bundle.index, tmp_path).valid
    write_manifest(manifest, tmp_path / "experiments", artifact_root=tmp_path)


@pytest.mark.parametrize("corporate_action", ["dividend", "split"])
def test_backtest_corporate_actions_can_be_indexed_without_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corporate_action: str
) -> None:
    session = date(2024, 7, 9)
    dataset = make_dataset(
        PRICES,
        dividend_sessions=(session,) if corporate_action == "dividend" else (),
        split_sessions=(session,) if corporate_action == "split" else (),
    )
    result = run_backtest(
        dataset,
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(1)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(100)),
            dividend_policy=DividendPolicy.CASH_DIVIDENDS,
        ),
    )
    strategy_records = (
        result.dividend_cashflows
        if corporate_action == "dividend"
        else result.split_adjustments
    )
    benchmark_records = (
        result.benchmark.dividend_cashflows
        if corporate_action == "dividend"
        else result.benchmark.split_adjustments
    )
    assert {record.account_id for record in strategy_records} == {"strategy"}
    assert {record.account_id for record in benchmark_records} == {"benchmark"}
    exported = export_backtest_result(result, tmp_path / "backtests")
    block_research(monkeypatch)
    bundle = inspect_study(StudyType.BACKTEST, exported, artifact_root=tmp_path)
    manifest = create_manifest(bundle, execution())
    assert verify_artifacts(bundle.index, tmp_path).valid
    path = write_manifest(manifest, tmp_path / "experiments", artifact_root=tmp_path)
    assert (
        read_manifest(path, artifact_root=tmp_path).serialize() == manifest.serialize()
    )


def test_qf29_features_schema_and_context_lineage_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _, _ = _build(tmp_path / "features")
    block_research(monkeypatch)
    bundle = inspect_study(
        StudyType.FEATURE_DATASET,
        tmp_path / "features" / result.dataset_id,
        artifact_root=tmp_path,
    )
    config = bundle.provenance.configuration.to_primitive()
    assert config["configuration"] == result.configuration
    assert config["feature_schema"] == result.schema.to_primitive()
    assert config["market_data"] == result.market_data.to_primitive()
    formats = {entry.file_format.value for entry in bundle.index.entries}
    assert formats == {"json", "csv", "parquet"}
    assert verify_artifacts(bundle.index, tmp_path).valid
    serialized = create_manifest(bundle, execution()).serialize()
    assert b"backend" in serialized
    assert b"family" in serialized
    assert b"forward" in serialized


def test_qf32_parameter_study_keeps_trials_and_search_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = _grid(tmp_path / "grid")
    result = grid.run()
    block_research(monkeypatch)
    bundle = inspect_study(
        StudyType.PARAMETER_STUDY,
        tmp_path / "grid" / result.study_id,
        artifact_root=tmp_path,
    )
    provenance = bundle.provenance.configuration.to_primitive()
    assert all(
        provenance[key] is not None
        for key in (
            "search_space",
            "parameter_constraints",
            "ranking",
            "stability",
            "indicator_backend",
            "study_factory",
        )
    )
    observations = bundle.provenance.observations.to_primitive()
    assert set(cast(list[str], observations["trial_ids"])) == {
        trial.trial_id for trial in result.trials
    }
    assert cast(PrimitiveMapping, observations["trial_counts"])["trials"] == len(
        result.trials
    )
    assert verify_artifacts(bundle.index, tmp_path).valid
    assert any(
        edge.relationship is RelationshipType.CONFIGURED_BY
        for edge in bundle.index.relationships
    )


def test_qf6_optimization_keeps_backtest_artifacts_and_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = GridSearchStudy(
        dataset=_dataset(),
        strategy_factory=MovingAverageCrossoverFactory(),
        config=_study_config(tmp_path / "optimization"),
    )
    result = grid.run()
    grid.export(result)
    block_research(monkeypatch)
    bundle = inspect_study(
        StudyType.OPTIMIZATION, grid.study_path, artifact_root=tmp_path
    )
    assert bundle.provenance.producer_study_id == result.study_id
    assert (
        bundle.provenance.configuration.to_primitive()
        == grid.manifest_primitive()["identity_inputs"]
    )
    assert any(
        entry.artifact_type is ArtifactType.BACKTEST_RESULT
        for entry in bundle.index.entries
    )
    assert verify_artifacts(bundle.index, tmp_path).valid


def test_prediction_identity_retains_historical_backend_and_unknown_metadata(
    tmp_path: Path,
) -> None:
    result = configured_result()
    path = tmp_path / "backtest.json"
    write_json(path, result.manifest_primitive())
    first = create_manifest(
        inspect_study(StudyType.BACKTEST, path, artifact_root=tmp_path), execution()
    )
    primitive = result.manifest_primitive()
    strategy = cast(PrimitiveMapping, primitive["strategy"])
    config = cast(PrimitiveMapping, strategy["configuration"])
    config["historical_backend"] = {
        "backend_id": "talib_v1",
        "library_version": "0.7.0",
        "function_name": "SMA",
    }
    # A historical configuration must carry its own consistent producer IDs.
    strategy["strategy_configuration_id"] = configuration_identity(config)
    market_data = cast(PrimitiveMapping, primitive["market_data"])
    primitive["run_id"] = configuration_identity(
        {
            "component": "quantforge_backtest",
            "engine_version": primitive["engine_version"],
            "result_schema_version": primitive["result_schema_version"],
            "market_data": {
                key: market_data[key]
                for key in (
                    "dataset_id",
                    "schema_version",
                    "adjustment_mode",
                    "calendar",
                    "corporate_action_snapshot_id",
                    "bars_fingerprint",
                )
            },
            "strategy": {
                key: value
                for key, value in strategy.items()
                if key != "warm_up_observations"
            },
            "backtest_configuration": primitive["backtest_configuration"],
        }
    )
    write_json(path, primitive)
    second = create_manifest(
        inspect_study(StudyType.BACKTEST, path, artifact_root=tmp_path), execution()
    )
    assert first.study_id != second.study_id
    assert b'"library_version":"0.7.0"' in second.serialize()
    unknown = replace(
        first, execution=replace(execution(), code=type(execution().code)())
    )
    assert unknown.execution.code.git_commit is None


def test_architecture_has_no_execution_or_rendering_imports() -> None:
    # Resolve from the repository, independent of the caller's cwd.
    source = Path(__file__).parents[3] / "src" / "quantforge" / "experiments"
    forbidden = {
        "run_prediction_study",
        "run_prediction_window",
        "build_signal_feature_dataset",
        "run_backtest",
        "aggregate_prediction",
        "aggregate_backtest",
        "configuration_stability",
        "build_study_inspection_report",
        "PredictionStudy",
        "GridSearchStudy",
        "PredictionGridStudy",
        "WalkForwardStudy",
        "MultiTimeframeContext",
    }
    files = tuple(source.glob("*.py"))
    assert files
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("talib") for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("talib", "quantforge.indicators", "quantforge.reporting")
                )
                assert not {alias.name for alias in node.names} & forbidden


def test_wrong_producer_type_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    write_json(path, {"component": "unrelated", "study_id": "a"})
    with pytest.raises(ManifestError):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


def test_successful_qf32_trials_link_the_original_qf42_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.oos.conftest import complete_study

    completed = complete_study(tmp_path, prediction=True)
    folder = (
        completed.study.study_path
        / "folds"
        / completed.source.folds[0].fold_id
        / "selection"
    )
    grid_path = next(path for path in folder.iterdir() if path.is_dir())
    block_research(monkeypatch)
    bundle = inspect_study(StudyType.PARAMETER_STUDY, grid_path, artifact_root=tmp_path)
    assert any(
        entry.artifact_type is ArtifactType.PREDICTION_RESULT
        for entry in bundle.index.entries
    )
    assert any(
        edge.relationship is RelationshipType.DERIVED_FROM
        for edge in bundle.index.relationships
    )
    assert b"decision_schedule" in create_manifest(bundle, execution()).serialize()


def test_qf7_feature_export_retains_dispositions_and_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quantforge.prediction import SignalDisposition
    from tests.unit.prediction.test_feature_dataset import (
        FixtureCandidateRule,
        _build_fixture,  # pyright: ignore[reportPrivateUsage]
    )

    result = _build_fixture(
        make_dataset(("100", "102", "101", "104")),
        FixtureCandidateRule(tuple(SignalDisposition)),
        tmp_path / "features",
    )
    block_research(monkeypatch)
    bundle = inspect_study(
        StudyType.FEATURE_DATASET,
        tmp_path / "features" / result.dataset_id,
        artifact_root=tmp_path,
    )
    assert (
        bundle.provenance.observations.to_primitive()["record_counts"]
        == result.summary.to_primitive()
    )
    assert (
        bundle.provenance.configuration.to_primitive()["configuration"]
        == result.configuration
    )
    assert all(entry.file_format.value != "parquet" for entry in bundle.index.entries)
    (tmp_path / "features" / result.dataset_id / "features.csv").unlink()
    with pytest.raises(ManifestError, match="missing"):
        inspect_study(
            StudyType.FEATURE_DATASET,
            tmp_path / "features" / result.dataset_id,
            artifact_root=tmp_path,
        )


def test_existing_qf34_inspection_can_be_indexed_without_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantforge.experiments import ArtifactIndex, index_artifact

    root = Path(__file__).parents[3]
    folder = next(
        (root / "examples/spy_multi_timeframe/study_inspection_reports").iterdir()
    )
    document = cast(
        PrimitiveMapping, json.loads((folder / "manifest.json").read_bytes())
    )
    block_research(monkeypatch)
    entry = index_artifact(
        root,
        path=(folder / "report.html").relative_to(root).as_posix(),
        artifact_type=ArtifactType.INSPECTION,
        schema_version=cast(str, document["schema_version"]),
        producer_study_id=cast(str, document["report_id"]),
        producer_artifact_id="inspection_html",
    )
    assert verify_artifacts(ArtifactIndex((entry,)), root).valid


def test_backtest_preexisting_corruption_is_not_blessed_by_a_new_manifest(
    tmp_path: Path,
) -> None:
    from quantforge.backtesting import ResultExportError

    path = export_backtest_result(configured_result(), tmp_path / "backtest")
    (path / "equity.csv").write_text("tampered\n")
    with pytest.raises(ResultExportError):
        inspect_study(StudyType.BACKTEST, path, artifact_root=tmp_path)
