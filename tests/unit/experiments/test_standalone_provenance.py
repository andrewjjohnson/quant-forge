from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import (
    BasisPointFees,
    BasisPointSlippage,
    EvaluationInterval,
    FixedCommission,
    run_backtest,
)
from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._json import mapping
from quantforge.prediction import PredictionContextFailurePolicy, run_prediction_study
from quantforge.strategies import (
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
)
from tests.unit.backtesting.test_runner import configured_result, zero_cost_config
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.helpers import SESSIONS, make_dataset
from tests.unit.indicators import test_timeframe_evaluation as timeframe_fixtures
from tests.unit.prediction import test_multi_timeframe_study as fixtures


def rehash_prediction(snapshot: PrimitiveMapping) -> None:
    manifest = mapping(snapshot["manifest"])
    identity = {
        "component": manifest["component"],
        "engine_version": manifest["engine_version"],
        "market_data": manifest["market_data"],
        "study_configuration": manifest["configuration"],
    }
    if "prediction_context" in manifest:
        identity["prediction_context"] = manifest["prediction_context"]
    manifest["study_id"] = configuration_identity(identity)
    for row in cast(list[PrimitiveMapping], snapshot["rows"]):
        row["study_id"] = manifest["study_id"]
        row["row_id"] = configuration_identity(
            {
                "record_type": "prediction_study_row",
                "study_id": manifest["study_id"],
                "outcome_id": mapping(row["outcome"])["outcome_id"],
                "evaluation_id": mapping(row["evaluation"])["evaluation_id"],
                "signal": {
                    "features": row["features"],
                    "prediction": row["prediction"],
                },
            }
        )


def context_result(*, skipped: bool = False, missing: bool = False) -> PrimitiveMapping:
    requirements = fixtures._requirements()  # pyright: ignore[reportPrivateUsage]
    if skipped:
        requirements = replace(
            requirements, failure_policy=PredictionContextFailurePolicy.SKIP
        )
    provider = (
        fixtures.InvalidContextProvider()
        if missing
        else fixtures.FixtureContextProvider(
            fixtures._prediction_context()  # pyright: ignore[reportPrivateUsage]
        )
    )
    return run_prediction_study(
        fixtures._prediction_dataset(),  # pyright: ignore[reportPrivateUsage]
        fixtures._study(fixtures.FixtureMultiTimeframeRule(requirements)),  # pyright: ignore[reportPrivateUsage]
        context_provider=provider,
    ).to_primitive()


@pytest.mark.parametrize("manifest_only", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        "skipped_predictions",
        "requirements",
        "missing_context",
        "source_id",
        "missing_source",
        "future_selected_bar",
        "selected_bars",
        "indicator",
        "invalid_status",
        "decision_session",
        "skipped_policy",
        "skipped_reason",
        "primary_age",
        "after_close",
    ],
)
def test_direct_prediction_rejects_rehashed_context_contradictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_only: bool, change: str
) -> None:
    result = context_result(skipped=change == "skipped_predictions")
    manifest = mapping(result["manifest"])
    context = mapping(manifest["prediction_context"])
    source = mapping(context["source_context"])
    if change.startswith("skipped_"):
        context["status"] = "skipped"
        context["reason"] = "context rejected"
        if change != "skipped_predictions":
            result["rows"] = []
            manifest["record_counts"] = {
                "generated_predictions": 0,
                "labeled_rows": 0,
                "unavailable_outcomes": 0,
            }
        if change == "skipped_reason":
            context.pop("reason")
            mapping(context["requirements"])["failure_policy"] = "skip"
            mapping(manifest["configuration"])["prediction_context_requirements"] = (
                deepcopy(context["requirements"])
            )
    elif change == "requirements":
        mapping(context["requirements"])["failure_policy"] = "skip"
    elif change == "missing_context":
        manifest.pop("prediction_context")
    elif change == "source_id":
        source["context_id"] = "0" * 64
    elif change == "missing_source":
        context["source_context"] = None
    elif change == "primary_age":
        primary_requirement = mapping(mapping(context["requirements"])["primary"])
        primary_requirement["maximum_age_microseconds"] = 1
        mapping(manifest["configuration"])["prediction_context_requirements"] = (
            deepcopy(context["requirements"])
        )
        cast(list[PrimitiveMapping], context["timeframes"])[0]["requirement"] = (
            deepcopy(primary_requirement)
        )
    elif change == "after_close":
        source["as_of"] = "2024-07-11T22:00:00+00:00"
        aligned = cast(list[PrimitiveMapping], source["timeframes"])
        aligned[0]["latest_completed_bar_timestamp"] = source["as_of"]
        for item in aligned:
            item["age_microseconds"] = (
                datetime.fromisoformat(source["as_of"])
                - datetime.fromisoformat(
                    cast(str, item["latest_completed_bar_timestamp"])
                )
            ) // timedelta(microseconds=1)
    elif change == "future_selected_bar":
        aligned = cast(list[PrimitiveMapping], source["timeframes"])
        boundary = datetime.fromisoformat(
            cast(str, aligned[0]["latest_completed_bar_timestamp"])
        )
        future = boundary + timedelta(minutes=1)
        aligned[1]["latest_completed_bar_timestamp"] = future.isoformat()
        aligned[1]["age_microseconds"] = (
            datetime.fromisoformat(cast(str, source["as_of"])) - future
        ) // timedelta(microseconds=1)
    elif change == "selected_bars":
        cast(list[PrimitiveMapping], context["timeframes"])[0]["visible_bar_ids"] = [
            "foreign-bar"
        ]
    elif change == "indicator":
        selected = cast(list[PrimitiveMapping], context["timeframes"])[0]
        cast(list[PrimitiveMapping], selected["indicators"])[0]["backend"] = {
            "backend_id": "changed"
        }
    elif change == "invalid_status":
        context["status"] = "unknown"
    else:
        context["decision_session"] = "2024-07-10"
    if change != "source_id":
        source["context_id"] = configuration_identity(
            {key: value for key, value in source.items() if key != "context_id"}
        )
    rehash_prediction(result)
    path = tmp_path / "prediction.json"
    write_json(path, manifest if manifest_only else result)
    block_research(monkeypatch)
    with pytest.raises(ManifestError):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("manifest_only", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_direct_prediction_preserves_available_and_skipped_contexts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_only: bool, missing: bool
) -> None:
    result = context_result(skipped=missing, missing=missing)
    manifest = mapping(result["manifest"])
    context = mapping(manifest["prediction_context"])
    if not missing:
        source = mapping(context["source_context"])
        primary = cast(list[PrimitiveMapping], source["timeframes"])[0]
        assert source["as_of"] != primary["latest_completed_bar_timestamp"]
    path = tmp_path / "prediction.json"
    write_json(path, manifest if manifest_only else result)
    block_research(monkeypatch)
    inspected = inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    assert (
        inspected.provenance.configuration.to_primitive()["prediction_context"]
        == context
    )


@pytest.mark.parametrize("missing", [True, False])
def test_detached_backtest_requires_sizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    manifest = configured_result().manifest_primitive()
    configuration = mapping(manifest["backtest_configuration"])
    if missing:
        del configuration["sizing"]
    else:
        configuration["sizing"] = None
    market = mapping(manifest["market_data"])
    strategy = mapping(manifest["strategy"])
    manifest["run_id"] = configuration_identity(
        {
            "component": "quantforge_backtest",
            "engine_version": manifest["engine_version"],
            "result_schema_version": manifest["result_schema_version"],
            "market_data": {
                key: market[key]
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
                key: strategy[key]
                for key in (
                    "strategy_id",
                    "strategy_implementation_version",
                    "strategy_configuration_id",
                    "configuration",
                )
            },
            "backtest_configuration": configuration,
        }
    )
    benchmark = mapping(manifest["benchmark"])
    benchmark["benchmark_id"] = configuration_identity(
        {
            "run_id": manifest["run_id"],
            "record_type": "benchmark",
            "configuration": benchmark["configuration"],
        }
    )
    path = tmp_path / "backtest.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError):
        inspect_study(StudyType.BACKTEST, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "initial_capital",
        "commission",
        "fees",
        "slippage",
        "dividend_policy",
        "split_policy",
        "corporate_action_snapshot_id",
        "start",
        "evaluation_interval",
        "implementation_version",
        "benchmark_id",
    ],
)
@pytest.mark.parametrize("rehash", [False, True])
def test_detached_backtest_benchmark_is_bound_to_recorded_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str, rehash: bool
) -> None:
    manifest = configured_result().manifest_primitive()
    benchmark = mapping(manifest["benchmark"])
    if change == "benchmark_id":
        benchmark["benchmark_id"] = "0" * 64
    else:
        mapping(benchmark["configuration"])[change] = "changed"
        if rehash:
            benchmark["benchmark_id"] = configuration_identity(
                {
                    "run_id": manifest["run_id"],
                    "record_type": "benchmark",
                    "configuration": benchmark["configuration"],
                }
            )
    path = tmp_path / "backtest.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError):
        inspect_study(StudyType.BACKTEST, path, artifact_root=tmp_path)


@pytest.mark.parametrize("bounded", [False, True])
def test_detached_backtest_preserves_costs_and_evaluation_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bounded: bool
) -> None:
    configuration = replace(
        zero_cost_config(),
        commission=FixedCommission(Decimal(1)),
        fees=BasisPointFees(Decimal(10)),
        slippage=BasisPointSlippage(Decimal(100)),
        evaluation_interval=EvaluationInterval(SESSIONS[3], SESSIONS[7])
        if bounded
        else None,
    )
    result = run_backtest(
        make_dataset(("3", "2", "1", "2", "3", "4", "3", "2", "1")),
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        configuration,
    )
    manifest = result.manifest_primitive()
    path = tmp_path / "backtest.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    inspected = inspect_study(StudyType.BACKTEST, path, artifact_root=tmp_path)
    captured = inspected.provenance.configuration.to_primitive()
    assert captured["backtest_configuration"] == configuration.to_primitive()
    assert (
        captured["benchmark_configuration"]
        == mapping(manifest["benchmark"])["configuration"]
    )


@pytest.mark.parametrize("manifest_only", [False, True])
def test_direct_context_preserves_rejected_future_bar_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_only: bool
) -> None:
    source = timeframe_fixtures._all_completed_context()  # pyright: ignore[reportPrivateUsage]
    requirements = fixtures._requirements(  # pyright: ignore[reportPrivateUsage]
        failure_policy=PredictionContextFailurePolicy.SKIP
    )
    result = run_prediction_study(
        fixtures._prediction_dataset(),  # pyright: ignore[reportPrivateUsage]
        fixtures._study(fixtures.FixtureMultiTimeframeRule(requirements)),  # pyright: ignore[reportPrivateUsage]
        context_provider=fixtures.FixtureContextProvider(source),
    ).to_primitive()
    manifest = mapping(result["manifest"])
    assert mapping(manifest["prediction_context"])["status"] == "skipped"
    path = tmp_path / "prediction.json"
    write_json(path, manifest if manifest_only else result)
    block_research(monkeypatch)
    inspected = inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    assert (
        inspected.provenance.configuration.to_primitive()["prediction_context"]
        == manifest["prediction_context"]
    )
