"""Benchmark execution evidence must belong to the manifest's QF-5 run."""

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import BacktestConfig, run_backtest
from quantforge.configuration import Primitive
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.experiments._json import mapping
from tests.unit.backtesting.test_runner import (
    PRICES,
    ManualTransitionStrategy,
    configured_result,
    zero_cost_config,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_backtest_manifest_schema import export as export
from tests.unit.experiments.test_backtest_manifest_schema import inspect_manifest
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record
from tests.unit.helpers import make_dataset


@pytest.mark.parametrize(
    ("record", "field", "invalid"),
    [
        ("order", "run_id", "foreign"),
        ("order", "order_id", "foreign"),
        ("order", "originating_signal_id", "foreign"),
        ("order", "symbol", "QQQ"),
        ("order", "strategy_id", "foreign"),
        ("order", "strategy_configuration_id", "foreign"),
        ("order", "side", "sell"),
        ("order", "order_type", "limit"),
        ("order", "target_position", "flat"),
        ("order", "target_weight", "0.5"),
        ("order", "signal_session", "2024-07-02"),
        ("order", "earliest_permitted_execution_session", None),
        ("order", "status", "rejected"),
        ("order", "reason", "foreign"),
        ("order", "requested_quantity", 0),
        ("fill", "fill_id", "foreign"),
        ("fill", "order_id", "foreign"),
        ("fill", "originating_signal_id", "foreign"),
        ("fill", "symbol", "QQQ"),
        ("fill", "side", "sell"),
        ("fill", "strategy_id", "foreign"),
        ("fill", "strategy_configuration_id", "foreign"),
        ("fill", "execution_session", "2024-07-02"),
        ("fill", "quantity", 0),
    ],
)
def test_saved_benchmark_provenance_and_fill_links_are_required(
    export: Path, record: str, field: str, invalid: Primitive
) -> None:
    manifest = read_record(export / "manifest.json")
    mapping(mapping(manifest["benchmark"])[record])[field] = invalid
    inspect_manifest(export, manifest)
    detached = export.parent / "detached.json"
    write_json(detached, manifest)
    with pytest.raises(ManifestError, match="backtest benchmark"):
        inspect_study(StudyType.BACKTEST, detached, artifact_root=export.parent)


@pytest.mark.parametrize("records", [("order",), ("fill",), ("order", "fill")])
def test_foreign_benchmark_records_cannot_be_transplanted_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, records: tuple[str, ...]
) -> None:
    target = configured_result().manifest_primitive()
    foreign = configured_result(PRICES, dataset_id="foreign").manifest_primitive()
    for name in records:
        mapping(target["benchmark"])[name] = deepcopy(
            mapping(foreign["benchmark"])[name]
        )
    path = tmp_path / "backtest.json"
    write_json(path, target)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="backtest benchmark"):
        inspect_study(StudyType.BACKTEST, path, artifact_root=tmp_path)


@pytest.mark.parametrize("field", ["fill", "quantity", "status", "reason", "valid"])
def test_rejected_benchmark_retains_its_zero_quantity_and_absent_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    dataset = make_dataset(("100", "100", "100", "100"))
    config = zero_cost_config()
    result = run_backtest(
        dataset,
        ManualTransitionStrategy(),
        BacktestConfig(Decimal(1), config.commission, config.fees, config.slippage),
    )
    manifest = result.manifest_primitive()
    benchmark = mapping(manifest["benchmark"])
    order = mapping(benchmark["order"])
    assert benchmark["fill"] is None
    if field == "fill":
        benchmark["fill"] = mapping(
            configured_result().manifest_primitive()["benchmark"]
        )["fill"]
    elif field == "quantity":
        order["requested_quantity"] = 1
    elif field == "status":
        order["status"] = "filled"
    elif field == "reason":
        order["reason"] = None
    path = tmp_path / "backtest.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    if field == "valid":
        inspected = inspect_study(StudyType.BACKTEST, path, artifact_root=tmp_path)
        assert verify_artifacts(inspected.index, tmp_path).valid
    else:
        with pytest.raises(ManifestError, match="backtest benchmark"):
            inspect_study(StudyType.BACKTEST, path, artifact_root=tmp_path)


def test_filled_benchmark_cannot_lose_its_fill(export: Path) -> None:
    manifest = read_record(export / "manifest.json")
    benchmark = mapping(manifest["benchmark"])
    assert cast(int, mapping(benchmark["order"])["requested_quantity"]) > 0
    benchmark["fill"] = None
    inspect_manifest(export, manifest)
