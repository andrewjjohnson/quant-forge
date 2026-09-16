"""Rehashed QF-5 manifests must retain counts from captured CSV records."""

import csv
from dataclasses import replace
from datetime import date
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import export_backtest_result, run_backtest
from quantforge.configuration import PrimitiveMapping
from quantforge.data.models import MarketDataset
from quantforge.experiments import StudyType, inspect_study, verify_artifacts
from quantforge.experiments._json import mapping
from quantforge.strategies import StrategyOutput
from tests.unit.backtesting.test_runner import (
    PRICES,
    ManualTransitionStrategy,
    configured_result,
    zero_cost_config,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_backtest_manifest_schema import inspect_manifest
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record
from tests.unit.helpers import make_dataset


@pytest.fixture(params=[False, True], ids=["ordinary", "corporate_actions"])
def export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Path:
    result = (
        run_backtest(
            make_dataset(
                ("100", "100", "50", "50"),
                dividends=((date(2024, 7, 3), "1"),),
                splits=((date(2024, 7, 3), "2"),),
            ),
            ManualTransitionStrategy(),
            zero_cost_config(),
        )
        if request.param
        else configured_result()
    )
    root = export_backtest_result(result, tmp_path / "backtests")
    block_research(monkeypatch)
    return root


def refresh_sidecar(export: Path) -> None:
    integrity = read_record(export / "integrity.json")
    for filename in mapping(integrity["files"]):
        mapping(integrity["files"])[filename] = sha256(
            (export / filename).read_bytes()
        ).hexdigest()
    write_json(export / "integrity.json", integrity)


@pytest.mark.parametrize(
    "name",
    [
        "signals",
        "orders",
        "fills",
        "positions",
        "completed_trades",
        "open_trades",
        "daily_equity",
        "benchmark_daily_equity",
        "dividend_cashflows",
        "split_adjustments",
        "benchmark_dividend_cashflows",
        "benchmark_split_adjustments",
    ],
)
def test_every_count_matches_its_integrity_checked_table(
    export: Path, name: str
) -> None:
    manifest = read_record(export / "manifest.json")
    counts = mapping(manifest["record_counts"])
    counts[name] = cast(int, counts[name]) + 1
    inspect_manifest(export, manifest)


def test_trade_count_redistribution_cannot_hide_behind_unchanged_total(
    export: Path,
) -> None:
    manifest = read_record(export / "manifest.json")
    counts = mapping(manifest["record_counts"])
    assert cast(int, counts["completed_trades"]) > 0
    counts["completed_trades"] = cast(int, counts["completed_trades"]) - 1
    counts["open_trades"] = cast(int, counts["open_trades"]) + 1
    inspect_manifest(export, manifest)


@pytest.mark.parametrize(
    "change", ["invalid_flag", "missing_flag", "short_row", "empty_file"]
)
def test_trade_tables_need_unambiguous_logical_records(
    export: Path, change: str
) -> None:
    path = export / "trades.csv"
    rows = list(csv.reader(StringIO(path.read_text(), newline="")))
    if change == "invalid_flag":
        rows[1][rows[0].index("is_open")] = "unknown"
    elif change == "missing_flag":
        index = rows[0].index("is_open")
        rows = [row[:index] + row[index + 1 :] for row in rows]
    elif change == "short_row":
        rows[1].pop()
    else:
        rows = []
    stream = StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    path.write_text(stream.getvalue())
    refresh_sidecar(export)
    inspect_manifest(export, read_record(export / "manifest.json"))


@pytest.mark.parametrize("corporate", [False, True])
@pytest.mark.parametrize("reason", ['first line\nsecond, "quoted" line', "", None])
def test_logical_counts_accept_quoted_newlines_and_empty_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corporate: bool, reason: str | None
) -> None:
    class ReasonedStrategy(ManualTransitionStrategy):
        name = "reasoned_fixture"

        def generate(self, dataset: MarketDataset) -> StrategyOutput:
            output = super().generate(dataset)
            return replace(
                output,
                decisions=tuple(
                    replace(row, reason=reason) for row in output.decisions
                ),
            )

    dataset = make_dataset(
        ("100", "100", "50", "50") if corporate else ("100", "100", "100", "100"),
        dividends=((date(2024, 7, 3), "1"),) if corporate else (),
        splits=((date(2024, 7, 3), "2"),) if corporate else (),
    )
    result = run_backtest(dataset, ReasonedStrategy(), zero_cost_config())
    export = export_backtest_result(result, tmp_path / "backtests")
    block_research(monkeypatch)
    path = export / "signals.csv"
    rows = list(csv.reader(StringIO(path.read_text(), newline="")))
    assert rows[1][rows[0].index("reason")] == (reason or "")
    stream = StringIO(newline="")
    csv.writer(stream, lineterminator="\r\n").writerows(rows)
    path.write_bytes(stream.getvalue().encode("utf-8"))
    refresh_sidecar(export)
    inspect_manifest(export, read_record(export / "manifest.json"), reject=False)


def test_detached_manifest_preserves_counts_without_claiming_table_verification(
    export: Path,
) -> None:
    manifest: PrimitiveMapping = read_record(export / "manifest.json")
    mapping(manifest["record_counts"])["signals"] = 999
    detached = export.parent / "detached.json"
    write_json(detached, manifest)
    inspected = inspect_study(StudyType.BACKTEST, detached, artifact_root=export.parent)
    assert verify_artifacts(inspected.index, export.parent).valid
    assert {entry.path for entry in inspected.index.entries} == {"detached.json"}


def test_native_open_trade_is_counted_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = configured_result(PRICES[:6])
    assert len(result.open_trades) == 1
    assert not result.completed_trades
    root = export_backtest_result(result, tmp_path / "backtests")
    block_research(monkeypatch)
    inspect_manifest(root, result.manifest_primitive(), reject=False)
