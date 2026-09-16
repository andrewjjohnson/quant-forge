"""Rehashed CSVs must retain producer schemas and a coherent execution graph."""

import csv
from io import StringIO
from pathlib import Path

import pytest

from quantforge.backtesting import export_backtest_result
from tests.unit.backtesting.test_runner import configured_result
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_backtest_manifest_schema import inspect_manifest
from tests.unit.experiments.test_backtest_table_counts import export as export
from tests.unit.experiments.test_backtest_table_counts import refresh_sidecar
from tests.unit.experiments.test_grid_integrity import read_record


def change_csv(export: Path, table: str, column: str, invalid: str) -> None:
    path = export / (table + ".csv")
    rows = list(csv.reader(StringIO(path.read_text(), newline="")))
    assert len(rows) > 1
    rows[1][rows[0].index(column)] = invalid
    stream = StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    path.write_bytes(stream.getvalue().encode("utf-8"))
    refresh_sidecar(export)


@pytest.mark.parametrize("table", ["signals", "orders", "fills", "trades"])
@pytest.mark.parametrize(
    "change", ["blank", "foreign", "missing_header", "unknown_header", "duplicate"]
)
def test_rehashed_execution_tables_require_complete_identifiable_records(
    export: Path, table: str, change: str
) -> None:
    path = export / (table + ".csv")
    rows = list(csv.reader(StringIO(path.read_text(), newline="")))
    assert len(rows) > 1
    if change in {"blank", "foreign"}:
        rows[1] = ["" if change == "blank" else "unrelated"] * len(rows[0])
    elif change == "missing_header":
        rows = [row[1:] for row in rows]
    elif change == "unknown_header":
        rows[0][0] = "unrelated"
    else:
        rows.append(rows[1])
    stream = StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    path.write_text(stream.getvalue())
    refresh_sidecar(export)
    inspect_manifest(export, read_record(export / "manifest.json"))


@pytest.mark.parametrize(
    ("table", "column", "invalid"),
    [
        ("signals", "signal_id", "foreign"),
        ("signals", "canonical_symbol", "QQQ"),
        ("signals", "strategy_configuration_id", "foreign"),
        ("signals", "strategy_parameters", "[]"),
        ("signals", "indicator_values", '{"x":"NaN"}'),
        ("signals", "target_weight", "NaN"),
        ("signals", "signal_session", "yesterday"),
        ("orders", "run_id", "foreign"),
        ("orders", "originating_signal_id", "foreign"),
        ("orders", "order_id", "foreign"),
        ("orders", "symbol", "QQQ"),
        ("orders", "strategy_id", "foreign"),
        ("orders", "strategy_configuration_id", "foreign"),
        ("orders", "side", "unknown"),
        ("orders", "order_type", "limit"),
        ("orders", "requested_quantity", "1.0"),
        ("orders", "requested_quantity", "-1"),
        ("orders", "status", "filled_elsewhere"),
        ("orders", "signal_session", "2024-07-09"),
        ("fills", "fill_id", "foreign"),
        ("fills", "order_id", "foreign"),
        ("fills", "originating_signal_id", "foreign"),
        ("fills", "symbol", "QQQ"),
        ("fills", "strategy_configuration_id", "foreign"),
        ("fills", "quantity", "0"),
        ("fills", "commission", "-1"),
        ("fills", "fill_price", "0"),
        ("fills", "execution_session", "1900-01-01"),
        ("trades", "trade_id", "foreign"),
        ("trades", "entry_signal_id", "foreign"),
        ("trades", "entry_order_id", "foreign"),
        ("trades", "entry_fill_id", "foreign"),
        ("trades", "exit_fill_id", "foreign"),
        ("trades", "strategy_id", "foreign"),
        ("trades", "strategy_implementation_version", "foreign"),
        ("trades", "entry_quantity", "0"),
        ("trades", "entry_price", "0"),
        ("trades", "exit_commission", ""),
        ("trades", "net_profit_loss", "Infinity"),
        ("trades", "is_open", "True"),
        ("positions", "symbol", "QQQ"),
        ("positions", "shares", "999"),
        ("equity", "fill_ids", '["foreign"]'),
        ("equity", "order_ids", '["foreign"]'),
        ("equity", "exposed", "true"),
        ("equity", "cash", "-1"),
        ("benchmark_equity", "order_ids", "[]"),
    ],
)
def test_csv_primitive_domains_and_cross_record_references(
    export: Path, table: str, column: str, invalid: str
) -> None:
    change_csv(export, table, column, invalid)
    inspect_manifest(export, read_record(export / "manifest.json"))


@pytest.mark.parametrize(
    "tables",
    [
        ("signals",),
        ("orders",),
        ("fills",),
        ("trades",),
        ("signals", "orders", "fills", "trades"),
    ],
)
def test_complete_foreign_execution_graph_is_still_bound_to_original_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tables: tuple[str, ...]
) -> None:
    target = export_backtest_result(configured_result(), tmp_path / "target")
    foreign = export_backtest_result(
        configured_result(dataset_id="foreign"), tmp_path / "foreign"
    )
    for table in tables:
        (target / (table + ".csv")).write_bytes(
            (foreign / (table + ".csv")).read_bytes()
        )
    refresh_sidecar(target)
    block_research(monkeypatch)
    inspect_manifest(target, read_record(target / "manifest.json"))


@pytest.mark.parametrize("export", [True], indirect=True)
@pytest.mark.parametrize(
    ("table", "column", "invalid"),
    [
        ("dividend_cashflows", "run_id", "foreign"),
        ("dividend_cashflows", "account_id", "benchmark"),
        ("dividend_cashflows", "source_dataset_id", "foreign"),
        ("dividend_cashflows", "symbol", "QQQ"),
        ("dividend_cashflows", "corporate_action_id", "foreign"),
        ("dividend_cashflows", "ex_dividend_session", "1900-01-01"),
        ("dividend_cashflows", "amount_per_share", "-1"),
        ("split_adjustments", "split_adjustment_id", "foreign"),
        ("split_adjustments", "split_factor", "0"),
        ("split_adjustments", "split_ratio_numerator", "0"),
        ("split_adjustments", "split_ratio_denominator", "0"),
        ("split_adjustments", "shares_after", "-1"),
        ("benchmark_dividend_cashflows", "total_dividend_cash", "999"),
        ("benchmark_split_adjustments", "shares_after", "999"),
        ("equity", "dividend_cashflow_ids", '["foreign"]'),
        ("equity", "split_adjustment_ids", '["foreign"]'),
    ],
)
def test_corporate_records_and_ledger_references_retain_saved_provenance(
    export: Path, table: str, column: str, invalid: str
) -> None:
    change_csv(export, table, column, invalid)
    inspect_manifest(export, read_record(export / "manifest.json"))


def test_logical_column_order_preserves_native_record_identities(export: Path) -> None:
    for path in export.glob("*.csv"):
        rows = list(csv.reader(StringIO(path.read_text(), newline="")))
        stream = StringIO(newline="")
        csv.writer(stream, lineterminator="\r\n").writerows(
            list(reversed(row)) for row in rows
        )
        path.write_bytes(stream.getvalue().encode("utf-8"))
    refresh_sidecar(export)
    inspect_manifest(export, read_record(export / "manifest.json"), reject=False)
