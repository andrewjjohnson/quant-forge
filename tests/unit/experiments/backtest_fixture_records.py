"""Refresh only persisted identity links when a metadata fixture changes run ID."""

import csv
from io import StringIO
from pathlib import Path
from typing import cast

from quantforge.backtesting.export import (
    _write_csv,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments._backtest_csv_schema import decode_csv_row
from quantforge.experiments._json import mapping, text


def refresh_export_records(export: Path, manifest: PrimitiveMapping) -> None:
    """Preserve quantities/prices/results while refreshing their deterministic IDs."""
    tables: dict[str, list[PrimitiveMapping]] = {}
    headers: dict[str, tuple[str, ...]] = {}
    for path in export.glob("*.csv"):
        name = {
            "equity": "daily_equity",
            "benchmark_equity": "benchmark_daily_equity",
        }.get(path.stem, path.stem)
        rows = csv.reader(StringIO(path.read_text(), newline=""))
        header = next(rows)
        headers[name] = tuple(header)
        tables[name] = [
            decode_csv_row(name, dict(zip(header, row, strict=True))) for row in rows
        ]
    run_id = manifest["run_id"]
    replacements: dict[str, Primitive] = {}
    for ordinal, signal in enumerate(tables["signals"]):
        replacements[text(signal["signal_id"])] = configuration_identity(
            {
                "run_id": run_id,
                "record_type": "signal",
                "ordinal": ordinal,
                "decision": {
                    key: value for key, value in signal.items() if key != "signal_id"
                },
            }
        )
    for order in tables["orders"]:
        replacements[text(order["run_id"])] = run_id
        replacements[text(order["order_id"])] = configuration_identity(
            {
                "run_id": run_id,
                "record_type": "order",
                "signal_id": replacements[text(order["originating_signal_id"])],
            }
        )
    for fill in tables["fills"]:
        replacements[text(fill["fill_id"])] = configuration_identity(
            {
                "run_id": run_id,
                "record_type": "fill",
                "order_id": replacements[text(fill["order_id"])],
            }
        )
    for trade in tables["trades"]:
        identity: PrimitiveMapping = {
            "run_id": run_id,
            "record_type": "trade",
            "entry_fill_id": replacements[text(trade["entry_fill_id"])],
        }
        if not trade["is_open"]:
            identity["exit_fill_id"] = replacements[text(trade["exit_fill_id"])]
        replacements[text(trade["trade_id"])] = configuration_identity(identity)
    for name in (
        "dividend_cashflows",
        "benchmark_dividend_cashflows",
        "split_adjustments",
        "benchmark_split_adjustments",
    ):
        kind = "dividend_cashflow" if "dividend" in name else "split_adjustment"
        for row in tables[name]:
            replacements[text(row[kind + "_id"])] = configuration_identity(
                {
                    "run_id": run_id,
                    "account_id": row["account_id"],
                    "record_type": kind,
                    "corporate_action_id": row["corporate_action_id"],
                }
            )
    first = tables["benchmark_daily_equity"][0]
    benchmark = mapping(manifest["benchmark"])
    replacements[text(cast(list[Primitive], first["order_ids"])[0])] = mapping(
        benchmark["order"]
    )["order_id"]
    if first["fill_ids"]:
        replacements[text(cast(list[Primitive], first["fill_ids"])[0])] = mapping(
            benchmark["fill"]
        )["fill_id"]

    def replace(value: Primitive) -> Primitive:
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    for name, rows in tables.items():
        filename = {
            "daily_equity": "equity",
            "benchmark_daily_equity": "benchmark_equity",
        }.get(name, name)
        _write_csv(
            export / (filename + ".csv"),
            [mapping(replace(row)) for row in rows],
            headers[name],
        )
    from tests.unit.experiments.test_backtest_table_counts import refresh_sidecar

    refresh_sidecar(export)
