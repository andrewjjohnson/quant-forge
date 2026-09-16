"""Reconcile captured QF-5 CSV row counts without reconstructing results."""

import csv
from io import StringIO
from pathlib import Path

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._backtest_csv_schema import (
    decode_csv_row,
    validate_csv_header,
)
from quantforge.experiments._backtest_manifest_integrity import (
    validate_backtest_manifest,
)
from quantforge.experiments._backtest_record_integrity import validate_backtest_records
from quantforge.experiments._json import ManifestError, mapping
from quantforge.experiments._producer_snapshot import ProducerReadSet


def validate_backtest_table_counts(export: Path, reads: ProducerReadSet) -> None:
    """Count logical CSV records from the same bytes pinned by the sidecar."""
    manifest, _ = reads.read(export / "manifest.json")
    validate_backtest_manifest(manifest)
    counts = mapping(manifest.get("record_counts"))
    observed: dict[str, int] = {}
    captured: dict[str, list[PrimitiveMapping]] = {}
    tables = {
        "signals": "signals.csv",
        "orders": "orders.csv",
        "fills": "fills.csv",
        "positions": "positions.csv",
        "daily_equity": "equity.csv",
        "benchmark_daily_equity": "benchmark_equity.csv",
        "dividend_cashflows": "dividend_cashflows.csv",
        "split_adjustments": "split_adjustments.csv",
        "benchmark_dividend_cashflows": "benchmark_dividend_cashflows.csv",
        "benchmark_split_adjustments": "benchmark_split_adjustments.csv",
        "trades": "trades.csv",
    }
    for name, filename in tables.items():
        path = export / filename
        try:
            content = path.read_bytes()
            reads.expect(path, content)
            rows = csv.reader(
                StringIO(content.decode("utf-8"), newline=""), strict=True
            )
            header = next(rows, list[str]())
            if (
                not header
                or len(set(header)) != len(header)
                or any(not field for field in header)
            ):
                raise ManifestError(f"backtest table {filename} has an invalid header")
            observed[name] = 0
            validate_csv_header(name, header)
            captured[name] = []
            if name == "trades":
                if "is_open" not in header:
                    raise ManifestError("backtest trades table is missing is_open")
                observed["open_trades"] = observed["completed_trades"] = 0
            for row in rows:
                if len(row) != len(header):
                    raise ManifestError(
                        f"backtest table {filename} has a malformed row"
                    )
                observed[name] += 1
                captured[name].append(
                    decode_csv_row(name, dict(zip(header, row, strict=True)))
                )
                if name == "trades":
                    is_open = row[header.index("is_open")]
                    if is_open not in {"True", "False"}:
                        raise ManifestError("backtest trade is_open must be boolean")
                    observed[
                        "open_trades" if is_open == "True" else "completed_trades"
                    ] += 1
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            raise ManifestError(f"cannot read backtest table {filename}") from error
    del observed["trades"]
    if set(counts) != set(observed) or any(
        type(counts[name]) is not int or counts[name] != count
        for name, count in observed.items()
    ):
        raise ManifestError("backtest record counts differ from captured tables")
    validate_backtest_records(manifest, captured)
    from quantforge.experiments._backtest_ledger_integrity import (
        validate_backtest_ledgers,
    )

    validate_backtest_ledgers(manifest, captured)
