"""Stable immutable structured backtest result export."""

import csv
import hmac
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import cast

from quantforge.backtesting.errors import ResultExportError
from quantforge.backtesting.models import BacktestResult
from quantforge.configuration import Primitive, PrimitiveMapping

BACKTEST_ARTIFACT_FILENAMES = (
    "manifest.json",
    "signals.csv",
    "orders.csv",
    "fills.csv",
    "positions.csv",
    "trades.csv",
    "equity.csv",
    "benchmark_equity.csv",
    "dividend_cashflows.csv",
    "split_adjustments.csv",
    "benchmark_dividend_cashflows.csv",
    "benchmark_split_adjustments.csv",
)

_BACKTEST_PAYLOAD_FILENAMES = tuple(
    filename for filename in BACKTEST_ARTIFACT_FILENAMES if filename != "manifest.json"
)

_FILL_CSV_FIELDS = (
    "fill_id",
    "order_id",
    "originating_signal_id",
    "symbol",
    "side",
    "quantity",
    "execution_session",
    "reference_price",
    "fill_price",
    "slippage_per_share",
    "slippage_basis_points",
    "gross_notional",
    "commission",
    "fees",
    "net_cash_effect",
    "strategy_id",
    "strategy_configuration_id",
)

_DIVIDEND_CASHFLOW_FIELDS = (
    "dividend_cashflow_id",
    "run_id",
    "account_id",
    "corporate_action_id",
    "symbol",
    "ex_dividend_session",
    "entitled_share_quantity",
    "amount_per_share",
    "total_dividend_cash",
    "resulting_cash_balance",
    "source_dataset_id",
)

_SPLIT_ADJUSTMENT_FIELDS = (
    "split_adjustment_id",
    "run_id",
    "account_id",
    "corporate_action_id",
    "symbol",
    "effective_session",
    "split_factor",
    "shares_before",
    "shares_after",
    "average_entry_cost_before",
    "average_entry_cost_after",
    "total_cost_basis_before",
    "total_cost_basis_after",
    "resulting_cash_balance",
    "source_dataset_id",
)


def _json_text(value: Primitive) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _csv_value(value: Primitive) -> str | int | float | bool | None:
    if isinstance(value, (dict, list)):
        return _json_text(value)
    return value


def _write_csv(
    path: Path, rows: Sequence[PrimitiveMapping], fieldnames: tuple[str, ...]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
        stream.flush()
        os.fsync(stream.fileno())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_backtest_result(result: BacktestResult, output_root: Path) -> Path:
    """Atomically create ``output_root/run_id`` without overwriting a prior run."""
    destination = output_root / result.run_id
    if destination.exists():
        raise ResultExportError(f"backtest run already exists: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{result.run_id}.", dir=str(output_root))
    )
    try:
        signals = [item.to_primitive() for item in result.signals]
        orders = [item.to_primitive() for item in result.orders]
        fills = [item.to_primitive() for item in result.fills]
        trades = [
            item.to_primitive()
            for item in (*result.completed_trades, *result.open_trades)
        ]
        positions = [item.to_primitive() for item in result.positions]
        equity = [item.to_primitive() for item in result.daily_equity]
        benchmark_equity = [
            item.to_primitive() for item in result.benchmark.daily_equity
        ]
        dividend_cashflows = [item.to_primitive() for item in result.dividend_cashflows]
        split_adjustments = [item.to_primitive() for item in result.split_adjustments]
        benchmark_dividend_cashflows = [
            item.to_primitive() for item in result.benchmark.dividend_cashflows
        ]
        benchmark_split_adjustments = [
            item.to_primitive() for item in result.benchmark.split_adjustments
        ]
        _write_csv(
            temporary / "signals.csv",
            signals,
            tuple(signals[0])
            if signals
            else (
                "signal_id",
                "canonical_symbol",
                "signal_session",
                "earliest_executable_session",
                "execution_timing",
                "execution_session_status",
                "target_position",
                "target_weight",
                "strategy_id",
                "strategy_configuration_id",
                "strategy_parameters",
                "reason",
                "indicator_values",
            ),
        )
        _write_csv(
            temporary / "orders.csv",
            orders,
            tuple(orders[0])
            if orders
            else tuple(result.benchmark.order.to_primitive()),
        )
        _write_csv(
            temporary / "fills.csv",
            fills,
            _FILL_CSV_FIELDS,
        )
        _write_csv(
            temporary / "trades.csv",
            trades,
            tuple(trades[0])
            if trades
            else (
                "trade_id",
                "symbol",
                "entry_signal_id",
                "entry_order_id",
                "entry_fill_id",
                "entry_session",
                "entry_price",
                "entry_quantity",
                "entry_commission",
                "entry_fees",
                "exit_signal_id",
                "exit_order_id",
                "exit_fill_id",
                "exit_session",
                "exit_price",
                "exit_commission",
                "exit_fees",
                "gross_profit_loss",
                "net_profit_loss",
                "return_percentage",
                "holding_period_sessions",
                "strategy_id",
                "strategy_implementation_version",
                "strategy_configuration_id",
                "is_open",
                "exit_quantity",
                "dividend_income",
                "total_economic_profit_loss",
                "total_economic_return",
            ),
        )
        _write_csv(
            temporary / "positions.csv",
            positions,
            tuple(positions[0]),
        )
        _write_csv(temporary / "equity.csv", equity, tuple(equity[0]))
        _write_csv(
            temporary / "benchmark_equity.csv",
            benchmark_equity,
            tuple(benchmark_equity[0]),
        )
        _write_csv(
            temporary / "dividend_cashflows.csv",
            dividend_cashflows,
            tuple(dividend_cashflows[0])
            if dividend_cashflows
            else _DIVIDEND_CASHFLOW_FIELDS,
        )
        _write_csv(
            temporary / "split_adjustments.csv",
            split_adjustments,
            tuple(split_adjustments[0])
            if split_adjustments
            else _SPLIT_ADJUSTMENT_FIELDS,
        )
        _write_csv(
            temporary / "benchmark_dividend_cashflows.csv",
            benchmark_dividend_cashflows,
            tuple(benchmark_dividend_cashflows[0])
            if benchmark_dividend_cashflows
            else _DIVIDEND_CASHFLOW_FIELDS,
        )
        _write_csv(
            temporary / "benchmark_split_adjustments.csv",
            benchmark_split_adjustments,
            tuple(benchmark_split_adjustments[0])
            if benchmark_split_adjustments
            else _SPLIT_ADJUSTMENT_FIELDS,
        )
        manifest = result.manifest_primitive()
        manifest["artifact_integrity"] = {
            "algorithm": "sha256",
            "files": {
                filename: _file_sha256(temporary / filename)
                for filename in _BACKTEST_PAYLOAD_FILENAMES
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.rename(temporary, destination)
    except (OSError, TypeError, ValueError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise ResultExportError("failed to export immutable backtest result") from error
    return destination


def validate_backtest_result_artifact(path: Path) -> Path:
    """Verify the exact file set and manifest-bound SHA-256 payload digests."""
    try:
        if path.is_symlink() or not path.is_dir():
            raise ResultExportError("invalid immutable backtest artifact")
        entries = {entry.name: entry for entry in path.iterdir()}
        if set(entries) != set(BACKTEST_ARTIFACT_FILENAMES) or any(
            entry.is_symlink() or not entry.is_file() for entry in entries.values()
        ):
            raise ResultExportError("invalid immutable backtest artifact")
        manifest = load_backtest_manifest(entries["manifest.json"])
        integrity_value = cast(object, manifest.get("artifact_integrity"))
        if not isinstance(integrity_value, dict):
            raise ResultExportError("invalid backtest artifact integrity manifest")
        integrity = cast(dict[object, object], integrity_value)
        files_value = integrity.get("files")
        if integrity.get("algorithm") != "sha256" or not isinstance(files_value, dict):
            raise ResultExportError("invalid backtest artifact integrity manifest")
        files = cast(dict[object, object], files_value)
        if set(files) != set(_BACKTEST_PAYLOAD_FILENAMES) or any(
            not isinstance(filename, str) or not isinstance(digest, str)
            for filename, digest in files.items()
        ):
            raise ResultExportError("invalid backtest artifact integrity manifest")
        if manifest.get("run_id") != path.name or any(
            not hmac.compare_digest(
                cast(str, files[filename]), _file_sha256(entries[filename])
            )
            for filename in _BACKTEST_PAYLOAD_FILENAMES
        ):
            raise ResultExportError("backtest artifact integrity validation failed")
    except ResultExportError:
        raise
    except OSError as error:
        raise ResultExportError(
            "failed to validate immutable backtest artifact"
        ) from error
    return path


def validate_backtest_result_export(result: BacktestResult, path: Path) -> Path:
    """Require an existing export to exactly match this immutable result."""
    try:
        if path.name != result.run_id:
            raise ResultExportError(
                "backtest export does not match the expected immutable result"
            )
        try:
            validate_backtest_result_artifact(path)
        except ResultExportError as error:
            raise ResultExportError(
                "backtest export does not match the expected immutable result"
            ) from error
        entries = {entry.name: entry for entry in path.iterdir()}
        with tempfile.TemporaryDirectory(
            prefix="quantforge-export-validation-"
        ) as temporary_root:
            expected_path = export_backtest_result(result, Path(temporary_root))
            if any(
                entries[filename].read_bytes()
                != (expected_path / filename).read_bytes()
                for filename in BACKTEST_ARTIFACT_FILENAMES
            ):
                raise ResultExportError(
                    "backtest export does not match the expected immutable result"
                )
    except ResultExportError:
        raise
    except OSError as error:
        raise ResultExportError(
            "failed to validate immutable backtest export"
        ) from error
    return path


def load_backtest_manifest(path: Path) -> PrimitiveMapping:
    """Load a previously exported manifest and reject non-object JSON."""
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultExportError("failed to load backtest manifest") from error
    if not isinstance(loaded, dict):
        raise ResultExportError("backtest manifest must be a JSON object")
    loaded_mapping = cast(dict[object, object], loaded)
    if any(not isinstance(key, str) for key in loaded_mapping):
        raise ResultExportError("backtest manifest keys must be strings")
    return cast(PrimitiveMapping, loaded_mapping)
