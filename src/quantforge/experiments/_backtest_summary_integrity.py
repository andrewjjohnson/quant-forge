"""Validate saved QF-40 backtest summary schemas without calculating metrics."""

from quantforge.experiments._aggregate_schema import (
    counter,
    decimal_field,
    record,
    strings,
    validate_completeness,
)
from quantforge.experiments._json import ManifestError

_COUNTS = {
    "trade_count",
    "open_trade_count",
    "winning_trades",
    "losing_trades",
    "oos_session_count",
}
_DECIMALS: dict[str, tuple[int | None, int | None]] = {
    "ending_index": (0, None),
    "total_return": (-1, None),
    "maximum_drawdown": (-1, 0),
    "benchmark_total_return": (-1, None),
    "benchmark_maximum_drawdown": (-1, 0),
    "win_rate": (0, 1),
    "profit_factor": (0, None),
    "gross_profit": (0, None),
    "gross_loss": (None, 0),
    "profitable_window_fraction": (0, 1),
    "exposure": (0, 1),
    "native_commissions": (0, None),
    "native_fees": (0, None),
    "native_slippage_cost": (0, None),
    "native_dividend_income": (None, None),
}


def validate_backtest_aggregate_summary(value: object) -> None:
    summary = record(
        value,
        _COUNTS
        | _DECIMALS.keys()
        | {
            "completeness",
            "windows",
            "stitching",
            "starting_index",
            "profitable_window_denominator",
            "warnings",
        },
        "backtest summary",
    )
    for field in _COUNTS:
        counter(summary[field])
    completeness = validate_completeness(summary["completeness"])
    if (
        summary["stitching"] != "reset_account_relative_equity_chain_v1"
        or summary["starting_index"] != "1"
        or summary["profitable_window_denominator"]
        != "completed_windows_only; see completeness"
    ):
        raise ManifestError("OOS aggregate backtest summary contract is unsupported")
    unavailable = {
        **dict.fromkeys(
            (
                "ending_index",
                "total_return",
                "maximum_drawdown",
                "benchmark_total_return",
                "benchmark_maximum_drawdown",
                "profitable_window_fraction",
            ),
            completeness["completed_windows"] == 0,
        ),
        "win_rate": summary["trade_count"] == 0,
        "exposure": summary["oos_session_count"] == 0,
    }
    for field, (minimum, maximum) in _DECIMALS.items():
        if unavailable.get(field, False) and summary[field] is not None:
            raise ManifestError(
                "OOS aggregate empty backtest metric must be unavailable"
            )
        decimal_field(
            summary[field],
            nullable=unavailable.get(field, False) or field == "profit_factor",
            minimum=minimum,
            maximum=maximum,
        )
    strings(summary["warnings"])
