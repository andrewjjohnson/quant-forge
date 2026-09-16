"""Validate QF-5's saved manifest schema without rebuilding a backtest result."""

from dataclasses import fields
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import get_args

from quantforge.backtesting.models import (
    DividendAccountingSummary,
    DividendCashflowRecord,
    FillRecord,
    MarketDataMetadata,
    OrderRecord,
    PerformanceSummary,
    ReturnBasis,
    SplitAdjustmentRecord,
)
from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._aggregate_schema import (
    counter,
    decimal_field,
    record,
    records,
    strings,
)
from quantforge.experiments._json import ManifestError, digest, mapping, text
from quantforge.experiments._prediction_sessions import session_text

_RECORD_COUNTS = {
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
}
_PERFORMANCE_COUNTS = {
    "trade_count",
    "open_trade_count",
    "winning_trades",
    "losing_trades",
    "annualization_factor",
    "dividend_event_count",
    "split_event_count",
}
_NULLABLE_METRICS = {
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "profit_factor",
    "win_rate",
    "average_trade_return",
    "benchmark_total_return",
}


def _timestamp(value: object) -> None:
    try:
        if datetime.fromisoformat(text(value)).utcoffset() is None:
            raise ValueError
    except ValueError as error:
        raise ManifestError(
            "backtest timestamp must be an aware ISO datetime"
        ) from error


def _manifest_row(
    value: object,
    model: type[
        OrderRecord | FillRecord | DividendCashflowRecord | SplitAdjustmentRecord
    ],
) -> None:
    """These four flat producer records serialize one primitive per dataclass field."""
    row = record(
        value, {field.name for field in fields(model)}, "backtest manifest row"
    )
    for field in fields(model):
        item = row[field.name]
        types = get_args(field.type) or (field.type,)
        if item is None and type(None) in types:
            continue
        if Decimal in types:
            decimal_field(item)
        elif int in types:
            counter(item)
        elif date in types:
            session_text(item)
        elif str in types:
            text(item)
        else:
            enum = next(
                (
                    kind
                    for kind in types
                    if isinstance(kind, type) and issubclass(kind, StrEnum)
                ),
                None,
            )
            if enum is None or item not in tuple(enum):
                raise ManifestError("backtest manifest row has an unsupported domain")


def _performance(value: object) -> None:
    performance = record(
        value,
        {field.name for field in fields(PerformanceSummary)},
        "backtest performance",
    )
    literals = {
        "volatility_standard_deviation": "sample",
        "maximum_drawdown_convention": "negative_decimal",
    }
    for name, metric in performance.items():
        if name in _PERFORMANCE_COUNTS:
            if counter(metric) == 0 and name == "annualization_factor":
                raise ManifestError("backtest annualization factor must be positive")
        elif name in literals:
            if metric != literals[name]:
                raise ManifestError("backtest performance convention is unsupported")
        else:
            minimum = (
                0
                if name
                in {
                    "annualized_volatility",
                    "profit_factor",
                    "exposure",
                    "win_rate",
                    "gross_profit",
                }
                else None
            )
            maximum = (
                1
                if name in {"exposure", "win_rate"}
                else 0
                if name in {"maximum_drawdown", "gross_loss"}
                else None
            )
            decimal_field(
                metric,
                nullable=name in _NULLABLE_METRICS,
                minimum=minimum,
                maximum=maximum,
            )


def _market(value: object) -> None:
    market = record(
        value,
        {field.name for field in fields(MarketDataMetadata)} | {"raw_snapshot_id"},
        "backtest market data",
    )
    counts = {"bar_count", "corporate_action_count", "dividend_count", "split_count"}
    dates = {
        "requested_start",
        "requested_end",
        "actual_first_session",
        "actual_last_session",
    }
    dates_arrays = {"missing_sessions", "split_sessions", "dividend_sessions"}
    hashes = {
        "bars_fingerprint",
        "raw_snapshot_id",
        "raw_sha256",
        "data_sha256",
        "corporate_action_snapshot_id",
    }
    for name, item in market.items():
        if name in counts:
            counter(item)
        elif name in dates:
            session_text(item)
        elif name in dates_arrays:
            if not isinstance(item, list):
                raise ManifestError("backtest market sessions must be arrays")
            for label in item:
                session_text(label)
        elif name in hashes:
            digest(item)
        elif name == "retrieved_at":
            _timestamp(item)
        elif name in {"corporate_actions_complete", "adjusted_fields_used"}:
            if type(item) is not bool:
                raise ManifestError("backtest market flags must be boolean")
        elif name != "provider_timezone" or item is not None:
            text(item)
    if market["raw_snapshot_id"] != market["raw_sha256"]:
        raise ManifestError("backtest raw snapshot identity is inconsistent")


def _dividends(value: object) -> None:
    dividends = record(
        value,
        {field.name for field in fields(DividendAccountingSummary)},
        "backtest dividends",
    )
    from quantforge.backtesting.config import DividendPolicy

    if dividends["dividend_policy"] not in tuple(DividendPolicy) or dividends[
        "return_basis"
    ] not in tuple(ReturnBasis):
        raise ManifestError("backtest dividend policy or return basis is unsupported")
    digest(dividends["corporate_action_snapshot_id"])
    for name in (
        "dividend_events_present",
        "dividend_events_credited",
        "dividend_events_ignored",
    ):
        counter(dividends[name])
    for name in ("total_dividend_cash_credited", "estimated_ignored_dividend_cash"):
        decimal_field(dividends[name])
    if dividends["warning"] is not None:
        text(dividends["warning"])


def validate_backtest_manifest(manifest: PrimitiveMapping) -> None:
    """Require all producer result fields, including those outside run identity."""
    record(
        manifest,
        {
            "run_id",
            "engine_version",
            "result_schema_version",
            "initiated_at",
            "market_data",
            "strategy",
            "backtest_configuration",
            "performance",
            "benchmark",
            "corporate_action_accounting",
            "record_counts",
            "warnings",
            "limitations",
        },
        "backtest manifest",
    )
    if manifest["initiated_at"] is not None:
        _timestamp(manifest["initiated_at"])
    _market(manifest["market_data"])
    strategy = record(
        manifest["strategy"],
        {
            "strategy_id",
            "strategy_implementation_version",
            "strategy_configuration_id",
            "configuration",
            "warm_up_observations",
        },
        "backtest strategy",
    )
    if counter(strategy["warm_up_observations"]) == 0:
        raise ManifestError("backtest strategy warm-up must be positive")
    for name in ("strategy_id", "strategy_implementation_version"):
        text(strategy[name])
    _performance(manifest["performance"])
    for count in record(
        manifest["record_counts"], _RECORD_COUNTS, "backtest record counts"
    ).values():
        counter(count)
    performance = mapping(manifest["performance"])
    counts = mapping(manifest["record_counts"])
    completed = counter(counts["completed_trades"])
    if (
        performance["trade_count"] != completed
        or performance["open_trade_count"] != counts["open_trades"]
        or counter(performance["winning_trades"])
        + counter(performance["losing_trades"])
        > completed
    ):
        raise ManifestError(
            "backtest performance trade counts differ from record counts"
        )
    strings(manifest["warnings"])
    strings(manifest["limitations"])
    benchmark = record(
        manifest["benchmark"],
        {
            "benchmark_id",
            "configuration",
            "order",
            "fill",
            "performance",
            "dividend_accounting",
            "dividend_cashflows",
            "split_adjustments",
        },
        "backtest benchmark",
    )
    _manifest_row(benchmark["order"], OrderRecord)
    if benchmark["fill"] is not None:
        _manifest_row(benchmark["fill"], FillRecord)
    for name, model in (
        ("dividend_cashflows", DividendCashflowRecord),
        ("split_adjustments", SplitAdjustmentRecord),
    ):
        for row in records(benchmark[name]):
            _manifest_row(row, model)
    _performance(benchmark["performance"])
    _dividends(benchmark["dividend_accounting"])
    corporate = record(
        manifest["corporate_action_accounting"],
        {
            "corporate_action_snapshot_id",
            "dividends",
            "splits",
        },
        "backtest corporate actions",
    )
    if corporate["corporate_action_snapshot_id"] != mapping(manifest["market_data"])[
        "corporate_action_snapshot_id"
    ] or corporate["splits"] != mapping(manifest["backtest_configuration"]).get(
        "split_policy"
    ):
        raise ManifestError("backtest corporate-action declarations are inconsistent")
    _dividends(corporate["dividends"])
