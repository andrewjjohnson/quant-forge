"""Decode the producer's flat CSV records into validated primitive snapshots."""

from dataclasses import fields
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import cast, get_args, get_origin

from quantforge.backtesting.models import (
    DailyPortfolioRecord,
    DividendCashflowRecord,
    FillRecord,
    OrderRecord,
    PositionRecord,
    SplitAdjustmentRecord,
    TradeRecord,
)
from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments._json import ManifestError, mapping, parse_json, text
from quantforge.experiments._prediction_sessions import session_text

MODELS = {
    "orders": OrderRecord,
    "fills": FillRecord,
    "trades": TradeRecord,
    "positions": PositionRecord,
    "daily_equity": DailyPortfolioRecord,
    "benchmark_daily_equity": DailyPortfolioRecord,
    "dividend_cashflows": DividendCashflowRecord,
    "benchmark_dividend_cashflows": DividendCashflowRecord,
    "split_adjustments": SplitAdjustmentRecord,
    "benchmark_split_adjustments": SplitAdjustmentRecord,
}
SIGNAL_FIELDS = (
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
)


def validate_csv_header(name: str, header: list[str]) -> None:
    expected = (
        set(SIGNAL_FIELDS)
        if name == "signals"
        else {field.name for field in fields(MODELS[name])}
    )
    if len(header) != len(expected) or set(header) != expected:
        raise ManifestError(f"backtest {name} header differs from producer schema")


def _decimal(raw: str) -> str:
    try:
        if not Decimal(raw).is_finite():
            raise ValueError
    except (InvalidOperation, ValueError) as error:
        raise ManifestError("backtest CSV decimal must be finite") from error
    return raw


def decode_csv_row(name: str, raw: dict[str, str]) -> PrimitiveMapping:
    """Retain decimals as recorded strings; never reconstruct execution objects."""
    if name == "signals":
        row: PrimitiveMapping = dict(raw)
        for key in SIGNAL_FIELDS:
            if key not in {"earliest_executable_session", "reason"}:
                text(raw[key])
        for key in ("strategy_parameters", "indicator_values"):
            row[key] = parse_json(raw[key].encode("utf-8"))
        parameters = mapping(row["strategy_parameters"])
        if any(isinstance(value, (dict, list)) for value in parameters.values()):
            raise ManifestError("backtest signal parameters must be scalars")
        for value in mapping(row["indicator_values"]).values():
            _decimal(text(value))
        session_text(raw["signal_session"])
        row["earliest_executable_session"] = (
            session_text(raw["earliest_executable_session"])
            if raw["earliest_executable_session"]
            else None
        )
        _decimal(raw["target_weight"])
        # The signal identity disambiguates CSV's shared empty/None encoding.
        row["reason"] = raw["reason"] or None
        return row
    result: PrimitiveMapping = {}
    for field in fields(MODELS[name]):
        value = raw[field.name]
        types = (
            get_args(field.type)
            if get_origin(field.type) is not tuple
            else (field.type,)
        )
        types = types or (field.type,)
        if not value and type(None) in types:
            result[field.name] = None
        elif Decimal in types:
            result[field.name] = _decimal(value)
            nonnegative = {
                "cash",
                "cost_basis",
                "market_value",
                "total_equity",
                "running_equity_peak",
                "exposure_weight",
                "average_entry_cost",
                "average_entry_cost_before",
                "average_entry_cost_after",
                "total_cost_basis_before",
                "total_cost_basis_after",
                "resulting_cash_balance",
                "amount_per_share",
                "total_dividend_cash",
                "dividend_income",
                "commission",
                "fees",
                "entry_commission",
                "entry_fees",
                "exit_commission",
                "exit_fees",
                "slippage_basis_points",
            }
            positive = {
                "reference_price",
                "fill_price",
                "entry_price",
                "exit_price",
                "closing_mark_price",
                "gross_notional",
                "split_factor",
            }
            number = Decimal(value)
            if (
                (field.name in nonnegative and number < 0)
                or (field.name in positive and number <= 0)
                or (field.name == "exposure_weight" and number > 1)
                or (field.name == "drawdown" and not -1 <= number <= 0)
            ):
                raise ManifestError("backtest CSV decimal is outside its domain")
        elif int in types:
            if not value.isascii() or not value.isdecimal() or str(int(value)) != value:
                raise ManifestError("backtest CSV integer is invalid")
            if field.name in {"split_ratio_numerator", "split_ratio_denominator"} and (
                int(value) == 0
            ):
                raise ManifestError("backtest CSV split ratio must be positive")
            result[field.name] = int(value)
        elif bool in types:
            if value not in {"True", "False"}:
                raise ManifestError("backtest CSV boolean is invalid")
            result[field.name] = value == "True"
        elif date in types:
            result[field.name] = session_text(value)
        elif str in types:
            result[field.name] = text(value)
        elif get_origin(field.type) is tuple:
            parsed = parse_json(('{"items":' + value + "}").encode("utf-8"))["items"]
            if not isinstance(parsed, list) or any(
                not isinstance(item, str) or not item for item in parsed
            ):
                raise ManifestError("backtest CSV references must be string arrays")
            result[field.name] = cast(list[Primitive], parsed)
        else:
            enum = next(
                (
                    kind
                    for kind in types
                    if isinstance(kind, type) and issubclass(kind, StrEnum)
                ),
                None,
            )
            if enum is None or value not in tuple(enum):
                raise ManifestError("backtest CSV enum value is invalid")
            result[field.name] = value
    return result
