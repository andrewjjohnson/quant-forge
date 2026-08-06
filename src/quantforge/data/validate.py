"""Deterministic canonical-bar validation."""

from datetime import date, datetime
from decimal import Decimal
from typing import cast

from quantforge.data.calendar import expected_sessions
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import dataset_identity_matches
from quantforge.data.models import (
    SCHEMA_VERSION,
    AdjustmentMode,
    CashDividend,
    DailyBar,
    DatasetMetadata,
    MarketDataset,
    StockSplit,
)


def _calendar_sessions(start: date, end: date, calendar: str) -> tuple[date, ...]:
    try:
        return expected_sessions(start, end, calendar)
    except Exception as error:
        raise ValidationError(
            f"cannot resolve expected sessions for calendar: {calendar}"
        ) from error


def validate_bars(
    bars: tuple[DailyBar, ...],
    symbol: str,
    start: date,
    end: date,
    calendar: str,
    *,
    strict: bool = True,
) -> tuple[DailyBar, ...]:
    """Stable-sort bars, reject invalid values, and optionally require every session."""
    if start > end:
        raise ValidationError("start must be on or before end")
    if not bars:
        raise ValidationError("provider returned no bars")
    ordered = tuple(sorted(bars, key=lambda bar: bar.session_date))
    sessions = [bar.session_date for bar in ordered]
    if len(sessions) != len(set(sessions)):
        raise ValidationError("duplicate trading sessions")
    for bar in ordered:
        if bar.symbol != symbol:
            raise ValidationError("inconsistent symbols")
        if not start <= bar.session_date <= end:
            raise ValidationError("session outside requested range")
        values = cast(
            tuple[object, ...],
            (bar.open, bar.high, bar.low, bar.close, bar.volume),
        )
        if any(not isinstance(value, Decimal) for value in values):
            raise ValidationError("OHLCV values must be Decimal instances")
        decimal_values = cast(tuple[Decimal, ...], values)
        if any(
            not value.is_finite() or value <= Decimal(0) for value in decimal_values
        ):
            raise ValidationError("OHLCV values must be positive and finite")
        if bar.high < max(bar.open, bar.low, bar.close):
            raise ValidationError("high is below another OHLC price")
        if bar.low > min(bar.open, bar.high, bar.close):
            raise ValidationError("low is above another OHLC price")
    expected = set(_calendar_sessions(start, end, calendar))
    unexpected = tuple(sorted(set(sessions) - expected))
    if unexpected:
        rendered = ", ".join(item.isoformat() for item in unexpected)
        raise ValidationError(f"non-session dates for {calendar}: {rendered}")
    missing = tuple(sorted(expected - set(sessions)))
    if strict and missing:
        rendered = tuple(item.isoformat() for item in missing)
        raise ValidationError(
            f"missing expected sessions: {', '.join(rendered)}",
            missing_sessions=rendered,
        )
    return ordered


def validate_market_dataset(dataset: MarketDataset) -> tuple[date, ...]:
    """Validate a complete QF-3 dataset and return recomputed missing sessions.

    This verifies every invariant derivable from the in-memory bars and metadata.
    Raw-provider bytes remain an immutable-cache responsibility because they are
    not carried by ``MarketDataset``.
    """
    dataset_value = cast(object, dataset)
    if not isinstance(dataset_value, MarketDataset):
        raise ValidationError("a QF-3 MarketDataset is required")
    bars_value = cast(object, dataset_value.bars)
    metadata_value = cast(object, dataset_value.metadata)
    if not isinstance(bars_value, tuple) or not bars_value:
        raise ValidationError("market bars must be a nonempty immutable tuple")
    if not isinstance(metadata_value, DatasetMetadata):
        raise ValidationError("canonical dataset metadata is required")
    metadata = metadata_value
    string_fields: tuple[tuple[str, object], ...] = (
        ("canonical symbol", cast(object, metadata.canonical_symbol)),
        ("provider name", cast(object, metadata.provider_name)),
        ("provider symbol", cast(object, metadata.provider_symbol)),
        ("calendar", cast(object, metadata.calendar)),
        ("adapter version", cast(object, metadata.adapter_version)),
        ("dataset ID", cast(object, metadata.dataset_id)),
        ("schema version", cast(object, metadata.schema_version)),
    )
    for field_name, field_value in string_fields:
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValidationError(f"{field_name} must be a nonempty string")
    if metadata.schema_version != SCHEMA_VERSION:
        raise ValidationError(f"market data schema {SCHEMA_VERSION} is required")
    adjustment_mode = cast(object, metadata.adjustment_mode)
    if not isinstance(adjustment_mode, AdjustmentMode):
        raise ValidationError("dataset adjustment mode is invalid")
    retrieved_at = cast(object, metadata.retrieved_at)
    if not isinstance(retrieved_at, datetime) or retrieved_at.utcoffset() is None:
        raise ValidationError("dataset retrieval timestamp must define a UTC offset")
    date_fields: tuple[tuple[str, object], ...] = (
        ("requested start", cast(object, metadata.requested_start)),
        ("requested end", cast(object, metadata.requested_end)),
        ("actual first session", cast(object, metadata.actual_first_session)),
        ("actual last session", cast(object, metadata.actual_last_session)),
    )
    for field_name, field_value in date_fields:
        if not isinstance(field_value, date):
            raise ValidationError(f"{field_name} must be a date")
    if metadata.requested_start > metadata.requested_end:
        raise ValidationError("requested start must be on or before requested end")
    bar_count = cast(object, metadata.bar_count)
    if isinstance(bar_count, bool) or not isinstance(bar_count, int) or bar_count <= 0:
        raise ValidationError("dataset bar count must be a positive integer")

    typed_bars: list[DailyBar] = []
    for bar_value in cast(tuple[object, ...], bars_value):
        if not isinstance(bar_value, DailyBar):
            raise ValidationError("dataset contains a noncanonical daily bar")
        if not isinstance(cast(object, bar_value.session_date), date):
            raise ValidationError("bar session must be a date")
        typed_bars.append(bar_value)
    bars = tuple(typed_bars)
    ordered = validate_bars(
        bars,
        metadata.canonical_symbol,
        metadata.requested_start,
        metadata.requested_end,
        metadata.calendar,
        strict=False,
    )
    if ordered != bars:
        raise ValidationError("market sessions must be strictly chronological")
    if metadata.bar_count != len(bars):
        raise ValidationError("dataset bar count does not match metadata")
    if (
        metadata.actual_first_session != bars[0].session_date
        or metadata.actual_last_session != bars[-1].session_date
    ):
        raise ValidationError("dataset session bounds do not match metadata")

    expected = _calendar_sessions(
        metadata.requested_start,
        metadata.requested_end,
        metadata.calendar,
    )
    actual_sessions = {bar.session_date for bar in bars}
    recomputed_missing = tuple(
        session for session in expected if session not in actual_sessions
    )
    missing_value = cast(object, metadata.missing_sessions)
    if not isinstance(missing_value, tuple) or any(
        not isinstance(item, date) for item in cast(tuple[object, ...], missing_value)
    ):
        raise ValidationError(
            "missing-session provenance must be an immutable date tuple"
        )
    if metadata.missing_sessions != recomputed_missing:
        computed = ", ".join(item.isoformat() for item in recomputed_missing) or "none"
        declared = (
            ", ".join(item.isoformat() for item in metadata.missing_sessions) or "none"
        )
        raise ValidationError(
            "dataset missing-session provenance does not match the calendar; "
            f"computed: {computed}; declared: {declared}"
        )

    for field_name, sessions in (
        ("split-session", metadata.split_sessions),
        ("dividend-session", metadata.dividend_sessions),
    ):
        sessions_value = cast(object, sessions)
        if not isinstance(sessions_value, tuple) or any(
            not isinstance(item, date)
            for item in cast(tuple[object, ...], sessions_value)
        ):
            raise ValidationError(
                f"{field_name} provenance must be an immutable date tuple"
            )
        if sessions != tuple(sorted(set(sessions))):
            raise ValidationError(
                f"{field_name} provenance must be unique and chronological"
            )
        if any(session not in actual_sessions for session in sessions):
            raise ValidationError(
                f"{field_name} provenance must reference an observed bar"
            )

    actions_value = cast(object, dataset_value.corporate_actions)
    if not isinstance(actions_value, tuple):
        raise ValidationError("corporate actions must be an immutable tuple")
    corporate_actions = cast(tuple[object, ...], actions_value)
    if not isinstance(cast(object, metadata.corporate_actions_complete), bool):
        raise ValidationError("corporate-action completeness must be explicit")
    count_fields = (
        ("corporate-action count", metadata.corporate_action_count),
        ("dividend count", metadata.dividend_count),
        ("split count", metadata.split_count),
    )
    for field_name, field_value in count_fields:
        if (
            isinstance(cast(object, field_value), bool)
            or not isinstance(cast(object, field_value), int)
            or field_value < 0
        ):
            raise ValidationError(f"{field_name} must be a nonnegative integer")
    dividends: list[CashDividend] = []
    splits: list[StockSplit] = []
    action_keys: list[tuple[str, date]] = []
    for action in corporate_actions:
        if isinstance(action, CashDividend):
            if not action.amount_per_share.is_finite() or action.amount_per_share <= 0:
                raise ValidationError("cash-dividend amount must be positive")
            action_session = action.ex_dividend_session
            action_key = ("cash_dividend", action_session)
            dividends.append(action)
        elif isinstance(action, StockSplit):
            if (
                not action.split_factor.is_finite()
                or action.split_factor <= 0
                or action.split_factor == Decimal(1)
            ):
                raise ValidationError(
                    "stock-split factor must be positive and nonneutral"
                )
            action_session = action.effective_session
            action_key = ("stock_split", action_session)
            splits.append(action)
        else:
            raise ValidationError("dataset contains an unsupported corporate action")
        if (
            not action.action_id
            or action.symbol != metadata.canonical_symbol
            or action.provider_name != metadata.provider_name
            or action.source_dataset_id != metadata.dataset_id
            or action_session not in actual_sessions
        ):
            raise ValidationError("corporate-action provenance is inconsistent")
        action_keys.append(action_key)
    if action_keys != sorted(action_keys, key=lambda item: (item[1], item[0])):
        raise ValidationError("corporate actions must be chronological")
    if len(action_keys) != len(set(action_keys)):
        raise ValidationError("duplicate corporate actions")
    if (
        metadata.corporate_action_count != len(corporate_actions)
        or metadata.dividend_count != len(dividends)
        or metadata.split_count != len(splits)
        or metadata.dividend_sessions
        != tuple(action.ex_dividend_session for action in dividends)
        or metadata.split_sessions
        != tuple(action.effective_session for action in splits)
    ):
        raise ValidationError(
            "corporate-action counts or sessions do not match records"
        )
    if not metadata.corporate_action_snapshot_id:
        raise ValidationError("corporate-action snapshot identity is required")
    if metadata.corporate_action_policy != (
        "separate_provider_reported_cash_dividends_and_splits"
    ):
        raise ValidationError("unsupported corporate-action dataset policy")
    if not isinstance(cast(object, metadata.adjusted_fields_used), bool):
        raise ValidationError("adjusted-field usage must be explicit")
    expected_basis = (
        "raw_provider"
        if metadata.adjustment_mode is AdjustmentMode.UNADJUSTED
        else "split_adjusted"
    )
    if metadata.ohlc_basis != expected_basis or metadata.volume_basis != expected_basis:
        raise ValidationError("dataset price or volume basis is inconsistent")

    if not dataset_identity_matches(dataset_value):
        raise ValidationError(
            "market bars or provenance do not match the QF-3 dataset identity"
        )
    return recomputed_missing
