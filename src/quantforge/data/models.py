"""Typed canonical daily-market-data records."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

SCHEMA_VERSION = "2"
type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
type ProviderRecord = Mapping[str, JsonValue]


class AdjustmentMode(StrEnum):
    """A single, explicit price and volume adjustment basis."""

    UNADJUSTED = "unadjusted"
    SPLIT_ADJUSTED = "split_adjusted"
    SPLIT_AND_DIVIDEND_ADJUSTED = "split_and_dividend_adjusted"


@dataclass(frozen=True, slots=True)
class DailyBar:
    """One completed exchange trading session, represented without a timezone."""

    symbol: str
    session_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Lossless JSON-compatible provider result and its provenance."""

    provider_name: str
    provider_symbol: str
    retrieved_at: datetime
    provider_timezone: str | None
    adjustment_mode: AdjustmentMode
    records: tuple[ProviderRecord, ...]
    metadata: dict[str, JsonValue]
    adapter_version: str


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    canonical_symbol: str
    provider_name: str
    provider_symbol: str
    retrieved_at: datetime
    requested_start: date
    requested_end: date
    actual_first_session: date
    actual_last_session: date
    calendar: str
    provider_timezone: str | None
    adjustment_mode: AdjustmentMode
    raw_location: str
    normalized_location: str
    dataset_id: str
    schema_version: str
    bar_count: int
    missing_sessions: tuple[date, ...]
    split_sessions: tuple[date, ...]
    adapter_version: str


@dataclass(frozen=True, slots=True)
class MarketDataset:
    """Canonical bars permanently associated with their immutable manifest."""

    bars: tuple[DailyBar, ...]
    metadata: DatasetMetadata
