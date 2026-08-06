"""Typed canonical daily-market-data records."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

SCHEMA_VERSION = "4"
type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
type ProviderRecord = Mapping[str, JsonValue]


class AdjustmentMode(StrEnum):
    """A single, explicit price and volume adjustment basis."""

    UNADJUSTED = "unadjusted"
    SPLIT_ADJUSTED = "split_adjusted"
    SPLIT_AND_DIVIDEND_ADJUSTED = "split_and_dividend_adjusted"


class CorporateActionType(StrEnum):
    """Supported immutable corporate-action record types."""

    CASH_DIVIDEND = "cash_dividend"
    STOCK_SPLIT = "stock_split"


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
class CashDividend:
    """Provider-reported cash amount per share on an ex-dividend session."""

    action_id: str
    symbol: str
    ex_dividend_session: date
    amount_per_share: Decimal
    provider_name: str
    source_dataset_id: str

    @property
    def action_type(self) -> CorporateActionType:
        return CorporateActionType.CASH_DIVIDEND

    def to_primitive(self) -> dict[str, JsonValue]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "symbol": self.symbol,
            "ex_dividend_session": self.ex_dividend_session.isoformat(),
            "amount_per_share": str(self.amount_per_share),
            "provider_name": self.provider_name,
            "source_dataset_id": self.source_dataset_id,
        }


@dataclass(frozen=True, slots=True)
class StockSplit:
    """Provider-reported shares-after/shares-before factor on its effective session."""

    action_id: str
    symbol: str
    effective_session: date
    split_factor: Decimal
    provider_name: str
    source_dataset_id: str

    @property
    def action_type(self) -> CorporateActionType:
        return CorporateActionType.STOCK_SPLIT

    def to_primitive(self) -> dict[str, JsonValue]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "symbol": self.symbol,
            "effective_session": self.effective_session.isoformat(),
            "split_factor": str(self.split_factor),
            "provider_name": self.provider_name,
            "source_dataset_id": self.source_dataset_id,
        }


type CorporateAction = CashDividend | StockSplit


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
    corporate_actions_location: str
    raw_sha256: str
    data_sha256: str
    dataset_id: str
    schema_version: str
    bar_count: int
    missing_sessions: tuple[date, ...]
    split_sessions: tuple[date, ...]
    dividend_sessions: tuple[date, ...]
    corporate_actions_complete: bool
    corporate_action_count: int
    dividend_count: int
    split_count: int
    corporate_action_snapshot_id: str
    ohlc_basis: str
    volume_basis: str
    adjusted_fields_used: bool
    corporate_action_policy: str
    adapter_version: str


@dataclass(frozen=True, slots=True)
class MarketDataset:
    """Canonical bars permanently associated with their immutable manifest."""

    bars: tuple[DailyBar, ...]
    metadata: DatasetMetadata
    corporate_actions: tuple[CorporateAction, ...] = ()
