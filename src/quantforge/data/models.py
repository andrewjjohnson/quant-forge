"""Typed canonical daily-market-data records."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import cast

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot

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


class CorporateActionAvailability(StrEnum):
    """Event supply is independent of whether source prices are adjusted."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class IntradayPredictionProvenance:
    """Canonical source lineage of a session projection used by predictions.

    The original intraday and session artifacts remain authoritative. This
    reference does not turn intraday observations into provider daily bars.
    """

    family_id: str
    source_dataset_id: str
    source_request_id: str
    source_raw_snapshot_ids: tuple[str, ...]
    session_dataset_id: str
    session_timeframe_configuration_id: str
    session_policy_id: str
    feed_scope_id: str
    corporate_action_availability: CorporateActionAvailability
    family_manifest: PrimitiveMappingSnapshot
    schema_version: str = "2"

    def __post_init__(self) -> None:
        if self.schema_version != "2":
            raise ValueError("unsupported intraday prediction provenance schema")
        if not isinstance(cast(object, self.family_manifest), PrimitiveMappingSnapshot):
            raise ValueError("intraday prediction family manifest must be immutable")
        if not isinstance(
            cast(object, self.corporate_action_availability),
            CorporateActionAvailability,
        ):
            raise ValueError("corporate-action availability must be explicit")
        for name in (
            "family_id",
            "source_dataset_id",
            "source_request_id",
            "session_dataset_id",
            "session_timeframe_configuration_id",
            "session_policy_id",
            "feed_scope_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"intraday prediction {name} must be nonempty text")
        if (
            not isinstance(cast(object, self.source_raw_snapshot_ids), tuple)
            or not self.source_raw_snapshot_ids
            or any(
                not isinstance(item, str) or not item
                for item in cast(tuple[object, ...], self.source_raw_snapshot_ids)
            )
            or len(set(self.source_raw_snapshot_ids))
            != len(self.source_raw_snapshot_ids)
        ):
            raise ValueError("intraday prediction raw snapshot IDs must be unique")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "source_dataset_id": self.source_dataset_id,
            "source_request_id": self.source_request_id,
            "source_raw_snapshot_ids": list(self.source_raw_snapshot_ids),
            "session_dataset_id": self.session_dataset_id,
            "session_timeframe_configuration_id": (
                self.session_timeframe_configuration_id
            ),
            "session_policy_id": self.session_policy_id,
            "feed_scope_id": self.feed_scope_id,
            "corporate_action_availability": self.corporate_action_availability.value,
            "family_manifest": self.family_manifest.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: object) -> "IntradayPredictionProvenance":
        if not isinstance(value, dict):
            raise ValueError("intraday prediction provenance must be a record")
        record = cast(dict[str, object], value)
        fields = set(cls.__dataclass_fields__)
        if set(record) != fields:
            raise ValueError("intraday prediction provenance fields are invalid")
        snapshots = record["source_raw_snapshot_ids"]
        if not isinstance(snapshots, list):
            raise ValueError("intraday prediction raw snapshots must be an array")
        family_manifest = record["family_manifest"]
        if not isinstance(family_manifest, dict):
            raise ValueError("intraday prediction family manifest must be a record")
        strings = fields - {
            "source_raw_snapshot_ids",
            "corporate_action_availability",
            "family_manifest",
        }
        if any(not isinstance(record[name], str) for name in strings):
            raise ValueError("intraday prediction provenance requires text identities")
        return cls(
            **cast(dict[str, str], {name: record[name] for name in strings}),
            source_raw_snapshot_ids=tuple(cast(list[str], snapshots)),
            corporate_action_availability=CorporateActionAvailability(
                cast(str, record["corporate_action_availability"])
            ),
            family_manifest=PrimitiveMappingSnapshot.capture(
                cast(PrimitiveMapping, family_manifest)
            ),
        )


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
    intraday_provenance: IntradayPredictionProvenance | None = None

    @property
    def corporate_action_availability(self) -> CorporateActionAvailability:
        """Event supply; available does not imply a complete observed history."""
        return (
            CorporateActionAvailability.AVAILABLE
            if self.intraday_provenance is None
            else self.intraday_provenance.corporate_action_availability
        )


@dataclass(frozen=True, slots=True)
class MarketDataset:
    """Canonical bars permanently associated with their immutable manifest."""

    bars: tuple[DailyBar, ...]
    metadata: DatasetMetadata
    corporate_actions: tuple[CorporateAction, ...] = ()
