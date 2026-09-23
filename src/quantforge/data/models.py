"""Typed canonical daily-market-data records."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)

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
    source_manifest: PrimitiveMappingSnapshot
    session_evidence: PrimitiveMappingSnapshot
    source_bar_evidence: PrimitiveMappingSnapshot
    schema_version: str = "5"

    def __post_init__(self) -> None:
        if self.schema_version != "5":
            raise ValueError("unsupported intraday prediction provenance schema")
        if not isinstance(cast(object, self.family_manifest), PrimitiveMappingSnapshot):
            raise ValueError("intraday prediction family manifest must be immutable")
        if not isinstance(cast(object, self.source_manifest), PrimitiveMappingSnapshot):
            raise ValueError("intraday prediction source manifest must be immutable")
        if not isinstance(
            cast(object, self.session_evidence), PrimitiveMappingSnapshot
        ):
            raise ValueError("intraday prediction session evidence must be immutable")
        if not isinstance(
            cast(object, self.source_bar_evidence), PrimitiveMappingSnapshot
        ):
            raise ValueError("intraday prediction source bars must be immutable")
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
            "source_manifest": self.source_manifest.to_primitive(),
            "session_evidence": self.session_evidence.to_primitive(),
            "source_bar_evidence": self.source_bar_evidence.to_primitive(),
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
        source_manifest = record["source_manifest"]
        if not isinstance(source_manifest, dict):
            raise ValueError("intraday prediction source manifest must be a record")
        session_evidence = record["session_evidence"]
        if not isinstance(session_evidence, dict):
            raise ValueError("intraday prediction session evidence must be a record")
        source_bar_evidence = record["source_bar_evidence"]
        if not isinstance(source_bar_evidence, dict):
            raise ValueError("intraday prediction source bars must be a record")
        strings = fields - {
            "source_raw_snapshot_ids",
            "corporate_action_availability",
            "family_manifest",
            "source_manifest",
            "session_evidence",
            "source_bar_evidence",
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
            source_manifest=PrimitiveMappingSnapshot.capture(
                cast(PrimitiveMapping, source_manifest)
            ),
            session_evidence=PrimitiveMappingSnapshot.capture(
                cast(PrimitiveMapping, session_evidence)
            ),
            source_bar_evidence=PrimitiveMappingSnapshot.capture(
                cast(PrimitiveMapping, source_bar_evidence)
            ),
        )


BOUNDED_PREDICTION_DATASET_PREFIX = "bounded-intraday-"
BOUNDED_PREDICTION_ADAPTER_VERSION = "quantforge_bounded_prediction_input_v1"


@dataclass(frozen=True, slots=True)
class BoundedPredictionProvenance:
    """A causal view, with opaque canonical ancestry and local session evidence.

    No canonical source manifest, coverage report, raw chunk list, or parent
    object is retained here. The family contains identities and static policies;
    session evidence contains only the completed observations in this view.
    Canonical validation happens before projection, outside the evaluator.
    """

    canonical_input_id: str
    canonical_provenance_id: str
    family_id: str
    source_dataset_id: str
    source_request_id: str
    session_dataset_id: str
    session_timeframe_configuration_id: str
    session_policy_id: str
    feed_scope_id: str
    provider_symbol: str
    source_retrieved_at: datetime
    causal_cutoff: datetime
    family_manifest: PrimitiveMappingSnapshot
    session_evidence: PrimitiveMappingSnapshot
    bars_fingerprint: str
    corporate_action_availability: CorporateActionAvailability = (
        CorporateActionAvailability.UNAVAILABLE
    )
    schema_version: str = "bounded-1"

    def __post_init__(self) -> None:
        if self.schema_version != "bounded-1":
            raise ValueError("unsupported bounded prediction provenance schema")
        for name in ("source_retrieved_at", "causal_cutoff"):
            instant = getattr(self, name)
            if not isinstance(instant, datetime) or instant.utcoffset() is None:
                raise ValueError(f"bounded prediction {name} must be timezone-aware")
            object.__setattr__(self, name, instant.astimezone(UTC))
        for name in ("family_manifest", "session_evidence"):
            if not isinstance(getattr(self, name), PrimitiveMappingSnapshot):
                raise ValueError(f"bounded prediction {name} must be immutable")
        if (
            self.corporate_action_availability
            is not CorporateActionAvailability.UNAVAILABLE
        ):
            raise ValueError(
                "bounded intraday corporate actions must remain unavailable"
            )
        for name in (
            "canonical_input_id",
            "canonical_provenance_id",
            "family_id",
            "source_dataset_id",
            "source_request_id",
            "session_dataset_id",
            "session_timeframe_configuration_id",
            "session_policy_id",
            "feed_scope_id",
            "provider_symbol",
            "bars_fingerprint",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"bounded prediction {name} must be nonempty text")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            name: (
                value.to_primitive()
                if isinstance(value, PrimitiveMappingSnapshot)
                else value.isoformat()
                if isinstance(value, datetime)
                else value.value
                if isinstance(value, CorporateActionAvailability)
                else value
            )
            for name in self.__dataclass_fields__
            for value in (getattr(self, name),)
        }

    @property
    def view_id(self) -> str:
        return BOUNDED_PREDICTION_DATASET_PREFIX + configuration_identity(
            self.to_primitive()
        )

    @classmethod
    def from_primitive(cls, value: object) -> "BoundedPredictionProvenance":
        if not isinstance(value, dict) or set(cast(dict[str, object], value)) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("bounded prediction provenance fields are invalid")
        record = cast(dict[str, object], value).copy()
        for name in ("family_manifest", "session_evidence"):
            if not isinstance(record[name], dict):
                raise ValueError(f"bounded prediction {name} must be a record")
            record[name] = PrimitiveMappingSnapshot.capture(
                cast(PrimitiveMapping, record[name])
            )
        for name in ("source_retrieved_at", "causal_cutoff"):
            if not isinstance(record[name], str):
                raise ValueError(f"bounded prediction {name} must be a timestamp")
            record[name] = datetime.fromisoformat(cast(str, record[name]))
        record["corporate_action_availability"] = CorporateActionAvailability(
            record["corporate_action_availability"]
        )
        restored = cls(
            canonical_input_id=cast(str, record["canonical_input_id"]),
            canonical_provenance_id=cast(str, record["canonical_provenance_id"]),
            family_id=cast(str, record["family_id"]),
            source_dataset_id=cast(str, record["source_dataset_id"]),
            source_request_id=cast(str, record["source_request_id"]),
            session_dataset_id=cast(str, record["session_dataset_id"]),
            session_timeframe_configuration_id=cast(
                str, record["session_timeframe_configuration_id"]
            ),
            session_policy_id=cast(str, record["session_policy_id"]),
            feed_scope_id=cast(str, record["feed_scope_id"]),
            provider_symbol=cast(str, record["provider_symbol"]),
            source_retrieved_at=cast(datetime, record["source_retrieved_at"]),
            causal_cutoff=cast(datetime, record["causal_cutoff"]),
            family_manifest=cast(PrimitiveMappingSnapshot, record["family_manifest"]),
            session_evidence=cast(PrimitiveMappingSnapshot, record["session_evidence"]),
            bars_fingerprint=cast(str, record["bars_fingerprint"]),
            corporate_action_availability=record["corporate_action_availability"],
            schema_version=cast(str, record["schema_version"]),
        )
        if restored.to_primitive() != value:
            raise ValueError("bounded prediction provenance must be canonical")
        return restored


type PredictionInputProvenance = (
    IntradayPredictionProvenance | BoundedPredictionProvenance
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
    intraday_provenance: PredictionInputProvenance | None = None

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
    """Session bars with canonical source or explicit bounded-view provenance."""

    bars: tuple[DailyBar, ...]
    metadata: DatasetMetadata
    corporate_actions: tuple[CorporateAction, ...] = ()
