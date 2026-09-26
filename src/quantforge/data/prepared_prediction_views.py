"""Bounded, invocation-owned QF-52 preparation; never persisted scientific state."""

from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime, timezone
from decimal import Decimal
from typing import cast
from zoneinfo import ZoneInfo

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import (
    serialize_bars_csv,
    sha256_hex,
)
from quantforge.data.models import (
    BOUNDED_PREDICTION_ADAPTER_VERSION,
    AdjustmentMode,
    BoundedPredictionProvenance,
    CorporateActionAvailability,
    DailyBar,
    DatasetMetadata,
    IntradayPredictionProvenance,
    MarketDataset,
)
from quantforge.data.prediction_inputs import validate_bounded_prediction_record
from quantforge.data.prediction_session_evidence import (
    session_bar_from_evidence,
    session_projection_bars,
)
from quantforge.data.prediction_views import bounded_prediction_view


def _immutable(value: object) -> bool:
    """Admit only reviewed immutable carriers; JSON-equivalent types may be unsafe."""
    kind = type(value)
    if kind in (
        type(None),
        str,
        int,
        bool,
        date,
        Decimal,
        AdjustmentMode,
        CorporateActionAvailability,
    ):
        return True
    if kind is datetime:
        instant = cast(datetime, value)
        return type(instant.tzinfo) in (timezone, ZoneInfo)
    if kind is tuple:
        return all(_immutable(item) for item in cast(tuple[object, ...], value))
    if (
        kind is DatasetMetadata
        and type(cast(DatasetMetadata, value).adjustment_mode) is not AdjustmentMode
    ):
        return False
    if kind is DailyBar:
        bar = cast(DailyBar, value)
        if (
            type(bar.symbol) is not str
            or type(bar.session_date) is not date
            or any(
                type(price) is not Decimal
                for price in (bar.open, bar.high, bar.low, bar.close, bar.volume)
            )
        ):
            return False
    if kind in (
        MarketDataset,
        DatasetMetadata,
        DailyBar,
        IntradayPredictionProvenance,
        BoundedPredictionProvenance,
        PrimitiveMappingSnapshot,
    ):
        return all(_immutable(getattr(value, item.name)) for item in fields(kind))
    return False


def _metadata_fingerprint(metadata: DatasetMetadata) -> str:
    """Bind every field without decoding the canonical source's large snapshots.

    This operational key supplements, never replaces, authoritative QF-51 IDs.
    Snapshot bytes are immutable; hashing them authenticates their current full
    contents even when a caller supplies unchanged IDs with different evidence.
    """
    original = metadata.intraday_provenance
    assert isinstance(original, IntradayPredictionProvenance)
    values = asdict(metadata)
    provenance = cast(dict[str, object], values["intraday_provenance"])
    for name in (
        "family_manifest",
        "source_manifest",
        "session_evidence",
        "source_bar_evidence",
    ):
        snapshot = cast(PrimitiveMappingSnapshot, getattr(original, name))
        provenance[name] = sha256_hex(snapshot.canonical_json.encode())
    for name in (
        "requested_start",
        "requested_end",
        "actual_first_session",
        "actual_last_session",
    ):
        values[name] = cast(date, values[name]).isoformat()
    values["retrieved_at"] = metadata.retrieved_at.astimezone(UTC).isoformat()
    for name in ("missing_sessions", "split_sessions", "dividend_sessions"):
        values[name] = [
            item.isoformat() for item in cast(tuple[date, ...], values[name])
        ]
    return configuration_identity(cast(PrimitiveMapping, values))


@dataclass(frozen=True, slots=True)
class _ProjectionKey:
    canonical_metadata_id: str
    scope: PrimitiveMappingSnapshot
    cutoff: datetime
    start: date | None
    policy: str


@dataclass(frozen=True, slots=True)
class _PreparedProjection:
    view: MarketDataset
    source_fingerprint: str


class PreparedProjectionRegistry:
    """Keep at most two bounded views, released at the invocation boundary.

    Keys authenticate current metadata/evidence, not caller-supplied IDs alone.
    ``scope`` must describe the plan/fold/window or standalone context environment.
    Strategy parameters are deliberately absent. This sequential registry belongs
    to orchestration and must never be supplied to research callbacks.
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[_ProjectionKey, _PreparedProjection] = OrderedDict()

    def clear(self) -> None:
        """Release all preparation, including on failed/interrupted execution."""
        self._entries.clear()

    @property
    def retained_count(self) -> int:
        return len(self._entries)

    def _key(
        self,
        metadata: DatasetMetadata,
        cutoff: datetime,
        start: date | None,
        scope: PrimitiveMappingSnapshot,
    ) -> _ProjectionKey:
        if not _immutable(metadata):
            raise ValidationError(
                "prepared prediction metadata must be deeply immutable"
            )
        if not isinstance(metadata.intraday_provenance, IntradayPredictionProvenance):
            raise ValidationError(
                "prepared prediction views require canonical ancestry"
            )
        if not isinstance(cast(object, cutoff), datetime) or cutoff.utcoffset() is None:
            raise ValidationError("bounded prediction cutoff must be timezone-aware")
        # Re-read current evidence even on hits: stale IDs or replaced ancestry
        # cannot authorize reuse. No canonical bar collection is retained here.
        metadata_id = _metadata_fingerprint(metadata)
        return _ProjectionKey(
            metadata_id,
            scope,
            cutoff.astimezone(UTC),
            start,
            BOUNDED_PREDICTION_ADAPTER_VERSION,
        )

    def _retain(
        self, key: _ProjectionKey, view: MarketDataset, source_fingerprint: str
    ) -> MarketDataset:
        entry = _PreparedProjection(view, source_fingerprint)
        self._entries[key] = entry
        self._entries.move_to_end(key)
        # Normalize an omitted start to the actual observed start, allowing the
        # independent lineage reader's explicit start to locate this same view.
        # Store one entry, not a second alias retaining duplicate evidence.
        while len(self._entries) > 2:
            self._entries.popitem(last=False)
        return view

    def _find(self, key: _ProjectionKey) -> _PreparedProjection | None:
        for existing, entry in tuple(self._entries.items()):
            if (
                existing.canonical_metadata_id == key.canonical_metadata_id
                and existing.scope == key.scope
                and existing.cutoff == key.cutoff
                and existing.policy == key.policy
                and (
                    existing.start == key.start
                    or (
                        existing.start is None
                        and key.start == entry.view.metadata.actual_first_session
                    )
                )
            ):
                self._entries.move_to_end(existing)
                return entry
        return None

    def project(
        self,
        dataset: MarketDataset,
        cutoff: datetime,
        *,
        scope: PrimitiveMappingSnapshot,
        start: date | None = None,
    ) -> MarketDataset:
        """Reuse only an exact authenticated source/scope/range projection."""
        if not _immutable(dataset):
            raise ValidationError("prepared prediction source must be deeply immutable")
        key = self._key(dataset.metadata, cutoff, start, scope)
        if dataset.corporate_actions:
            raise ValidationError("canonical intraday events must remain unavailable")
        fingerprint = sha256_hex(serialize_bars_csv(dataset.bars))
        entry = self._find(key)
        if entry is not None:
            if entry.source_fingerprint != fingerprint:
                raise ValidationError("prepared prediction source content changed")
            return entry.view
        view = bounded_prediction_view(dataset, cutoff, start=start)
        return self._retain(key, view, fingerprint)

    def prepare_lineage(
        self,
        canonical_metadata: DatasetMetadata,
        cutoff: datetime,
        *,
        scope: PrimitiveMappingSnapshot,
        start: date | None = None,
    ) -> MarketDataset:
        """Construct expected immutable evidence once when no projection exists."""
        key = self._key(canonical_metadata, cutoff, start, scope)
        entry = self._find(key)
        if entry is not None:
            return entry.view
        original = canonical_metadata.intraday_provenance
        assert isinstance(original, IntradayPredictionProvenance)
        evidence = cast(
            PrimitiveMapping, original.session_evidence.to_primitive()["bars"]
        )
        bars = session_projection_bars(
            tuple(
                session_bar_from_evidence(item)
                for item in cast(list[Primitive], evidence["bars"])
            )
        )
        view = bounded_prediction_view(
            MarketDataset(bars, canonical_metadata), cutoff, start=start
        )
        return self._retain(key, view, sha256_hex(serialize_bars_csv(bars)))

    def verify_lineage(
        self,
        record: PrimitiveMapping,
        canonical_metadata: DatasetMetadata,
        cutoff: datetime,
        *,
        scope: PrimitiveMappingSnapshot,
        start: date | None = None,
    ) -> None:
        """Always validate supplied evidence and bind it to the prepared ancestor."""
        provenance = validate_bounded_prediction_record(record)
        expected = self.prepare_lineage(
            canonical_metadata, cutoff, scope=scope, start=start
        )
        assert isinstance(
            expected.metadata.intraday_provenance, BoundedPredictionProvenance
        )
        if expected.metadata.intraday_provenance != provenance:
            raise ValidationError(
                "bounded prediction view differs from canonical plan ancestry or cutoff"
            )
