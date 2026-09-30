"""Execution-local authenticated canonical dataset preparation (QF-65).

A research load session authenticates each canonical intraday source once, at
its immutable-cache trust boundary, and each derived artifact once, by the
existing validators. Later consumers in the same session reuse that proof only
when the object they present is proven identical in content: its request and
metadata are equal by value and every bar field (and every nested record field)
equals the pristine authenticated value. Otherwise they run the unchanged
reference validation, which fails closed on corruption.

Nothing here is scientific identity. Authoritative dataset, batch, bar, manifest,
report and view identities are the unchanged persisted values. Preparation is
never persisted, never process-global and never required to validate a
persisted artifact after restart; a new process authenticates again.
"""

from collections import Counter, OrderedDict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from operator import attrgetter
from pathlib import Path
from sys import getsizeof
from typing import TYPE_CHECKING, Protocol, cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.identity import serialize_bars_csv, sha256_hex
from quantforge.data.models import (
    BoundedPredictionProvenance,
    DatasetMetadata,
    IntradayPredictionProvenance,
    MarketDataset,
)
from quantforge.timeframes import Timeframe, TimeframeMemo, timeframe_memo

if TYPE_CHECKING:
    from quantforge.data.intraday import IntradayBarRequest
    from quantforge.data.intraday_ingestion import IntradayDataset

type _Getter = Callable[[object], tuple[object, ...]]

# Immutable leaves compared by value. Calendar timestamps are datetime subclasses
# whose instance dictionaries never participate in equality or identities.
_SCALAR_TYPES = (type(None), str, int, Decimal, date, time, timedelta, Enum)
_MARKET_MEMO_LIMIT = 64
_COUNTERS = (
    "source_authentications",
    "source_not_admitted",
    "source_cache_reuses",
    "source_cache_byte_mismatches",
    "source_reuses",
    "source_rejections",
    "source_integrity_failures",
    "derivations",
    "derivation_reuses",
    "derivation_rejections",
    "derived_validations",
    "derived_validation_reuses",
    "derived_rejections",
    "derived_integrity_failures",
    "derived_not_admitted",
    "market_validations",
    "market_validation_reuses",
    "session_window_resolutions",
    "session_window_reuses",
)


class _DerivedDataset(Protocol):
    """Structural view of QF-18/QF-19 derived datasets (no import cycle)."""

    @property
    def bars(self) -> tuple[object, ...]: ...

    @property
    def metadata(self) -> object: ...


def _field_getter(record_type: type) -> _Getter:
    names = tuple(item.name for item in fields(record_type))
    if not names:
        return lambda record: ()
    if len(names) == 1:
        name = names[0]
        return lambda record: (getattr(record, name),)
    return cast(_Getter, attrgetter(*names))


def _collect_records(value: object, found: dict[int, object]) -> bool:
    """Collect frozen dataclass records below a field; False for mutable carriers."""
    if isinstance(value, _SCALAR_TYPES):
        return True
    if type(value) is tuple:
        return all(
            _collect_records(item, found) for item in cast(tuple[object, ...], value)
        )
    if not is_dataclass(value) or isinstance(value, type):
        return False
    if not getattr(type(value), "__dataclass_params__").frozen:
        return False
    if id(value) in found:
        return True
    found[id(value)] = value
    return all(
        _collect_records(getattr(value, item.name), found) for item in fields(value)
    )


@dataclass(frozen=True, slots=True, eq=False)
class _ContentIntegrity:
    """Pristine field values of authenticated bars and every nested record.

    Frozen records change only through a bypass such as ``object.__setattr__``,
    which replaces a field value. Comparing current field values with these
    pristine ones proves unchanged content as completely as reserializing the
    bars, at C-level map/compare cost (the QF-63 backing-integrity technique).
    Presented bars that are different but equal objects compare equal by value.
    """

    bar_getter: _Getter
    bar_values: tuple[tuple[object, ...], ...]
    records: tuple[tuple[_Getter, tuple[object, ...], tuple[object, ...]], ...]

    @classmethod
    def capture(cls, bars: tuple[object, ...]) -> "_ContentIntegrity | None":
        """Return ``None`` for heterogeneous or mutable graphs (reference path)."""
        if not bars:
            return cls(lambda record: (), (), ())
        bar_type = type(bars[0])
        if (
            not is_dataclass(bar_type)
            or not getattr(bar_type, "__dataclass_params__").frozen
            or any(type(bar) is not bar_type for bar in bars)
        ):
            return None
        getter = _field_getter(bar_type)
        values = tuple(map(getter, bars))
        found: dict[int, object] = {}
        for column in zip(*values, strict=True):
            if all(issubclass(kind, _SCALAR_TYPES) for kind in set(map(type, column))):
                continue
            for value in {id(value): value for value in column}.values():
                if not _collect_records(value, found):
                    return None
        by_type: dict[type, list[object]] = {}
        for record in found.values():
            by_type.setdefault(type(record), []).append(record)
        records: list[tuple[_Getter, tuple[object, ...], tuple[object, ...]]] = []
        for record_type, members in by_type.items():
            record_getter = _field_getter(record_type)
            owned = tuple(members)
            records.append((record_getter, owned, tuple(map(record_getter, owned))))
        return cls(getter, values, tuple(records))

    def intact(self, bars: tuple[object, ...]) -> bool:
        return (
            len(bars) == len(self.bar_values)
            and tuple(map(self.bar_getter, bars)) == self.bar_values
            and all(
                tuple(map(getter, owned)) == values
                for getter, owned, values in self.records
            )
        )


@dataclass(frozen=True, slots=True, eq=False)
class PreparedCanonicalDataset:
    """One canonical intraday source authenticated by its immutable cache.

    ``batch_id`` and ``bar_ids`` are the authenticated persisted identities (the
    manifest batch identity and each verified bar identity, by position). The
    compatibility key is ``(dataset_id, request_id)`` plus the resolved cache
    root; reuse additionally requires value-equal request/metadata and intact
    bar content. Coverage and session evidence are the authenticated metadata.
    ``file_digests`` are the SHA-256 digests of every artifact file the loader
    authenticated (manifest, normalized bars and raw extracts), by relative path.
    """

    dataset: "IntradayDataset"
    cache_root: Path
    batch_id: str
    bar_ids: tuple[str, ...]
    integrity: _ContentIntegrity
    file_digests: tuple[tuple[str, str], ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset.metadata.dataset_id, self.dataset.request.request_id


@dataclass(frozen=True, slots=True, eq=False)
class _AuthenticatedDerived:
    dataset: _DerivedDataset
    integrity: _ContentIntegrity

    def matches(self, dataset: _DerivedDataset) -> bool:
        return (
            type(dataset) is type(self.dataset)
            and dataset.metadata == self.dataset.metadata
            and getattr(dataset, "request", None)
            == getattr(self.dataset, "request", None)
            and self.integrity.intact(dataset.bars)
        )


def _plain(value: object) -> object:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_plain(item) for item in cast(tuple[object, ...], value)]
    if isinstance(value, PrimitiveMappingSnapshot):
        # Immutable evidence bytes are hashed without decoding (QF-60 technique).
        return {"sha256": sha256_hex(value.canonical_json.encode())}
    return value


def _market_fingerprint(dataset: MarketDataset) -> str | None:
    """Content key of one intraday-derived QF-3 dataset, or ``None`` (reference).

    Every metadata and provenance field is bound; snapshot evidence by the hash
    of its canonical bytes and bars by their canonical QF-3 serialization.
    """
    from quantforge.data.prepared_prediction_views import (
        _immutable,  # pyright: ignore[reportPrivateUsage]
    )

    if type(dataset) is not MarketDataset or dataset.corporate_actions:
        return None
    metadata = dataset.metadata
    provenance = metadata.intraday_provenance
    if type(provenance) not in (
        IntradayPredictionProvenance,
        BoundedPredictionProvenance,
    ) or not _immutable(dataset):
        return None
    values = {
        item.name: _plain(getattr(metadata, item.name))
        for item in fields(DatasetMetadata)
        if item.name != "intraday_provenance"
    }
    evidence = {
        item.name: _plain(getattr(provenance, item.name))
        for item in fields(cast(type, type(provenance)))
    }
    return configuration_identity(
        cast(
            PrimitiveMapping,
            {
                "metadata": values,
                "provenance_type": type(provenance).__name__,
                "provenance": evidence,
                "bars_sha256": sha256_hex(serialize_bars_csv(dataset.bars)),
            },
        )
    )


class CanonicalPreparation:
    """Authenticated canonical/derived state for one research load session.

    Owned by the orchestration that opens it with ``canonical_preparation()``
    and released when that scope exits, including on errors. It retains only
    datasets that the session itself authenticated (never a global cache), and
    returns ``None``/``False`` whenever compatibility cannot be proven so the
    caller performs independent reference validation.
    """

    def __init__(self) -> None:
        self.sessions = TimeframeMemo()
        self._sources: dict[tuple[str, str], PreparedCanonicalDataset] = {}
        self._validated: dict[tuple[type, str], _AuthenticatedDerived] = {}
        self._derivations: dict[tuple[str, ...], _AuthenticatedDerived] = {}
        self._markets: OrderedDict[str, tuple[date, ...]] = OrderedDict()
        self._windows: dict[tuple[date, Timeframe], tuple[object, ...]] = {}
        self.counts: Counter[str] = Counter()

    def clear(self) -> None:
        """Release every retained dataset, evidence record and session."""
        self._sources.clear()
        self._validated.clear()
        self._derivations.clear()
        self._markets.clear()
        self._windows.clear()
        self.sessions.clear()

    def session_windows[T](
        self,
        session_date: date,
        timeframe: Timeframe,
        resolve: Callable[[date, Timeframe], tuple[T, ...]],
    ) -> tuple[T, ...]:
        """Resolve one exact (session, timeframe) value's immutable windows once."""
        key = (session_date, timeframe)
        found = self._windows.get(key)
        if found is None:
            windows = resolve(session_date, timeframe)
            self._windows[key] = windows
            self.counts["session_window_resolutions"] += 1
            return windows
        self.counts["session_window_reuses"] += 1
        return cast(tuple[T, ...], found)

    # -- canonical sources -------------------------------------------------

    def retain_source(
        self,
        dataset: "IntradayDataset",
        *,
        cache_root: Path,
        batch_id: str,
        bar_ids: tuple[str, ...],
        file_digests: tuple[tuple[str, str], ...],
    ) -> None:
        """Record a source just authenticated from ``cache_root`` by the loader."""
        integrity = _ContentIntegrity.capture(cast(tuple[object, ...], dataset.bars))
        if integrity is None or len(bar_ids) != len(dataset.bars):
            self.counts["source_not_admitted"] += 1
            return
        entry = PreparedCanonicalDataset(
            dataset,
            cache_root.resolve(),
            batch_id,
            bar_ids,
            integrity,
            file_digests,
        )
        self._sources[entry.key] = entry
        self.counts["source_authentications"] += 1

    def cached_source(
        self, cache_root: Path, dataset_id: str, request: "IntradayBarRequest"
    ) -> "IntradayDataset | None":
        """Return the retained dataset for an identical cache load, if intact.

        Every artifact file is read and hashed again, so changed cache bytes are
        never masked; only decoding and validation of identical bytes is reused.
        """
        entry = self._sources.get((dataset_id, request.request_id))
        root = cache_root.resolve()
        if (
            entry is None
            or entry.cache_root != root
            or entry.dataset.request != request
        ):
            return None
        if not entry.integrity.intact(cast(tuple[object, ...], entry.dataset.bars)):
            self._evict_source(entry)
            return None
        for location, digest in entry.file_digests:
            try:
                current = sha256_hex((root / location).read_bytes())
            except OSError:
                current = None
            if current != digest:
                self.counts["source_cache_byte_mismatches"] += 1
                self._evict_source(entry)
                return None
        self.counts["source_cache_reuses"] += 1
        return entry.dataset

    def source(self, dataset: "IntradayDataset") -> PreparedCanonicalDataset | None:
        """Return authenticated facts for a content-identical presented source."""
        entry = self._sources.get(
            (dataset.metadata.dataset_id, dataset.request.request_id)
        )
        if entry is None:
            return None
        if not entry.integrity.intact(cast(tuple[object, ...], entry.dataset.bars)):
            self._evict_source(entry)
            return None
        if (
            dataset.request != entry.dataset.request
            or dataset.metadata != entry.dataset.metadata
            or not entry.integrity.intact(cast(tuple[object, ...], dataset.bars))
        ):
            self.counts["source_rejections"] += 1
            return None
        self.counts["source_reuses"] += 1
        return entry

    def _evict_source(self, entry: PreparedCanonicalDataset) -> None:
        self.counts["source_integrity_failures"] += 1
        self._sources.pop(entry.key, None)
        source_id = entry.dataset.metadata.dataset_id
        for key in [key for key in self._derivations if key[0] == source_id]:
            del self._derivations[key]

    # -- derived artifacts -------------------------------------------------

    def retain_validated(self, dataset: _DerivedDataset, dataset_id: str) -> None:
        """Record a derived dataset that just passed its reference ``validate``."""
        integrity = self._integrity(dataset)
        if integrity is None:
            self.counts["derived_not_admitted"] += 1
            return
        self._validated[(type(dataset), dataset_id)] = _AuthenticatedDerived(
            dataset, integrity
        )
        self.counts["derived_validations"] += 1

    def validated(self, dataset: _DerivedDataset, dataset_id: str) -> bool:
        entry = self._validated.get((type(dataset), dataset_id))
        if entry is None:
            return False
        if not entry.integrity.intact(entry.dataset.bars):
            self.counts["derived_integrity_failures"] += 1
            del self._validated[(type(dataset), dataset_id)]
            return False
        if not entry.matches(dataset):
            self.counts["derived_rejections"] += 1
            return False
        self.counts["derived_validation_reuses"] += 1
        return True

    def retain_derivation(
        self,
        source: PreparedCanonicalDataset,
        target_timeframe_id: str,
        policy_id: str,
        dataset: _DerivedDataset,
    ) -> None:
        """Record the exact output of deriving an authenticated source once."""
        integrity = self._integrity(dataset)
        if integrity is None:
            return
        key = (*source.key, type(dataset).__name__, target_timeframe_id, policy_id)
        self._derivations[key] = _AuthenticatedDerived(dataset, integrity)
        self.counts["derivations"] += 1

    def _integrity(self, dataset: _DerivedDataset) -> _ContentIntegrity | None:
        """Share one pristine snapshot per retained derived object (memory)."""
        for entry in (*self._derivations.values(), *self._validated.values()):
            if entry.dataset is dataset and entry.integrity.intact(dataset.bars):
                return entry.integrity
        return _ContentIntegrity.capture(dataset.bars)

    def derivation_matches(
        self,
        source: "IntradayDataset",
        target_timeframe_id: str,
        policy_id: str,
        dataset: _DerivedDataset,
    ) -> bool:
        """Whether ``dataset`` is this session's derivation of an intact source."""
        prepared = self.source(source)
        if prepared is None:
            return False
        key = (*prepared.key, type(dataset).__name__, target_timeframe_id, policy_id)
        entry = self._derivations.get(key)
        if entry is None or not entry.integrity.intact(entry.dataset.bars):
            return False
        if not entry.matches(dataset):
            self.counts["derivation_rejections"] += 1
            return False
        self.counts["derivation_reuses"] += 1
        return True

    # -- QF-3 prediction inputs --------------------------------------------

    def validated_market_dataset(
        self,
        dataset: MarketDataset,
        validate: Callable[[MarketDataset], tuple[date, ...]],
    ) -> tuple[date, ...]:
        """Reuse a verdict only for byte-identical intraday-derived content."""
        fingerprint = _market_fingerprint(dataset)
        if fingerprint is None:
            return validate(dataset)
        found = self._markets.get(fingerprint)
        if found is not None:
            self._markets.move_to_end(fingerprint)
            self.counts["market_validation_reuses"] += 1
            return found
        result = validate(dataset)
        self._markets[fingerprint] = result
        while len(self._markets) > _MARKET_MEMO_LIMIT:
            self._markets.popitem(last=False)
        self.counts["market_validations"] += 1
        return result

    # -- reporting -----------------------------------------------------------

    def statistics(self) -> dict[str, int]:
        """Explicit invocation counts (zero when an event never happened)."""
        return {
            **dict.fromkeys(_COUNTERS, 0),
            **self.counts,
            "session_resolutions": self.sessions.resolved,
            "session_reuses": self.sessions.reused,
            "timeframe_identity_computations": self.sessions.identities_computed,
            "timeframe_identity_reuses": self.sessions.identities_reused,
            "retained_sources": len(self._sources),
            "retained_validated_derived": len(self._validated),
            "retained_derivations": len(self._derivations),
            "retained_market_verdicts": len(self._markets),
        }

    def memory_report(self) -> dict[str, int]:
        """Approximate bytes owned by this preparation (bar backings are shared).

        Bars, metadata and nested records belong to the authenticated datasets
        the research session already holds; only identity tuples, integrity
        snapshots and memo entries are owned here.
        """

        def integrity_bytes(integrity: _ContentIntegrity) -> int:
            return (
                getsizeof(integrity.bar_values)
                + sum(map(getsizeof, integrity.bar_values))
                + sum(
                    getsizeof(owned) + getsizeof(values) + sum(map(getsizeof, values))
                    for _, owned, values in integrity.records
                )
            )

        derived = {
            id(entry.integrity): entry
            for entry in (*self._validated.values(), *self._derivations.values())
        }.values()
        return {
            "source_bar_ids_bytes": sum(
                getsizeof(entry.bar_ids) + sum(map(getsizeof, entry.bar_ids))
                for entry in self._sources.values()
            ),
            "source_integrity_bytes": sum(
                integrity_bytes(entry.integrity) for entry in self._sources.values()
            ),
            "derived_integrity_bytes": sum(
                integrity_bytes(entry.integrity) for entry in derived
            ),
            "session_memo_entries": len(self.sessions),
            "session_window_entries": len(self._windows),
            "session_window_bytes": sum(
                getsizeof(windows) + sum(map(getsizeof, windows))
                for windows in self._windows.values()
            ),
            "market_verdict_entries": len(self._markets),
        }


_ACTIVE: ContextVar[CanonicalPreparation | None] = ContextVar(
    "quantforge_canonical_preparation", default=None
)


def active_canonical_preparation() -> CanonicalPreparation | None:
    """The preparation of the enclosing research load session, if any."""
    return _ACTIVE.get()


@contextmanager
def canonical_preparation() -> Generator[CanonicalPreparation]:
    """Open one execution-local research load session, or join the active one.

    Only the outermost scope owns the preparation; it clears all retained state
    on exit, including failures and interruption.
    """
    active = _ACTIVE.get()
    if active is not None:
        yield active
        return
    preparation = CanonicalPreparation()
    token = _ACTIVE.set(preparation)
    try:
        with timeframe_memo(preparation.sessions):
            yield preparation
    finally:
        _ACTIVE.reset(token)
        preparation.clear()


__all__ = [
    "CanonicalPreparation",
    "PreparedCanonicalDataset",
    "active_canonical_preparation",
    "canonical_preparation",
]
