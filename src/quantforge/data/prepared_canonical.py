"""Execution-local authenticated canonical dataset preparation (QF-65).

A research load session authenticates each canonical intraday source once, at
its immutable-cache trust boundary, and each derived artifact once, by the
existing validators. Later consumers in the same session reuse that proof only
when the object they present is proven identical in content: its request,
metadata and every bar field, with every nested record field, equal the
pristine authenticated values, and the retained dataset itself is still intact.
Otherwise they run the unchanged reference validation, which fails closed on
corruption.

Nothing here is scientific identity. Authoritative dataset, batch, bar, manifest,
report and view identities are the unchanged persisted values. Preparation is
never persisted, never process-global and never required to validate a
persisted artifact after restart; a new process authenticates again.
"""

from collections import Counter, OrderedDict
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from itertools import chain
from operator import attrgetter, is_
from pathlib import Path
from sys import getsizeof
from typing import TYPE_CHECKING, Any, Protocol, cast
from zoneinfo import ZoneInfo

from pandas import Timestamp  # pyright: ignore[reportMissingTypeStubs]

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
from quantforge.timeframes import (
    Timeframe,
    TimeframeMemo,
    memoizable_timeframe_state,
    timeframe_memo,
)

if TYPE_CHECKING:
    from quantforge.data.intraday import IntradayBarRequest
    from quantforge.data.intraday_ingestion import IntradayDataset

type _Getter = Callable[[object], tuple[object, ...]]
type _RecordSnapshot = tuple[
    type, _Getter, tuple[object, ...], tuple[tuple[object, ...], ...]
]

# Exact immutable leaf types, compared by value. Subclasses are refused because
# their class or behavior can change without replacing the field object. The
# exchange calendar's Timestamp is admitted exactly; its instance dictionary
# never participates in equality or identities. Enum members are process-global
# singletons whose state is snapshotted separately.
_SCALAR_TYPES: frozenset[type] = frozenset(
    {
        type(None),
        str,
        int,
        bool,
        Decimal,
        date,
        datetime,
        time,
        timedelta,
        cast(type, Timestamp),
    }
)
# A zone's offset rule must not change behind an unchanged datetime object.
_ZONED_TYPES: frozenset[type] = frozenset({datetime, time, cast(type, Timestamp)})
_ZONE_TYPES = (timezone, ZoneInfo)
_MARKET_MEMO_LIMIT = 64
_VARIANT_LIMIT = 4
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
    "session_window_declines",
)


class _DerivedDataset(Protocol):
    """Structural view of QF-18/QF-19 derived datasets (no import cycle)."""

    @property
    def bars(self) -> tuple[object, ...]: ...

    @property
    def metadata(self) -> object: ...


def _field_getter(record_type: type) -> _Getter:
    return _names_getter(tuple(item.name for item in fields(record_type)))


def _names_getter(names: tuple[str, ...]) -> _Getter:
    if not names:
        return lambda record: ()
    if len(names) == 1:
        name = names[0]
        return lambda record: (getattr(record, name),)
    return cast(_Getter, attrgetter(*names))


class _Reach:
    """Records and enum members reachable from authenticated fields, by id."""

    __slots__ = ("enums", "records")

    def __init__(self) -> None:
        self.records: dict[int, object] = {}
        self.enums: dict[int, Enum] = {}


def _admitted_zones(values: Iterable[object]) -> bool:
    """Datetime-like leaves may carry only immutable reviewed zones (QF-60).

    Zones are deduplicated by identity: an unreviewed zone is rejected here,
    never hashed (it may be unhashable or hash arbitrarily).
    """
    zones = {id(zone): zone for zone in map(attrgetter("tzinfo"), values)}
    return all(zone is None or type(zone) in _ZONE_TYPES for zone in zones.values())


def _collect(value: object, reach: _Reach) -> bool:
    """Collect reachable frozen records and enum members; False if unadmitted."""
    kind = type(value)
    if kind in _SCALAR_TYPES:
        return kind not in _ZONED_TYPES or _admitted_zones((value,))
    if isinstance(value, Enum):
        reach.enums.setdefault(id(value), value)
        return True
    if kind is tuple:
        return all(_collect(item, reach) for item in cast(tuple[object, ...], value))
    if not _frozen_dataclass(kind):
        return False
    if id(value) in reach.records:
        return True
    reach.records[id(value)] = value
    return all(
        _collect(getattr(value, item.name), reach) for item in fields(cast(Any, kind))
    )


def _quantforge_enum_members() -> tuple[Enum, ...]:
    """Every member of every QuantForge enum class.

    Serialization also reads members that no field reaches, through computed
    properties (an interval's ``kind``, a report's ``status``). Mutating such a
    process-global member changes canonical bytes, so all are snapshotted.
    """
    members: dict[int, Enum] = {}
    pending: list[type[Enum]] = [Enum]
    seen: set[type[Enum]] = set()
    while pending:
        enum_type = pending.pop()
        if enum_type in seen:
            continue
        seen.add(enum_type)
        pending.extend(enum_type.__subclasses__())
        if enum_type.__module__.partition(".")[0] == "quantforge":
            for member in enum_type.__members__.values():
                members.setdefault(id(member), member)
    return tuple(members.values())


def _enum_state(member: Enum) -> tuple[object, ...]:
    """What canonical serialization reads from a member: class, name and value."""
    return type(member), member._name_, member._value_


def _exact_text(values: tuple[object, ...]) -> bool:
    """Memo key parts must be exact ``str`` before they are ever hashed."""
    return all(type(value) is str for value in values)


def _frozen_dataclass(kind: type) -> bool:
    return is_dataclass(kind) and getattr(kind, "__dataclass_params__").frozen


def _strict_equal(left: object, right: object) -> bool:
    """Equal content with equal types and exact representation.

    Python numeric and datetime equality ignore type and timezone representation
    (``100 == Decimal(100)``, equal instants in different zones), but canonical
    serialization and validation do not. Such a substitution is never equal here.
    """
    if left is right:
        return True
    kind = type(left)
    if kind is not type(right):
        return False
    if kind is tuple:
        return _same(cast(tuple[object, ...], left), cast(tuple[object, ...], right))
    if _frozen_dataclass(kind):
        getter = _field_getter(kind)
        return _same(getter(left), getter(right))
    if kind in _ZONED_TYPES and type(getattr(left, "tzinfo")) is not type(
        getattr(right, "tzinfo")
    ):
        return False  # A zone type the pristine value lacks (its repr may mimic).
    return bool(left == right) and repr(left) == repr(right)


def _same(current: tuple[object, ...], pristine: tuple[object, ...]) -> bool:
    """Identical field objects (the normal case), else strictly equal ones."""
    return len(current) == len(pristine) and (
        all(map(is_, current, pristine)) or all(map(_strict_equal, current, pristine))
    )


def _same_rows(
    current: tuple[tuple[object, ...], ...], pristine: tuple[tuple[object, ...], ...]
) -> bool:
    """Row-wise ``_same``; unchanged backings pass one C-level identity scan."""
    if len(current) != len(pristine):
        return False
    if all(map(is_, chain.from_iterable(current), chain.from_iterable(pristine))):
        return True
    return all(map(_same, current, pristine))


@dataclass(frozen=True, slots=True, eq=False)
class _ContentIntegrity:
    """Pristine state of one authenticated dataset and everything it reaches.

    Covers the dataset's own fields (request, metadata), every bar and every
    record reachable from them (provenance, timeframe, coverage and aggregation
    reports, family lineage, ...). For each it retains the exact class and the
    field values, plus the class, name and value of every reachable enum member
    and of every QuantForge enum member (some reach serialization via properties).
    Frozen records change only through bypasses such as ``object.__setattr__``,
    which replace a field value or the object's class; comparing current state
    with this snapshot proves unchanged content as completely as reserializing,
    at C-level map/compare cost (the QF-63 backing-integrity technique). A
    presented dataset of different but strictly equal objects also passes.
    """

    dataset_type: type
    field_getter: _Getter
    field_values: tuple[object, ...]
    bar_type: type | None
    bar_getter: _Getter
    bar_values: tuple[tuple[object, ...], ...]
    records: tuple[_RecordSnapshot, ...]
    enums: tuple[Enum, ...]
    enum_states: tuple[tuple[object, ...], ...]

    @classmethod
    def capture(cls, dataset: object) -> "_ContentIntegrity | None":
        """Return ``None`` for heterogeneous or mutable graphs (reference path)."""
        dataset_type = type(dataset)
        if not _frozen_dataclass(dataset_type):
            return None
        bars = cast(object, getattr(dataset, "bars", None))
        if type(bars) is not tuple:
            return None
        bars = cast(tuple[object, ...], bars)
        names = tuple(
            item.name for item in fields(cast(Any, dataset_type)) if item.name != "bars"
        )
        field_getter = _names_getter(names)
        field_values = field_getter(dataset)
        reach = _Reach()
        if not all(_collect(value, reach) for value in field_values):
            return None
        bar_type: type | None = None
        bar_getter: _Getter = lambda record: ()  # noqa: E731 - empty collection.
        bar_values: tuple[tuple[object, ...], ...] = ()
        if bars:
            bar_type = type(bars[0])
            if not _frozen_dataclass(bar_type) or set(map(type, bars)) != {bar_type}:
                return None
            bar_getter = _field_getter(bar_type)
            bar_values = tuple(map(bar_getter, bars))
            for column in zip(*bar_values, strict=True):
                kinds = set(map(type, column))
                if kinds <= _SCALAR_TYPES:
                    if kinds & _ZONED_TYPES and not _admitted_zones(column):
                        return None
                    continue
                for value in {id(value): value for value in column}.values():
                    if not _collect(value, reach):
                        return None
        by_type: dict[type, list[object]] = {}
        for record in reach.records.values():
            by_type.setdefault(type(record), []).append(record)
        records: list[_RecordSnapshot] = []
        for record_type, members in by_type.items():
            record_getter = _field_getter(record_type)
            owned = tuple(members)
            records.append(
                (record_type, record_getter, owned, tuple(map(record_getter, owned)))
            )
        for member in _quantforge_enum_members():
            reach.enums.setdefault(id(member), member)
        enums = tuple(reach.enums.values())
        return cls(
            dataset_type,
            field_getter,
            field_values,
            bar_type,
            bar_getter,
            bar_values,
            tuple(records),
            enums,
            tuple(map(_enum_state, enums)),
        )

    def intact(self, dataset: object) -> bool:
        """Whether ``dataset`` has exactly the authenticated content."""
        if type(dataset) is not self.dataset_type:
            return False
        bars = cast(object, getattr(dataset, "bars", None))
        if type(bars) is not tuple:
            return False
        bars = cast(tuple[object, ...], bars)
        if bars and set(map(type, bars)) != {self.bar_type}:
            return False
        return (
            _same(self.field_getter(dataset), self.field_values)
            and _same_rows(tuple(map(self.bar_getter, bars)), self.bar_values)
            and all(
                set(map(type, owned)) == {record_type}
                and _same_rows(tuple(map(getter, owned)), values)
                for record_type, getter, owned, values in self.records
            )
            and _same_rows(tuple(map(_enum_state, self.enums)), self.enum_states)
        )


@dataclass(frozen=True, slots=True, eq=False)
class PreparedCanonicalDataset:
    """One canonical intraday source authenticated by its immutable cache.

    ``batch_id`` and ``bar_ids`` are the authenticated persisted identities (the
    manifest batch identity and each verified bar identity, by position). The
    compatibility key is ``(dataset_id, request_id)`` plus the resolved cache
    root; reuse additionally requires the presented request, metadata and bars
    (with every nested record) to strictly equal their pristine authenticated
    values. Lookups compare against the snapshot and never recompute the key
    from presented content.
    Coverage and session evidence are the authenticated metadata.
    ``file_digests`` are the SHA-256 digests of every artifact file the loader
    authenticated (manifest, normalized bars and raw extracts), by relative path.
    ``key`` is ``(dataset_id, request_id)`` as admitted: it is never recomputed
    from the retained dataset, which may be corrupted by the time it is evicted.
    """

    dataset: "IntradayDataset"
    key: tuple[str, str]
    cache_root: Path
    batch_id: str
    bar_ids: tuple[str, ...]
    integrity: _ContentIntegrity
    file_digests: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True, eq=False)
class _AuthenticatedDerived:
    dataset: _DerivedDataset
    integrity: _ContentIntegrity

    def intact(self) -> bool:
        """The retained dataset itself still has its authenticated content."""
        return self.integrity.intact(self.dataset)

    def matches(self, dataset: _DerivedDataset) -> bool:
        """Request, metadata, bars and nested records equal the pristine values."""
        return self.integrity.intact(dataset)


def _plain(value: object) -> object:
    """Type- and representation-exact fingerprint input for one field value."""
    kind = type(value)
    if value is None or kind in (str, int, bool):
        return value
    if kind is tuple:
        return [_plain(item) for item in cast(tuple[object, ...], value)]
    if kind is PrimitiveMappingSnapshot:
        # Immutable evidence bytes are hashed without decoding (QF-60 technique).
        snapshot = cast(PrimitiveMappingSnapshot, value)
        return {"sha256": sha256_hex(snapshot.canonical_json.encode())}
    # Decimal, dates, datetimes (with their zone), enums: equal values with
    # another type or representation must not share a verdict.
    return {"type": f"{kind.__module__}.{kind.__qualname__}", "repr": repr(value)}


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
        # Strictly distinct authenticated presentations of one derived dataset.
        self._validated: dict[tuple[type, str], list[_AuthenticatedDerived]] = {}
        self._derivations: dict[tuple[str, ...], _AuthenticatedDerived] = {}
        self._markets: OrderedDict[str, tuple[date, ...]] = OrderedDict()
        self._windows: dict[
            tuple[date, Timeframe], tuple[tuple[object, ...], tuple[object, ...]]
        ] = {}
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
        """Resolve one exact (session, timeframe) value's immutable windows once.

        As for timeframe identities, only exact reviewed leaf types are keyed
        (never hashed otherwise) and each hit rechecks the enum member state.
        """
        state = memoizable_timeframe_state(timeframe)
        if type(session_date) is not date or state is None:
            self.counts["session_window_declines"] += 1
            return resolve(session_date, timeframe)
        key = (session_date, timeframe)
        found = self._windows.get(key)
        if found is not None and all(map(is_, state, found[1])):
            self.counts["session_window_reuses"] += 1
            return cast(tuple[T, ...], found[0])
        windows = resolve(session_date, timeframe)
        self._windows[key] = (windows, state)
        self.counts["session_window_resolutions"] += 1
        return windows

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
        integrity = _ContentIntegrity.capture(dataset)
        key = (
            cast(object, dataset.metadata.dataset_id),
            cast(object, dataset.request.request_id),
        )
        if (
            integrity is None
            or len(bar_ids) != len(dataset.bars)
            or not _exact_text(key)
        ):
            self.counts["source_not_admitted"] += 1
            return
        entry = PreparedCanonicalDataset(
            dataset,
            cast(tuple[str, str], key),
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
        Candidates are found by their admitted key and cache root alone. The
        presented request may be the retained object itself, so none of it is
        evaluated (not even ``request_id``) until the retained dataset is proven
        intact; it is then only compared strictly with the pristine request,
        because the reference load returns the presented request.
        """
        if type(dataset_id) is not str:
            return None  # Never hash an unreviewed key; retain_source declines it.
        root = cache_root.resolve()
        candidates = [
            entry
            for entry in self._sources.values()
            if entry.key[0] == dataset_id and entry.cache_root == root
        ]
        for entry in candidates:
            if not entry.integrity.intact(entry.dataset):
                self._evict_source(entry)
                continue
            if not _strict_equal(request, entry.dataset.request):
                continue
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
        return None

    def source(self, dataset: "IntradayDataset") -> PreparedCanonicalDataset | None:
        """Return authenticated facts for a content-identical presented source.

        Each retained dataset is first proven intact (a corrupted one is
        evicted); the presented dataset must then be that object or strictly
        equal to its pristine snapshot. None of the presented content (not even
        ``request_id``) is evaluated otherwise, since it may be, or share
        records with, a corrupted retained dataset.
        """
        compared = False
        for entry in tuple(self._sources.values()):
            if not entry.integrity.intact(entry.dataset):
                self._evict_source(entry)
                continue
            if dataset is entry.dataset or entry.integrity.intact(dataset):
                self.counts["source_reuses"] += 1
                return entry
            compared = True
        if compared:
            self.counts["source_rejections"] += 1
        return None

    def _evict_source(self, entry: PreparedCanonicalDataset) -> None:
        self.counts["source_integrity_failures"] += 1
        # Only the admitted key: the retained dataset is corrupted by now.
        self._sources.pop(entry.key, None)
        for key in [key for key in self._derivations if key[:2] == entry.key]:
            del self._derivations[key]

    # -- derived artifacts -------------------------------------------------

    def retain_validated(self, dataset: _DerivedDataset, dataset_id: str) -> None:
        """Record a derived dataset that just passed its reference ``validate``."""
        integrity = None if type(dataset_id) is not str else self._integrity(dataset)
        if integrity is None:
            self.counts["derived_not_admitted"] += 1
            return
        variants = self._validated.setdefault((type(dataset), dataset_id), [])
        variants[:] = [entry for entry in variants if entry.dataset is not dataset]
        variants.append(_AuthenticatedDerived(dataset, integrity))
        del variants[:-_VARIANT_LIMIT]
        self.counts["derived_validations"] += 1

    def validated(self, dataset: _DerivedDataset, dataset_id: str) -> bool:
        """Whether a strictly identical presentation already passed ``validate``.

        Each retained variant is checked against its own pristine snapshot first;
        a changed one is dropped. Variants keep a presentation decoded from
        evidence (plain datetimes) from evicting the calendar-built original.
        """
        if type(dataset_id) is not str:
            return False  # Unreviewed key: never hashed, never retained.
        key = (type(dataset), dataset_id)
        variants = self._validated.get(key)
        if not variants:
            return False
        intact = [entry for entry in variants if entry.intact()]
        if len(intact) != len(variants):
            self.counts["derived_integrity_failures"] += len(variants) - len(intact)
            if intact:
                self._validated[key] = intact
            else:
                del self._validated[key]
        for entry in intact:
            if dataset is entry.dataset or entry.matches(dataset):
                self.counts["derived_validation_reuses"] += 1
                return True
        if intact:
            self.counts["derived_rejections"] += 1
        return False

    def _validated_entries(self) -> tuple[_AuthenticatedDerived, ...]:
        return tuple(
            entry for variants in self._validated.values() for entry in variants
        )

    def retain_derivation(
        self,
        source: PreparedCanonicalDataset,
        target_timeframe_id: str,
        policy_id: str,
        dataset: _DerivedDataset,
    ) -> None:
        """Record the exact output of deriving an authenticated source once."""
        key = (*source.key, type(dataset).__name__, target_timeframe_id, policy_id)
        integrity = self._integrity(dataset) if _exact_text(key) else None
        if integrity is None:
            return
        self._derivations[key] = _AuthenticatedDerived(dataset, integrity)
        self.counts["derivations"] += 1

    def _integrity(self, dataset: _DerivedDataset) -> _ContentIntegrity | None:
        """Share one pristine snapshot per retained derived object (memory)."""
        for entry in (*self._derivations.values(), *self._validated_entries()):
            if entry.dataset is dataset and entry.intact():
                return entry.integrity
        return _ContentIntegrity.capture(dataset)

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
        entry = self._derivations.get(key) if _exact_text(key) else None
        if entry is None:
            return False
        if not entry.intact():
            self.counts["derived_integrity_failures"] += 1
            del self._derivations[key]
            return False
        if dataset is not entry.dataset and not entry.matches(dataset):
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
            "session_resolution_declines": self.sessions.sessions_declined,
            "timeframe_identity_computations": self.sessions.identities_computed,
            "timeframe_identity_reuses": self.sessions.identities_reused,
            "timeframe_identity_declines": self.sessions.identities_declined,
            "retained_sources": len(self._sources),
            "retained_validated_derived": sum(map(len, self._validated.values())),
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
                getsizeof(integrity.field_values)
                + getsizeof(integrity.bar_values)
                + sum(map(getsizeof, integrity.bar_values))
                + sum(
                    getsizeof(owned) + getsizeof(values) + sum(map(getsizeof, values))
                    for _, _, owned, values in integrity.records
                )
            )

        derived = {
            id(entry.integrity): entry
            for entry in (*self._validated_entries(), *self._derivations.values())
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
                getsizeof(windows) + sum(map(getsizeof, windows)) + getsizeof(state)
                for windows, state in self._windows.values()
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
