"""QF-65 preparation mechanics: scoped memos, content integrity and lifecycle."""

from dataclasses import dataclass, fields, replace
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import configuration_identity
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    FeedScope,
    IntradayBar,
    IntradayBarProvenance,
    IntradayBarRequest,
    MarketDataset,
    intraday_session_windows,
)
from quantforge.data.prepared_canonical import (
    _ContentIntegrity,  # pyright: ignore[reportPrivateUsage]
    active_canonical_preparation,
    canonical_preparation,
)
from quantforge.timeframes import (
    BarCompletion,
    ExchangeSessionPolicy,
    IntervalKind,
    IntradayAnchor,
    IntradayBarWindow,
    IntradayInterval,
    SessionScope,
    Timeframe,
    TimeframeMemo,
    TimeframeValidationError,
    resolve_exchange_session,
    timeframe_memo,
)
from tests.unit.helpers import make_dataset

ONE_MINUTE = Timeframe.us_equity(IntradayInterval(timedelta(minutes=1)))
EXTENDED = ExchangeSessionPolicy(
    scope=SessionScope.EXTENDED_HOURS,
    extended_hours_start=time(4),
    extended_hours_end=time(20),
)
REGULAR = ExchangeSessionPolicy()
NORMAL, EARLY_CLOSE, BEFORE_DST, AFTER_DST = (
    date(2024, 7, 5),
    date(2024, 7, 3),
    date(2024, 3, 8),
    date(2024, 3, 11),
)


def test_session_memo_returns_reference_sessions_once_per_value() -> None:
    dates = (NORMAL, EARLY_CLOSE, BEFORE_DST, AFTER_DST)
    reference = [resolve_exchange_session(value) for value in dates]
    memo = TimeframeMemo()
    with timeframe_memo(memo):
        first = [resolve_exchange_session(value) for value in dates]
        # An equal policy value, not the same object, shares the resolution.
        again = [
            resolve_exchange_session(value, ExchangeSessionPolicy()) for value in dates
        ]
        extended = resolve_exchange_session(NORMAL, EXTENDED)
    assert first == again == reference
    assert (memo.resolved, memo.reused) == (5, 4)
    # Early close and DST keep exact exchange boundaries.
    assert reference[1].close_timestamp == datetime(2024, 7, 3, 17, tzinfo=UTC)
    assert reference[2].open_timestamp == datetime(2024, 3, 8, 14, 30, tzinfo=UTC)
    assert reference[3].open_timestamp == datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
    assert extended != reference[0]
    assert extended == resolve_exchange_session(NORMAL, EXTENDED)


def test_session_memo_never_retains_or_masks_invalid_sessions() -> None:
    memo = TimeframeMemo()
    with timeframe_memo(memo):
        for _ in range(2):
            with pytest.raises(TimeframeValidationError, match="not an exchange"):
                resolve_exchange_session(date(2024, 7, 4))  # Independence Day.
        with pytest.raises(TimeframeValidationError, match="must belong"):
            # A bar outside its declared session still fails with the memo.
            IntradayBarWindow(
                ONE_MINUTE,
                NORMAL,
                datetime(2024, 7, 5, 12, tzinfo=UTC),
                datetime(2024, 7, 5, 12, 1, tzinfo=UTC),
                intraday_session_windows(NORMAL, ONE_MINUTE)[0].completion,
            )
    assert len(memo) == 1
    assert memo.resolved == 1


def test_timeframe_identity_memo_is_value_keyed_and_exact() -> None:
    two_minutes = Timeframe.us_equity(IntradayInterval(timedelta(minutes=2)))
    reference = configuration_identity(ONE_MINUTE.to_primitive())
    memo = TimeframeMemo()
    with timeframe_memo(memo):
        equal = Timeframe.us_equity(IntradayInterval(timedelta(minutes=1)))
        assert equal is not ONE_MINUTE
        assert ONE_MINUTE.configuration_id == equal.configuration_id == reference
        assert two_minutes.configuration_id == configuration_identity(
            two_minutes.to_primitive()
        )
    assert (memo.identities_computed, memo.identities_reused) == (2, 1)


def test_memos_are_scoped_released_and_absent_outside_a_session() -> None:
    assert active_canonical_preparation() is None
    with canonical_preparation() as preparation:
        resolve_exchange_session(NORMAL)
        assert ONE_MINUTE.configuration_id
        intraday_session_windows(NORMAL, ONE_MINUTE)
        intraday_session_windows(NORMAL, ONE_MINUTE)
        statistics = preparation.statistics()
    assert statistics["session_resolutions"] == 1
    assert statistics["timeframe_identity_computations"] == 1
    assert statistics["session_window_resolutions"] == 1
    assert statistics["session_window_reuses"] == 1
    assert len(preparation.sessions) == 0
    assert active_canonical_preparation() is None
    # Outside any session the reference path resolves independently.
    resolve_exchange_session(NORMAL)
    assert preparation.statistics()["session_resolutions"] == 1


def _bar(minute: int, provenance: IntradayBarProvenance) -> IntradayBar:
    window = intraday_session_windows(NORMAL, ONE_MINUTE)[minute]
    return IntradayBar(
        "SPY",
        NORMAL,
        window.start_timestamp,
        window.end_timestamp,
        ONE_MINUTE,
        window.completion,
        Decimal(100),
        Decimal(101),
        Decimal(99),
        Decimal(100),
        Decimal(10),
        provenance,
    )


def _provenance() -> IntradayBarProvenance:
    return IntradayBarProvenance(
        "fixture",
        "SPY",
        "v1",
        datetime(2024, 7, 6, tzinfo=UTC),
        "request",
        "snapshot",
        FeedScope.consolidated(),
        AdjustmentBasis(
            AdjustmentMode.UNADJUSTED,
            "raw_provider",
            "raw_provider",
            "not_provided_for_intraday_bars",
            False,
        ),
    )


@dataclass(frozen=True)
class _Window:
    output_bar_id: str


@dataclass(frozen=True)
class _Metadata:
    dataset_id: str
    bar_count: int
    windows: tuple[_Window, ...]


@dataclass(frozen=True)
class _Dataset:
    request: IntradayBarRequest
    bars: tuple[IntradayBar, ...]
    metadata: _Metadata


def _request() -> IntradayBarRequest:
    return IntradayBarRequest(
        "SPY",
        datetime(2024, 7, 5, 13, 30, tzinfo=UTC),
        datetime(2024, 7, 5, 20, tzinfo=UTC),
        ONE_MINUTE,
        FeedScope.consolidated(),
        _provenance().adjustment_basis,
    )


def _dataset(provenance: IntradayBarProvenance) -> _Dataset:
    return _Dataset(
        _request(),
        tuple(_bar(minute, provenance) for minute in range(3)),
        _Metadata("dataset", 3, (_Window("a"), _Window("b"))),
    )


def test_content_integrity_accepts_equal_content_and_detects_bypass() -> None:
    provenance = _provenance()
    dataset = _dataset(provenance)
    integrity = _ContentIntegrity.capture(dataset)
    assert integrity is not None
    assert integrity.intact(dataset)
    assert integrity.intact(_dataset(_provenance()))  # Equal values, new objects.
    bars = dataset.bars
    for presented in (
        replace(dataset, bars=bars[:2]),
        replace(dataset, bars=(bars[0], bars[2], bars[1])),
        replace(dataset, metadata=replace(dataset.metadata, bar_count=4)),
        replace(dataset, request=replace(dataset.request, symbol="QQQ")),
    ):
        assert not integrity.intact(presented)
    object.__setattr__(bars[1], "close", Decimal("100.5"))
    assert not integrity.intact(dataset)
    object.__setattr__(bars[1], "close", Decimal(100))
    assert integrity.intact(dataset)
    # Nested shared records are snapshotted too: provenance and its feed scope.
    object.__setattr__(provenance, "source_snapshot_id", "other")
    assert not integrity.intact(dataset)
    object.__setattr__(provenance, "source_snapshot_id", "snapshot")
    object.__setattr__(provenance.feed_scope, "provider_scope", "altered")
    assert not integrity.intact(dataset)


@pytest.mark.parametrize("target", ["metadata", "window", "request", "container"])
def test_content_integrity_detects_metadata_and_request_bypass(target: str) -> None:
    dataset = _dataset(_provenance())
    integrity = _ContentIntegrity.capture(dataset)
    assert integrity is not None
    # The same retained object: self-comparison alone would miss all of these.
    if target == "metadata":
        object.__setattr__(dataset.metadata, "bar_count", 4)
    elif target == "window":
        object.__setattr__(dataset.metadata.windows[1], "output_bar_id", "c")
    elif target == "request":
        object.__setattr__(dataset.request, "symbol", "QQQ")
    else:
        object.__setattr__(
            dataset, "metadata", replace(dataset.metadata, dataset_id="other")
        )
    assert not integrity.intact(dataset)


@pytest.mark.parametrize("target", ["bar_price", "bar_zone", "count", "presented"])
def test_content_integrity_rejects_equal_valued_type_or_zone_swaps(target: str) -> None:
    dataset = _dataset(_provenance())
    integrity = _ContentIntegrity.capture(dataset)
    assert integrity is not None
    bar = dataset.bars[1]
    if target == "bar_price":
        assert bar.close == 100
        object.__setattr__(bar, "close", 100)  # int, equal to Decimal(100)
    elif target == "bar_zone":
        local = bar.start_timestamp.astimezone(ZoneInfo("America/New_York"))
        assert local == bar.start_timestamp
        object.__setattr__(bar, "start_timestamp", local)  # same instant
    elif target == "count":
        object.__setattr__(dataset.metadata, "bar_count", Decimal(3))
    else:
        # A different presented object is compared strictly, not by ``==``.
        dataset = replace(
            dataset, metadata=replace(dataset.metadata, bar_count=Decimal(3))
        )
        assert dataset.metadata.bar_count == 3
    assert not integrity.intact(dataset)


def _shadow[T](record: object, kind: type[T]) -> T:
    """An instance of ``kind`` holding ``record``'s exact field objects."""
    shadow = object.__new__(kind)
    for item in fields(cast(Any, kind)):
        object.__setattr__(shadow, item.name, getattr(record, item.name))
    return shadow


@dataclass(frozen=True, slots=True)
class _ShadowBar:  # IntradayBar's exact field layout, another class.
    symbol: str
    session_date: date
    start_timestamp: datetime
    end_timestamp: datetime
    timeframe: Timeframe
    completion: BarCompletion
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    provenance: IntradayBarProvenance
    schema_version: str


@dataclass(frozen=True, slots=True)
class _ShadowProvenance:  # IntradayBarProvenance's field layout.
    provider_name: str
    provider_symbol: str
    adapter_version: str
    retrieved_at: datetime
    source_request_id: str
    source_snapshot_id: str
    feed_scope: FeedScope
    adjustment_basis: AdjustmentBasis


@pytest.mark.parametrize("target", ["presented_bars", "bar_class", "record_class"])
def test_content_integrity_checks_exact_bar_and_record_classes(target: str) -> None:
    provenance = _provenance()
    dataset = _dataset(provenance)
    integrity = _ContentIntegrity.capture(dataset)
    assert integrity is not None
    if target == "presented_bars":
        # Same field objects, another class: equal rows, not the same bars.
        dataset = replace(
            dataset, bars=tuple(_shadow(bar, _ShadowBar) for bar in dataset.bars)
        )
        assert integrity.intact(dataset) is False
        return
    record = dataset.bars[1] if target == "bar_class" else provenance
    shadow = _ShadowBar if target == "bar_class" else _ShadowProvenance
    object.__setattr__(record, "__class__", shadow)
    assert type(record) is shadow
    assert not integrity.intact(dataset)


@pytest.mark.parametrize("reached_by", ["field", "property"])
def test_content_integrity_detects_mutated_enum_members(reached_by: str) -> None:
    dataset = _dataset(_provenance())
    integrity = _ContentIntegrity.capture(dataset)
    assert integrity is not None
    # BarCompletion is a bar field; IntervalKind reaches serialization only via
    # the interval's ``kind`` property, never through a field.
    member = (
        dataset.bars[0].completion
        if reached_by == "field"
        else ONE_MINUTE.interval.kind
    )
    assert member in (BarCompletion.COMPLETED, IntervalKind.INTRADAY)
    original = member._value_
    try:
        # Every snapshot still holds this singleton; only its value changed.
        object.__setattr__(member, "_value_", "altered")
        assert not integrity.intact(dataset)
    finally:
        object.__setattr__(member, "_value_", original)
    assert integrity.intact(dataset)


class _Text(str):
    pass


def test_content_integrity_refuses_scalar_subclasses() -> None:
    dataset = _dataset(_provenance())
    altered = replace(
        dataset, metadata=replace(dataset.metadata, dataset_id=_Text("dataset"))
    )
    assert _ContentIntegrity.capture(dataset) is not None
    assert _ContentIntegrity.capture(altered) is None


class _ShiftingZone(tzinfo):
    """A tzinfo whose offset rule can change behind an unchanged datetime."""

    def __init__(self) -> None:
        self.offset = timedelta(0)

    def utcoffset(self, dt: datetime | None) -> timedelta:
        return self.offset

    def dst(self, dt: datetime | None) -> timedelta:
        return timedelta(0)


class _UnhashableZone(tzinfo):
    """An equality-defining zone without ``__hash__`` (so it is unhashable)."""

    def utcoffset(self, dt: datetime | None) -> timedelta:
        return timedelta(0)

    def dst(self, dt: datetime | None) -> timedelta:
        return timedelta(0)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _UnhashableZone)


@pytest.mark.parametrize("zone", [_ShiftingZone, _UnhashableZone])
@pytest.mark.parametrize("target", ["bar", "provenance"])
def test_content_integrity_refuses_unreviewed_time_zones(
    target: str, zone: type[tzinfo]
) -> None:
    provenance = _provenance()
    dataset = _dataset(provenance)
    assert _ContentIntegrity.capture(dataset) is not None
    # Domain constructors normalize to UTC; a record can still hold such a zone
    # (bypass or custom producer). Its offset could later change in place.
    record: object = dataset.bars[0] if target == "bar" else provenance
    name = "start_timestamp" if target == "bar" else "retrieved_at"
    moment = cast(datetime, getattr(record, name))
    object.__setattr__(record, name, moment.replace(tzinfo=zone()))
    # Rejected by type before any zone is hashed (never a TypeError).
    assert _ContentIntegrity.capture(dataset) is None


@pytest.mark.parametrize("member_name", ["anchor", "kind"])
def test_timeframe_identity_memo_rechecks_enum_state(member_name: str) -> None:
    reference = configuration_identity(ONE_MINUTE.to_primitive())
    memo = TimeframeMemo()
    member = getattr(cast(IntradayInterval, ONE_MINUTE.interval), member_name)
    original = member._value_
    with timeframe_memo(memo):
        assert ONE_MINUTE.configuration_id == reference
        try:
            object.__setattr__(member, "_value_", "altered")
            fresh = configuration_identity(ONE_MINUTE.to_primitive())
            assert fresh != reference
            assert ONE_MINUTE.configuration_id == fresh  # Never the stale identity.
        finally:
            object.__setattr__(member, "_value_", original)
        assert ONE_MINUTE.configuration_id == reference
    assert memo.identities_computed == 3


class _ShiftedTime(time):
    """Equal and hash-identical to its value, but serializes differently."""

    def isoformat(self, timespec: str = "auto") -> str:
        return "00:00:00.000000"


class _UnhashableText(str):
    __hash__ = None  # pyright: ignore[reportAssignmentType]


def _clock_timeframe() -> Timeframe:
    return Timeframe(
        IntradayInterval(
            timedelta(minutes=5), IntradayAnchor.CLOCK, clock_anchor=time(9, 30)
        ),
        EXTENDED,
    )


@pytest.mark.parametrize("leaf", ["clock_anchor", "extended_hours_end"])
def test_timeframe_identity_memo_declines_unreviewed_leaf_types(leaf: str) -> None:
    timeframe = _clock_timeframe()
    reference = configuration_identity(timeframe.to_primitive())
    record = timeframe.interval if leaf == "clock_anchor" else timeframe.session_policy
    original = cast(time, getattr(record, leaf))
    memo = TimeframeMemo()
    with timeframe_memo(memo):
        assert timeframe.configuration_id == reference
        # Equal and hash-identical: dictionary lookup alone would reuse the
        # cached identity although serialization now differs.
        shifted = _ShiftedTime(original.hour, original.minute)
        assert shifted == original
        assert hash(shifted) == hash(original)
        object.__setattr__(record, leaf, shifted)
        fresh = configuration_identity(timeframe.to_primitive())
        assert fresh != reference
        assert timeframe.configuration_id == fresh  # Never the stale identity.
        object.__setattr__(record, leaf, original)
        assert timeframe.configuration_id == reference
    assert (memo.identities_computed, memo.identities_reused) == (1, 1)
    assert memo.identities_declined == 1


def test_timeframe_memos_never_hash_unreviewed_leaves() -> None:
    policy = ExchangeSessionPolicy()
    timeframe = Timeframe.us_equity(IntradayInterval(timedelta(minutes=2)))
    reference_id = configuration_identity(timeframe.to_primitive())
    reference_session = resolve_exchange_session(NORMAL)
    reference_windows = intraday_session_windows(NORMAL, timeframe)
    # Regular-hours resolution never reads the zone name, so the reference path
    # succeeds; a memo that hashed its key first would raise TypeError instead.
    for record in (policy, timeframe.session_policy):
        object.__setattr__(record, "timezone_name", _UnhashableText("America/New_York"))
    with pytest.raises(TypeError):
        hash(timeframe)
    assert configuration_identity(timeframe.to_primitive()) == reference_id
    with canonical_preparation() as preparation:
        for _ in range(2):
            assert timeframe.configuration_id == reference_id
            assert resolve_exchange_session(NORMAL, policy) == reference_session
            assert intraday_session_windows(NORMAL, timeframe) == reference_windows
        statistics = preparation.statistics()
    assert statistics["timeframe_identity_declines"] == 2
    assert statistics["session_resolution_declines"] >= 2
    assert statistics["session_resolutions"] == 0
    assert statistics["session_window_declines"] == 2
    assert statistics["timeframe_identity_computations"] == 0
    assert statistics["session_window_resolutions"] == 0


def test_session_window_memo_rechecks_enum_state() -> None:
    timeframe = Timeframe.us_equity(IntradayInterval(timedelta(minutes=2)))
    member = cast(IntradayInterval, timeframe.interval).anchor
    original = member._value_
    with canonical_preparation() as preparation:
        first = intraday_session_windows(NORMAL, timeframe)
        assert intraday_session_windows(NORMAL, timeframe) is first
        try:
            object.__setattr__(member, "_value_", "altered")
            intraday_session_windows(NORMAL, timeframe)  # Resolved again.
        finally:
            object.__setattr__(member, "_value_", original)
        assert intraday_session_windows(NORMAL, timeframe) == first
        statistics = preparation.statistics()
    assert statistics["session_window_resolutions"] == 3
    assert statistics["session_window_reuses"] == 1


def test_preparation_never_hashes_unreviewed_dataset_ids(tmp_path: Path) -> None:
    dataset = _dataset(_provenance())
    unhashable = _UnhashableText("dataset")
    presented = replace(
        dataset, metadata=replace(dataset.metadata, dataset_id=unhashable)
    )
    derived = cast(Any, presented)
    with canonical_preparation() as preparation:
        assert preparation.cached_source(tmp_path, unhashable, dataset.request) is None
        assert preparation.source(derived) is None
        assert not preparation.validated(derived, unhashable)
        preparation.retain_validated(derived, unhashable)
        assert not preparation.validated(derived, unhashable)
        statistics = preparation.statistics()
    assert statistics["derived_not_admitted"] == 1
    assert statistics["retained_validated_derived"] == 0


@dataclass(frozen=True)
class _Mutable:
    values: list[int]


@dataclass(frozen=True)
class _MutableDataset:
    bars: tuple[IntradayBar, ...]
    metadata: _Mutable


def test_content_integrity_refuses_mutable_or_mixed_graphs() -> None:
    provenance = _provenance()
    bars = (_bar(0, provenance),)
    assert _ContentIntegrity.capture(_MutableDataset(bars, _Mutable([1]))) is None
    mixed = _Dataset(
        _request(),
        (bars[0], provenance),  # pyright: ignore[reportArgumentType]
        _Metadata("dataset", 2, ()),
    )
    assert _ContentIntegrity.capture(mixed) is None
    assert _ContentIntegrity.capture(bars) is None  # Not a dataset record.


def test_non_intraday_datasets_always_use_the_reference_validator() -> None:
    dataset = make_dataset(("100", "101", "102"))
    calls: list[MarketDataset] = []

    def validate(value: MarketDataset) -> tuple[date, ...]:
        calls.append(value)
        return ()

    with canonical_preparation() as preparation:
        for _ in range(2):
            assert preparation.validated_market_dataset(dataset, validate) == ()
        assert preparation.statistics()["market_validations"] == 0
    assert calls == [dataset, dataset]
