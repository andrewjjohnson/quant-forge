"""QF-65 preparation mechanics: scoped memos, content integrity and lifecycle."""

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

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
    ExchangeSessionPolicy,
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
