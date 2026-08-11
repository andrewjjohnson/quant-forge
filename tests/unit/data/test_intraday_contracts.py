import json
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import get_type_hints
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import configuration_identity
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    DatasetFamilyValidationError,
    FeedCoverage,
    FeedScope,
    IntradayBar,
    IntradayBarProvenance,
    IntradayBarRequest,
    IntradayContractValidationError,
    IntradayProviderCapabilities,
    UnsupportedDateRangeError,
    UnsupportedFeedError,
    UnsupportedIntervalError,
    UnsupportedSessionScopeError,
)
from quantforge.data.models import SCHEMA_VERSION, DailyBar, ProviderResponse
from quantforge.data.providers import DailyBarProvider, IntradayBarProvider
from quantforge.timeframes import (
    BarCompletion,
    ExchangeSessionPolicy,
    IntradayInterval,
    SessionInterval,
    SessionScope,
    Timeframe,
)

NEW_YORK = ZoneInfo("America/New_York")


def _raw_adjustment_basis() -> AdjustmentBasis:
    return AdjustmentBasis(
        AdjustmentMode.UNADJUSTED,
        "raw_provider",
        "raw_provider",
        "separate_provider_reported_cash_dividends_and_splits",
        False,
    )


def _split_adjusted_basis() -> AdjustmentBasis:
    return AdjustmentBasis(
        AdjustmentMode.SPLIT_ADJUSTED,
        "split_adjusted",
        "split_adjusted",
        "separate_provider_reported_cash_dividends_and_splits",
        True,
    )


def _timeframe(
    duration: timedelta = timedelta(hours=1),
    *,
    session_scope: SessionScope = SessionScope.REGULAR_HOURS,
) -> Timeframe:
    session_policy = (
        ExchangeSessionPolicy()
        if session_scope is SessionScope.REGULAR_HOURS
        else ExchangeSessionPolicy(
            scope=SessionScope.EXTENDED_HOURS,
            extended_hours_start=time(4),
            extended_hours_end=time(20),
        )
    )
    return Timeframe(
        IntradayInterval(duration),
        session_policy=session_policy,
    )


def _request(
    *,
    timeframe: Timeframe | None = None,
    feed_scope: FeedScope | None = None,
    adjustment_basis: AdjustmentBasis | None = None,
    start_timestamp: datetime | None = None,
    end_timestamp: datetime | None = None,
) -> IntradayBarRequest:
    return IntradayBarRequest(
        "SPY",
        start_timestamp or datetime(2024, 7, 1, 9, 30, tzinfo=NEW_YORK),
        end_timestamp or datetime(2024, 7, 1, 16, tzinfo=NEW_YORK),
        timeframe or _timeframe(),
        feed_scope or FeedScope.consolidated(),
        adjustment_basis or _raw_adjustment_basis(),
    )


def _provenance(
    request: IntradayBarRequest,
    *,
    retrieved_at: datetime = datetime(2024, 7, 1, 22, tzinfo=UTC),
    source_snapshot_id: str = "source-snapshot-1",
) -> IntradayBarProvenance:
    return IntradayBarProvenance(
        provider_name="fixture",
        provider_symbol="SPY",
        adapter_version="test-1",
        retrieved_at=retrieved_at,
        source_request_id=request.request_id,
        source_snapshot_id=source_snapshot_id,
        feed_scope=request.feed_scope,
        adjustment_basis=request.adjustment_basis,
    )


def _bar(
    *,
    timeframe: Timeframe | None = None,
    start_timestamp: datetime | None = None,
    end_timestamp: datetime | None = None,
    completion: BarCompletion = BarCompletion.COMPLETED,
    high: Decimal = Decimal("102"),
    low: Decimal = Decimal("99"),
    volume: Decimal = Decimal("1000"),
    provenance: IntradayBarProvenance | None = None,
) -> IntradayBar:
    selected_timeframe = timeframe or _timeframe()
    request = _request(timeframe=selected_timeframe)
    return IntradayBar(
        symbol="spy",
        session_date=date(2024, 7, 1),
        start_timestamp=start_timestamp or datetime(2024, 7, 1, 9, 30, tzinfo=NEW_YORK),
        end_timestamp=end_timestamp or datetime(2024, 7, 1, 10, 30, tzinfo=NEW_YORK),
        timeframe=selected_timeframe,
        completion=completion,
        open=Decimal("100"),
        high=high,
        low=low,
        close=Decimal("101"),
        volume=volume,
        provenance=provenance or _provenance(request),
    )


def test_feed_scope_distinguishes_iex_consolidated_and_unknown() -> None:
    iex = FeedScope.iex_only()
    consolidated = FeedScope.consolidated()
    unknown = FeedScope.unknown()

    assert iex.coverage is FeedCoverage.SINGLE_VENUE
    assert iex.market_center == "IEX"
    assert consolidated.to_primitive() == {
        "coverage": "consolidated",
        "market_center": None,
        "provider_scope": None,
    }
    assert unknown.to_primitive() == {
        "coverage": "unknown",
        "market_center": None,
        "provider_scope": None,
    }
    assert len({iex, consolidated, unknown}) == 3

    with pytest.raises(DatasetFamilyValidationError, match="unknown feed"):
        FeedScope(FeedCoverage.UNKNOWN, market_center="IEX")


def test_intraday_request_is_typed_serializable_and_deterministic() -> None:
    request = _request()
    equivalent = _request(
        start_timestamp=datetime(2024, 7, 1, 13, 30, tzinfo=UTC),
        end_timestamp=datetime(2024, 7, 1, 20, tzinfo=UTC),
    )

    assert request.symbol == "SPY"
    assert request.start_timestamp == datetime(2024, 7, 1, 13, 30, tzinfo=UTC)
    assert request.end_timestamp == datetime(2024, 7, 1, 20, tzinfo=UTC)
    assert request.source_interval == IntradayInterval(timedelta(hours=1))
    assert json.loads(request.serialize()) == request.to_primitive()
    assert request.request_id == configuration_identity(request.to_primitive())
    assert request.request_id == equivalent.request_id


def test_request_identity_changes_with_every_material_intraday_policy() -> None:
    baseline = _request()
    variants = (
        _request(timeframe=_timeframe(timedelta(minutes=30))),
        _request(timeframe=_timeframe(session_scope=SessionScope.EXTENDED_HOURS)),
        _request(feed_scope=FeedScope.iex_only()),
        _request(feed_scope=FeedScope.unknown()),
        _request(adjustment_basis=_split_adjusted_basis()),
        _request(start_timestamp=datetime(2024, 7, 1, 10, tzinfo=NEW_YORK)),
        _request(end_timestamp=datetime(2024, 7, 1, 15, tzinfo=NEW_YORK)),
    )

    identities = {baseline.request_id, *(variant.request_id for variant in variants)}
    assert len(identities) == len(variants) + 1


def test_intraday_request_rejects_naive_ordered_or_nonintraday_boundaries() -> None:
    with pytest.raises(IntradayContractValidationError, match="timezone-aware"):
        _request(start_timestamp=datetime(2024, 7, 1, 9, 30))
    with pytest.raises(IntradayContractValidationError, match="earlier than end"):
        _request(
            start_timestamp=datetime(2024, 7, 1, 16, tzinfo=NEW_YORK),
            end_timestamp=datetime(2024, 7, 1, 9, 30, tzinfo=NEW_YORK),
        )
    with pytest.raises(IntradayContractValidationError, match="intraday timeframe"):
        _request(timeframe=Timeframe.us_equity(SessionInterval()))


def test_provider_capabilities_are_serializable_and_order_independent() -> None:
    start = datetime(2024, 1, 1, tzinfo=NEW_YORK)
    end = datetime(2025, 1, 1, tzinfo=NEW_YORK)
    first = IntradayProviderCapabilities(
        "fixture",
        (
            IntradayInterval(timedelta(hours=1)),
            IntradayInterval(timedelta(minutes=5)),
        ),
        (FeedScope.unknown(), FeedScope.consolidated(), FeedScope.iex_only()),
        (SessionScope.REGULAR_HOURS, SessionScope.EXTENDED_HOURS),
        start,
        end,
    )
    second = IntradayProviderCapabilities(
        "fixture",
        tuple(reversed(first.supported_intervals)),
        tuple(reversed(first.supported_feed_scopes)),
        tuple(reversed(first.supported_session_scopes)),
        start.astimezone(UTC),
        end.astimezone(UTC),
    )

    assert first == second
    assert first.configuration_id == second.configuration_id
    assert first.serialize() == second.serialize()
    assert json.loads(first.serialize()) == first.to_primitive()
    assert first.available_start_timestamp == start.astimezone(UTC)


def test_provider_capabilities_raise_precise_unsupported_request_errors() -> None:
    capabilities = IntradayProviderCapabilities(
        "fixture",
        (IntradayInterval(timedelta(hours=1)),),
        (FeedScope.consolidated(),),
        (SessionScope.REGULAR_HOURS,),
        datetime(2024, 7, 1, tzinfo=UTC),
        datetime(2024, 7, 3, tzinfo=UTC),
    )
    capabilities.validate_request(_request())

    with pytest.raises(UnsupportedIntervalError):
        capabilities.validate_request(
            _request(timeframe=_timeframe(timedelta(minutes=5)))
        )
    with pytest.raises(UnsupportedFeedError):
        capabilities.validate_request(_request(feed_scope=FeedScope.iex_only()))
    with pytest.raises(UnsupportedSessionScopeError):
        capabilities.validate_request(
            _request(timeframe=_timeframe(session_scope=SessionScope.EXTENDED_HOURS))
        )
    with pytest.raises(UnsupportedDateRangeError):
        capabilities.validate_request(
            _request(start_timestamp=datetime(2024, 6, 30, 9, 30, tzinfo=NEW_YORK))
        )


def test_provider_capabilities_reject_invalid_metadata() -> None:
    interval = IntradayInterval(timedelta(minutes=5))
    with pytest.raises(IntradayContractValidationError, match="unique"):
        IntradayProviderCapabilities(
            "fixture",
            (interval, interval),
            (FeedScope.consolidated(),),
            (SessionScope.REGULAR_HOURS,),
        )
    with pytest.raises(IntradayContractValidationError, match="timezone-aware"):
        IntradayProviderCapabilities(
            "fixture",
            (interval,),
            (FeedScope.consolidated(),),
            (SessionScope.REGULAR_HOURS,),
            datetime(2024, 1, 1),
        )
    with pytest.raises(IntradayContractValidationError, match="earlier than end"):
        IntradayProviderCapabilities(
            "fixture",
            (interval,),
            (FeedScope.consolidated(),),
            (SessionScope.REGULAR_HOURS,),
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        )


def test_canonical_intraday_bar_validates_time_and_serializes_identity() -> None:
    bar = _bar()
    equivalent = replace(
        bar,
        start_timestamp=datetime(2024, 7, 1, 13, 30, tzinfo=UTC),
        end_timestamp=datetime(2024, 7, 1, 14, 30, tzinfo=UTC),
    )

    assert bar.symbol == "SPY"
    assert bar.start_timestamp == datetime(2024, 7, 1, 13, 30, tzinfo=UTC)
    assert bar.end_timestamp == datetime(2024, 7, 1, 14, 30, tzinfo=UTC)
    assert bar.nominal_duration == bar.actual_duration == timedelta(hours=1)
    assert bar.session_identifier == "2024-07-01"
    assert json.loads(bar.serialize()) == bar.to_primitive()
    assert bar.bar_id == configuration_identity(bar.to_primitive())
    assert bar.bar_id == equivalent.bar_id


def test_bar_identity_binds_values_and_provider_neutral_provenance() -> None:
    bar = _bar()
    changed_close = replace(bar, close=Decimal("101.5"))
    changed_source = replace(
        bar,
        provenance=replace(bar.provenance, source_snapshot_id="source-snapshot-2"),
    )

    assert len({bar.bar_id, changed_close.bar_id, changed_source.bar_id}) == 3


def test_bar_rejects_naive_invalid_duration_and_impossible_provenance_time() -> None:
    with pytest.raises(IntradayContractValidationError, match="timezone-aware"):
        _bar(start_timestamp=datetime(2024, 7, 1, 9, 30))
    with pytest.raises(IntradayContractValidationError, match="nominal duration"):
        _bar(end_timestamp=datetime(2024, 7, 1, 10, tzinfo=NEW_YORK))
    with pytest.raises(IntradayContractValidationError, match="earlier than bar end"):
        _bar(end_timestamp=datetime(2024, 7, 1, 9, 30, tzinfo=NEW_YORK))

    request = _request()
    with pytest.raises(IntradayContractValidationError, match="cannot precede"):
        _bar(
            provenance=_provenance(
                request,
                retrieved_at=datetime(2024, 7, 1, 14, tzinfo=UTC),
            )
        )


def test_bar_preserves_completed_partial_actual_duration() -> None:
    timeframe = _timeframe(timedelta(hours=4))
    bar = _bar(
        timeframe=timeframe,
        start_timestamp=datetime(2024, 7, 1, 13, 30, tzinfo=NEW_YORK),
        end_timestamp=datetime(2024, 7, 1, 16, tzinfo=NEW_YORK),
        completion=BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL,
    )

    assert bar.nominal_duration == timedelta(hours=4)
    assert bar.actual_duration == timedelta(hours=2, minutes=30)
    assert bar.to_primitive()["actual_duration_microseconds"] == 9_000_000_000


@pytest.mark.parametrize(
    ("high", "low", "volume", "message"),
    [
        (Decimal("100.5"), Decimal("99"), Decimal("1000"), "high"),
        (Decimal("102"), Decimal("101.5"), Decimal("1000"), "low"),
        (Decimal("102"), Decimal("99"), Decimal("-1"), "nonnegative"),
        (Decimal("NaN"), Decimal("99"), Decimal("1000"), "finite"),
    ],
)
def test_bar_rejects_invalid_ohlcv(
    high: Decimal,
    low: Decimal,
    volume: Decimal,
    message: str,
) -> None:
    with pytest.raises(IntradayContractValidationError, match=message):
        _bar(high=high, low=low, volume=volume)


def test_intraday_adapter_boundary_is_canonical_and_daily_contract_is_unchanged() -> (
    None
):
    intraday_hints = get_type_hints(IntradayBarProvider.fetch_intraday_bars)
    daily_hints = get_type_hints(DailyBarProvider.fetch_daily_bars)
    daily_bar = DailyBar(
        "SPY",
        date(2024, 7, 1),
        Decimal("100"),
        Decimal("102"),
        Decimal("99"),
        Decimal("101"),
        Decimal("1000"),
    )

    assert intraday_hints["request"] is IntradayBarRequest
    assert intraday_hints["return"] == tuple[IntradayBar, ...]
    assert daily_hints["return"] is ProviderResponse
    assert SCHEMA_VERSION == "4"
    assert daily_bar.session_date == date(2024, 7, 1)
