import json
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import configuration_identity
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    CacheError,
    FeedScope,
    IntradayBar,
    IntradayBarBatch,
    IntradayBarProvenance,
    IntradayBarRequest,
    IntradayDataset,
    IntradayDatasetMetadata,
    IntradayValidationMode,
    MissingConstituentPolicy,
    SessionAggregationCache,
    SessionAggregationPolicy,
    SessionAggregationQualityError,
    SessionAggregationValidationError,
    aggregate_session_dataset,
    validate_intraday_coverage,
)
from quantforge.data.identity import sha256_hex
from quantforge.timeframes import (
    BarCompletion,
    CrossSessionPolicy,
    ExchangeSessionPolicy,
    IntradayInterval,
    SessionInterval,
    SessionScope,
    Timeframe,
    TradingWeekInterval,
    resolve_exchange_session,
)

FIXTURE = (
    Path(__file__).parents[2]
    / "fixtures"
    / "intraday"
    / "spy_4h_holiday_week_hand_audit.json"
)
NEW_YORK = ZoneInfo("America/New_York")
RETRIEVED_AT = datetime(2025, 1, 1, tzinfo=UTC)


def _adjustment_basis() -> AdjustmentBasis:
    return AdjustmentBasis(
        adjustment_mode=AdjustmentMode.UNADJUSTED,
        ohlc_basis="raw_provider",
        volume_basis="raw_provider",
        corporate_action_policy="not_provided_for_intraday_bars",
        adjusted_fields_used=False,
    )


def _dataset(*, omit_index: int | None = None) -> IntradayDataset:
    records = cast(list[dict[str, str]], json.loads(FIXTURE.read_text()))
    source_timeframe = Timeframe.us_equity(IntradayInterval(timedelta(hours=4)))
    first_session = resolve_exchange_session(
        date.fromisoformat(records[0]["session_date"])
    )
    last_session = resolve_exchange_session(
        date.fromisoformat(records[-1]["session_date"])
    )
    request = IntradayBarRequest(
        symbol="SPY",
        start_timestamp=first_session.open_timestamp,
        end_timestamp=last_session.close_timestamp,
        timeframe=source_timeframe,
        feed_scope=FeedScope.consolidated(),
        adjustment_basis=_adjustment_basis(),
    )
    provenance = IntradayBarProvenance(
        provider_name="fixture-provider",
        provider_symbol="SPY",
        adapter_version="fixture-1",
        retrieved_at=RETRIEVED_AT,
        source_request_id=request.request_id,
        source_snapshot_id="raw-holiday-week",
        feed_scope=request.feed_scope,
        adjustment_basis=request.adjustment_basis,
    )
    bars = tuple(
        IntradayBar(
            symbol="SPY",
            session_date=session_date,
            start_timestamp=datetime.combine(
                session_date, time.fromisoformat(record["start_local"]), NEW_YORK
            ),
            end_timestamp=datetime.combine(
                session_date, time.fromisoformat(record["end_local"]), NEW_YORK
            ),
            timeframe=source_timeframe,
            completion=BarCompletion(record["completion"]),
            open=Decimal(record["open"]),
            high=Decimal(record["high"]),
            low=Decimal(record["low"]),
            close=Decimal(record["close"]),
            volume=Decimal(record["volume"]),
            provenance=provenance,
        )
        for index, record in enumerate(records)
        if index != omit_index
        for session_date in (date.fromisoformat(record["session_date"]),)
    )
    batch = IntradayBarBatch(request, bars)
    report = validate_intraday_coverage(batch, mode=IntradayValidationMode.DIAGNOSTIC)
    dataset_id = configuration_identity(
        {
            "fixture": "holiday-week",
            "request_id": request.request_id,
            "batch_id": batch.batch_id,
        }
    )
    metadata = IntradayDatasetMetadata(
        dataset_id=dataset_id,
        request_id=request.request_id,
        provider_name="fixture-provider",
        provider_symbol="SPY",
        adapter_version="fixture-1",
        retrieved_at=RETRIEVED_AT,
        capabilities_configuration_id="fixture-capabilities",
        batch_id=batch.batch_id,
        bar_count=len(bars),
        raw_snapshot_ids=("raw-holiday-week",),
        raw_locations=("intraday/raw/raw-holiday-week.json",),
        normalized_location=f"intraday/datasets/{dataset_id}/bars.json",
        data_sha256=sha256_hex(batch.serialize()),
        quality_report=report,
    )
    return IntradayDataset(request, bars, metadata)


def _daily() -> Timeframe:
    return Timeframe.us_equity(SessionInterval())


def _weekly() -> Timeframe:
    return Timeframe.us_equity(TradingWeekInterval())


def _extended_dataset() -> tuple[IntradayDataset, ExchangeSessionPolicy]:
    session_policy = ExchangeSessionPolicy(
        scope=SessionScope.EXTENDED_HOURS,
        extended_hours_start=time(4),
        extended_hours_end=time(20),
    )
    source_timeframe = Timeframe(IntradayInterval(timedelta(hours=4)), session_policy)
    session_date = date(2024, 7, 1)
    session = resolve_exchange_session(session_date, session_policy)
    request = IntradayBarRequest(
        symbol="SPY",
        start_timestamp=session.open_timestamp,
        end_timestamp=session.close_timestamp,
        timeframe=source_timeframe,
        feed_scope=FeedScope.consolidated(),
        adjustment_basis=_adjustment_basis(),
    )
    provenance = IntradayBarProvenance(
        provider_name="fixture-provider",
        provider_symbol="SPY",
        adapter_version="fixture-1",
        retrieved_at=RETRIEVED_AT,
        source_request_id=request.request_id,
        source_snapshot_id="raw-extended-hours",
        feed_scope=request.feed_scope,
        adjustment_basis=request.adjustment_basis,
    )
    bars = tuple(
        IntradayBar(
            symbol="SPY",
            session_date=session_date,
            start_timestamp=session.open_timestamp + timedelta(hours=4 * index),
            end_timestamp=session.open_timestamp + timedelta(hours=4 * (index + 1)),
            timeframe=source_timeframe,
            completion=BarCompletion.COMPLETED,
            open=Decimal(100 + index),
            high=Decimal(102 + index),
            low=Decimal(99 + index),
            close=Decimal(101 + index),
            volume=Decimal(1000 + index),
            provenance=provenance,
        )
        for index in range(4)
    )
    batch = IntradayBarBatch(request, bars)
    report = validate_intraday_coverage(batch, mode=IntradayValidationMode.DIAGNOSTIC)
    dataset_id = configuration_identity(
        {
            "fixture": "extended-hours",
            "request_id": request.request_id,
            "batch_id": batch.batch_id,
        }
    )
    metadata = IntradayDatasetMetadata(
        dataset_id=dataset_id,
        request_id=request.request_id,
        provider_name="fixture-provider",
        provider_symbol="SPY",
        adapter_version="fixture-1",
        retrieved_at=RETRIEVED_AT,
        capabilities_configuration_id="fixture-capabilities",
        batch_id=batch.batch_id,
        bar_count=len(bars),
        raw_snapshot_ids=("raw-extended-hours",),
        raw_locations=("intraday/raw/raw-extended-hours.json",),
        normalized_location=f"intraday/datasets/{dataset_id}/bars.json",
        data_sha256=sha256_hex(batch.serialize()),
        quality_report=report,
    )
    return IntradayDataset(request, bars, metadata), session_policy


def _cross_session_dataset() -> IntradayDataset:
    source_timeframe = Timeframe.us_equity(
        IntradayInterval(
            timedelta(hours=4),
            cross_session_policy=CrossSessionPolicy.PERMITTED,
        )
    )
    session_date = date(2024, 7, 1)
    session = resolve_exchange_session(session_date)
    request = IntradayBarRequest(
        symbol="SPY",
        start_timestamp=session.open_timestamp,
        end_timestamp=session.open_timestamp + timedelta(hours=8),
        timeframe=source_timeframe,
        feed_scope=FeedScope.consolidated(),
        adjustment_basis=_adjustment_basis(),
    )
    provenance = IntradayBarProvenance(
        provider_name="fixture-provider",
        provider_symbol="SPY",
        adapter_version="fixture-1",
        retrieved_at=RETRIEVED_AT,
        source_request_id=request.request_id,
        source_snapshot_id="raw-cross-session",
        feed_scope=request.feed_scope,
        adjustment_basis=request.adjustment_basis,
    )
    bars = tuple(
        IntradayBar(
            symbol="SPY",
            session_date=session_date,
            start_timestamp=session.open_timestamp + timedelta(hours=4 * index),
            end_timestamp=session.open_timestamp + timedelta(hours=4 * (index + 1)),
            timeframe=source_timeframe,
            completion=BarCompletion.COMPLETED,
            open=Decimal(100 + index),
            high=Decimal(102 + index),
            low=Decimal(99 + index),
            close=Decimal(101 + index),
            volume=Decimal(1000 + index),
            provenance=provenance,
        )
        for index in range(2)
    )
    batch = IntradayBarBatch(request, bars)
    report = validate_intraday_coverage(batch, mode=IntradayValidationMode.DIAGNOSTIC)
    dataset_id = configuration_identity(
        {
            "fixture": "cross-session",
            "request_id": request.request_id,
            "batch_id": batch.batch_id,
        }
    )
    metadata = IntradayDatasetMetadata(
        dataset_id=dataset_id,
        request_id=request.request_id,
        provider_name="fixture-provider",
        provider_symbol="SPY",
        adapter_version="fixture-1",
        retrieved_at=RETRIEVED_AT,
        capabilities_configuration_id="fixture-capabilities",
        batch_id=batch.batch_id,
        bar_count=len(bars),
        raw_snapshot_ids=("raw-cross-session",),
        raw_locations=("intraday/raw/raw-cross-session.json",),
        normalized_location=f"intraday/datasets/{dataset_id}/bars.json",
        data_sha256=sha256_hex(batch.serialize()),
        quality_report=report,
    )
    return IntradayDataset(request, bars, metadata)


def test_daily_bars_are_one_session_and_hand_auditable() -> None:
    derived = aggregate_session_dataset(_dataset(), _daily())

    assert [bar.session_dates for bar in derived.bars] == [
        (date(2024, 7, 1),),
        (date(2024, 7, 2),),
        (date(2024, 7, 3),),
        (date(2024, 7, 5),),
    ]
    first = derived.bars[0]
    assert (first.open, first.high, first.low, first.close, first.volume) == (
        Decimal("100"),
        Decimal("106"),
        Decimal("99"),
        Decimal("105"),
        Decimal("2500"),
    )
    early_close = derived.bars[2]
    assert early_close.end_timestamp.astimezone(NEW_YORK).time() == time(13)
    assert early_close.completion is BarCompletion.COMPLETED
    assert all(
        len(window.session_dates) == 1 for window in derived.aggregation_report.windows
    )


def test_holiday_shortened_week_is_one_complete_hand_auditable_bar() -> None:
    derived = aggregate_session_dataset(_dataset(), _weekly())

    assert len(derived.bars) == 1
    weekly = derived.bars[0]
    assert weekly.period_start_date == date(2024, 7, 1)
    assert weekly.session_dates == (
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
    )
    assert date(2024, 7, 4) not in weekly.session_dates
    assert (weekly.open, weekly.high, weekly.low, weekly.close, weekly.volume) == (
        Decimal("100"),
        Decimal("116"),
        Decimal("99"),
        Decimal("115"),
        Decimal("17500"),
    )
    assert weekly.end_timestamp.astimezone(NEW_YORK).time() == time(16)
    assert derived.aggregation_report.excluded_partial_periods == ()


def test_source_bars_are_counted_exactly_once() -> None:
    source = _dataset()
    daily = aggregate_session_dataset(source, _daily())
    weekly = aggregate_session_dataset(source, _weekly())

    daily_ids = tuple(
        source_id for bar in daily.bars for source_id in bar.source_bar_ids
    )
    weekly_ids = weekly.bars[0].source_bar_ids
    expected_ids = tuple(bar.bar_id for bar in source.bars)
    assert daily_ids == expected_ids
    assert weekly_ids == expected_ids
    assert len(daily_ids) == len(set(daily_ids))
    with pytest.raises(
        SessionAggregationValidationError, match="cannot be counted twice"
    ):
        replace(daily, bars=(daily.bars[0], daily.bars[0])).validate()


def test_missing_constituents_fail_strictly_and_remain_disclosed_diagnostically() -> (
    None
):
    source = _dataset(omit_index=1)

    with pytest.raises(SessionAggregationQualityError) as raised:
        aggregate_session_dataset(source, _daily())

    assert raised.value.report.missing_constituent_count == 1
    diagnostic = aggregate_session_dataset(
        source,
        _daily(),
        policy=SessionAggregationPolicy(MissingConstituentPolicy.DIAGNOSTIC),
    )
    first = diagnostic.aggregation_report.windows[0]
    assert not diagnostic.aggregation_report.is_complete
    assert first.expected_constituent_count == 2
    assert first.observed_constituent_count == 1
    assert len(first.missing_constituents) == 1
    manifest_source = cast(
        dict[str, object], diagnostic.to_manifest()["source_dataset"]
    )
    quality = cast(dict[str, object], manifest_source["quality_report"])
    assert quality["report"] == source.quality_report.to_primitive()


def test_identities_bind_source_scope_policy_and_are_deterministic() -> None:
    source = _dataset()

    first = aggregate_session_dataset(source, _weekly())
    second = aggregate_session_dataset(source, _weekly())

    assert first == second
    assert first.serialize_bars() == second.serialize_bars()
    assert first.serialize_manifest() == second.serialize_manifest()
    manifest = first.to_manifest()
    assert manifest["source_dataset_family"] == {
        "family_id": first.dataset_family.family_id,
        "canonical_source_snapshot_id": source.metadata.dataset_id,
    }
    assert (
        manifest["session_scope"]
        == source.request.timeframe.session_policy.to_primitive()
    )
    assert (
        cast(dict[str, object], manifest["aggregation_policy"])["configuration_id"]
        == first.metadata.aggregation_policy.configuration_id
    )
    assert all(bar.producer_name == "quantforge" for bar in first.bars)
    changed_source = replace(
        source, metadata=replace(source.metadata, dataset_id="another-source")
    )
    changed = aggregate_session_dataset(changed_source, _weekly())
    assert changed.metadata.dataset_id != first.metadata.dataset_id
    diagnostic = aggregate_session_dataset(
        source,
        _weekly(),
        policy=SessionAggregationPolicy(MissingConstituentPolicy.DIAGNOSTIC),
    )
    assert diagnostic.metadata.dataset_id != first.metadata.dataset_id


def test_session_scope_must_match_and_provider_native_bar_is_rejected() -> None:
    source = _dataset()
    extended_policy = ExchangeSessionPolicy(
        scope=SessionScope.EXTENDED_HOURS,
        extended_hours_start=time(4),
        extended_hours_end=time(20),
    )
    mismatched = Timeframe(SessionInterval(), extended_policy)

    with pytest.raises(SessionAggregationValidationError, match="policies must match"):
        aggregate_session_dataset(source, mismatched)

    derived = aggregate_session_dataset(source, _daily())
    with pytest.raises(SessionAggregationValidationError, match="provider-native"):
        replace(derived.bars[0], producer_name="fixture-provider")


def test_extended_hours_scope_controls_boundaries_and_identity() -> None:
    source, session_policy = _extended_dataset()
    target = Timeframe(SessionInterval(), session_policy)

    derived = aggregate_session_dataset(source, target)

    assert len(derived.bars) == 1
    bar = derived.bars[0]
    assert bar.start_timestamp.astimezone(NEW_YORK).time() == time(4)
    assert bar.end_timestamp.astimezone(NEW_YORK).time() == time(20)
    assert bar.volume == Decimal("4006")
    assert derived.to_manifest()["session_scope"] == session_policy.to_primitive()


def test_partial_exchange_week_is_excluded_not_mislabeled_complete() -> None:
    source = _dataset()
    partial_bars = tuple(
        bar for bar in source.bars if bar.session_date >= date(2024, 7, 2)
    )
    request = replace(
        source.request,
        start_timestamp=resolve_exchange_session(date(2024, 7, 2)).open_timestamp,
    )
    rebound_bars = tuple(
        replace(
            bar,
            provenance=replace(bar.provenance, source_request_id=request.request_id),
        )
        for bar in partial_bars
    )
    batch = IntradayBarBatch(request, rebound_bars)
    quality = validate_intraday_coverage(batch, mode=IntradayValidationMode.DIAGNOSTIC)
    partial = IntradayDataset(
        request,
        rebound_bars,
        replace(
            source.metadata,
            dataset_id="partial-week-source",
            request_id=request.request_id,
            batch_id=batch.batch_id,
            bar_count=len(rebound_bars),
            data_sha256=sha256_hex(batch.serialize()),
            quality_report=quality,
        ),
    )

    derived = aggregate_session_dataset(partial, _weekly())

    assert derived.bars == ()
    assert derived.aggregation_report.excluded_partial_periods == (date(2024, 7, 1),)


def test_immutable_cache_round_trips_and_rejects_collisions(tmp_path: Path) -> None:
    source = _dataset()
    target = _weekly()
    derived = aggregate_session_dataset(source, target)
    cache = SessionAggregationCache(tmp_path)

    cache.persist(derived)
    assert (
        cache.load(
            derived.metadata.dataset_id,
            source_dataset=source,
            target_timeframe=target,
        )
        == derived
    )
    manifest_path = tmp_path / derived.metadata.manifest_location
    manifest_path.write_text("{}")
    with pytest.raises(CacheError, match="collision"):
        cache.persist(derived)


def test_cache_rejects_replaced_content_before_writing(tmp_path: Path) -> None:
    source = _dataset()
    derived = aggregate_session_dataset(source, _daily())
    altered = replace(
        derived,
        bars=(
            replace(derived.bars[0], volume=derived.bars[0].volume + Decimal(1)),
            *derived.bars[1:],
        ),
    )
    cache = SessionAggregationCache(tmp_path)

    with pytest.raises(SessionAggregationValidationError, match="content digest"):
        cache.persist(altered)

    assert not (tmp_path / "session").exists()


def test_invalid_targets_remain_out_of_scope() -> None:
    source = _dataset()

    with pytest.raises(
        SessionAggregationValidationError, match="daily or exchange-weekly"
    ):
        aggregate_session_dataset(
            source, Timeframe.us_equity(IntradayInterval(timedelta(hours=1)))
        )
    with pytest.raises(SessionAggregationValidationError, match="exactly one"):
        aggregate_session_dataset(source, Timeframe.us_equity(SessionInterval(2)))
    with pytest.raises(SessionAggregationValidationError, match="exactly one"):
        aggregate_session_dataset(source, Timeframe.us_equity(TradingWeekInterval(2)))


def test_source_bars_cannot_continue_beyond_session_close() -> None:
    source = _cross_session_dataset()
    assert (
        source.bars[-1].end_timestamp
        > resolve_exchange_session(date(2024, 7, 1)).close_timestamp
    )

    with pytest.raises(
        SessionAggregationValidationError,
        match="prohibited source cross-session continuation",
    ):
        aggregate_session_dataset(source, _daily())
