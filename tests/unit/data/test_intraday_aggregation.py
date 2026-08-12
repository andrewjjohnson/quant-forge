import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
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
    IntradayAggregationCache,
    IntradayAggregationPolicy,
    IntradayAggregationQualityError,
    IntradayAggregationValidationError,
    IntradayBar,
    IntradayBarBatch,
    IntradayBarProvenance,
    IntradayBarRequest,
    IntradayDataset,
    IntradayDatasetMetadata,
    IntradayValidationMode,
    MissingConstituentPolicy,
    aggregate_intraday_dataset,
    validate_intraday_coverage,
)
from quantforge.data.identity import sha256_hex
from quantforge.timeframes import (
    BarCompletion,
    CrossSessionPolicy,
    IntradayInterval,
    SessionInterval,
    Timeframe,
    resolve_exchange_session,
)

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "intraday"
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


def _request(start_timestamp: datetime, end_timestamp: datetime) -> IntradayBarRequest:
    return IntradayBarRequest(
        symbol="SPY",
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        timeframe=Timeframe.us_equity(IntradayInterval(timedelta(minutes=5))),
        feed_scope=FeedScope.consolidated(),
        adjustment_basis=_adjustment_basis(),
    )


def _provenance(request: IntradayBarRequest) -> IntradayBarProvenance:
    return IntradayBarProvenance(
        provider_name="fixture-provider",
        provider_symbol="SPY",
        adapter_version="fixture-1",
        retrieved_at=RETRIEVED_AT,
        source_request_id=request.request_id,
        source_snapshot_id="raw-fixture",
        feed_scope=request.feed_scope,
        adjustment_basis=request.adjustment_basis,
    )


def _bar(
    request: IntradayBarRequest,
    session_date: date,
    start_timestamp: datetime,
    end_timestamp: datetime,
    completion: BarCompletion,
    *,
    sequence: int,
    open_price: Decimal | None = None,
    high_price: Decimal | None = None,
    low_price: Decimal | None = None,
    close_price: Decimal | None = None,
    volume: Decimal | None = None,
) -> IntradayBar:
    base = Decimal(100 + sequence)
    return IntradayBar(
        symbol=request.symbol,
        session_date=session_date,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        timeframe=request.timeframe,
        completion=completion,
        open=base if open_price is None else open_price,
        high=base + 2 if high_price is None else high_price,
        low=base - 1 if low_price is None else low_price,
        close=base + 1 if close_price is None else close_price,
        volume=Decimal(10 + sequence) if volume is None else volume,
        provenance=_provenance(request),
    )


def _dataset(
    request: IntradayBarRequest, bars: tuple[IntradayBar, ...]
) -> IntradayDataset:
    batch = IntradayBarBatch(request, bars)
    quality_report = validate_intraday_coverage(
        batch, mode=IntradayValidationMode.DIAGNOSTIC
    )
    dataset_id = configuration_identity(
        {
            "fixture": "intraday-source",
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
        raw_snapshot_ids=("raw-fixture",),
        raw_locations=("intraday/raw/raw-fixture.json",),
        normalized_location=f"intraday/datasets/{dataset_id}/bars.json",
        data_sha256=sha256_hex(batch.serialize()),
        quality_report=quality_report,
    )
    return IntradayDataset(request, bars, metadata)


def _session_dataset(
    session_dates: tuple[date, ...],
    *,
    missing_start: datetime | None = None,
) -> IntradayDataset:
    first_session = resolve_exchange_session(session_dates[0])
    last_session = resolve_exchange_session(session_dates[-1])
    request = _request(first_session.open_timestamp, last_session.close_timestamp)
    bars: list[IntradayBar] = []
    sequence = 0
    for session_date in session_dates:
        session = resolve_exchange_session(session_date)
        start_timestamp = session.open_timestamp
        while start_timestamp < session.close_timestamp:
            end_timestamp = min(
                start_timestamp + timedelta(minutes=5), session.close_timestamp
            )
            completion = (
                BarCompletion.COMPLETED
                if end_timestamp - start_timestamp == timedelta(minutes=5)
                else BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL
            )
            if start_timestamp != missing_start:
                bars.append(
                    _bar(
                        request,
                        session_date,
                        start_timestamp,
                        end_timestamp,
                        completion,
                        sequence=sequence,
                    )
                )
            sequence += 1
            start_timestamp = end_timestamp
    return _dataset(request, tuple(bars))


def _fixture_dataset() -> IntradayDataset:
    fixture = cast(
        list[dict[str, str]],
        json.loads((FIXTURE_ROOT / "spy_5min_hand_audit.json").read_text()),
    )
    request = _request(
        datetime.fromisoformat(fixture[0]["start_timestamp"]),
        datetime.fromisoformat(fixture[-1]["end_timestamp"]),
    )
    bars = tuple(
        _bar(
            request,
            date(2024, 7, 1),
            datetime.fromisoformat(record["start_timestamp"]),
            datetime.fromisoformat(record["end_timestamp"]),
            BarCompletion.COMPLETED,
            sequence=index,
            open_price=Decimal(record["open"]),
            high_price=Decimal(record["high"]),
            low_price=Decimal(record["low"]),
            close_price=Decimal(record["close"]),
            volume=Decimal(record["volume"]),
        )
        for index, record in enumerate(fixture)
    )
    return _dataset(request, bars)


def _target(minutes: int) -> Timeframe:
    return Timeframe.us_equity(IntradayInterval(timedelta(minutes=minutes)))


def test_hand_auditable_fixture_uses_canonical_ohlcv_aggregation() -> None:
    derived = aggregate_intraday_dataset(_fixture_dataset(), _target(15))

    assert len(derived.bars) == 1
    bar = derived.bars[0]
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == (
        Decimal("100"),
        Decimal("106"),
        Decimal("98"),
        Decimal("104"),
        Decimal("600"),
    )
    assert bar.completion is BarCompletion.COMPLETED
    quality = derived.aggregation_report.windows[0]
    assert quality.expected_constituent_count == 3
    assert quality.observed_constituent_count == 3
    assert quality.is_complete
    assert len(quality.source_bar_ids) == 3


@pytest.mark.parametrize(
    ("target_minutes", "expected_bar_count"),
    [(15, 26), (20, 20), (30, 13), (60, 7), (120, 4), (240, 2)],
)
def test_five_minute_source_supports_required_exact_multiples(
    target_minutes: int, expected_bar_count: int
) -> None:
    source = _session_dataset((date(2024, 7, 1),))

    derived = aggregate_intraday_dataset(source, _target(target_minutes))

    assert len(derived.bars) == expected_bar_count
    assert all(bar.session_date == date(2024, 7, 1) for bar in derived.bars)
    assert derived.aggregation_report.is_complete


def test_normal_and_early_close_four_hour_windows_end_at_actual_session_close() -> None:
    normal = aggregate_intraday_dataset(
        _session_dataset((date(2024, 7, 1),)), _target(240)
    )
    early_close = aggregate_intraday_dataset(
        _session_dataset((date(2024, 11, 29),)), _target(240)
    )

    assert [
        (
            bar.start_timestamp.astimezone(NEW_YORK).time(),
            bar.end_timestamp.astimezone(NEW_YORK).time(),
            bar.completion,
        )
        for bar in normal.bars
    ] == [
        (
            datetime(2024, 7, 1, 9, 30).time(),
            datetime(2024, 7, 1, 13, 30).time(),
            BarCompletion.COMPLETED,
        ),
        (
            datetime(2024, 7, 1, 13, 30).time(),
            datetime(2024, 7, 1, 16).time(),
            BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL,
        ),
    ]
    assert len(early_close.bars) == 1
    assert early_close.bars[0].actual_duration == timedelta(hours=3, minutes=30)
    assert (
        early_close.bars[0].end_timestamp.astimezone(NEW_YORK).time()
        == datetime(2024, 11, 29, 13).time()
    )
    assert (
        early_close.bars[0].completion
        is BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL
    )


def test_sessions_never_mix_and_dst_uses_exchange_local_boundaries() -> None:
    derived = aggregate_intraday_dataset(
        _session_dataset((date(2024, 3, 8), date(2024, 3, 11))),
        _target(240),
    )

    assert len(derived.bars) == 4
    assert [bar.session_date for bar in derived.bars] == [
        date(2024, 3, 8),
        date(2024, 3, 8),
        date(2024, 3, 11),
        date(2024, 3, 11),
    ]
    assert derived.bars[0].start_timestamp == datetime(2024, 3, 8, 14, 30, tzinfo=UTC)
    assert derived.bars[2].start_timestamp == datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
    assert all(
        quality.session_date
        == next(
            bar.session_date
            for bar in derived.bars
            if bar.bar_id == quality.output_bar_id
        )
        for quality in derived.aggregation_report.windows
    )


def test_missing_constituents_raise_strictly_and_are_bound_diagnostically() -> None:
    missing_start = datetime(2024, 7, 1, 13, 35, tzinfo=UTC)
    source = _session_dataset((date(2024, 7, 1),), missing_start=missing_start)

    with pytest.raises(IntradayAggregationQualityError) as raised:
        aggregate_intraday_dataset(source, _target(15))

    assert raised.value.report.missing_constituent_count == 1
    diagnostic = aggregate_intraday_dataset(
        source,
        _target(15),
        policy=IntradayAggregationPolicy(MissingConstituentPolicy.DIAGNOSTIC),
    )
    first_quality = diagnostic.aggregation_report.windows[0]
    assert not diagnostic.aggregation_report.is_complete
    assert first_quality.expected_constituent_count == 3
    assert first_quality.observed_constituent_count == 2
    assert first_quality.missing_constituents[0].start_timestamp == missing_start
    assert first_quality.output_bar_id == diagnostic.bars[0].bar_id
    source_manifest = cast(
        dict[str, object], diagnostic.to_manifest()["source_dataset"]
    )
    source_quality = cast(dict[str, object], source_manifest["quality_report"])
    assert source_quality["report_id"] == source.quality_report.report_id
    assert source_quality["report"] == source.quality_report.to_primitive()


def test_reaggregation_is_deterministic_and_binds_policy_source_and_lineage() -> None:
    source = _session_dataset((date(2024, 7, 1),))

    first = aggregate_intraday_dataset(source, _target(240))
    second = aggregate_intraday_dataset(source, _target(240))

    assert first == second
    assert first.serialize_bars() == second.serialize_bars()
    assert first.serialize_manifest() == second.serialize_manifest()
    assert first.metadata.dataset_id == second.metadata.dataset_id
    assert [bar.bar_id for bar in first.bars] == [bar.bar_id for bar in second.bars]
    assert all(bar.provenance.provider_name == "quantforge" for bar in first.bars)
    assert all(
        bar.provenance.source_snapshot_id == source.metadata.dataset_id
        for bar in first.bars
    )
    assert (
        first.dataset_family.canonical_source_snapshot_id == source.metadata.dataset_id
    )
    assert first.dataset_family.reference(first.metadata.dataset_id).family_id == (
        first.dataset_family.family_id
    )
    assert {entry.dataset_id for entry in first.dataset_family.datasets} == {
        source.metadata.dataset_id,
        first.metadata.dataset_id,
    }
    diagnostic = aggregate_intraday_dataset(
        source,
        _target(240),
        policy=IntradayAggregationPolicy(MissingConstituentPolicy.DIAGNOSTIC),
    )
    assert diagnostic.metadata.dataset_id != first.metadata.dataset_id
    changed_source = replace(
        source,
        metadata=replace(source.metadata, dataset_id="another-source-snapshot"),
    )
    changed = aggregate_intraday_dataset(changed_source, _target(240))
    assert changed.metadata.dataset_id != first.metadata.dataset_id
    assert changed.bars[0].bar_id != first.bars[0].bar_id


def test_derived_cache_persists_reloads_and_rejects_collisions(tmp_path: Path) -> None:
    source = _session_dataset((date(2024, 7, 1),))
    target = _target(60)
    derived = aggregate_intraday_dataset(source, target)
    cache = IntradayAggregationCache(tmp_path)

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


def test_invalid_target_semantics_fail_closed() -> None:
    source = _session_dataset((date(2024, 7, 1),))

    with pytest.raises(IntradayAggregationValidationError, match="exact multiple"):
        aggregate_intraday_dataset(source, _target(7))
    with pytest.raises(IntradayAggregationValidationError, match="intraday"):
        aggregate_intraday_dataset(source, Timeframe.us_equity(SessionInterval()))
    cross_session = Timeframe.us_equity(
        IntradayInterval(
            timedelta(minutes=15),
            cross_session_policy=CrossSessionPolicy.PERMITTED,
        )
    )
    with pytest.raises(IntradayAggregationValidationError, match="cross-session"):
        aggregate_intraday_dataset(source, cross_session)
