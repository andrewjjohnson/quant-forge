"""QF-65 authenticated canonical loading: equivalence, reuse and fail-closed reuse.

Every test loads synthetic immutable caches offline through the same generic path
as QF-45 (canonical 1m source, derived 2m and daily artifacts, the QF-51 prediction
input and a QF-52 bounded view), with and without a canonical preparation session.
"""

import shutil
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    AggregatedIntradayDataset,
    AggregatedSessionDataset,
    AggregationPolicy,
    CacheError,
    DatasetFamily,
    DatasetLineage,
    FeedScope,
    IntradayAggregationCache,
    IntradayBar,
    IntradayBarBatch,
    IntradayBarProvenance,
    IntradayBarRequest,
    IntradayDataset,
    IntradayFetchResult,
    IntradayMarketDataCache,
    IntradayMarketDataService,
    IntradayRawSnapshot,
    MarketDataCache,
    MarketDataError,
    MarketDataset,
    SessionAggregationCache,
    TimeframeBarSeries,
    ValidationError,
    aggregate_intraday_dataset,
    aggregate_session_dataset,
    intraday_session_windows,
    validate_market_dataset,
)
from quantforge.data.prediction_inputs import prediction_dataset_from_intraday
from quantforge.data.prediction_views import bounded_prediction_view
from quantforge.data.prepared_canonical import (
    CanonicalPreparation,
    active_canonical_preparation,
    canonical_preparation,
)
from quantforge.timeframes import (
    BarCompletion,
    IntradayInterval,
    SessionInterval,
    Timeframe,
    resolve_exchange_session,
)
from quantforge.validation import DatasetProvenance
from tests.unit.data.test_prepared_canonical import (
    _shadow,  # pyright: ignore[reportPrivateUsage]
    _ShadowBar,  # pyright: ignore[reportPrivateUsage]
)

ONE_MINUTE = Timeframe.us_equity(IntradayInterval(timedelta(minutes=1)))
TWO_MINUTES = Timeframe.us_equity(IntradayInterval(timedelta(minutes=2)))
DAILY = Timeframe.us_equity(SessionInterval(1))
BASIS = AdjustmentBasis(
    AdjustmentMode.UNADJUSTED,
    "raw_provider",
    "raw_provider",
    "not_provided_for_intraday_bars",
    False,
)
OTHER_BASIS = AdjustmentBasis(
    AdjustmentMode.SPLIT_ADJUSTED,
    "split_adjusted",
    "split_adjusted",
    "not_provided_for_intraday_bars",
    True,
)
PROVIDER = "fixture"
# DST begins 2024-03-10 (UTC open 14:30 -> 13:30). 2024-07-03 closes early at
# 13:00 New York and 2024-07-04 is an exchange holiday between the sessions.
DST = (date(2024, 3, 8), date(2024, 3, 11))
EARLY_CLOSE_HOLIDAY = (date(2024, 7, 3), date(2024, 7, 5))


def persist_source(root: Path, sessions: tuple[date, ...]) -> IntradayBarRequest:
    """Persist one synthetic canonical 1m source; return its request."""
    request = IntradayBarRequest(
        "SPY",
        resolve_exchange_session(sessions[0], ONE_MINUTE.session_policy).open_timestamp,
        resolve_exchange_session(
            sessions[-1], ONE_MINUTE.session_policy
        ).close_timestamp,
        ONE_MINUTE,
        FeedScope.consolidated(),
        BASIS,
    )
    retrieved = datetime.combine(
        sessions[-1] + timedelta(days=1), datetime.min.time(), UTC
    )
    raw = IntradayRawSnapshot(
        PROVIDER,
        "SPY",
        "fixture-v1",
        "offline-fixture",
        request.request_id,
        request.start_timestamp,
        request.end_timestamp,
        retrieved,
        (),
        ({"fixture": "synthetic canonical SPY minutes"},),
    )
    provenance = IntradayBarProvenance(
        PROVIDER,
        "SPY",
        "fixture-v1",
        retrieved,
        request.request_id,
        raw.snapshot_id,
        request.feed_scope,
        BASIS,
    )
    bars: list[IntradayBar] = []
    for session in sessions:
        for index, interval in enumerate(intraday_session_windows(session, ONE_MINUTE)):
            base = Decimal(100) + Decimal(index % 11) / 100
            bars.append(
                IntradayBar(
                    "SPY",
                    session,
                    interval.start_timestamp,
                    interval.end_timestamp,
                    ONE_MINUTE,
                    interval.completion,
                    base,
                    base + Decimal("0.05"),
                    base - Decimal("0.04"),
                    base + Decimal(index % 3) / 100,
                    Decimal(1000 + index),
                    provenance,
                )
            )
    IntradayMarketDataCache(root).persist(
        IntradayFetchResult(
            IntradayBarBatch(request, tuple(bars)), (raw,), "fixture-capabilities"
        )
    )
    return request


@dataclass(frozen=True)
class Loaded:
    source: IntradayDataset
    primary: AggregatedIntradayDataset
    daily: AggregatedSessionDataset
    dataset: MarketDataset
    primary_series: TimeframeBarSeries
    daily_series: TimeframeBarSeries
    view: MarketDataset


def load(root: Path, request: IntradayBarRequest) -> Loaded:
    """The QF-45 input composition through generic caches and validators."""
    cache = IntradayMarketDataCache(root)
    source = IntradayMarketDataService(cache, provider_name=PROVIDER).get_intraday_bars(
        request
    )
    primary = IntradayAggregationCache(root).persist(
        aggregate_intraday_dataset(source, TWO_MINUTES)
    )
    daily = SessionAggregationCache(root).persist(
        aggregate_session_dataset(source, DAILY)
    )
    source_id = source.metadata.dataset_id
    children = (primary.metadata.dataset_id, daily.metadata.dataset_id)
    family = DatasetFamily(
        "SPY",
        PROVIDER,
        request.feed_scope,
        BASIS,
        AggregationPolicy(
            "quantforge_context_artifact_set",
            "1",
            cast(
                PrimitiveMapping,
                {
                    "artifact_family_manifest_ids": sorted(
                        (
                            primary.dataset_family.manifest_id,
                            daily.dataset_family.manifest_id,
                        )
                    )
                },
            ),
        ),
        source_id,
        (
            DatasetLineage(source_id, ONE_MINUTE, source_id, None, children),
            DatasetLineage(children[0], TWO_MINUTES, source_id, source_id),
            DatasetLineage(children[1], DAILY, source_id, source_id),
        ),
    )
    dataset = prediction_dataset_from_intraday(
        source, daily, cache=MarketDataCache(root), intraday_cache=cache, family=family
    )
    for _ in range(3):  # QF-45 plan, adapter and partition provenance captures.
        DatasetProvenance.from_market_dataset(dataset)
    first_close = resolve_exchange_session(
        daily.bars[0].session_dates[0], DAILY.session_policy
    ).close_timestamp
    return Loaded(
        source,
        primary,
        daily,
        dataset,
        TimeframeBarSeries.from_aggregated_intraday_dataset(primary, family=family),
        TimeframeBarSeries.from_aggregated_session_dataset(daily, family=family),
        bounded_prediction_view(dataset, first_close),
    )


@pytest.fixture(scope="module", params=[DST, EARLY_CLOSE_HOLIDAY], ids=["dst", "early"])
def cached(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, IntradayBarRequest, Loaded]:
    root = tmp_path_factory.mktemp("canonical")
    source_request = persist_source(root, cast(tuple[date, ...], request.param))
    return root, source_request, load(root, source_request)


def copy_cache(root: Path, destination: Path) -> Path:
    shutil.copytree(root, destination)
    return destination


def test_prepared_loading_is_scientifically_identical(
    cached: tuple[Path, IntradayBarRequest, Loaded],
) -> None:
    root, request, reference = cached
    with canonical_preparation() as preparation:
        prepared = load(root, request)
        statistics = preparation.statistics()
    # Bars, metadata, coverage/session evidence, derived artifacts, family
    # provenance, prediction input and QF-52 view (identity and ancestry).
    assert prepared == reference
    for name in ("source", "primary", "daily", "dataset", "view"):
        assert (
            getattr(prepared, name).metadata.dataset_id
            == getattr(reference, name).metadata.dataset_id
        )
    assert prepared.view.metadata.intraday_provenance == (
        reference.view.metadata.intraday_provenance
    )
    # The bounded view exposes only completed sessions through its cutoff.
    assert len(prepared.view.bars) == 1
    assert prepared.source.quality_report.is_complete
    assert statistics["session_resolutions"] == len(prepared.daily.bars)


def test_session_and_calendar_boundaries_are_exact(
    cached: tuple[Path, IntradayBarRequest, Loaded],
) -> None:
    root, request, reference = cached
    with canonical_preparation():
        prepared = load(root, request)
        memo_sessions = [
            resolve_exchange_session(bar.session_dates[0], DAILY.session_policy)
            for bar in prepared.daily.bars
        ]
    sessions = [
        resolve_exchange_session(bar.session_dates[0], DAILY.session_policy)
        for bar in reference.daily.bars
    ]
    assert memo_sessions == sessions
    for bar, session in zip(prepared.daily.bars, sessions, strict=True):
        # Completed daily bars use actual exchange opens/closes (early close,
        # DST) and never cross sessions.
        assert (bar.start_timestamp, bar.end_timestamp) == (
            session.open_timestamp,
            session.close_timestamp,
        )
        in_session = [
            item
            for item in prepared.primary.bars
            if item.session_date == bar.session_dates[0]
        ]
        assert in_session[0].start_timestamp == session.open_timestamp
        assert in_session[-1].end_timestamp == session.close_timestamp
        assert len(bar.source_bar_ids) == len(
            intraday_session_windows(bar.session_dates[0], ONE_MINUTE)
        )


def test_session_authenticates_source_once_and_reuses_its_proof(
    cached: tuple[Path, IntradayBarRequest, Loaded],
) -> None:
    root, request, _ = cached
    with canonical_preparation() as preparation:
        load(root, request)
        statistics = preparation.statistics()
    assert statistics["source_authentications"] == 1
    # prediction_dataset_from_intraday's immutable-cache binding reload.
    assert statistics["source_cache_reuses"] == 1
    # 2m derivation, daily derivation and the daily derivation proof.
    assert statistics["source_reuses"] == 3
    assert statistics["derivations"] == 2
    assert statistics["derivation_reuses"] == 1
    # 2m, daily, and the daily rebuilt from QF-51 evidence: its timestamps are
    # plain datetimes, not calendar Timestamps, so it is strictly distinct and
    # validated once as its own variant (never evicting the original).
    assert statistics["derived_validations"] == 3
    assert statistics["derived_rejections"] == 1
    assert statistics["derived_validation_reuses"] >= 4
    # QF-51 source evidence rebuilt once; later provenance captures reuse it.
    assert statistics["market_validations"] >= 1
    assert statistics["market_validation_reuses"] >= 3


def test_preparation_is_execution_local_and_restart_is_deterministic(
    cached: tuple[Path, IntradayBarRequest, Loaded],
) -> None:
    root, request, reference = cached
    runs: list[tuple[Loaded, CanonicalPreparation]] = []
    for _ in range(2):
        with canonical_preparation() as preparation:
            with canonical_preparation() as joined:
                assert joined is preparation  # Inner scopes join; never nest.
            runs.append((load(root, request), preparation))
        assert active_canonical_preparation() is None
        statistics = preparation.statistics()
        assert statistics["retained_sources"] == 0
        assert statistics["retained_market_verdicts"] == 0
    assert runs[0][1] is not runs[1][1]
    assert runs[0][0] == runs[1][0] == reference


def test_failed_session_releases_its_preparation(
    cached: tuple[Path, IntradayBarRequest, Loaded],
) -> None:
    root, request, _ = cached
    scopes: list[CanonicalPreparation] = []

    def interrupted_run() -> None:
        with canonical_preparation() as preparation:
            scopes.append(preparation)
            load(root, request)
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted_run()
    (preparation,) = scopes
    assert active_canonical_preparation() is None
    assert preparation.statistics()["retained_sources"] == 0


@pytest.mark.parametrize("artifact", ["bars", "manifest", "raw", "truncated"])
def test_changed_cache_bytes_are_never_reused(
    cached: tuple[Path, IntradayBarRequest, Loaded], tmp_path: Path, artifact: str
) -> None:
    root, request, reference = cached
    copied = copy_cache(root, tmp_path / "cache")
    metadata = reference.source.metadata
    path = {
        "bars": copied / metadata.normalized_location,
        "truncated": copied / metadata.normalized_location,
        "manifest": copied / f"intraday/datasets/{metadata.dataset_id}/manifest.json",
        "raw": copied / metadata.raw_locations[0],
    }[artifact]
    cache = IntradayMarketDataCache(copied)
    with canonical_preparation() as preparation:
        cache.load(metadata.dataset_id, request)
        content = path.read_bytes()
        path.chmod(0o600)
        if artifact == "truncated":
            path.write_bytes(content[: len(content) // 2])
        else:
            path.write_bytes(content.replace(b"SPY", b"SPZ", 1))
        with pytest.raises(CacheError):
            cache.load(metadata.dataset_id, request)
        statistics = preparation.statistics()
    assert statistics["source_cache_byte_mismatches"] == 1
    assert statistics["source_cache_reuses"] == 0


def test_other_identity_request_or_root_is_authenticated_independently(
    cached: tuple[Path, IntradayBarRequest, Loaded], tmp_path: Path
) -> None:
    root, request, reference = cached
    metadata = reference.source.metadata
    corrupt_root = copy_cache(root, tmp_path / "other")
    bars = corrupt_root / metadata.normalized_location
    bars.chmod(0o600)
    bars.write_bytes(bars.read_bytes().replace(b'"100.05"', b'"100.06"', 1))
    other_basis = replace(request, adjustment_basis=OTHER_BASIS)
    with canonical_preparation() as preparation:
        IntradayMarketDataCache(root).load(metadata.dataset_id, request)
        with pytest.raises(CacheError):  # Same identity, different root.
            IntradayMarketDataCache(corrupt_root).load(metadata.dataset_id, request)
        with pytest.raises(CacheError, match="request mismatch"):
            IntradayMarketDataCache(root).load(metadata.dataset_id, other_basis)
        with pytest.raises(CacheError):
            IntradayMarketDataCache(root).load("0" * 64, request)
        assert preparation.statistics()["source_cache_reuses"] == 0


def _changed_source(source: IntradayDataset, change: str) -> IntradayDataset:
    bars = source.bars
    if change == "missing_bar":
        return replace(source, bars=bars[1:])
    if change == "duplicate_bar":
        return replace(source, bars=(bars[0], *bars))
    if change == "extra_bar":
        # A valid observation a week earlier, outside the authenticated request.
        early = replace(
            bars[0],
            start_timestamp=bars[0].start_timestamp - timedelta(days=7),
            end_timestamp=bars[0].end_timestamp - timedelta(days=7),
            session_date=bars[0].session_date - timedelta(days=7),
        )
        return replace(source, bars=(early, *bars))
    if change == "changed_bar":
        return replace(
            source,
            bars=(replace(bars[0], close=bars[0].close + 1 / Decimal(100)), *bars[1:]),
        )
    request = source.request
    if change == "wrong_symbol":
        return replace(source, request=replace(request, symbol="QQQ"))
    if change == "wrong_timeframe":
        return replace(source, request=replace(request, timeframe=TWO_MINUTES))
    if change == "wrong_adjustment":
        return replace(
            source,
            request=replace(request, adjustment_basis=OTHER_BASIS),
        )
    if change == "wrong_feed":
        return replace(
            source, request=replace(request, feed_scope=FeedScope.iex_only())
        )
    if change == "wrong_bar_count":
        return replace(
            source, metadata=replace(source.metadata, bar_count=len(bars) + 1)
        )
    assert change == "wrong_batch_id"
    return replace(source, metadata=replace(source.metadata, batch_id="0" * 64))


@pytest.mark.parametrize(
    "change",
    [
        "missing_bar",
        "duplicate_bar",
        "extra_bar",
        "changed_bar",
        "wrong_symbol",
        "wrong_timeframe",
        "wrong_adjustment",
        "wrong_feed",
        "wrong_bar_count",
        "wrong_batch_id",
    ],
)
@pytest.mark.parametrize("target", ["2m", "daily"])
def test_presented_source_must_match_authenticated_content(
    cached: tuple[Path, IntradayBarRequest, Loaded], change: str, target: str
) -> None:
    root, request, _ = cached
    with canonical_preparation() as preparation:
        source = IntradayMarketDataCache(root).load(_source_id(root, request), request)
        changed = _changed_source(source, change)
        aggregate = (
            partial(aggregate_intraday_dataset, target_timeframe=TWO_MINUTES)
            if target == "2m"
            else partial(aggregate_session_dataset, target_timeframe=DAILY)
        )
        with pytest.raises((ValueError, TypeError, MarketDataError)):
            aggregate(changed)
        assert preparation.statistics()["source_reuses"] == 0


def _source_id(root: Path, request: IntradayBarRequest) -> str:
    index = root / "intraday" / "requests" / PROVIDER / f"{request.request_id}.json"
    return index.read_text().split('"')[3]


def test_bypass_mutated_backing_is_revalidated_and_rejected(
    cached: tuple[Path, IntradayBarRequest, Loaded],
) -> None:
    root, request, _ = cached
    with canonical_preparation() as preparation:
        source = IntradayMarketDataCache(root).load(_source_id(root, request), request)
        bar = source.bars[3]
        object.__setattr__(bar, "close", bar.high)  # Unsupported bypass mutation.
        with pytest.raises(MarketDataError, match="batch identity"):
            aggregate_intraday_dataset(source, TWO_MINUTES)
        statistics = preparation.statistics()
        assert statistics["source_integrity_failures"] == 1
        assert statistics["source_reuses"] == 0
        # Bars decoded from one raw chunk share one immutable provenance record.
        assert source.bars[0].provenance is source.bars[1].provenance


@pytest.mark.parametrize("field", ["provider_symbol", "retrieved_at", "batch_id"])
def test_bypass_mutated_source_metadata_is_never_reused(
    cached: tuple[Path, IntradayBarRequest, Loaded], field: str
) -> None:
    root, request, _ = cached
    dataset_id = _source_id(root, request)
    cache = IntradayMarketDataCache(root)
    with canonical_preparation() as preparation:
        source = cache.load(dataset_id, request)
        value = {
            "provider_symbol": "QQQ",
            "retrieved_at": source.metadata.retrieved_at + timedelta(days=1),
            "batch_id": "0" * 64,
        }[field]
        # The retained object itself: comparing it with itself cannot detect this.
        object.__setattr__(source.metadata, field, value)
        if field == "batch_id":
            with pytest.raises(MarketDataError, match="batch identity"):
                aggregate_intraday_dataset(source, TWO_MINUTES)
        else:
            # The reference path does not check this field, so it derives from the
            # mutated value; the session must not substitute its own proof.
            derived = aggregate_intraday_dataset(source, TWO_MINUTES)
            assert getattr(derived.bars[0].provenance, field) == value
        statistics = preparation.statistics()
        assert statistics["source_integrity_failures"] == 1
        assert statistics["source_reuses"] == 0
        # A later identical cache load re-authenticates from disk instead of
        # returning the mutated retained object.
        reloaded = cache.load(dataset_id, request)
        assert reloaded is not source
        assert getattr(reloaded.metadata, field) != value
        assert preparation.statistics()["source_cache_reuses"] == 0


@pytest.mark.parametrize(
    "target", ["2m_bar_count", "2m_request", "daily_location", "daily_report"]
)
def test_bypass_mutated_derived_metadata_is_revalidated(
    cached: tuple[Path, IntradayBarRequest, Loaded], tmp_path: Path, target: str
) -> None:
    root, request, _ = cached
    with canonical_preparation() as preparation:
        loaded = load(root, request)  # Both derived artifacts validated once.
        if target == "2m_bar_count":
            object.__setattr__(loaded.primary.metadata, "bar_count", 1)
        elif target == "2m_request":
            object.__setattr__(loaded.primary.request, "symbol", "QQQ")
        elif target == "daily_location":
            object.__setattr__(loaded.daily.metadata, "normalized_location", "x")
        else:
            window = loaded.daily.metadata.aggregation_report.windows[0]
            object.__setattr__(window, "observed_constituent_count", 1)
        artifact = loaded.primary if target.startswith("2m") else loaded.daily
        # Whatever the reference validation raises: a domain or dataset error.
        with pytest.raises((MarketDataError, ValueError)):
            artifact.validate()
        if not target.startswith("2m"):
            with pytest.raises(MarketDataError):
                prediction_dataset_from_intraday(
                    loaded.source,
                    loaded.daily,
                    cache=MarketDataCache(tmp_path / "projection"),
                    intraday_cache=IntradayMarketDataCache(root),
                )
        assert preparation.statistics()["derived_integrity_failures"] >= 1


@pytest.mark.parametrize(
    "target", ["source_volume", "source_zone", "daily_count", "prediction_count"]
)
def test_equal_valued_type_or_zone_substitution_is_never_reused(
    cached: tuple[Path, IntradayBarRequest, Loaded], target: str
) -> None:
    root, request, _ = cached
    with canonical_preparation() as preparation:
        loaded = load(root, request)
        bar = loaded.source.bars[1]
        # Each replacement equals the original under ``==``; only its type or
        # time-zone representation changes, which canonical serialization and
        # the reference validation reject.
        if target == "source_volume":
            assert int(bar.volume) == bar.volume
            object.__setattr__(bar, "volume", int(bar.volume))
            with pytest.raises(AttributeError):
                aggregate_intraday_dataset(loaded.source, TWO_MINUTES)
            failures = preparation.statistics()["source_integrity_failures"]
        elif target == "source_zone":
            zone = ZoneInfo("America/New_York")
            object.__setattr__(
                bar, "start_timestamp", bar.start_timestamp.astimezone(zone)
            )
            with pytest.raises(MarketDataError, match="batch identity"):
                aggregate_intraday_dataset(loaded.source, TWO_MINUTES)
            failures = preparation.statistics()["source_integrity_failures"]
        elif target == "daily_count":
            window = loaded.daily.metadata.aggregation_report.windows[0]
            count = window.observed_constituent_count
            object.__setattr__(window, "observed_constituent_count", Decimal(count))
            with pytest.raises(TypeError, match="JSON serializable"):
                loaded.daily.validate()
            failures = preparation.statistics()["derived_integrity_failures"]
        else:
            metadata = loaded.dataset.metadata
            object.__setattr__(metadata, "bar_count", Decimal(metadata.bar_count))
            with pytest.raises(ValidationError, match="bar count"):
                validate_market_dataset(loaded.dataset)
            failures = 1
        assert failures == 1


@pytest.mark.parametrize("target", ["presented_class", "retained_class", "enum"])
def test_class_or_enum_substitution_is_never_reused(
    cached: tuple[Path, IntradayBarRequest, Loaded], target: str
) -> None:
    root, request, _ = cached
    with canonical_preparation() as preparation:
        source = IntradayMarketDataCache(root).load(_source_id(root, request), request)
        if target == "presented_class":
            # Another class carrying the same field objects for every bar.
            shadow = replace(
                source, bars=tuple(_shadow(bar, _ShadowBar) for bar in source.bars)
            )
            with pytest.raises(ValueError, match="invalid bar"):
                aggregate_intraday_dataset(shadow, TWO_MINUTES)
            assert preparation.statistics()["source_rejections"] == 1
        elif target == "retained_class":
            object.__setattr__(source.bars[2], "__class__", _ShadowBar)
            with pytest.raises(ValueError, match="invalid bar"):
                aggregate_intraday_dataset(source, TWO_MINUTES)
            assert preparation.statistics()["source_integrity_failures"] == 1
        else:
            member = BarCompletion.COMPLETED
            original = member._value_
            try:
                # A process-global singleton: serialization changes, the field
                # objects do not. The reference batch identity no longer matches.
                object.__setattr__(member, "_value_", "altered")
                with pytest.raises(MarketDataError, match="batch identity"):
                    aggregate_intraday_dataset(source, TWO_MINUTES)
            finally:
                object.__setattr__(member, "_value_", original)
            assert preparation.statistics()["source_integrity_failures"] == 1
        assert preparation.statistics()["source_reuses"] == 0


def test_corrupted_or_rebound_derived_artifacts_fail_closed(
    cached: tuple[Path, IntradayBarRequest, Loaded], tmp_path: Path
) -> None:
    root, request, reference = cached
    copied = copy_cache(root, tmp_path / "derived")
    with canonical_preparation():
        loaded = load(copied, request)
        primary_path = copied / loaded.primary.metadata.normalized_location
        primary_path.chmod(0o600)
        primary_path.write_bytes(primary_path.read_bytes().replace(b"SPY", b"SPZ", 1))
        with pytest.raises(CacheError, match="collision"):
            IntradayAggregationCache(copied).persist(loaded.primary)
        with pytest.raises(CacheError, match="normalized artifact mismatch"):
            IntradayAggregationCache(copied).load(
                loaded.primary.metadata.dataset_id,
                source_dataset=loaded.source,
                target_timeframe=TWO_MINUTES,
            )
        rebound = replace(
            loaded.primary,
            metadata=replace(loaded.primary.metadata, source_dataset_id="0" * 64),
        )
        with pytest.raises(MarketDataError):
            rebound.validate()
        daily_bar = loaded.daily.bars[0]
        changed_daily = replace(
            loaded.daily,
            bars=(replace(daily_bar, close=daily_bar.high), *loaded.daily.bars[1:]),
        )
        with pytest.raises(MarketDataError):
            changed_daily.validate()
        with pytest.raises(ValidationError, match="differs from its intraday source"):
            prediction_dataset_from_intraday(
                loaded.source,
                changed_daily,
                cache=MarketDataCache(tmp_path / "projection"),
                intraday_cache=IntradayMarketDataCache(copied),
            )
    assert reference.primary == loaded.primary


def test_daily_artifact_bound_to_another_source_is_rederived_and_rejected(
    tmp_path: Path,
) -> None:
    dst_request = persist_source(tmp_path / "a", DST)
    early_request = persist_source(tmp_path / "b", EARLY_CLOSE_HOLIDAY)
    with canonical_preparation() as preparation:
        dst = load(tmp_path / "a", dst_request)
        early = load(tmp_path / "b", early_request)
        with pytest.raises(ValidationError, match="differs from its intraday source"):
            prediction_dataset_from_intraday(
                dst.source,
                early.daily,
                cache=MarketDataCache(tmp_path / "projection"),
                intraday_cache=IntradayMarketDataCache(tmp_path / "a"),
                family=dst.daily.dataset_family,
            )
        assert preparation.statistics()["retained_sources"] == 2


def test_mutated_prediction_input_is_revalidated_and_rejected(
    cached: tuple[Path, IntradayBarRequest, Loaded],
) -> None:
    root, request, _ = cached
    with canonical_preparation() as preparation:
        loaded = load(root, request)
        reused = preparation.statistics()["market_validation_reuses"]
        validate_market_dataset(loaded.dataset)
        assert preparation.statistics()["market_validation_reuses"] == reused + 1
        bar = loaded.dataset.bars[0]
        object.__setattr__(bar, "close", bar.high)  # Test-local dataset object.
        with pytest.raises(ValidationError):
            validate_market_dataset(loaded.dataset)
        changed = replace(
            loaded.dataset,
            metadata=replace(loaded.dataset.metadata, provider_symbol="QQQ"),
        )
        with pytest.raises(ValidationError):
            validate_market_dataset(changed)
