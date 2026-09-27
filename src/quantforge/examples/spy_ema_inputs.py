"""QF-45 offline source configuration and ordinary canonical artifact composition."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    AggregationPolicy,
    DatasetFamily,
    DatasetLineage,
    FeedScope,
    IntradayAggregationCache,
    IntradayBarRequest,
    IntradayDataset,
    IntradayMarketDataCache,
    IntradayMarketDataService,
    MarketDataCache,
    MarketDataset,
    SessionAggregationCache,
    TimeframeBarSeries,
    aggregate_intraday_dataset,
    aggregate_session_dataset,
)
from quantforge.data.prediction_inputs import prediction_dataset_from_intraday
from quantforge.examples.spy_ema import DAILY, ONE_MINUTE, TWO_MINUTES

SOURCE_ID = "7e396b640d4387c324ab9a25194a5b046d8dcaf1bc7a345783f993f6a2477972"
RAW_SNAPSHOT_ID = "6d85c98d75be72c7d5e4fba8bbc78066c09967eead34aacca9ad8c673315e029"
TWO_MINUTE_ID = "0447ab4da91ae57b6acca49b41f13501140f0e18565afa5c1376dd05c3a0ac8d"
DAILY_ID = "18936f6834a19be1be8112a7e0756199079a5bf9a0d7f4ffa3ae3b9a6367b08e"


def source_request() -> IntradayBarRequest:
    """The verified 2025 SPY/XNYS/RTH raw-price request, with explicit bounds."""
    return IntradayBarRequest(
        "SPY",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
        ONE_MINUTE,
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
class SmokeInputs:
    source: IntradayDataset
    dataset: MarketDataset
    primary: TimeframeBarSeries
    daily: TimeframeBarSeries
    family: DatasetFamily


def load_inputs(cache_root: Path) -> SmokeInputs:
    """Revalidate the immutable source and derive through generic QF-18/QF-19."""
    cache = IntradayMarketDataCache(cache_root)
    source = IntradayMarketDataService(
        cache, provider_name="massive"
    ).get_intraday_bars(source_request())
    if source.metadata.dataset_id != SOURCE_ID or not source.quality_report.is_complete:
        raise ValueError("QF-45 requires its frozen complete 2025 source snapshot")
    if (
        source.metadata.raw_snapshot_ids != (RAW_SNAPSHOT_ID,)
        or len(source.bars) != 96_960
        or len({bar.session_date for bar in source.bars}) != 250
    ):
        raise ValueError("QF-45 source identity or complete-year counts differ")
    print("Validated frozen source from cache", flush=True)
    primary = IntradayAggregationCache(cache_root).persist(
        aggregate_intraday_dataset(source, TWO_MINUTES)
    )
    print("Validated/persisted 48,480 two-minute bars", flush=True)
    daily = SessionAggregationCache(cache_root).persist(
        aggregate_session_dataset(source, DAILY)
    )
    if (
        primary.metadata.dataset_id != TWO_MINUTE_ID
        or len(primary.bars) != 48_480
        or daily.metadata.dataset_id != DAILY_ID
        or len(daily.bars) != 250
    ):
        raise ValueError("QF-45 derived identities or complete-year counts differ")
    print("Validated/persisted 250 daily bars", flush=True)
    source_id = source.metadata.dataset_id
    children = (primary.metadata.dataset_id, daily.metadata.dataset_id)
    family = DatasetFamily(
        "SPY",
        source.metadata.provider_name,
        source.request.feed_scope,
        source.request.adjustment_basis,
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
        source,
        daily,
        cache=MarketDataCache(cache_root),
        intraday_cache=cache,
        family=family,
    )
    print("Validated canonical prediction metadata", flush=True)
    return SmokeInputs(
        source,
        dataset,
        TimeframeBarSeries.from_aggregated_intraday_dataset(primary, family=family),
        TimeframeBarSeries.from_aggregated_session_dataset(daily, family=family),
        family,
    )
