"""Opt-in QF-54 historical SPY verification through the normal cache service."""

import argparse
import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

from quantforge.data import (
    FeedScope,
    IntradayBarRequest,
    IntradayMarketDataCache,
    IntradayMarketDataService,
)
from quantforge.data.models import JsonValue
from quantforge.data.providers import MassiveProvider, create_intraday_provider
from quantforge.timeframes import IntradayInterval, Timeframe


def verify_range(
    cache: IntradayMarketDataCache,
    provider: MassiveProvider | None,
    start: date,
    end: date,
    *,
    label: str,
) -> dict[str, JsonValue]:
    """Require complete SPY coverage for verification using the generic report."""
    request = IntradayBarRequest(
        "SPY",
        datetime(start.year, start.month, start.day, tzinfo=UTC),
        datetime(end.year, end.month, end.day, tzinfo=UTC),
        Timeframe.us_equity(IntradayInterval(timedelta(minutes=1))),
        FeedScope.consolidated(),
        MassiveProvider.intraday_adjustment_basis,
    )
    before = 0 if provider is None else provider.http_request_count
    dataset = IntradayMarketDataService(
        cache, provider=provider, provider_name="massive"
    ).get_intraday_bars(request)
    http_count = 0 if provider is None else provider.http_request_count - before
    # No provider object or credential is available to this replay service.
    offline = IntradayMarketDataService(cache, provider_name="massive")
    replay = offline.get_intraday_bars(request)
    if replay != dataset:
        raise RuntimeError("Massive offline replay verification failed")
    # This SPY verification requirement is not a Massive stock-provider invariant.
    if not dataset.quality_report.is_complete:
        raise RuntimeError("SPY verification requires complete RTH coverage")
    page_count = 0
    raw_count = 0
    for location in dataset.metadata.raw_locations:
        snapshot = cast(
            dict[str, JsonValue], json.loads((cache.root / location).read_bytes())
        )
        for page in cast(list[dict[str, JsonValue]], snapshot["records"]):
            response = cast(dict[str, JsonValue], page["response"])
            raw_count += len(cast(list[JsonValue], response.get("results", [])))
            page_count += 1
    return {
        "label": label,
        "requested_start_inclusive": request.start_timestamp.isoformat(),
        "requested_end_exclusive": request.end_timestamp.isoformat(),
        "first_bar_start": dataset.bars[0].start_timestamp.isoformat(),
        "last_bar_end": dataset.bars[-1].end_timestamp.isoformat(),
        "provider": dataset.metadata.provider_name,
        "dataset_id": dataset.metadata.dataset_id,
        "raw_snapshot_ids": list(dataset.metadata.raw_snapshot_ids),
        "http_attempts_this_run": http_count,
        "retained_pages": page_count,
        "raw_result_count": raw_count,
        "canonical_rth_count": len(dataset.bars),
        "coverage_complete": dataset.quality_report.is_complete,
        "offline_cache_replay": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("data/market-data"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--history-start", type=date.fromisoformat, default=date(2025, 1, 1)
    )
    parser.add_argument(
        "--history-end", type=date.fromisoformat, default=date(2026, 1, 1)
    )
    arguments = parser.parse_args()
    if not arguments.offline and not os.environ.get("MASSIVE_API_KEY"):
        parser.error("MASSIVE_API_KEY is required for live verification")
    provider = (
        None
        if arguments.offline
        else cast(MassiveProvider, create_intraday_provider("massive"))
    )
    cache = IntradayMarketDataCache(arguments.cache_root)
    cases = (
        ("normal_session", date(2025, 7, 1), date(2025, 7, 2)),
        ("early_close", date(2025, 7, 3), date(2025, 7, 4)),
        ("one_month", date(2025, 7, 1), date(2025, 8, 1)),
        ("pagination", date(2025, 1, 1), date(2025, 5, 1)),
        ("continuous_history", arguments.history_start, arguments.history_end),
    )
    for label, start, end in cases:
        result = verify_range(cache, provider, start, end, label=label)
        print(json.dumps(result, sort_keys=True), flush=True)
        if label == "normal_session" and result["canonical_rth_count"] != 390:
            raise RuntimeError("SPY normal-session bar count differs from 390")
        if label == "early_close" and result["canonical_rth_count"] != 210:
            raise RuntimeError("SPY early-close bar count differs from 210")
        if label == "pagination" and cast(int, result["retained_pages"]) < 2:
            raise RuntimeError("verification range did not exercise pagination")


if __name__ == "__main__":
    main()
