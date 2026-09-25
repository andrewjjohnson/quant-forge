"""Large calendar schedule: one source index, exact nonsequential lookups."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest

from quantforge.data import TimeframeBarSeries
from quantforge.data.intraday_aggregation import intraday_session_windows
from quantforge.prediction import PredictionDecisionSchedule
from quantforge.validation import PredictionSourceIndex
from tests.unit.walk_forward.timestamp_fixtures import timestamp_fixture


def test_5760_exact_calendar_lookups_build_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, adapter = timestamp_fixture(tmp_path)
    source = next(
        item for item in adapter.series if item.timeframe == adapter.primary_timeframe
    )
    first = source.bars[0].end_timestamp
    broad = PredictionDecisionSchedule(
        source.timeframe, first, first + timedelta(days=150)
    )
    schedule = PredictionDecisionSchedule(
        source.timeframe, first, broad.decision_timestamps[5759]
    )
    prototype = source.bars[0]
    bars = tuple(
        replace(
            prototype,
            session_date=session,
            start_timestamp=interval.start_timestamp,
            end_timestamp=interval.end_timestamp,
            completion=interval.completion,
        )
        for session in sorted(set(schedule.decision_sessions))
        for interval in intraday_session_windows(session, source.timeframe)
        if first <= interval.end_timestamp <= schedule.end_timestamp
    )
    series = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        source.dataset_reference,
        source.timeframe,
        bars,
        dataset_family_manifest_id=source.dataset_family_manifest_id,
    )
    began = perf_counter()
    index = PredictionSourceIndex.build(series, schedule.decision_timestamps[3], 3)
    construction = perf_counter() - began

    # Once built, any renewed full-source validation is an architectural failure.
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("indexed lookup must not revalidate the full source")

    monkeypatch.setattr(
        "quantforge.validation.prepared_context.validate_source_observations", forbidden
    )
    began = perf_counter()
    for position in (*range(3, 5760), *range(5759, 2, -113)):
        start, stop = index.bounds(schedule.decision_timestamps[position])
        assert (start, stop) == (0, position + 1)
        assert (
            index.source.bars[stop - 1].end_timestamp
            == schedule.decision_timestamps[position]
        )
    elapsed = perf_counter() - began
    print(
        {
            "bars": len(bars),
            "index_builds": 1,
            "construction_seconds": construction,
            "lookup_seconds": elapsed,
        }
    )
