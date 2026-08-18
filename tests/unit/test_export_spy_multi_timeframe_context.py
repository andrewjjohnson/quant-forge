import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import scripts.export_spy_multi_timeframe_context as spy_context_script

from quantforge.data import (
    AggregatedSessionBar,
    DevelopingBar,
    IntradayFetchResult,
    IntradayMarketDataCache,
)
from quantforge.timeframes import (
    BarCompletion,
    IntradayInterval,
    SessionInterval,
    Timeframe,
    TradingWeekInterval,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPOSITORY_ROOT / "examples" / "spy_multi_timeframe" / "fixture.json"
EXPECTED_EXPORT_ROOT = REPOSITORY_ROOT / "examples" / "spy_multi_timeframe" / "exports"


@pytest.fixture(scope="module")
def example_result(
    tmp_path_factory: pytest.TempPathFactory,
) -> spy_context_script.ExampleResult:
    return spy_context_script.build_example(
        FIXTURE_PATH, tmp_path_factory.mktemp("qf30-cache")
    )


def _timeframes() -> tuple[Timeframe, Timeframe, Timeframe, Timeframe]:
    return (
        Timeframe.us_equity(IntradayInterval(timedelta(minutes=5))),
        Timeframe.us_equity(IntradayInterval(timedelta(hours=4))),
        Timeframe.us_equity(SessionInterval()),
        Timeframe.us_equity(TradingWeekInterval()),
    )


def test_example_uses_one_cached_family_and_derives_all_timeframes(
    example_result: spy_context_script.ExampleResult,
) -> None:
    datasets = example_result.datasets

    assert datasets.cache_status == "seeded_local_cache_from_committed_fixture"
    assert len(datasets.source.bars) == 1056
    assert len(datasets.four_hour.bars) == 27
    assert len(datasets.daily.bars) == 14
    assert len(datasets.weekly.bars) == 3
    assert datasets.source.quality_report.is_complete
    assert datasets.four_hour.aggregation_report.is_complete
    assert datasets.daily.aggregation_report.is_complete
    assert datasets.weekly.aggregation_report.is_complete

    source_id = datasets.source.metadata.dataset_id
    assert datasets.family.canonical_source_snapshot_id == source_id
    assert {
        reference.canonical_source_snapshot_id
        for _, context in (
            *example_result.completed_contexts,
            *example_result.developing_contexts,
        )
        for aligned in context.timeframes
        if (reference := aligned.dataset_reference) is not None
    } == {source_id}
    assert {
        context.source_consistency.family_id
        for _, context in (
            *example_result.completed_contexts,
            *example_result.developing_contexts,
        )
    } == {datasets.family.family_id}


def test_four_hour_boundaries_and_early_close_values_are_hand_auditable(
    example_result: spy_context_script.ExampleResult,
) -> None:
    early_close = next(
        bar
        for bar in example_result.datasets.four_hour.bars
        if bar.session_date == date(2024, 7, 3)
    )

    assert early_close.start_timestamp == datetime(2024, 7, 3, 13, 30, tzinfo=UTC)
    assert early_close.end_timestamp == datetime(2024, 7, 3, 17, 0, tzinfo=UTC)
    assert early_close.actual_duration == timedelta(hours=3, minutes=30)
    assert early_close.completion is BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL
    assert (
        early_close.open,
        early_close.high,
        early_close.low,
        early_close.close,
        early_close.volume,
    ) == (
        Decimal("505.46"),
        Decimal("505.92"),
        Decimal("505.43"),
        Decimal("505.89"),
        Decimal("65793"),
    )

    normal_session = tuple(
        bar
        for bar in example_result.datasets.four_hour.bars
        if bar.session_date == date(2024, 7, 2)
    )
    assert tuple(bar.actual_duration for bar in normal_session) == (
        timedelta(hours=4),
        timedelta(hours=2, minutes=30),
    )
    assert normal_session[-1].completion is (
        BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL
    )


def test_completed_and_developing_contexts_are_causally_separate(
    example_result: spy_context_script.ExampleResult,
) -> None:
    _, four_hour, daily, weekly = _timeframes()
    completed = next(
        context
        for scenario, context in example_result.completed_contexts
        if scenario.name == "early_close_near_close"
    )
    developing = next(
        context
        for scenario, context in example_result.developing_contexts
        if scenario.name == "early_close_near_close"
    )

    completed_daily = completed.latest_bar_for(daily)
    completed_weekly = completed.latest_bar_for(weekly)
    assert isinstance(completed_daily, AggregatedSessionBar)
    assert isinstance(completed_weekly, AggregatedSessionBar)
    assert completed_daily.period_start_date == date(2024, 7, 2)
    assert completed_weekly.period_start_date == date(2024, 6, 24)
    assert all(
        bar.completion is not BarCompletion.DEVELOPING
        for aligned in completed.timeframes
        for bar in aligned.bars
    )

    for timeframe in (four_hour, daily, weekly):
        bar = developing.latest_bar_for(timeframe)
        assert isinstance(bar, DevelopingBar)
        assert bar.observed_end_timestamp == datetime(2024, 7, 3, 16, 55, tzinfo=UTC)
        assert bar.as_of == developing.as_of
        assert bar.expected_completion_boundary > developing.as_of

    developing_daily = developing.latest_bar_for(daily)
    assert isinstance(developing_daily, DevelopingBar)
    assert developing_daily.source_bar_count == 41
    assert (
        developing_daily.open,
        developing_daily.high,
        developing_daily.low,
        developing_daily.close,
        developing_daily.volume,
    ) == (
        Decimal("505.46"),
        Decimal("505.91"),
        Decimal("505.43"),
        Decimal("505.88"),
        Decimal("64206"),
    )


def test_cache_replay_and_export_are_deterministic_and_immutable(
    example_result: spy_context_script.ExampleResult, tmp_path: Path
) -> None:
    output_root = tmp_path / "exports"
    destination, status = spy_context_script.export_example(example_result, output_root)
    assert status == "created_immutable_export"
    assert destination.name == example_result.example_id
    first_bytes = {
        path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()
    }

    repeated, repeated_status = spy_context_script.export_example(
        example_result, output_root
    )
    assert repeated == destination
    assert repeated_status == "reused_immutable_export"
    assert {
        path.name: path.read_bytes() for path in repeated.iterdir() if path.is_file()
    } == first_bytes

    replay = spy_context_script.build_example(
        FIXTURE_PATH, example_result.datasets.cache.root
    )
    assert replay.datasets.cache_status == "replayed_local_cache"
    assert replay.example_id == example_result.example_id
    assert tuple(
        context.context_id for _, context in replay.completed_contexts
    ) == tuple(context.context_id for _, context in example_result.completed_contexts)

    (destination / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        spy_context_script.SpyContextExampleError,
        match="immutable export content differs",
    ):
        spy_context_script.export_example(example_result, output_root)


def test_cache_replay_rejects_changed_capabilities_identity(tmp_path: Path) -> None:
    fixture = spy_context_script.load_fixture(FIXTURE_PATH)
    expected = spy_context_script._source_fetch_result(  # pyright: ignore[reportPrivateUsage]
        fixture
    )
    cache_root = tmp_path / "cache"
    IntradayMarketDataCache(cache_root).persist(
        IntradayFetchResult(
            batch=expected.batch,
            raw_snapshots=expected.raw_snapshots,
            capabilities_configuration_id="different-fixture-capabilities",
        )
    )

    with pytest.raises(
        spy_context_script.SpyContextExampleError,
        match="capabilities_configuration_id",
    ):
        spy_context_script.build_example(FIXTURE_PATH, cache_root)


def test_checked_in_export_matches_a_fresh_offline_run(
    example_result: spy_context_script.ExampleResult, tmp_path: Path
) -> None:
    generated, _ = spy_context_script.export_example(
        example_result, tmp_path / "golden"
    )
    expected = EXPECTED_EXPORT_ROOT / example_result.example_id

    assert expected.is_dir()
    assert {
        path.name: path.read_bytes() for path in generated.iterdir() if path.is_file()
    } == {path.name: path.read_bytes() for path in expected.iterdir() if path.is_file()}


def test_script_runs_without_loading_an_indicator_backend(tmp_path: Path) -> None:
    check = """
import json
import sys
from pathlib import Path
import scripts.export_spy_multi_timeframe_context as example

result = example.build_example(Path(sys.argv[1]), Path(sys.argv[2]))
destination, status = example.export_example(result, Path(sys.argv[3]))
loaded = sorted(
    name
    for name in sys.modules
    if name == "quantforge.indicators"
    or name.startswith("quantforge.indicators.")
)
print(
    json.dumps(
        {"loaded": loaded, "status": status, "destination": str(destination)}
    )
)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            check,
            str(FIXTURE_PATH),
            str(tmp_path / "cache"),
            str(tmp_path / "exports"),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)

    assert summary["loaded"] == []
    assert summary["status"] == "created_immutable_export"
