"""QF-63 large-history invocation counts and scale, without timing thresholds.

Every decision in this fixture sees at least 2,500 completed two-minute bars and
51 completed daily bars. Timings are recorded as descriptive properties only; the
assertions are structural (invocation counts, reuse and exactness).
"""

from collections import Counter
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from time import perf_counter
from typing import Any, cast

import pytest

import quantforge.prediction.study as study_module
from quantforge.data import IntradayBar, TimeframeBarSeries
from quantforge.data.multi_timeframe import MultiTimeframeContext
from quantforge.data.session_aggregation import AggregatedSessionBar
from quantforge.examples.spy_ema import TWO_MINUTES
from quantforge.indicators.timeframe import (
    ConfiguredTimeframeIndicator,
    TimeframeIndicatorOutput,
)
from quantforge.prediction.context import PredictionRuleContext
from quantforge.prediction.prepared_features import PreparedContextScope
from quantforge.prediction.window import iter_prediction_window_decisions
from quantforge.walk_forward.prediction import (
    _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.prepared_feature_fixtures import (
    SmokeScope,
    smoke_inputs,
    smoke_scope,
)

WARM_UP = 2_500


@pytest.fixture(scope="module")
def large(tmp_path_factory: pytest.TempPathFactory) -> SmokeScope:
    root = tmp_path_factory.mktemp("qf63-scale")
    return smoke_scope(
        smoke_inputs(root / "cache"),
        root / "plan",
        fixed=False,
        primary_warm_up=WARM_UP,
    )


@contextmanager
def counted() -> Generator[Counter[str]]:
    """Count the expensive operations QF-63 targets."""
    counts: Counter[str] = Counter()

    def wrap(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapper

    def wrap_property(name: str, original: Any) -> property:
        return property(wrap(name, cast(property, original).fget))  # pyright: ignore[reportArgumentType]

    targets: list[tuple[object, str, object]] = [
        (IntradayBar, "bar_id", wrap_property("bar_id", IntradayBar.bar_id)),
        (
            AggregatedSessionBar,
            "bar_id",
            wrap_property("bar_id", AggregatedSessionBar.bar_id),
        ),
        (
            PredictionRuleContext,
            "values_primitive",
            wrap("values_primitive", PredictionRuleContext.values_primitive),
        ),
        (
            ConfiguredTimeframeIndicator,
            "calculate",
            wrap("indicator_calculation", ConfiguredTimeframeIndicator.calculate),
        ),
        (
            TimeframeIndicatorOutput,
            "__post_init__",
            wrap("indicator_array_validation", TimeframeIndicatorOutput.__post_init__),
        ),
        (
            MultiTimeframeContext,
            "__post_init__",
            wrap("full_context_validation", MultiTimeframeContext.__post_init__),
        ),
        (
            MultiTimeframeContext,
            "context_id",
            wrap_property("context_identity", MultiTimeframeContext.context_id),
        ),
        (
            TimeframeBarSeries,
            "_from_validated_artifact",
            classmethod(
                wrap(
                    "series_materialization",
                    TimeframeBarSeries._from_validated_artifact.__func__,  # pyright: ignore[reportPrivateUsage,reportFunctionMemberAccess]
                )
            ),
        ),
        (
            study_module,
            "validate_prediction_context_sources",
            wrap(
                "full_source_validation",
                study_module.validate_prediction_context_sources,
            ),
        ),
    ]
    with ExitStack() as stack:
        for owner, name, replacement in targets:
            patch = pytest.MonkeyPatch()
            stack.callback(patch.undo)
            patch.setattr(owner, name, replacement)
        yield counts


def execute(
    scope: SmokeScope,
    provider: _PermittedContextProvider,
    study: Any,
    *,
    start: int,
    count: int,
    session: Any = None,
) -> list[float]:
    decisions = iter_prediction_window_decisions(
        scope.session() if session is None else session,
        study,
        schedule=scope.schedule,
        context_provider=provider,
        dataset_family_fingerprint=scope.adapter.series[0].dataset_reference.family_id,
        start_sequence=start,
    )
    elapsed: list[float] = []
    for _ in range(count):
        began = perf_counter()
        decision = next(decisions)
        decision.to_primitive()
        elapsed.append(perf_counter() - began)
    return elapsed


def test_large_history_invocations_startup_and_steady_state(
    large: SmokeScope, record_property: Callable[[str, object], None]
) -> None:
    study = large.study("8/48")
    decisions = len(large.schedule.decision_timestamps)
    assert decisions == 64
    reference = large.provider(reference=True)
    reference_session = large.session()
    # Reference: a bounded late sample (every decision re-derives all history).
    # One warm decision absorbs the session's one-time outcome preparation.
    execute(large, reference, study, start=40, count=1, session=reference_session)
    with counted() as reference_counts:
        reference_seconds = execute(
            large, reference, study, start=41, count=5, session=reference_session
        )
    began = perf_counter()
    provider = large.provider()
    preparation_seconds = perf_counter() - began
    prepared = cast(PreparedContextScope, provider._prepared_features)  # pyright: ignore[reportPrivateUsage]
    run = prepared.runs[TWO_MINUTES.configuration_id]
    assert len(run.bars) >= WARM_UP + decisions
    session = large.session()
    execute(large, provider, study, start=0, count=1, session=session)
    with counted() as prepared_counts:
        began = perf_counter()
        prepared_seconds = execute(
            large, provider, study, start=1, count=decisions - 1, session=session
        )
        prepared_total = perf_counter() - began
    statistics = prepared.statistics()
    # No per-decision re-hashing, re-validation, full traversal or TA-Lib work.
    # The only calculations: the new daily EMA50 series when the rolling daily
    # start shifts at the first close (one build plus its first-use proof).
    assert prepared_counts["values_primitive"] == 0
    assert prepared_counts["indicator_calculation"] == 2
    assert prepared_counts["indicator_array_validation"] == 2
    assert prepared_counts["full_context_validation"] == 0
    assert prepared_counts["full_source_validation"] == 0
    # The single candidate's bounded QF-11 outcome path is unchanged work.
    assert prepared_counts["series_materialization"] <= 3
    assert prepared_counts["bar_id"] < 200
    assert prepared_counts["context_identity"] == decisions - 1
    # Reference work grows with the >= 2,500-bar history at every decision.
    assert reference_counts["values_primitive"] == 3 * 5
    assert reference_counts["indicator_calculation"] == 3 * 5
    assert reference_counts["full_context_validation"] == 5
    assert reference_counts["bar_id"] >= 5 * 10 * WARM_UP
    assert statistics["contexts"] == decisions
    assert statistics["unique_series"] == statistics["series_built"] == 4
    assert statistics["series_reused"] == 3 * decisions - 4
    measurements: dict[str, object] = {
        "reference_seconds_per_decision": sum(reference_seconds) / 5,
        "prepared_seconds_per_decision": sum(prepared_seconds) / len(prepared_seconds),
        "prepared_total_seconds": prepared_total,
        "preparation_seconds": preparation_seconds,
        "reference_bar_ids_per_decision": reference_counts["bar_id"] / 5,
        "prepared_bar_ids_total": prepared_counts["bar_id"],
        "prepared_run_bars": len(run.bars),
        **{f"statistic_{name}": value for name, value in statistics.items()},
    }
    for name, value in measurements.items():
        record_property(name, value)
    print(measurements)


def test_three_trials_share_immutable_backing_and_series(
    large: SmokeScope, record_property: Callable[[str, object], None]
) -> None:
    provider = large.provider()
    prepared = cast(PreparedContextScope, provider._prepared_features)  # pyright: ignore[reportPrivateUsage]
    session = large.session()
    trial_seconds: dict[str, float] = {}
    for pair in ("8/40", "8/48", "12/60"):
        began = perf_counter()
        execute(large, provider, large.study(pair), start=0, count=64, session=session)
        trial_seconds[pair] = perf_counter() - began
    statistics = prepared.statistics()
    assert statistics["unique_series"] == statistics["series_built"] == 7
    assert statistics["series_reused"] == 3 * 64 * 3 - 7
    run = prepared.runs[TWO_MINUTES.configuration_id]
    series = prepared._series.values()  # pyright: ignore[reportPrivateUsage]
    # Structural sharing: series reuse the run's bar objects and identity strings.
    for entry in series:
        output = entry.output
        if output.source_timeframe != TWO_MINUTES:
            continue
        offset = len(run.bars) - len(output.bar_ids)
        assert all(
            a is b for a, b in zip(output.bar_ids, run.bar_ids[offset:], strict=True)
        )
        assert all(
            a is b
            for a, b in zip(
                output.bar_end_timestamps, run.end_timestamps[offset:], strict=True
            )
        )
    assert all(
        bar is source for bar, source in zip(run.bars, run.source.bars[run.offset :])
    )
    report = prepared.memory_report()
    runs = cast(dict[str, dict[str, int]], report["runs"])
    series_report = cast(list[dict[str, Any]], report["series"])
    total = sum(
        sum(item[key] for key in item if key.endswith("_bytes"))
        for item in runs.values()
    )
    total += sum(item["values_bytes"] + item["index_bytes"] for item in series_report)
    assert len(series_report) == 7
    for name, value in {
        **{f"trial_{pair}_seconds": seconds for pair, seconds in trial_seconds.items()},
        "prepared_state_bytes": total,
        "prepared_series": len(series_report),
    }.items():
        record_property(name, value)
    print(trial_seconds, total)
