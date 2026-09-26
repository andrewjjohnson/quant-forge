"""Exact indexed/reference context selection, immutable scope and operation counts."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

import quantforge.validation.context as reference_module
import quantforge.validation.prepared_context as prepared_module
from quantforge.data import TimeframeBarSeries
from quantforge.timeframes import IntradayInterval, Timeframe
from quantforge.validation import (
    PreparedPredictionContext,
    PreparedValidationPlan,
    TimestampBoundary,
    ValidationPlan,
    ValidationPlanError,
    ValidationWindow,
    select_prediction_context_observations,
)
from tests.integration.test_validation_multi_timeframe_prediction import (
    PredictionCase,
)
from tests.integration.test_validation_multi_timeframe_prediction import (
    prediction_case as session_case,
)
from tests.unit.walk_forward.timestamp_fixtures import timestamp_fixture

__all__ = ["session_case"]


@pytest.fixture(scope="module")
def timestamp_case(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return timestamp_fixture(tmp_path_factory.mktemp("prepared-timestamps"))


def capture(
    plan: ValidationPlan,
    window: ValidationWindow,
    series: tuple[TimeframeBarSeries, ...],
) -> PreparedPredictionContext:
    result = PreparedPredictionContext.capture(
        plan, window, series=series, input_identity="validated-view"
    )
    assert result is not None
    return result


def equivalent(
    prepared: PreparedPredictionContext,
    plan: ValidationPlan,
    window: ValidationWindow,
    sources: tuple[TimeframeBarSeries, ...],
    timestamp: datetime,
) -> None:
    for source in sources:
        reference = select_prediction_context_observations(
            plan, window, source=source, as_of=TimestampBoundary(timestamp)
        )
        indexed, selected = prepared.select(
            source.timeframe, as_of=TimestampBoundary(timestamp)
        )
        assert indexed == reference
        assert indexed.to_primitive() == reference.to_primitive()
        keys = {
            *reference.source_selection.warm_up_context,
            *reference.source_selection.study_observations,
        }
        assert selected.bars == tuple(
            bar for bar in source.bars if TimestampBoundary(bar.end_timestamp) in keys
        )
        assert all(bar.end_timestamp <= timestamp for bar in selected.bars)
        assert selected.dataset_reference == source.dataset_reference
        assert selected.dataset_family_manifest_id == source.dataset_family_manifest_id


def test_timestamp_adjacent_irregular_reverse_and_daily_equivalence(
    timestamp_case: Any,
) -> None:
    config, adapter = timestamp_case
    plan = config.plan
    window = plan.folds[0].selection
    prepared = capture(plan, window, adapter.series)
    membership = plan.prediction_membership
    timestamps = tuple(
        t
        for t in membership.schedule.decision_timestamps
        if window.interval.contains(TimestampBoundary(t))
    )
    for t in (*timestamps, *timestamps[::3], *reversed(timestamps)):
        equivalent(prepared, plan, window, adapter.series, t)
    daily = adapter.series[1]
    anchors = [
        prepared.select(daily.timeframe, as_of=TimestampBoundary(t))[1].bars[-1]
        for t in timestamps
    ]
    assert all(bar == anchors[0] for bar in anchors)
    assert anchors[0].end_timestamp < timestamps[0]
    with pytest.raises(ValidationPlanError, match="captured schedule"):
        prepared.select(
            adapter.series[0].timeframe,
            as_of=TimestampBoundary(timestamps[0] + timedelta(seconds=1)),
        )


def test_session_weekly_anchor_and_holiday_gap(session_case: PredictionCase) -> None:
    case = session_case
    # The existing generic confluence fixture includes four timeframes and the
    # July 3 early close / July 4 holiday. Keep original source values unchanged.
    for window, instants in (
        (
            case.plan.folds[0].selection,
            (
                datetime(2024, 7, 3, 13, 30, tzinfo=UTC),
                datetime(2024, 7, 3, 17, tzinfo=UTC),
            ),
        ),
        (
            case.plan.folds[0].test,
            (
                datetime(2024, 7, 5, 13, 30, tzinfo=UTC),
                datetime(2024, 7, 5, 14, tzinfo=UTC),
                datetime(2024, 7, 5, 20, tzinfo=UTC),
            ),
        ),
    ):
        assert window is not None
        prepared = capture(case.plan, window, case.series)
        for timestamp in instants:
            # Some timeframes intentionally have insufficient warm-up at July 3;
            # both implementations must reject exactly that availability case.
            for source in case.series:
                try:
                    select_prediction_context_observations(
                        case.plan,
                        window,
                        source=source,
                        as_of=TimestampBoundary(timestamp),
                    )
                except ValidationPlanError:
                    with pytest.raises(ValidationPlanError):
                        prepared.select(
                            source.timeframe, as_of=TimestampBoundary(timestamp)
                        )
                else:
                    equivalent(prepared, case.plan, window, (source,), timestamp)


def test_plan_capture_is_deterministic_and_does_not_memoize_mutable_plans(
    timestamp_case: Any,
) -> None:
    config, adapter = timestamp_case
    plan = config.plan
    prepared = capture(plan, plan.folds[0].selection, adapter.series)
    assert prepared.plan.plan_id == plan.plan_id
    assert PreparedValidationPlan.capture(replace(plan)) == prepared.plan
    changed = replace(plan, name="different scientific scope")
    assert PreparedValidationPlan.capture(changed).plan_id != prepared.plan.plan_id
    with pytest.raises(ValidationPlanError, match="plan differs"):
        prepared.plan.validate_compatible(changed)
    with pytest.raises(FrozenInstanceError):
        setattr(prepared.plan, "plan_id", "forged")

    # A mutable nested runtime subclass cannot poison a cached live-plan ID:
    # capture is a detached boundary, and subsequent reuse rechecks the live ID.
    class MutablePlan(ValidationPlan):
        def _identity_primitive(self) -> Any:
            result = super()._identity_primitive()
            result["runtime_revision"] = self.__dict__.get("revision", 0)
            return result

    mutable = MutablePlan(
        plan.name,
        plan.environment,
        plan.folds,
        plan.final_holdout,
        plan.purge_policy,
        plan.training_window_mode,
        prediction_membership=plan.prediction_membership,
    )
    previous = PreparedValidationPlan.capture(mutable)
    mutable.__dict__["revision"] = 1
    assert previous.plan_id != mutable.plan_id
    with pytest.raises(ValidationPlanError):
        previous.validate_compatible(mutable)


@pytest.mark.parametrize("change", ["plan", "window", "view", "source", "timeframe"])
def test_incompatible_preparation_fails_closed(
    timestamp_case: Any, change: str
) -> None:
    config, adapter = timestamp_case
    plan, window, sources = config.plan, config.plan.folds[0].selection, adapter.series
    prepared = capture(plan, window, sources)
    identity = "validated-view"
    if change == "plan":
        plan = replace(plan, name="foreign")
    elif change == "window":
        window = plan.folds[0].test
    elif change == "view":
        identity = "same-prices-different-ancestry"
    elif change == "timeframe":
        with pytest.raises(ValidationPlanError, match="timeframe/session"):
            prepared.select(
                Timeframe.us_equity(IntradayInterval(timedelta(minutes=10))),
                as_of=window.interval.start,
            )
        return
    else:
        source = sources[0]
        source = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
            replace(source.dataset_reference, dataset_id="foreign-source")
            if change == "source"
            else source.dataset_reference,
            Timeframe.us_equity(IntradayInterval(timedelta(minutes=10)))
            if change == "timeframe"
            else source.timeframe,
            source.bars,
            dataset_family_manifest_id=source.dataset_family_manifest_id,
        )
        sources = (source, *sources[1:])
    with pytest.raises(ValidationPlanError):
        prepared.validate_compatible(
            plan, window, series=sources, input_identity=identity
        )


def test_full_hash_and_source_scan_counts_do_not_scale_per_decision(
    timestamp_case: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, adapter = timestamp_case
    plan, window, sources = config.plan, config.plan.folds[0].selection, adapter.series
    calls = {"hash": 0, "scan": 0, "build": 0, "lookup": 0}
    original_id = ValidationPlan.plan_id.fget  # pyright: ignore[reportFunctionMemberAccess]
    assert original_id is not None
    scan = prepared_module.validate_source_observations
    build = prepared_module.PredictionSourceIndex.build.__func__
    lookup = prepared_module.PredictionSourceIndex.bounds

    def plan_id(self: ValidationPlan) -> str:
        calls["hash"] += 1
        return cast(str, original_id(self))

    def scan_count(*args: Any, **kwargs: Any) -> Any:
        calls["scan"] += 1
        return scan(*args, **kwargs)

    def build_count(cls: Any, *args: Any, **kwargs: Any) -> Any:
        calls["build"] += 1
        return build(cls, *args, **kwargs)

    def lookup_count(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["lookup"] += 1
        return lookup(self, *args, **kwargs)

    monkeypatch.setattr(ValidationPlan, "plan_id", property(plan_id))
    monkeypatch.setattr(reference_module, "validate_source_observations", scan_count)
    monkeypatch.setattr(prepared_module, "validate_source_observations", scan_count)
    monkeypatch.setattr(
        prepared_module.PredictionSourceIndex, "build", classmethod(build_count)
    )
    monkeypatch.setattr(prepared_module.PredictionSourceIndex, "bounds", lookup_count)
    timestamp = window.interval.start
    for _ in range(7):
        for source in sources:
            select_prediction_context_observations(
                plan, window, source=source, as_of=timestamp
            )
    assert calls == {"hash": 14, "scan": 14, "build": 0, "lookup": 0}
    calls.update(dict.fromkeys(calls, 0))
    prepared = capture(plan, window, sources)
    for _ in range(1000):
        for source in sources:
            prepared.select(source.timeframe, as_of=timestamp)
    assert calls == {"hash": 1, "scan": 2, "build": 2, "lookup": 2000}
    assert tuple(index.source for index in prepared.indexes) == sources


def test_early_close_cross_session_holiday_and_daily_open(timestamp_case: Any) -> None:
    from datetime import date

    from quantforge.validation import ValidationFold, ValidationInterval
    from tests.unit.data.test_multi_timeframe import (
        _session_bar,  # pyright: ignore[reportPrivateUsage]
    )

    config, adapter = timestamp_case
    plan = config.plan
    primary = next(
        source
        for source in adapter.series
        if isinstance(source.timeframe.interval, IntradayInterval)
    )
    daily = next(source for source in adapter.series if source is not primary)
    # Extend only the named synthetic daily artifact with preceding warm-up.
    daily = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        daily.dataset_reference,
        daily.timeframe,
        (
            _session_bar(daily.timeframe, date(2024, 6, 27)),
            _session_bar(daily.timeframe, date(2024, 6, 28)),
            *daily.bars,
        ),
        dataset_family_manifest_id=daily.dataset_family_manifest_id,
    )

    def interval(start: str, end: str) -> ValidationInterval:
        return ValidationInterval(
            TimestampBoundary(datetime.fromisoformat(start)),
            TimestampBoundary(datetime.fromisoformat(end)),
        )

    development = replace(
        plan.folds[0].development,
        interval=interval("2024-07-01T14:00:00+00:00", "2024-07-02T16:00:00+00:00"),
    )
    test = replace(
        plan.folds[0].test,
        interval=interval("2024-07-03T13:35:00+00:00", "2024-07-05T20:00:00+00:00"),
    )
    plan = replace(plan, folds=(ValidationFold("calendar", development, test),))
    prepared = capture(plan, test, (primary, daily))
    times = tuple(
        datetime.fromisoformat(t)
        for t in (
            "2024-07-03T13:35:00+00:00",  # first completed primary
            "2024-07-03T13:40:00+00:00",
            "2024-07-03T16:55:00+00:00",  # early close approaching
            "2024-07-03T17:00:00+00:00",  # actual early close
            "2024-07-05T13:35:00+00:00",  # holiday gap / next session
            "2024-07-05T13:40:00+00:00",
            "2024-07-05T19:55:00+00:00",
            "2024-07-05T20:00:00+00:00",  # normal close
        )
    )
    for timestamp in (*times, *reversed(times)):
        equivalent(prepared, plan, test, (primary, daily), timestamp)
    anchors = [
        prepared.select(daily.timeframe, as_of=TimestampBoundary(t))[1]
        .bars[-1]
        .end_timestamp
        for t in times
    ]
    assert anchors[0].date() == date(2024, 7, 2)
    assert anchors[0] == anchors[1] == anchors[2]
    assert anchors[3].date() == date(2024, 7, 3)
    assert anchors[3] == anchors[4] == anchors[5] == anchors[6]
    assert anchors[7].date() == date(2024, 7, 5)
    with pytest.raises(ValidationPlanError, match="captured schedule"):
        prepared.select(
            primary.timeframe,
            as_of=TimestampBoundary(datetime(2024, 7, 4, 16, tzinfo=UTC)),
        )


def test_source_gap_exact_lookup_and_insufficient_warmup(timestamp_case: Any) -> None:
    config, adapter = timestamp_case
    plan, window = config.plan, config.plan.folds[0].selection
    primary = next(
        source
        for source in adapter.series
        if isinstance(source.timeframe.interval, IntradayInterval)
    )
    start = window.interval.start.timestamp
    source = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        primary.dataset_reference,
        primary.timeframe,
        tuple(bar for bar in primary.bars if bar.end_timestamp != start),
        dataset_family_manifest_id=primary.dataset_family_manifest_id,
    )
    prepared = capture(plan, window, (source,))
    equivalent(prepared, plan, window, (source,), start)
    _, selected = prepared.select(source.timeframe, as_of=window.interval.start)
    assert selected.bars[-1].end_timestamp < start
    # QF-11/QF-42 still reject a missing exact primary at execution; indexed
    # context selection itself preserves the reference's preceding-anchor rule.
    short = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        primary.dataset_reference,
        primary.timeframe,
        tuple(bar for bar in primary.bars if bar.end_timestamp >= start),
        dataset_family_manifest_id=primary.dataset_family_manifest_id,
    )
    insufficient = capture(plan, window, (short,))
    for select in (
        lambda: insufficient.select(short.timeframe, as_of=window.interval.start),
        lambda: select_prediction_context_observations(
            plan, window, source=short, as_of=window.interval.start
        ),
    ):
        with pytest.raises(ValidationPlanError, match="insufficient"):
            select()


def test_mutable_source_uses_safe_reference_fallback(timestamp_case: Any) -> None:
    config, adapter = timestamp_case

    class MutableSeries(TimeframeBarSeries):
        pass

    source = adapter.series[0]
    mutable = MutableSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        source.dataset_reference,
        source.timeframe,
        source.bars,
        dataset_family_manifest_id=source.dataset_family_manifest_id,
    )
    assert (
        PreparedPredictionContext.capture(
            config.plan,
            config.plan.folds[0].selection,
            series=(mutable,),
            input_identity="validated-view",
        )
        is None
    )


def test_source_session_policy_cannot_reuse_index(timestamp_case: Any) -> None:
    from datetime import time

    from quantforge.timeframes import SessionScope

    config, adapter = timestamp_case
    window = config.plan.folds[0].selection
    prepared = capture(config.plan, window, adapter.series)
    timeframe = adapter.series[0].timeframe
    changed = replace(
        timeframe,
        session_policy=replace(
            timeframe.session_policy,
            scope=SessionScope.EXTENDED_HOURS,
            extended_hours_start=time(4),
            extended_hours_end=time(20),
        ),
    )
    with pytest.raises(ValidationPlanError, match="timeframe/session"):
        prepared.select(changed, as_of=window.interval.start)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2024-07-04T16:00:00+00:00",  # holiday rejects through calendar boundary
        "2024-07-03T16:00:00+00:00",  # observed session outside window
        "2024-07-05T13:29:00+00:00",  # pre-open
        "2024-07-05T20:01:00+00:00",  # post-close
    ],
)
def test_session_rejection_type_and_message_match_reference(
    session_case: PredictionCase,
    timestamp: str,
) -> None:
    case = session_case
    window = case.plan.folds[0].test
    source = case.series[0]
    prepared = capture(case.plan, window, case.series)
    as_of = TimestampBoundary(datetime.fromisoformat(timestamp))
    with pytest.raises(ValueError, match=r".+") as old:
        select_prediction_context_observations(
            case.plan, window, source=source, as_of=as_of
        )
    with pytest.raises(type(old.value)) as new:
        prepared.select(source.timeframe, as_of=as_of)
    assert str(new.value) == str(old.value)


def test_timestamp_window_rejection_matches_reference(timestamp_case: Any) -> None:
    config, adapter = timestamp_case
    window = config.plan.folds[0].selection
    source = adapter.series[0]
    prepared = capture(config.plan, window, adapter.series)
    as_of = TimestampBoundary(window.interval.start.timestamp - timedelta(minutes=5))
    with pytest.raises(ValidationPlanError) as old:
        select_prediction_context_observations(
            config.plan, window, source=source, as_of=as_of
        )
    with pytest.raises(ValidationPlanError) as new:
        prepared.select(source.timeframe, as_of=as_of)
    assert str(new.value) == str(old.value)
