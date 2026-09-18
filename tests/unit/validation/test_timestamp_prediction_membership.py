"""QF-42 observations and QF-46 reach enter existing QF-8 selection/purge rules."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from quantforge.prediction import PredictionDecisionSchedule
from quantforge.validation import (
    BoundaryAxis,
    PartitionRole,
    PredictionMembershipSource,
    TimestampBoundary,
    ValidationPlanError,
    purge_partition_observations,
    select_window_observations,
)
from quantforge.walk_forward.partitions import partition
from tests.unit.walk_forward.fixtures import prediction_fixture
from tests.unit.walk_forward.timestamp_fixtures import (
    PRIMARY,
    instant,
    timestamp_fixture,
)


def test_explicit_axis_and_legacy_serialization(tmp_path: Path) -> None:
    legacy, _ = prediction_fixture(tmp_path)
    config, adapter = timestamp_fixture(tmp_path)
    assert legacy.plan.membership_axis is BoundaryAxis.EXCHANGE_SESSION
    assert "prediction_membership" not in legacy.plan.to_manifest()
    source = config.plan.prediction_membership
    assert source is not None
    assert config.plan.membership_axis is BoundaryAxis.TIMESTAMP
    assert config.plan.plan_id == replace(config.plan).plan_id
    assert config.plan.to_manifest()["prediction_membership"] == source.to_primitive()
    assert source.to_primitive()["schedule_id"] == source.schedule.schedule_id
    assert (
        tuple(k.timestamp for k in source.observations)
        == source.schedule.decision_timestamps
    )
    assert source.schedule.decision_timestamps == tuple(
        sorted(set(source.schedule.decision_timestamps))
    )
    assert len(set(source.schedule.decision_sessions)) < len(source.observations)
    assert source.family_manifest_id == adapter.series[0].dataset_family_manifest_id
    with pytest.raises(ValidationPlanError, match="observed-session"):
        replace(config.plan, prediction_membership=None)


@pytest.mark.parametrize(
    ("role", "start", "end"),
    [
        (PartitionRole.DEVELOPMENT, "10:00", "11:55"),
        (PartitionRole.SELECTION, "12:00", "12:55"),
        (PartitionRole.WALK_FORWARD_TEST, "13:00", "13:55"),
        (PartitionRole.FINAL_HOLDOUT, "14:00", "15:55"),
    ],
)
def test_exact_closed_boundaries_and_timezone(
    tmp_path: Path, role: PartitionRole, start: str, end: str
) -> None:
    config, _ = timestamp_fixture(tmp_path)
    source = config.plan.prediction_membership
    assert source is not None
    fold = config.plan.folds[1]
    window = {
        PartitionRole.DEVELOPMENT: fold.development,
        PartitionRole.SELECTION: fold.selection,
        PartitionRole.WALK_FORWARD_TEST: fold.test,
        PartitionRole.FINAL_HOLDOUT: config.plan.final_holdout.window,
    }[role]
    assert window is not None
    selected = select_window_observations(
        window, source.observations, source=source, source_timeframe=PRIMARY
    )
    assert selected.study_observations[0] == TimestampBoundary(
        instant(5, start).astimezone(ZoneInfo("America/New_York"))
    )
    assert selected.study_observations[-1] == TimestampBoundary(instant(5, end))
    assert not window.interval.contains(
        TimestampBoundary(instant(5, start) - timedelta(microseconds=1))
    )
    assert not window.interval.contains(
        TimestampBoundary(instant(5, end) + timedelta(microseconds=1))
    )
    assert not set(selected.warm_up_context).intersection(selected.study_observations)


@pytest.mark.parametrize(
    ("role", "retained", "purged"),
    [
        (PartitionRole.DEVELOPMENT, "11:20", "11:25"),
        (PartitionRole.SELECTION, "12:20", "12:25"),
        (PartitionRole.WALK_FORWARD_TEST, "13:20", "13:25"),
    ],
)
@pytest.mark.parametrize("embargo", [0, 10])
def test_reach_and_embargo_purge_exact_overlap(
    tmp_path: Path, role: PartitionRole, retained: str, purged: str, embargo: int
) -> None:
    config, adapter = timestamp_fixture(tmp_path, embargo=timedelta(minutes=embargo))
    plan = config.plan
    source = plan.prediction_membership
    assert source is not None
    assert plan.purge_policy.label_horizon.elapsed == timedelta(minutes=35)
    assert plan.purge_policy.label_horizon.exchange_sessions is None
    result = purge_partition_observations(
        plan, 1, role, source.observations, source=source
    )
    shift = timedelta(minutes=embargo)
    assert result.retained[-1] == TimestampBoundary(instant(5, retained) - shift)
    assert result.purged[0] == TimestampBoundary(instant(5, purged) - shift)
    projected = partition(adapter.dataset, plan, 1, role, minimum_observations=1)
    assert projected.decision_timestamps == tuple(
        key.timestamp for key in result.retained if isinstance(key, TimestampBoundary)
    )
    assert len(projected.sessions) == len(result.retained)
    assert len(set(projected.sessions)) == 1
    # Legacy metadata contains no future full daily bar at an intraday decision.
    assert projected.dataset.bars[-1].session_date < projected.sessions[0]


def test_schedule_certification_rejects_missing_bars_and_foreign_sources(
    tmp_path: Path,
) -> None:
    config, adapter = timestamp_fixture(tmp_path)
    membership = config.plan.prediction_membership
    assert membership is not None
    source = next(s for s in adapter.series if s.timeframe == PRIMARY)
    shortened = type(
        source
    )._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        source.dataset_reference,
        source.timeframe,
        source.bars[1:],
        dataset_family_manifest_id=source.dataset_family_manifest_id,
    )
    with pytest.raises(ValidationPlanError, match="missing a scheduled"):
        PredictionMembershipSource.capture(membership.schedule, shortened)
    with pytest.raises(ValidationPlanError, match="source"):
        purge_partition_observations(
            config.plan,
            0,
            PartitionRole.DEVELOPMENT,
            membership.observations,
            source=adapter.dataset,
        )
    with pytest.raises(ValidationPlanError, match="exact prefix"):
        purge_partition_observations(
            config.plan,
            0,
            PartitionRole.DEVELOPMENT,
            membership.observations[1:],
            source=membership,
        )


def test_prefix_membership_is_stable_with_unrelated_future_bars(tmp_path: Path) -> None:
    config, adapter = timestamp_fixture(tmp_path)
    membership = config.plan.prediction_membership
    assert membership is not None
    source = next(s for s in adapter.series if s.timeframe == PRIMARY)
    historical = type(source)._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        source.dataset_reference,
        source.timeframe,
        tuple(
            b
            for b in source.bars
            if b.end_timestamp <= membership.schedule.end_timestamp
        ),
        dataset_family_manifest_id=source.dataset_family_manifest_id,
    )
    assert (
        PredictionMembershipSource.capture(membership.schedule, historical)
        == membership
    )
    changed = PredictionMembershipSource.capture(
        PredictionDecisionSchedule(
            PRIMARY,
            membership.schedule.start_timestamp + timedelta(minutes=5),
            membership.schedule.end_timestamp,
        ),
        source,
    )
    assert (
        replace(config.plan, prediction_membership=changed).plan_id
        != config.plan.plan_id
    )
    prefix = tuple(
        key for key in membership.observations if key.timestamp <= instant(5, "14:00")
    )
    baseline = purge_partition_observations(
        config.plan,
        1,
        PartitionRole.WALK_FORWARD_TEST,
        membership.observations,
        source=membership,
    )
    assert (
        purge_partition_observations(
            config.plan, 1, PartitionRole.WALK_FORWARD_TEST, prefix, source=membership
        )
        == baseline
    )


def test_declared_duration_cannot_understate_qf46_alignment_reach(
    tmp_path: Path,
) -> None:
    from quantforge.prediction.outcome_temporal import OutcomeTemporalConfiguration
    from quantforge.validation import OutcomeProvenance
    from tests.unit.walk_forward.timestamp_fixtures import MetadataLabeler

    class UnderstatedLabeler(MetadataLabeler):
        @property
        def required_future_duration(self) -> timedelta:
            return timedelta(minutes=30)

    component = UnderstatedLabeler(
        OutcomeTemporalConfiguration.elapsed_duration(timedelta(minutes=30), PRIMARY)
    )
    with pytest.raises(ValidationPlanError, match="QF-46 temporal reach"):
        OutcomeProvenance.capture_timestamp(component)


@pytest.mark.parametrize("preceding_observations", [0, 2, 3])
def test_plan_requires_scheduled_primary_warm_up_at_first_window(
    tmp_path: Path, preceding_observations: int
) -> None:
    config, adapter = timestamp_fixture(tmp_path)
    membership = config.plan.prediction_membership
    assert membership is not None
    first_window = config.plan.folds[0].development
    required = first_window.warm_up_observations_for(PRIMARY)
    assert required == 3
    schedule = PredictionDecisionSchedule(
        PRIMARY,
        instant(4, "10:00") - timedelta(minutes=5 * preceding_observations),
        membership.schedule.end_timestamp,
    )
    captured = PredictionMembershipSource.capture(schedule, adapter.series[0])
    # The complete artifact has more history, but only the captured schedule
    # certifies prediction membership and its preceding warm-up observations.
    if preceding_observations < required:
        with pytest.raises(ValidationPlanError, match=r"preceding QF-42.*warm-up"):
            replace(config.plan, prediction_membership=captured)
    else:
        plan = replace(config.plan, prediction_membership=captured)
        selected = select_window_observations(
            first_window,
            captured.observations,
            source=captured,
            source_timeframe=PRIMARY,
        )
        assert len(selected.warm_up_context) == required
        assert plan.to_manifest()["prediction_membership"] == captured.to_primitive()


@pytest.mark.parametrize(
    ("fold_index", "role"),
    [
        (0, PartitionRole.DEVELOPMENT),
        (0, PartitionRole.SELECTION),
        (0, PartitionRole.WALK_FORWARD_TEST),
        (1, PartitionRole.DEVELOPMENT),
        (1, PartitionRole.SELECTION),
        (1, PartitionRole.WALK_FORWARD_TEST),
        (1, PartitionRole.FINAL_HOLDOUT),
    ],
)
def test_plan_checks_scheduled_warm_up_for_every_window(
    tmp_path: Path, fold_index: int, role: PartitionRole
) -> None:
    config, _ = timestamp_fixture(tmp_path)
    plan = config.plan
    membership = plan.prediction_membership
    assert membership is not None
    fold = plan.folds[fold_index]
    window = {
        PartitionRole.DEVELOPMENT: fold.development,
        PartitionRole.SELECTION: fold.selection,
        PartitionRole.WALK_FORWARD_TEST: fold.test,
        PartitionRole.FINAL_HOLDOUT: plan.final_holdout.window,
    }[role]
    assert window is not None
    preceding = membership.observations.index(window.interval.start)
    invalid = replace(
        window,
        warm_up_by_timeframe=tuple(
            replace(requirement, observations=preceding + 1)
            if requirement.timeframe == PRIMARY
            else requirement
            for requirement in window.warm_up_by_timeframe
        ),
    )
    folds = plan.folds
    holdout = plan.final_holdout
    if role is PartitionRole.FINAL_HOLDOUT:
        holdout = replace(holdout, window=invalid)
    else:
        field = {
            PartitionRole.DEVELOPMENT: "development",
            PartitionRole.SELECTION: "selection",
            PartitionRole.WALK_FORWARD_TEST: "test",
        }[role]
        changed = replace(fold, **{field: invalid})
        folds = tuple(
            changed if index == fold_index else item
            for index, item in enumerate(plan.folds)
        )
    with pytest.raises(ValidationPlanError, match=rf"{window.name!r}.*warm-up"):
        replace(plan, folds=folds, final_holdout=holdout)
