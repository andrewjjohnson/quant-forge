"""Exact QF-8 observations survive QF-39 freezes, resume, and QF-40 readback."""

import shutil
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from quantforge.configuration import configuration_identity
from quantforge.oos import (
    HoldoutEvaluation,
    HoldoutLedger,
    HoldoutState,
    aggregate_prediction,
    load_oos_source,
)
from quantforge.oos._records import mapping, records
from quantforge.oos.source import _selection  # pyright: ignore[reportPrivateUsage]
from quantforge.validation import (
    PartitionRole,
    PredictionMembershipSource,
    TimestampBoundary,
)
from quantforge.walk_forward import (
    FoldStatus,
    PredictionEvaluator,
    WalkForwardPersistenceError,
    WalkForwardStudy,
)
from quantforge.walk_forward.partitions import partition
from tests.unit.walk_forward.fixtures import prediction_fixture
from tests.unit.walk_forward.timestamp_fixtures import (
    PRIMARY,
    instant,
    timestamp_fixture,
)


def test_elapsed_execution_freeze_resume_oos_and_reserved_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, adapter = timestamp_fixture(tmp_path)
    study = WalkForwardStudy(config, adapter, tmp_path / "studies")
    result = study.run()
    assert all(f.status is FoldStatus.COMPLETED for f in result.folds)
    for index, fold in enumerate(result.folds):
        assert fold.selection is not None
        assert fold.artifact is not None
        frozen = fold.selection.to_primitive()
        for role, key in (
            (PartitionRole.DEVELOPMENT, "development"),
            (PartitionRole.SELECTION, "selection"),
            (PartitionRole.WALK_FORWARD_TEST, "test"),
        ):
            expected = partition(
                adapter.dataset, config.plan, index, role, minimum_observations=1
            )
            assert mapping(frozen["membership"])[key] == expected.to_primitive()
        payload = fold.artifact.snapshot.to_primitive()
        decisions = records(payload["decisions"])
        retained = mapping(mapping(frozen["membership"])["test"])[
            "evaluation_timestamps"
        ]
        assert [d["decision_timestamp"] for d in decisions] == retained
        for decision in decisions:
            prediction = mapping(decision["prediction_study"])
            wrapper = mapping(
                mapping(mapping(prediction["manifest"])["configuration"])[
                    "outcome_labeler"
                ]
            )
            assert "required_future_sessions" not in wrapper
            for row in records(prediction["rows"]):
                resolution = mapping(mapping(row["outcome"])["temporal_resolution"])
                assert (
                    mapping(mapping(resolution["request"])["anchor"])[
                        "decision_timestamp"
                    ]
                    == decision["decision_timestamp"]
                )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("completed resume / OOS readback must not execute research")

    monkeypatch.setattr(PredictionEvaluator, "select", forbidden)
    monkeypatch.setattr(PredictionEvaluator, "evaluate", forbidden)
    assert study.resume() == result
    source = load_oos_source(config.plan, study.study_path)
    # QF-40 reads the frozen membership without calling QF-8 partition builders.
    monkeypatch.setattr("quantforge.validation.select_window_observations", forbidden)
    monkeypatch.setattr("quantforge.validation.purge_partition_observations", forbidden)
    assert load_oos_source(config.plan, study.study_path) == source
    aggregate = aggregate_prediction(source)
    assert aggregate is not None
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    assert ledger.reserve(source).state is HoldoutState.RESERVED
    evaluation = HoldoutEvaluation.prepare(
        source, adapter, selection_fold_id=source.folds[-1].fold_id
    )
    assert evaluation.permitted.decision_timestamps[0] == instant(5, "14:00")
    assert evaluation.permitted.decision_timestamps[-1] == instant(5, "15:20")
    assert (
        evaluation.permitted.to_primitive()["prediction_membership"]
        == config.plan.to_manifest()["prediction_membership"]
    )
    from quantforge.experiments._holdout_membership_integrity import (
        validate_holdout_membership,
    )

    validate_holdout_membership(source, evaluation.permitted.to_primitive())
    assert ledger.state(source).state is HoldoutState.RESERVED


@pytest.mark.parametrize("change", ["axis", "schedule", "source", "boundary", "reach"])
def test_material_changes_cannot_reuse_completed_membership(
    tmp_path: Path, change: str
) -> None:
    config, adapter = timestamp_fixture(tmp_path)
    baseline = WalkForwardStudy(config, adapter, tmp_path / "base")
    baseline.run()
    if change == "axis":
        changed, other = prediction_fixture(tmp_path)
    elif change == "reach":
        changed, other = timestamp_fixture(tmp_path, horizon=timedelta(minutes=25))
    elif change == "source":
        changed, other = timestamp_fixture(
            tmp_path, source_id="changed-immutable-snapshot"
        )
    elif change == "schedule":
        membership = config.plan.prediction_membership
        assert membership is not None
        source = next(s for s in adapter.series if s.timeframe == PRIMARY)
        schedule = replace(
            membership.schedule,
            start_timestamp=membership.schedule.start_timestamp + timedelta(minutes=5),
        )
        changed = replace(
            config,
            plan=replace(
                config.plan,
                prediction_membership=PredictionMembershipSource.capture(
                    schedule, source
                ),
            ),
        )
        other = adapter
    else:
        folds = tuple(
            replace(
                f,
                test=replace(
                    f.test,
                    interval=replace(
                        f.test.interval, end=TimestampBoundary(instant(4 + i, "13:50"))
                    ),
                ),
            )
            for i, f in enumerate(config.plan.folds)
        )
        changed = replace(config, plan=replace(config.plan, folds=folds))
        other = adapter
    candidate = WalkForwardStudy(changed, other, tmp_path / "changed")
    assert candidate.study_id != baseline.study_id
    assert changed.plan.plan_id != config.plan.plan_id
    shutil.copytree(baseline.study_path, candidate.study_path)
    with pytest.raises(WalkForwardPersistenceError):
        candidate.resume()


@pytest.mark.parametrize("tamper", ["axis", "timestamp", "source"])
def test_rehashed_membership_must_match_timestamp_lineage(
    tmp_path: Path, tamper: str
) -> None:
    config, adapter = timestamp_fixture(tmp_path)
    study = WalkForwardStudy(config, adapter, tmp_path / "study")
    result = study.run()
    selection = result.folds[0].selection
    assert selection is not None
    record = selection.to_primitive()
    part = mapping(mapping(record["membership"])["test"])
    if tamper == "axis":
        mapping(part["prediction_membership"])["axis"] = "exchange_session"
    elif tamper == "timestamp":
        part["evaluation_timestamps"] = [instant(4, "13:05").isoformat()]
    else:
        mapping(mapping(part["prediction_membership"])["source_reference"])[
            "dataset_id"
        ] = "foreign"
    record["selection_id"] = configuration_identity(
        {k: v for k, v in record.items() if k != "selection_id"}
    )
    with pytest.raises(ValueError, match="timestamp membership"):
        _selection(
            record, mapping(record["study_definition"]), config.plan, 0, study.study_id
        )


def test_qf40_synthetic_holdout_keeps_exact_membership_through_consumption(
    tmp_path: Path,
) -> None:
    from quantforge.experiments._holdout_integrity import validate_holdout_artifact

    config, adapter = timestamp_fixture(tmp_path)
    study = WalkForwardStudy(config, adapter, tmp_path / "study")
    study.run()
    source = load_oos_source(config.plan, study.study_path)
    evaluation = HoldoutEvaluation.prepare(
        source, adapter, selection_fold_id=source.folds[-1].fold_id
    )
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    assert ledger.reserve(source).state is HoldoutState.RESERVED
    consumed = ledger.consume(evaluation, run_id="synthetic-metadata-only")
    assert consumed.state is HoldoutState.CONSUMED
    assert consumed.consumption is not None
    exported = ledger.result(evaluation).to_primitive()
    validate_holdout_artifact(source, consumed.consumption.to_primitive(), exported)
    assert ledger.consume(evaluation, run_id="retry") == consumed
