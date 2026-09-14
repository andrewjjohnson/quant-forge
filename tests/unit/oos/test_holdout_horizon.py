"""Timestamp holdouts retain only decisions whose full duration fits inside."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from quantforge.oos import (
    HoldoutEvaluation,
    HoldoutLedger,
    OOSIntegrityError,
    load_oos_source,
)
from quantforge.oos._records import mapping, records
from quantforge.validation import (
    PurgePolicy,
    TemporalOffset,
    TimestampBoundary,
    ValidationInterval,
)
from quantforge.walk_forward import WalkForwardStudy
from tests.unit.validation.test_validation_plans import (
    _timestamp_outcome,  # pyright: ignore[reportPrivateUsage]
)

from .test_holdout_scope import midnight_study


@pytest.mark.parametrize(
    ("horizon", "minimum", "retained_count"),
    [
        (timedelta(0), 2, 2),
        (timedelta(hours=23), 1, 1),
        (timedelta(days=1), 1, 1),
        (timedelta(days=1), 2, 1),
        (timedelta(days=1, microseconds=1), 1, 0),
    ],
    ids=["zero", "subday", "exact-end", "below-minimum", "past-end"],
)
def test_timestamp_holdout_applies_duration_before_minimum_and_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    horizon: timedelta,
    minimum: int,
    retained_count: int,
) -> None:
    baseline = midnight_study(tmp_path / "baseline", timestamp_axis=True)
    original_plan = baseline.source.plan
    original_window = original_plan.final_holdout.window
    end = original_window.interval.end
    assert isinstance(end, TimestampBoundary)
    window = replace(
        original_window,
        interval=ValidationInterval(
            TimestampBoundary(end.timestamp - timedelta(days=1)), end
        ),
    )
    plan = replace(
        original_plan,
        environment=replace(
            original_plan.environment, outcomes=(_timestamp_outcome(horizon),)
        ),
        final_holdout=replace(original_plan.final_holdout, window=window),
        purge_policy=PurgePolicy(
            TemporalOffset.duration(horizon), TemporalOffset.duration(timedelta(0))
        ),
    )
    study = WalkForwardStudy(
        replace(baseline.study.config, plan=plan, minimum_test_observations=minimum),
        baseline.evaluator,
        tmp_path / "duration-study",
    )
    study.run()
    source = load_oos_source(plan, study.study_path)
    assert all(fold.artifact for fold in source.folds)
    selection = source.folds[-1].selection
    assert selection is not None
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    reserved = ledger.reserve(source)
    if retained_count < minimum:

        def forbidden(*args: object, **kwargs: object) -> None:
            pytest.fail("invalid duration holdout must not be evaluated")

        monkeypatch.setattr(HoldoutEvaluation, "_evaluate", forbidden)
        error = "minimum_test_observations" if retained_count else "outcome horizon"
        with pytest.raises(OOSIntegrityError, match=error):
            HoldoutEvaluation.prepare(
                source, baseline.evaluator, selection_fold_id=source.folds[-1].fold_id
            )
        baseline_evaluation = HoldoutEvaluation.prepare(
            baseline.source,
            baseline.evaluator,
            selection_fold_id=baseline.source.folds[-1].fold_id,
        )
        # Direct construction cannot bypass preparation during consumption.
        with pytest.raises(OOSIntegrityError, match=error):
            ledger.consume(
                replace(baseline_evaluation, source=source, selection=selection),
                run_id="invalid",
            )
        assert ledger.state(source) == reserved
        assert not tuple((ledger.root / "exposures").iterdir())
        return

    evaluation = HoldoutEvaluation.prepare(
        source, baseline.evaluator, selection_fold_id=source.folds[-1].fold_id
    )
    membership = evaluation.permitted.membership.study_observations
    assert len(membership) == 2
    assert len(evaluation.permitted.sessions) == retained_count
    final_session = baseline.evaluator.dataset.bars[-1].session_date
    expected_sessions = (final_session - timedelta(days=1), final_session)[
        :retained_count
    ]
    assert evaluation.permitted.sessions == expected_sessions
    assert evaluation.permitted.dataset.bars[-1].session_date == final_session
    assert (
        ledger.consume(evaluation, run_id="duration-holdout").result_reference
        is not None
    )
    payload = mapping(
        mapping(ledger.result(evaluation).to_primitive()["artifact"])["result"]
    )
    assert [row["session"] for row in records(payload["daily_equity"])] == [
        session.isoformat() for session in expected_sessions
    ]
