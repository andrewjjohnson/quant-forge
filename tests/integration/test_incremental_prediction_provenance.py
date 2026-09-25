"""Actual QF-11/42/52/48 and QF-49/47 contracts through incremental persistence."""

from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from quantforge.data.prediction_views import bounded_prediction_view
from quantforge.prediction import (
    PredictionDecisionSchedule,
    PredictionStudy,
    SignalFeatureCandidate,
    intraday_excursion_outcome,
    intraday_forward_return_outcome,
    intraday_target_stop_outcome,
)
from quantforge.prediction.study import prepare_prediction_study_dataset
from quantforge.prediction.window_compact import CompactPredictionWindowResult
from quantforge.prediction.window_execution import (
    run_incremental_prediction_window_in_session,
)
from quantforge.validation import PredictionMembershipSource
from quantforge.walk_forward import FoldStatus, WalkForwardStudy
from tests.integration.test_bounded_prediction_workflow import workflow_fixture
from tests.integration.test_compact_prediction_provenance import (
    WindowProvider,
    make_window,
    verify,
)
from tests.integration.test_intraday_prediction_provenance import (
    DECISION,
    TWO_MINUTES,
    Fixture,
    cached_fixture,
    study_inputs,
)
from tests.integration.test_intraday_prediction_provenance import fixture as fixture
from tests.unit.helpers import SESSIONS
from tests.unit.walk_forward.test_incremental_prediction import compact_adapter


@pytest.mark.parametrize(
    "outcome_name",
    ["forward10", "forward30", "forward60", "forward120", "excursion", "target_stop"],
)
def test_real_incremental_contracts(
    fixture: Fixture, tmp_path: Path, outcome_name: str
) -> None:
    if outcome_name.startswith("forward"):
        outcome = intraday_forward_return_outcome(
            timedelta(minutes=int(outcome_name.removeprefix("forward"))),
            fixture.primary,
        )
    elif outcome_name == "excursion":
        outcome = intraday_excursion_outcome(timedelta(minutes=60), fixture.primary)
    else:
        outcome = intraday_target_stop_outcome(
            timedelta(minutes=60), fixture.primary, Decimal("0.003"), Decimal("0.002")
        )
    rule, _ = study_inputs(fixture)
    study = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
        rule,
        outcome.labeler,
        outcome.evaluator,
        outcome_source=fixture.primary,
    )
    view = bounded_prediction_view(fixture.dataset, DECISION)
    schedule = PredictionDecisionSchedule(
        TWO_MINUTES, DECISION, DECISION + timedelta(minutes=4)
    )
    reader = run_incremental_prediction_window_in_session(
        prepare_prediction_study_dataset(view),
        study,
        path=tmp_path / "window.jsonl",
        schedule=schedule,
        context_provider=WindowProvider(fixture),
        dataset_family_fingerprint=fixture.primary.dataset_reference.family_id,
        context_environment={"provider": "immutable_synthetic_fixture"},
        canonical_metadata=fixture.dataset.metadata,
    )
    expected = make_window(fixture, outcome_name)
    assert (
        reader.path.read_bytes()
        == CompactPredictionWindowResult.from_window(expected).serialize()
    )
    membership = PredictionMembershipSource.capture(schedule, fixture.primary)
    assert reader.evidence.schedule == membership.schedule
    verify(reader, fixture, expected)


def test_bounded_walk_forward_produces_compact_windows(tmp_path: Path) -> None:
    source_fixture = cached_fixture(tmp_path / "cache", session_dates=SESSIONS[:7])
    config, adapter = workflow_fixture(tmp_path, source_fixture)
    study = WalkForwardStudy(config, compact_adapter(adapter), tmp_path / "studies")
    result = study.run()
    assert all(f.status is FoldStatus.COMPLETED for f in result.folds)
    assert list(study.study_path.rglob("prediction-window.jsonl"))
    assert study.resume() == result
