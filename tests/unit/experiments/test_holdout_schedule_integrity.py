"""A canonical partial window is not the full requested final holdout."""

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, inspect_validation
from quantforge.experiments._json import mapping, text
from quantforge.experiments._window_integrity import validate_window_snapshot
from quantforge.oos import HoldoutEvaluation, HoldoutLedger, load_oos_source
from quantforge.prediction import PredictionDecisionSchedule
from quantforge.timeframes import resolve_exchange_session
from quantforge.validation import ExchangeSessionBoundary
from quantforge.walk_forward import WalkForwardStudy
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_window_session_integrity import refresh_window_manifest
from tests.unit.helpers import SESSIONS
from tests.unit.oos.conftest import CompletedStudy
from tests.unit.walk_forward.fixtures import prediction_fixture


@pytest.mark.parametrize(
    "shortened",
    [
        "first_session",
        "last_session",
        "first_bar",
        "last_bar",
        "same_decisions",
        "empty",
    ],
)
def test_consumed_holdout_requires_full_request_derived_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shortened: str
) -> None:
    config, evaluator = prediction_fixture(tmp_path)
    holdout = config.plan.final_holdout
    holdout_window = replace(
        holdout.window,
        interval=replace(
            holdout.window.interval, start=ExchangeSessionBoundary(SESSIONS[12])
        ),
    )
    config = replace(
        config,
        plan=replace(
            config.plan,
            folds=config.plan.folds[:1],
            final_holdout=replace(holdout, window=holdout_window),
        ),
    )
    study = WalkForwardStudy(config, evaluator, tmp_path / "walk-forward")
    study.run()
    completed = CompletedStudy(
        study, load_oos_source(config.plan, study.study_path), evaluator
    )
    source = completed.source
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    consumed = ledger.consume(
        HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        ),
        run_id="complete-holdout-schedule",
    )
    assert consumed.result_reference is not None
    path = ledger.root / text(consumed.result_reference.to_primitive()["path"])
    result = read_record(path)
    artifact = mapping(result["artifact"])
    window = mapping(artifact["result"])
    manifest = mapping(window["manifest"])
    recorded = mapping(manifest["schedule"])
    primary = next(
        timeframe
        for timeframe in source.plan.environment.timeframes
        if timeframe.to_primitive() == recorded["primary_timeframe"]
    )
    start = datetime.fromisoformat(text(recorded["start_timestamp"]))
    end = datetime.fromisoformat(text(recorded["end_timestamp"]))
    full = PredictionDecisionSchedule(primary, start, end)
    assert len(set(full.decision_sessions)) > 1
    if shortened == "first_session":
        session = sorted(set(full.decision_sessions))[1]
        start = resolve_exchange_session(session, primary.session_policy).open_timestamp
    elif shortened == "last_session":
        session = sorted(set(full.decision_sessions))[-2]
        end = resolve_exchange_session(session, primary.session_policy).close_timestamp
    elif shortened == "first_bar":
        start = full.decision_timestamps[0] + timedelta(microseconds=1)
    elif shortened == "last_bar":
        end = full.decision_timestamps[-1] - timedelta(microseconds=1)
    elif shortened == "same_decisions":
        start = full.decision_timestamps[0]
    else:
        end = start
    replacement = PredictionDecisionSchedule(primary, start, end)
    block_research(monkeypatch)
    inspect_validation(
        source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
    )
    manifest["schedule"] = replacement.to_primitive()
    window["decisions"] = [
        decision
        for decision in cast(list[PrimitiveMapping], window["decisions"])
        if datetime.fromisoformat(text(decision["decision_timestamp"]))
        in replacement.decision_timestamps
    ]
    refresh_window_manifest(window)
    manifest["window_result_id"] = configuration_identity(
        {"window_id": manifest["window_id"], "decisions": window["decisions"]}
    )
    artifact["result_id"] = manifest["window_result_id"]
    result["artifact_sha256"] = configuration_identity(artifact)
    write_record(path, result)
    validate_window_snapshot(window)  # All internal identities/calendar checks pass.
    assert ledger.state(source).state.value == "consumed"
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ManifestError, match=r"holdout prediction.*schedule"):
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )
    assert {path: path.read_bytes() for path in before} == before
    assert ledger.state(source).state.value == "consumed"
