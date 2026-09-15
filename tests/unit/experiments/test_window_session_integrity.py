from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._window_sessions import scheduled_sessions
from quantforge.prediction import PredictionDecisionSchedule
from quantforge.timeframes import ExchangeSessionPolicy, IntradayInterval, Timeframe
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record
from tests.unit.prediction.test_prediction_window import (
    START,
    _provider_with_final_session,  # pyright: ignore[reportPrivateUsage]
    _rewrite_decision_identities,  # pyright: ignore[reportPrivateUsage]
    _rewrite_window_checksums,  # pyright: ignore[reportPrivateUsage]
    grid,
    schedule,
)


@pytest.mark.parametrize("nested", [False, True], ids=["standalone", "grid"])
@pytest.mark.parametrize("no_prediction", [False, True])
def test_rehashed_context_and_signal_cannot_change_scheduled_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool, no_prediction: bool
) -> None:
    tomorrow = START + timedelta(days=1)
    study = grid(
        tmp_path / "grid",
        _provider_with_final_session(),
        decision_schedule=schedule(tomorrow, tomorrow),
    )
    trial = study.run().trials[0]
    root = tmp_path / "grid" / study.study_id
    artifact_path = root / cast(str, trial.artifact_location)
    trial_path = root / "trials" / f"{trial.trial_id}.json"
    artifact = read_record(artifact_path)
    window = cast(PrimitiveMapping, artifact["prediction_window"])
    decision = cast(list[PrimitiveMapping], window["decisions"])[0]
    manifest = cast(
        PrimitiveMapping,
        cast(PrimitiveMapping, decision["prediction_study"])["manifest"],
    )
    cast(PrimitiveMapping, manifest["prediction_context"])["decision_session"] = (
        START.date().isoformat()
    )
    signals = cast(list[PrimitiveMapping], decision["generated_signals"])
    cast(PrimitiveMapping, signals[0]["prediction"])["signal_session"] = (
        START.date().isoformat()
    )
    if no_prediction:
        decision["generated_signals"] = []
        decision["status"] = "no_prediction"
    _rewrite_decision_identities(decision)
    _rewrite_window_checksums(artifact_path, trial_path, artifact)
    block_research(monkeypatch)
    if nested:
        source, study_type = root, StudyType.PARAMETER_STUDY
    else:
        source, study_type = tmp_path / "window.json", StudyType.PREDICTION_WINDOW
        write_json(source, window)
    with pytest.raises(ManifestError, match="scheduled exchange session"):
        inspect_study(study_type, source, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("calendar", "timezone", "start", "end"),
    [
        (
            "XNYS",
            "America/New_York",
            "2024-07-03T12:55:00-04:00",
            "2024-07-03T13:00:00-04:00",
        ),
        (
            "CMES",
            "America/Chicago",
            "2024-07-07T17:05:00-05:00",
            "2024-07-07T17:10:00-05:00",
        ),
        ("24/5", "UTC", "2024-07-13T00:00:00+00:00", "2024-07-13T00:00:00+00:00"),
        (
            "XNYS",
            "America/New_York",
            "2024-07-06T12:00:00-04:00",
            "2024-07-06T13:00:00-04:00",
        ),
    ],
)
def test_recorded_session_labels_match_producer_calendar_boundaries(
    calendar: str, timezone: str, start: str, end: str
) -> None:
    decision_schedule = PredictionDecisionSchedule(
        Timeframe(
            IntradayInterval(timedelta(minutes=5)),
            ExchangeSessionPolicy(calendar_name=calendar, timezone_name=timezone),
        ),
        datetime.fromisoformat(start),
        datetime.fromisoformat(end),
    )
    assert scheduled_sessions(decision_schedule.to_primitive()) == {
        timestamp.isoformat(): label.isoformat()
        for timestamp, label in zip(
            decision_schedule.decision_timestamps,
            decision_schedule.decision_sessions,
            strict=True,
        )
    }
