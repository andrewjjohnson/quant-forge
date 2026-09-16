from datetime import datetime, time, timedelta
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._window_sessions import scheduled_sessions
from quantforge.prediction import PredictionDecisionSchedule
from quantforge.prediction.window import (
    _window_record_counts,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.timeframes import (
    BarLabel,
    ExchangeSessionPolicy,
    IntradayAnchor,
    IntradayInterval,
    SessionScope,
    Timeframe,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record
from tests.unit.prediction.test_prediction_window import (
    START,
    WindowProvider,
    _provider_with_final_session,  # pyright: ignore[reportPrivateUsage]
    _rewrite_decision_identities,  # pyright: ignore[reportPrivateUsage]
    _rewrite_window_checksums,  # pyright: ignore[reportPrivateUsage]
    grid,
    schedule,
)


def refresh_window_manifest(window: PrimitiveMapping) -> None:
    manifest = cast(PrimitiveMapping, window["manifest"])
    manifest["schedule_id"] = configuration_identity(
        cast(PrimitiveMapping, manifest["schedule"])
    )
    manifest["window_id"] = configuration_identity(
        {
            key: value
            for key, value in manifest.items()
            if key
            not in {"window_id", "window_result_id", "schedule_id", "record_counts"}
        }
    )
    manifest["record_counts"] = _window_record_counts(
        cast(list[PrimitiveMapping], window["decisions"])
    )


@pytest.mark.parametrize("extended", [False, True])
def test_canonical_calendar_keeps_clock_anchors_partial_bars_and_holidays(
    extended: bool,
) -> None:
    primary = Timeframe(
        IntradayInterval(
            timedelta(hours=1), anchor=IntradayAnchor.CLOCK, clock_anchor=time(9)
        ),
        session_policy=ExchangeSessionPolicy(
            scope=SessionScope.EXTENDED_HOURS
            if extended
            else SessionScope.REGULAR_HOURS,
            extended_hours_start=time(4, 30) if extended else None,
            extended_hours_end=time(19, 15) if extended else None,
        ),
        bar_label=BarLabel.START,
    )
    original = PredictionDecisionSchedule(
        primary,
        datetime.fromisoformat("2024-07-03T00:00:00+00:00"),
        datetime.fromisoformat("2024-07-04T23:59:59+00:00"),
    )
    assert len(original.decision_timestamps) >= 2
    assert scheduled_sessions(original.to_primitive()) == {
        instant.isoformat(): label.isoformat()
        for instant, label in zip(
            original.decision_timestamps, original.decision_sessions, strict=True
        )
    }


@pytest.mark.parametrize("nested", [False, True], ids=["standalone", "grid"])
@pytest.mark.parametrize("omitted", ["first", "middle", "last", "all"])
def test_rehashed_window_cannot_omit_canonical_calendar_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
    omitted: str,
) -> None:
    study = grid(tmp_path / "grid", WindowProvider())
    trial = study.run().trials[0]
    root = tmp_path / "grid" / study.study_id
    artifact_path = root / cast(str, trial.artifact_location)
    trial_path = root / "trials" / f"{trial.trial_id}.json"
    artifact = read_record(artifact_path)
    window = cast(PrimitiveMapping, artifact["prediction_window"])
    decisions = cast(list[Primitive], window["decisions"])
    manifest = cast(PrimitiveMapping, window["manifest"])
    timestamps = cast(
        list[Primitive],
        cast(PrimitiveMapping, manifest["schedule"])["decision_timestamps"],
    )
    assert len(decisions) >= 3
    if omitted == "all":
        decisions.clear()
        timestamps.clear()
    else:
        index = {"first": 0, "middle": len(decisions) // 2, "last": -1}[omitted]
        decisions.pop(index)
        timestamps.pop(index)
    refresh_window_manifest(window)
    _rewrite_window_checksums(artifact_path, trial_path, artifact)
    block_research(monkeypatch)
    if nested:
        source, study_type = root, StudyType.PARAMETER_STUDY
    else:
        source, study_type = tmp_path / "window.json", StudyType.PREDICTION_WINDOW
        write_json(source, window)
    with pytest.raises(ManifestError, match="canonical completed bar boundaries"):
        inspect_study(study_type, source, artifact_root=tmp_path)


@pytest.mark.parametrize("nested", [False, True], ids=["standalone", "grid"])
@pytest.mark.parametrize("target", ["manifest", "schedule"])
@pytest.mark.parametrize("version", ["0", "2", "corrupt"])
def test_rehashed_window_rejects_unsupported_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
    target: str,
    version: str,
) -> None:
    study = grid(tmp_path / "grid", WindowProvider())
    trial = study.run().trials[0]
    root = tmp_path / "grid" / study.study_id
    artifact_path = root / cast(str, trial.artifact_location)
    trial_path = root / "trials" / f"{trial.trial_id}.json"
    artifact = read_record(artifact_path)
    window = cast(PrimitiveMapping, artifact["prediction_window"])
    manifest = cast(PrimitiveMapping, window["manifest"])
    record = (
        manifest
        if target == "manifest"
        else cast(PrimitiveMapping, manifest["schedule"])
    )
    record["schema_version"] = version
    refresh_window_manifest(window)
    _rewrite_window_checksums(artifact_path, trial_path, artifact)
    block_research(monkeypatch)
    if nested:
        source, study_type = root, StudyType.PARAMETER_STUDY
    else:
        source, study_type = tmp_path / "window.json", StudyType.PREDICTION_WINDOW
        write_json(source, window)
    with pytest.raises(ManifestError, match=r"unsupported prediction window.*schema"):
        inspect_study(study_type, source, artifact_root=tmp_path)


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
