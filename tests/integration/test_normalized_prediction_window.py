"""QF-62 on validated intraday caches: session transitions and rolling daily bounds.

The fixture spans the 2024-07-03 early close and the 2024-07-04 holiday. A
two-session daily lookback reproduces QF-45's rolling daily start-bound shift at
the session-close decision, while 2m membership keeps growing across sessions.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.data import (
    MultiTimeframeContext,
    TimeframeBarSeries,
    build_multi_timeframe_context,
)
from quantforge.data.models import BoundedPredictionProvenance
from quantforge.data.prediction_views import bounded_prediction_view
from quantforge.examples.spy_ema import EmaWindowAnalyzer
from quantforge.experiments import (
    StudyType,
    create_manifest,
    inspect_study,
    verify_artifacts,
    write_manifest,
)
from quantforge.prediction import (
    PredictionContextRequirements,
    PredictionDecisionSchedule,
    PredictionStudy,
    SignalFeatureCandidate,
    intraday_forward_return_outcome,
)
from quantforge.prediction.study import prepare_prediction_study_dataset
from quantforge.prediction.window_compact_validation import (
    validate_prediction_window_reader,
)
from quantforge.prediction.window_encoding import canonical, mapping
from quantforge.prediction.window_execution import (
    run_incremental_prediction_window_in_session,
)
from quantforge.prediction.window_membership import prediction_context
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.reporting import build_research_report, export_research_report
from tests.integration.test_intraday_prediction_provenance import (
    TWO_MINUTES,
    Fixture,
    cached_fixture,
    study_inputs,
)
from tests.unit.experiments.test_contracts import execution

SESSIONS = (date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3), date(2024, 7, 5))
FIRST = datetime(2024, 7, 3, 16, 52, tzinfo=UTC)  # 12:52 New York, early-close day
LAST = datetime(2024, 7, 5, 13, 36, tzinfo=UTC)  # 09:36 New York, after the holiday


@dataclass(frozen=True)
class RollingDailyProvider:
    """Expose only the latest two completed daily bars, like a bounded lookback."""

    fixture: Fixture
    calls: list[datetime]

    def get_context_at(
        self, requirements: PredictionContextRequirements, *, as_of: datetime
    ) -> MultiTimeframeContext:
        self.calls.append(as_of)
        daily = self.fixture.daily
        visible = tuple(bar for bar in daily.bars if bar.end_timestamp <= as_of)[-2:]
        rolling = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
            daily.dataset_reference,
            daily.timeframe,
            visible,
            dataset_family_manifest_id=daily.dataset_family_manifest_id,
        )
        return build_multi_timeframe_context(
            series=(self.fixture.primary, rolling),
            primary_timeframe=requirements.primary.timeframe,
            required_timeframes=requirements.context_timeframe_requirements(),
            completion_policy=requirements.context_completion_policy,
            as_of=as_of,
        )


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return cached_fixture(
        tmp_path_factory.mktemp("qf62-sessions"), "massive", session_dates=SESSIONS
    )


def execute(
    inputs: Fixture, path: Path, version: str, calls: list[datetime]
) -> PredictionWindowReader:
    rule, _ = study_inputs(inputs)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), inputs.primary)
    study = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
        rule, outcome.labeler, outcome.evaluator, outcome_source=inputs.primary
    )
    view = bounded_prediction_view(inputs.dataset, FIRST)
    provenance = view.metadata.intraday_provenance
    assert isinstance(provenance, BoundedPredictionProvenance)
    return run_incremental_prediction_window_in_session(
        prepare_prediction_study_dataset(view),
        study,
        path=path,
        schedule=PredictionDecisionSchedule(TWO_MINUTES, FIRST, LAST),
        context_provider=RollingDailyProvider(inputs, calls),
        dataset_family_fingerprint=provenance.family_id,
        context_environment={"provider": "rolling_two_session_daily_fixture"},
        canonical_metadata=inputs.dataset.metadata,
        schema_version=version,
    )


def validate(inputs: Fixture, reader: PredictionWindowReader) -> None:
    rule, _ = study_inputs(inputs)
    validate_prediction_window_reader(
        reader,
        expected_identity=reader.evidence.identity_snapshot,
        schedule=reader.evidence.schedule,
        outcome_sessions=(),
        strategy_parameters=rule.parameters.to_primitive(),
        canonical_metadata=inputs.dataset.metadata,
    )


def ranges(record: dict[str, Any]) -> list[tuple[int, int]]:
    context = prediction_context(record)
    source = cast(
        list[dict[str, Any]], mapping(context["source_context"])["timeframes"]
    )
    return [
        (
            item["visible_bar_range"]["start_index"],
            item["visible_bar_range"]["stop_index"],
        )
        for item in source
    ]


def test_session_transition_and_rolling_daily_bounds_are_exactly_equivalent(
    inputs: Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = PredictionDecisionSchedule(TWO_MINUTES, FIRST, LAST)
    stamps = [item.isoformat() for item in schedule.decision_timestamps]
    # 13:00 early close on 07-03, no 07-04 session, then the 07-05 open.
    assert stamps[4] == "2024-07-03T17:00:00+00:00"
    assert stamps[5] == "2024-07-05T13:32:00+00:00"
    assert len(stamps) == 8
    old = execute(inputs, tmp_path / "v2" / "prediction-window.jsonl", "2", [])
    calls: list[datetime] = []
    path = tmp_path / "v3" / "prediction-window.jsonl"
    original = RollingDailyProvider.get_context_at

    def interrupted(
        provider: RollingDailyProvider,
        requirements: PredictionContextRequirements,
        *,
        as_of: datetime,
    ) -> MultiTimeframeContext:
        if len(calls) == 5:  # Interrupt immediately after the rolling shift.
            raise KeyboardInterrupt
        return original(provider, requirements, as_of=as_of)

    monkeypatch.setattr(RollingDailyProvider, "get_context_at", interrupted)
    with pytest.raises(KeyboardInterrupt):
        execute(inputs, path, "3", calls)
    monkeypatch.setattr(RollingDailyProvider, "get_context_at", original)
    committed = len(calls)
    new = execute(inputs, path, "3", calls)
    assert calls[committed:] == list(schedule.decision_timestamps[committed:])
    assert (old.schema_version, new.schema_version) == ("2", "3")
    validate(inputs, old)
    validate(inputs, new)
    physical = ("decision_id", "shared_evidence_id")
    new_records: list[dict[str, Any]] = []
    for left, right in zip(
        old.iterate_decisions(), new.iterate_decisions(), strict=True
    ):
        expected = {k: v for k, v in left.to_primitive().items() if k not in physical}
        actual = {k: v for k, v in right.expanded_record().items() if k not in physical}
        assert canonical(actual) == canonical(expected)
        new_records.append(right.to_primitive())
    found = [ranges(record) for record in new_records]
    primary = [item[0] for item in found]
    daily = [item[1] for item in found]
    assert all(start == 0 for start, _ in primary)
    stops = [stop for _, stop in primary]
    assert stops == list(range(stops[0], stops[0] + 8))  # Across early close/holiday.
    # Rolling start bound: [07-01, 07-02] -> [07-02, 07-03] at the 13:00 close.
    assert daily == [(0, 2)] * 4 + [(1, 3)] * 4
    counts = mapping(new.header()["record_counts"])
    assert counts == old.header()["record_counts"]
    assert counts["generated_predictions"] == 8
    statuses = [
        mapping(mapping(row)["outcome"])["temporal_resolution"]
        for record in new_records
        for row in cast(list[Any], mapping(record["prediction_study"])["rows"])
    ]
    # 30 minutes after the 13:00 early close overflows; after the holiday it resolves.
    assert [mapping(item)["status"] for item in statuses] == [
        "session_overflow"
    ] * 5 + ["available"] * 3
    assert EmaWindowAnalyzer().analyze_compact_window(
        new
    ) == EmaWindowAnalyzer().analyze_compact_window(old)
    assert new.path.stat().st_size < old.path.stat().st_size

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("normalized reuse and inspection must not execute research")

    monkeypatch.setattr(RollingDailyProvider, "get_context_at", forbidden)
    monkeypatch.setattr(
        "quantforge.prediction.window.run_prediction_study_in_session", forbidden
    )
    assert execute(inputs, path, "3", []).header() == new.header()
    artifacts = inspect_study(
        StudyType.PREDICTION_WINDOW,
        path,
        artifact_root=tmp_path,
        canonical_metadata=inputs.dataset.metadata,
    )
    assert {entry.schema_version for entry in artifacts.index.entries} == {"3"}
    verify_artifacts(artifacts.index, tmp_path).require_valid()
    manifest_path = write_manifest(
        create_manifest(artifacts, execution()),
        tmp_path / "manifests",
        artifact_root=tmp_path,
    )
    report = build_research_report(manifest_path, artifact_root=tmp_path)
    assert export_research_report(report, tmp_path / "reports").is_file()
