"""QF-64 sparse event windows on validated intraday caches, offline.

The actual QF-45 EMA rule persists only no-prediction receipts while its daily
EMA50 warms up. A sparse candidate rule then spans the 2024-07-03 early close,
the 2024-07-04 holiday and QF-62's rolling two-session daily lookback, so rich
observations keep normalized membership while ordinary decisions are receipts.
"""

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from quantforge.data import MultiTimeframeContext
from quantforge.data.models import BoundedPredictionProvenance
from quantforge.data.prediction_views import bounded_prediction_view
from quantforge.examples.spy_ema import EmaStudyFactory, EmaWindowAnalyzer
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
    PredictionRuleContext,
    PredictionStudy,
    SignalFeatureCandidate,
    SignalFeatureCandidateOutput,
    intraday_forward_return_outcome,
)
from quantforge.prediction.study import prepare_prediction_study_dataset
from quantforge.prediction.window_encoding import canonical, mapping
from quantforge.prediction.window_execution import (
    run_incremental_prediction_window_in_session,
)
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.reporting import build_research_report, export_research_report
from tests.integration.test_compact_prediction_provenance import WindowProvider
from tests.integration.test_intraday_prediction_provenance import (
    DECISION,
    TWO_MINUTES,
    Fixture,
    cached_fixture,
    study_inputs,
)
from tests.integration.test_normalized_prediction_window import (
    FIRST,
    LAST,
    SESSIONS,
    RollingDailyProvider,
    validate,
)
from tests.unit.experiments.test_contracts import execution
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _FixtureCandidateRule,  # pyright: ignore[reportPrivateUsage]
)

PHYSICAL = ("decision_id", "shared_evidence_id")


class SparseCandidateRule(_FixtureCandidateRule):
    """Emit the fixture candidate only at bar ends whose minute ends in 4."""

    name = "fixture_sparse_multi_timeframe_candidates"

    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> SignalFeatureCandidateOutput:
        output = super().generate_with_context(context)
        primary = context.latest_bar_for(self.context_requirements.primary.timeframe)
        if primary.end_timestamp.minute % 10 == 4:
            return output
        return replace(output, signals=())


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return cached_fixture(
        tmp_path_factory.mktemp("qf64-sessions"), "massive", session_dates=SESSIONS
    )


def execute(
    inputs: Fixture, path: Path, version: str, provider: RollingDailyProvider
) -> PredictionWindowReader:
    fixture_rule, _ = study_inputs(inputs)
    rule = SparseCandidateRule(fixture_rule.context_requirements)
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
        context_provider=provider,
        dataset_family_fingerprint=provenance.family_id,
        context_environment={"provider": "rolling_two_session_daily_fixture"},
        canonical_metadata=inputs.dataset.metadata,
        schema_version=version,
    )


def report(inputs: Fixture, path: Path, root: Path) -> None:
    artifacts = inspect_study(
        StudyType.PREDICTION_WINDOW,
        path,
        artifact_root=root,
        canonical_metadata=inputs.dataset.metadata,
    )
    assert {entry.schema_version for entry in artifacts.index.entries} == {"4"}
    verify_artifacts(artifacts.index, root).require_valid()
    manifest_path = write_manifest(
        create_manifest(artifacts, execution()), root / "manifests", artifact_root=root
    )
    built = build_research_report(manifest_path, artifact_root=root)
    assert export_research_report(built, root / "reports").is_file()


@pytest.mark.parametrize("interrupt_after", [2, 3, 5])
def test_sparse_events_across_sessions_match_normalized_decisions(
    inputs: Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_after: int,
) -> None:
    schedule = PredictionDecisionSchedule(TWO_MINUTES, FIRST, LAST)
    old = execute(
        inputs,
        tmp_path / "v3" / "prediction-window.jsonl",
        "3",
        RollingDailyProvider(inputs, []),
    )
    path = tmp_path / "v4" / "prediction-window.jsonl"
    calls: list[datetime] = []
    original = RollingDailyProvider.get_context_at

    def interrupted(
        provider: RollingDailyProvider,
        requirements: PredictionContextRequirements,
        *,
        as_of: datetime,
    ) -> MultiTimeframeContext:
        # 2: after an observation; 3: after a no-prediction receipt; 5: just after
        # the rolling daily shift at the early close.
        if len(provider.calls) == interrupt_after:
            raise KeyboardInterrupt
        return original(provider, requirements, as_of=as_of)

    monkeypatch.setattr(RollingDailyProvider, "get_context_at", interrupted)
    with pytest.raises(KeyboardInterrupt):
        execute(inputs, path, "4", RollingDailyProvider(inputs, calls))
    monkeypatch.setattr(RollingDailyProvider, "get_context_at", original)
    journal = path.with_name(path.name + ".in-progress") / "decisions.jsonl"
    assert len(journal.read_bytes().splitlines()) == interrupt_after
    resumed: list[datetime] = []
    new = execute(inputs, path, "4", RollingDailyProvider(inputs, resumed))
    assert resumed == list(schedule.decision_timestamps[interrupt_after:])
    uninterrupted = execute(
        inputs,
        tmp_path / "once" / "prediction-window.jsonl",
        "4",
        RollingDailyProvider(inputs, []),
    )
    assert path.read_bytes() == uninterrupted.path.read_bytes()
    assert new.schema_version == "4"
    # Same fixture parameters: QF-62's trusted validator applies unchanged.
    validate(inputs, old)
    validate(inputs, new)
    counts = mapping(new.header()["record_counts"])
    assert counts == old.header()["record_counts"]
    assert (counts["no_prediction_decisions"], counts["generated_predictions"]) == (
        6,
        2,
    )
    for left, right in zip(
        old.iterate_decision_receipts(), new.iterate_decision_receipts(), strict=True
    ):
        assert (left.sequence, left.decision_timestamp, left.status) == (
            right.sequence,
            right.decision_timestamp,
            right.status,
        )
        assert (left.context_id, left.prediction_study_id) == (
            right.context_id,
            right.prediction_study_id,
        )
        assert (left.observation_start, left.observation_count) == (
            right.observation_start,
            right.observation_count,
        )
        if right.decision is None:
            assert right.status == "no_prediction"
            continue
        assert left.decision is not None
        expected = {
            k: v
            for k, v in left.decision.expanded_record().items()
            if k not in PHYSICAL
        }
        actual = {
            k: v
            for k, v in right.decision.expanded_record().items()
            if k not in PHYSICAL
        }
        assert canonical(actual) == canonical(expected)
    observations = list(new.iterate_observations())
    assert [item.decision_timestamp.isoformat() for item in observations] == [
        "2024-07-03T16:54:00+00:00",
        "2024-07-05T13:34:00+00:00",
    ]
    statuses = [
        mapping(mapping(item.row() or {})["outcome"])["temporal_resolution"]
        for item in observations
    ]
    # 30 minutes after the 13:00 early close overflows; after the holiday resolves.
    assert [mapping(item)["status"] for item in statuses] == [
        "session_overflow",
        "available",
    ]
    assert [(o.signal(), o.row()) for o in observations] == [
        (o.signal(), o.row()) for o in old.iterate_observations()
    ]
    assert EmaWindowAnalyzer().analyze_compact_window(
        new
    ) == EmaWindowAnalyzer().analyze_compact_window(old)
    old_lines = old.path.read_bytes().splitlines()[2:]
    new_lines = new.path.read_bytes().splitlines()[2:]
    receipts = [
        line
        for line, item in zip(new_lines, new.iterate_decision_receipts(), strict=True)
        if item.decision is None
    ]
    # Fixture no-prediction records are ~9.9 KB in schema 3; receipts ~0.34 KB.
    assert len(receipts) == 6
    assert all(len(line) < 400 for line in receipts)
    assert sum(map(len, new_lines)) < 0.6 * sum(map(len, old_lines))

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("sparse reuse and inspection must not execute research")

    monkeypatch.setattr(RollingDailyProvider, "get_context_at", forbidden)
    monkeypatch.setattr(
        "quantforge.prediction.window.run_prediction_study_in_session", forbidden
    )
    assert execute(inputs, path, "4", RollingDailyProvider(inputs, [])).header() == (
        new.header()
    )
    report(inputs, path, tmp_path)


def test_actual_qf45_rule_persists_no_prediction_receipts_offline(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    fixture = cached_fixture(tmp_path_factory.mktemp("qf64-ema"), provider="massive")
    study = EmaStudyFactory(fixture.primary).build({"ema_pair": "8/48"})
    prepared = prepare_prediction_study_dataset(
        bounded_prediction_view(fixture.dataset, DECISION)
    )
    schedule = PredictionDecisionSchedule(
        TWO_MINUTES, DECISION, DECISION + timedelta(minutes=4)
    )
    readers: dict[str, PredictionWindowReader] = {}
    for version in ("2", "4"):
        readers[version] = run_incremental_prediction_window_in_session(
            prepared,
            study,
            path=tmp_path / version / "prediction-window.jsonl",
            schedule=schedule,
            context_provider=WindowProvider(fixture),
            dataset_family_fingerprint=fixture.primary.dataset_reference.family_id,
            context_environment={"provider": "synthetic_offline_massive"},
            canonical_metadata=fixture.dataset.metadata,
            schema_version=version,
        )
    old, new = readers["2"], readers["4"]
    assert mapping(new.header()["record_counts"])["no_prediction_decisions"] == 3
    assert new.header()["record_counts"] == old.header()["record_counts"]
    coverage = [
        (r.sequence, r.status, r.context_id, r.prediction_study_id)
        for r in new.iterate_decision_receipts()
    ]
    assert coverage == [
        (r.sequence, r.status, r.context_id, r.prediction_study_id)
        for r in old.iterate_decision_receipts()
    ]
    assert all(r.decision is None for r in new.iterate_decision_receipts())
    assert not list(new.iterate_observations())
    analysis = EmaWindowAnalyzer().analyze_compact_window(new)
    assert analysis == EmaWindowAnalyzer().analyze_compact_window(old)
    assert analysis.prediction_count == 0  # Daily EMA50 is still warming up.
    lines = new.path.read_bytes().splitlines()[2:]
    assert all(len(line) < 400 for line in lines)
    # Shared evidence is unchanged; per-decision records shrink from full QF-11
    # contexts to identity receipts.
    assert sum(map(len, lines)) * 50 < sum(
        map(len, old.path.read_bytes().splitlines()[2:])
    )
    report(fixture, new.path, tmp_path / "4")
