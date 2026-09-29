"""QF-45 uses incremental compact records and downstream inspection offline."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from quantforge.data import MultiTimeframeContext
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
)
from quantforge.prediction.study import prepare_prediction_study_dataset
from quantforge.prediction.window_encoding import mapping
from quantforge.prediction.window_execution import (
    run_incremental_prediction_window_in_session,
)
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.reporting import build_research_report, export_research_report
from tests.integration.test_compact_prediction_provenance import (
    WindowProvider,
    make_window,
)
from tests.integration.test_intraday_prediction_provenance import (
    DECISION,
    TWO_MINUTES,
    Fixture,
    cached_fixture,
)
from tests.unit.experiments.test_contracts import execution


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return cached_fixture(tmp_path_factory.mktemp("qf45-compact"), provider="massive")


def test_actual_rule_persists_zero_candidate_decisions_and_reuses_final(
    inputs: Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = EmaStudyFactory(inputs.primary).build({"ema_pair": "8/48"})
    prepared = prepare_prediction_study_dataset(
        bounded_prediction_view(inputs.dataset, DECISION)
    )
    schedule = PredictionDecisionSchedule(
        TWO_MINUTES, DECISION, DECISION + timedelta(minutes=4)
    )
    path = tmp_path / "prediction-window.jsonl"

    def run() -> PredictionWindowReader:
        return run_incremental_prediction_window_in_session(
            prepared,
            study,
            path=path,
            schedule=schedule,
            context_provider=WindowProvider(inputs),
            dataset_family_fingerprint=inputs.primary.dataset_reference.family_id,
            context_environment={"provider": "synthetic_offline_massive"},
            canonical_metadata=inputs.dataset.metadata,
        )

    original_context = WindowProvider.get_context_at
    completed_calls: list[datetime] = []

    def interrupted_context(
        provider: WindowProvider,
        requirements: PredictionContextRequirements,
        *,
        as_of: datetime,
    ) -> MultiTimeframeContext:
        if completed_calls:
            raise KeyboardInterrupt
        completed_calls.append(as_of)
        return original_context(provider, requirements, as_of=as_of)

    monkeypatch.setattr(WindowProvider, "get_context_at", interrupted_context)
    with pytest.raises(KeyboardInterrupt):
        run()
    journal = path.with_name(path.name + ".in-progress") / "decisions.jsonl"
    assert len(journal.read_bytes().splitlines()) == 1

    def resumed_context(
        provider: WindowProvider,
        requirements: PredictionContextRequirements,
        *,
        as_of: datetime,
    ) -> MultiTimeframeContext:
        completed_calls.append(as_of)
        return original_context(provider, requirements, as_of=as_of)

    monkeypatch.setattr(WindowProvider, "get_context_at", resumed_context)
    reader = run()
    assert completed_calls == list(schedule.decision_timestamps)
    assert reader.schema_version == "2"
    assert reader.decision_count == 3
    analysis = EmaWindowAnalyzer().analyze_compact_window(reader)
    assert analysis.prediction_count == 0  # Daily EMA50 is still warming up.
    assert "mean_30m_return" not in analysis.metrics
    assert mapping(reader.header()["record_counts"])["no_prediction_decisions"] == 3
    encoded = path.read_bytes()
    assert encoded.count(b'"record_type":"shared_evidence"') == 1
    assert not path.with_name(path.name + ".in-progress").exists()

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("compatible compact reuse/report must not execute research")

    monkeypatch.setattr(WindowProvider, "get_context_at", forbidden)
    monkeypatch.setattr(
        "quantforge.prediction.window.run_prediction_study_in_session", forbidden
    )
    monkeypatch.setattr(
        "quantforge.prediction.window_compact.CompactPredictionWindowResult.from_window",
        forbidden,
    )
    assert run().header() == reader.header()
    assert path.read_bytes() == encoded
    artifacts = inspect_study(
        StudyType.PREDICTION_WINDOW,
        path,
        artifact_root=tmp_path,
        canonical_metadata=inputs.dataset.metadata,
    )
    verify_artifacts(artifacts.index, tmp_path).require_valid()
    manifest_path = write_manifest(
        create_manifest(artifacts, execution()),
        tmp_path / "manifests",
        artifact_root=tmp_path,
    )
    monkeypatch.setattr(type(reader), "iterate_decisions", forbidden)
    monkeypatch.setattr(type(reader), "_coverage", forbidden)
    report = build_research_report(manifest_path, artifact_root=tmp_path)
    assert export_research_report(report, tmp_path / "reports").is_file()


def test_compact_analysis_matches_original_typed_forward_returns(
    inputs: Fixture, tmp_path: Path
) -> None:
    from quantforge.prediction.window_compact import CompactPredictionWindowResult
    from quantforge.prediction.window_reader import PredictionWindowReader

    # A tiny upstream fixture supplies nonempty QF-49 rows. Only the compatibility
    # test constructs v1; QF-45 execution and the production analyzer stay compact.
    legacy = make_window(inputs)
    path = tmp_path / "comparison.jsonl"
    with path.open("wb") as stream:
        stream.writelines(
            CompactPredictionWindowResult.from_window(legacy).iter_serialized_records()
        )
    analyzer = EmaWindowAnalyzer()
    actual = analyzer.analyze_compact_window(PredictionWindowReader.open(path))
    assert actual.prediction_count > 0
    assert actual == analyzer.analyze_window(legacy)
