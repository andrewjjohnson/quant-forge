"""QF-49 composes existing replay/export, validation, OOS, and integrity hooks."""

import csv
import shutil
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import MarketDataset, TimeframeBarSeries
from quantforge.experiments import (
    ManifestError,
    StudyType,
    create_manifest,
    inspect_study,
    inspect_validation,
    read_manifest,
    write_manifest,
)
from quantforge.indicators import SIMPLE_MOVING_AVERAGE_OUTPUT
from quantforge.oos import aggregate_prediction, load_oos_source
from quantforge.oos._records import mapping, records
from quantforge.prediction import (
    IntradayForwardReturnOutcomeLabeler,
    IntradayForwardReturnValues,
    MultiTimeframeFeatureRequest,
    PredictionContextRequirements,
    PredictionRuleContext,
    PredictionStudy,
    PredictionStudyOutcome,
    PredictionTrialAnalysis,
    PredictionWindowResult,
    SchemaFieldCategory,
    SignalFeatureCandidate,
    SignalFeatureCandidateOutput,
    SignalFeaturePersistenceError,
    build_signal_feature_dataset,
    intraday_forward_return_outcome,
    run_prediction_study,
)
from quantforge.timeframes import IntradayInterval, Timeframe
from quantforge.validation import OutcomeProvenance, PurgePolicy, TemporalOffset
from quantforge.walk_forward import (
    FoldStatus,
    PredictionEvaluator,
    WalkForwardConfig,
    WalkForwardPersistenceError,
    WalkForwardStudy,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import execution, write_json
from tests.unit.experiments.test_elapsed_prediction_integrity import contextual_study
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _FixtureCandidateRule,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_timestamp_replay import population
from tests.unit.walk_forward.timestamp_fixtures import (
    TimestampFactory,
    instant,
    timestamp_fixture,
)

type ReturnStudy = PredictionStudy[
    SignalFeatureCandidate, IntradayForwardReturnValues, IntradayForwardReturnValues
]
type ReturnOutcome = PredictionStudyOutcome[
    IntradayForwardReturnValues, IntradayForwardReturnValues
]


def replay_study(
    tmp_path: Path, *, minutes: int = 30, source: TimeframeBarSeries | None = None
) -> tuple[MarketDataset, ReturnStudy, ReturnOutcome]:
    dataset, replay, _, original = population(tmp_path)
    source = original.outcome_source if source is None else source
    assert source is not None
    outcome = intraday_forward_return_outcome(timedelta(minutes=minutes), source)
    return (
        dataset,
        PredictionStudy[
            SignalFeatureCandidate,
            IntradayForwardReturnValues,
            IntradayForwardReturnValues,
        ].create(replay, outcome.labeler, outcome.evaluator, outcome_source=source),
        outcome,
    )


def forbidden(*args: object, **kwargs: object) -> None:
    pytest.fail("completed artifacts must not recalculate intraday returns")


def test_multiple_horizons_exact_replay_export_and_completed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, study, _ = replay_study(tmp_path)
    source = study.outcome_source
    assert source is not None
    outcomes = tuple(
        intraday_forward_return_outcome(timedelta(minutes=m), source)
        for m in (10, 30, 60, 120)
    )
    assert len({o.namespace for o in outcomes}) == 4
    assert len({o.configuration_id for o in outcomes}) == 4
    first = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=outcomes,
        output_root=tmp_path / "features",
        chunk_size=1,
    )
    assert len(first.rows) == 2
    assert len({r.candidate_id for r in first.rows}) == 2
    for row, decision in zip(
        first.rows, (instant(4, "10:00"), instant(4, "10:05")), strict=True
    ):
        values = row.to_primitive()
        assert values["decision_timestamp"] == decision.isoformat()
        for outcome, minutes in zip(outcomes, (10, 30, 60, 120), strict=True):
            prefix = f"outcome_{outcome.namespace}_"
            assert values[prefix + "decision_timestamp"] == decision.isoformat()
            assert (
                values[prefix + "resolved_observation_timestamp"]
                == (decision + timedelta(minutes=minutes)).isoformat()
            )
            assert (
                values[prefix + "elapsed_duration_microseconds"] == minutes * 60_000_000
            )
            assert values[prefix + "available"] is True
            assert values[prefix + "raw_return"] == "0"
            assert values[
                prefix + "source_reference"
            ] == source.dataset_reference.to_primitive(include_feed_scope=True)
    assert all(
        field.category is SchemaFieldCategory.FUTURE_OUTCOME
        for field in first.schema.fields
        if "raw_return" in field.name
    )
    path = tmp_path / "features" / first.dataset_id
    with (path / "features.csv").open(newline="") as file:
        assert len(tuple(csv.DictReader(file))) == 2
    assert len(tuple((path / "rows").glob("*.json"))) == 2
    monkeypatch.setattr(IntradayForwardReturnOutcomeLabeler, "label_request", forbidden)
    resumed = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=outcomes,
        output_root=tmp_path / "features",
        chunk_size=20,
    )
    assert resumed == first
    assert (
        inspect_study(
            StudyType.FEATURE_DATASET, path, artifact_root=tmp_path
        ).provenance.producer_study_id
        == first.dataset_id
    )


@pytest.mark.parametrize("truncate", [False, True])
def test_unavailable_endpoint_exports_null_return_with_original_resolution(
    tmp_path: Path, truncate: bool
) -> None:
    dataset, template, _ = replay_study(tmp_path)
    complete = template.outcome_source
    assert complete is not None
    endpoints = (instant(4, "10:30"), instant(4, "10:35"))
    source = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        complete.dataset_reference,
        complete.timeframe,
        tuple(
            bar
            for bar in complete.bars
            if (
                bar.end_timestamp < endpoints[0]
                if truncate
                else bar.end_timestamp not in endpoints
            )
        ),
        dataset_family_manifest_id=complete.dataset_family_manifest_id,
    )
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), source)
    study = PredictionStudy[
        SignalFeatureCandidate, IntradayForwardReturnValues, IntradayForwardReturnValues
    ].create(
        template.strategy, outcome.labeler, outcome.evaluator, outcome_source=source
    )
    result = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(outcome,),
        output_root=tmp_path / "unavailable",
    )
    prefix = f"outcome_{outcome.namespace}_"
    for row, expected in zip(result.rows, endpoints, strict=True):
        values = row.to_primitive()
        assert values[prefix + "available"] is False
        assert values[prefix + "raw_return"] is None
        assert values[prefix + "outcome_price"] is None
        assert values[prefix + "reference_price"] is not None
        assert values[prefix + "expected_observation_timestamp"] == expected.isoformat()
        assert values[prefix + "resolved_observation_timestamp"] is None
        assert values[prefix + "status"] == (
            "dataset_end" if truncate else "missing_required_observation"
        )
    path = tmp_path / "unavailable" / result.dataset_id
    assert (
        inspect_study(
            StudyType.FEATURE_DATASET, path, artifact_root=tmp_path
        ).provenance.producer_study_id
        == result.dataset_id
    )


@pytest.mark.parametrize("change", ["horizon", "source", "timeframe"])
def test_material_outcome_changes_reject_copied_feature_checkpoints(
    tmp_path: Path, change: str
) -> None:
    dataset, study, outcome = replay_study(tmp_path)
    original = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(outcome,),
        output_root=tmp_path / "features",
    )
    source = study.outcome_source
    assert source is not None
    if change == "source":
        source = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
            replace(source.dataset_reference, dataset_id="other-source"),
            source.timeframe,
            source.bars,
            dataset_family_manifest_id=source.dataset_family_manifest_id,
        )
    elif change == "timeframe":
        from tests.unit.helpers import SESSIONS
        from tests.unit.prediction.test_intraday_forward_return import priced_source

        changed_timeframe = priced_source(
            session=SESSIONS[4],
            timeframe=Timeframe.us_equity(IntradayInterval(timedelta(minutes=1))),
        )
        source = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
            changed_timeframe.dataset_reference,
            changed_timeframe.timeframe,
            changed_timeframe.bars,
            dataset_family_manifest_id=source.dataset_family_manifest_id,
        )
    minutes = 60 if change == "horizon" else 30
    modified = intraday_forward_return_outcome(timedelta(minutes=minutes), source)
    changed_study = PredictionStudy[
        SignalFeatureCandidate, IntradayForwardReturnValues, IntradayForwardReturnValues
    ].create(
        study.strategy, modified.labeler, modified.evaluator, outcome_source=source
    )
    changed = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=changed_study,
        contextual_features=(),
        outcomes=(modified,),
        output_root=tmp_path / "changed",
    )
    assert changed.dataset_id != original.dataset_id
    assert (
        modified.labeler_configuration_id != outcome.labeler_configuration_id
        or change == "source"
    )
    destination = tmp_path / "copied" / changed.dataset_id
    shutil.copytree(tmp_path / "features" / original.dataset_id, destination)
    with pytest.raises(SignalFeaturePersistenceError):
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=changed_study,
            contextual_features=(),
            outcomes=(modified,),
            output_root=tmp_path / "copied",
        )


def test_qf29_parquet_keeps_future_labels_separate_from_causal_features(
    tmp_path: Path,
) -> None:
    dataset, original, provider = contextual_study(tmp_path)
    source = original.outcome_source
    assert source is not None
    requirements = cast(
        PredictionContextRequirements,
        getattr(original.strategy, "context_requirements"),
    )

    class CausalRule(_FixtureCandidateRule):
        def generate_with_context(
            self, context: PredictionRuleContext
        ) -> SignalFeatureCandidateOutput:
            assert all(
                bar.end_timestamp <= context.as_of
                for requirement in requirements.all_timeframes
                for bar in context.bars_for(requirement.timeframe)
            )
            assert not hasattr(context, "outcome_source")
            return super().generate_with_context(context)

    outcome = intraday_forward_return_outcome(timedelta(minutes=30), source)
    study = PredictionStudy[
        SignalFeatureCandidate, IntradayForwardReturnValues, IntradayForwardReturnValues
    ].create(
        CausalRule(requirements),
        outcome.labeler,
        outcome.evaluator,
        outcome_source=source,
    )
    indicator = requirements.primary.indicators[0]
    result = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(outcome,),
        output_root=tmp_path / "contextual",
        context_provider=provider,
        multi_timeframe_features=(
            MultiTimeframeFeatureRequest(
                "primary",
                source.timeframe,
                indicator.alias,
                SIMPLE_MOVING_AVERAGE_OUTPUT,
                "price_per_share",
                False,
            ),
        ),
    )
    path = tmp_path / "contextual" / result.dataset_id
    assert (path / "features.parquet").is_file()
    assert len(result.rows) == 1
    assert (
        result.rows[0].to_primitive()["decision_timestamp"]
        == instant(4, "10:00").isoformat()
    )
    assert (
        inspect_study(
            StudyType.FEATURE_DATASET, path, artifact_root=tmp_path
        ).provenance.producer_study_id
        == result.dataset_id
    )


def test_qf9_manifest_preserves_concrete_configuration_without_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, study, outcome = replay_study(tmp_path)
    result = run_prediction_study(dataset, study)
    path = tmp_path / "prediction.json"
    write_json(path, result.to_primitive())
    block_research(monkeypatch)
    monkeypatch.setattr(IntradayForwardReturnOutcomeLabeler, "label_request", forbidden)
    bundle = inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    manifest = create_manifest(bundle, execution())
    saved = write_manifest(manifest, tmp_path / "manifests", artifact_root=tmp_path)
    assert (
        read_manifest(saved, artifact_root=tmp_path).manifest_id == manifest.manifest_id
    )
    configuration = mapping(
        manifest.provenance.configuration.to_primitive()["configuration"]
    )
    wrapper = mapping(configuration["outcome_labeler"])
    assert wrapper["configuration_id"] == outcome.labeler_configuration_id
    assert "required_future_sessions" not in wrapper
    assert (
        mapping(wrapper["configuration"])["parameters"]
        == mapping(outcome.labeler.configuration())["parameters"]
    )


class ReturnWindowAnalyzer:
    """Fixed fixture score exercises selection without a research hypothesis."""

    name = "qf49_fixture_score"
    version = "1"

    def configuration(self) -> PrimitiveMapping:
        return {"component_name": self.name, "implementation_version": "1"}

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def analyze_window(
        self, result: PredictionWindowResult[Any, Any, Any]
    ) -> PredictionTrialAnalysis:
        count = sum(len(decision.result.rows) for decision in result.decisions)
        return PredictionTrialAnalysis.create(
            prediction_count=count,
            metrics={"accuracy": "1"},
            period_comparisons=({"period": "fixture", "count": count},),
            weekday_comparisons=({"weekday": 0, "count": count},),
            matched_baseline_comparisons=(
                {"baseline_name": "always_up", "count": count},
            ),
        )


class ReturnFactory(TimestampFactory):
    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        template = super().build(parameters)
        outcome = intraday_forward_return_outcome(self.horizon, self.source)
        return PredictionStudy[
            Any, IntradayForwardReturnValues, IntradayForwardReturnValues
        ].create(
            template.strategy,
            outcome.labeler,
            outcome.evaluator,
            outcome_source=self.source,
        )


def walk_forward_fixture(
    tmp_path: Path, *, minutes: int = 30
) -> tuple[WalkForwardConfig, PredictionEvaluator]:
    config, previous = timestamp_fixture(tmp_path, horizon=timedelta(minutes=minutes))
    factory = ReturnFactory(previous.series[0], timedelta(minutes=minutes))
    adapter = PredictionEvaluator(
        dataset=previous.dataset,
        series=previous.series,
        primary_timeframe=previous.primary_timeframe,
        study_factory=factory,
        analyzer=ReturnWindowAnalyzer(),
        indicator_backend=previous.backend,
        grid_config=previous.grid_config,
    )
    outcome = OutcomeProvenance.capture_timestamp(
        factory.build({"window": 2}).outcome_labeler
    )
    plan = replace(
        config.plan,
        environment=replace(config.plan.environment, outcomes=(outcome,)),
        purge_policy=PurgePolicy(
            outcome.future_horizon, TemporalOffset.duration(timedelta(0))
        ),
    )
    return replace(config, plan=plan), adapter


def test_qf48_qf39_qf40_and_manifest_use_real_returns_without_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, adapter = walk_forward_fixture(tmp_path)
    study = WalkForwardStudy(config, adapter, tmp_path / "walk-forward")
    result = study.run()
    assert all(f.status is FoldStatus.COMPLETED for f in result.folds)
    decisions_seen: list[str] = []
    for fold in result.folds:
        assert fold.artifact is not None
        for decision in records(fold.artifact.snapshot.to_primitive()["decisions"]):
            for row in records(mapping(decision["prediction_study"])["rows"]):
                values = mapping(mapping(row["outcome"])["values"])
                assert values["decision_timestamp"] == decision["decision_timestamp"]
                assert values["raw_return"] == "0"
                assert values["available"] is True
                decisions_seen.append(str(values["decision_timestamp"]))
    assert len(decisions_seen) == len(set(decisions_seen)) > 2
    monkeypatch.setattr(IntradayForwardReturnOutcomeLabeler, "label_request", forbidden)
    monkeypatch.setattr(PredictionEvaluator, "select", forbidden)
    monkeypatch.setattr(PredictionEvaluator, "evaluate", forbidden)
    assert study.resume() == result
    source = load_oos_source(config.plan, study.study_path)
    aggregate = aggregate_prediction(source)
    assert len(aggregate.observations) == len(decisions_seen)
    for observation in aggregate.observations:
        row = mapping(observation.to_primitive()["row"])
        assert mapping(mapping(row["evaluation"])["values"])["raw_return"] == "0"
    assert (
        inspect_validation(
            source, study.study_path, artifact_root=tmp_path
        ).provenance.producer_study_id
        == study.study_id
    )
    incompatible, other = walk_forward_fixture(tmp_path, minutes=25)
    changed = WalkForwardStudy(incompatible, other, tmp_path / "changed-walk-forward")
    assert changed.study_id != study.study_id
    shutil.copytree(study.study_path, changed.study_path)
    with pytest.raises(WalkForwardPersistenceError):
        changed.resume()
    with pytest.raises((ValueError, ManifestError)):
        load_oos_source(incompatible.plan, study.study_path)
