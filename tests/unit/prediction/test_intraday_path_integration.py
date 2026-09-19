"""Exercise path outcomes through existing export, validation, resume and integrity."""

import csv
import shutil
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping
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
    IntradayExcursionEvaluator,
    IntradayPathOutcomeLabeler,
    IntradayPathValues,
    IntradayTargetStopEvaluator,
    MultiTimeframeFeatureRequest,
    PredictionContextRequirements,
    PredictionRuleContext,
    PredictionStudy,
    PredictionStudyOutcome,
    SchemaFieldCategory,
    SignalFeatureCandidate,
    SignalFeatureCandidateOutput,
    SignalFeaturePersistenceError,
    build_signal_feature_dataset,
    intraday_excursion_outcome,
    intraday_target_stop_outcome,
    run_prediction_study,
)
from quantforge.validation import (
    OutcomeProvenance,
    PurgePolicy,
    TemporalOffset,
    TimestampBoundary,
    ValidationFold,
)
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
from tests.unit.prediction.test_intraday_forward_return_integration import (
    ReturnWindowAnalyzer,
)
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _FixtureCandidateRule,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_timestamp_replay import population
from tests.unit.walk_forward.timestamp_fixtures import (
    TimestampFactory,
    instant,
    timestamp_fixture,
)

type PathStudy = PredictionStudy[SignalFeatureCandidate, IntradayPathValues, Any]
type PathOutcome = PredictionStudyOutcome[IntradayPathValues, Any]


def configured(
    source: TimeframeBarSeries,
    kind: str,
    *,
    minutes: int = 60,
    target: str = "0.003",
    stop: str = "0.002",
) -> PathOutcome:
    if kind == "excursion":
        return intraday_excursion_outcome(timedelta(minutes=minutes), source)
    return intraday_target_stop_outcome(
        timedelta(minutes=minutes), source, Decimal(target), Decimal(stop)
    )


def replay_study(
    tmp_path: Path, kind: str = "excursion"
) -> tuple[MarketDataset, PathStudy, PathOutcome]:
    dataset, replay, _, template = population(tmp_path)
    source = template.outcome_source
    assert source is not None
    outcome = configured(source, kind)
    return (
        dataset,
        PredictionStudy[SignalFeatureCandidate, IntradayPathValues, Any].create(
            replay,
            outcome.labeler,
            outcome.evaluator,
            outcome_source=source,
        ),
        outcome,
    )


def forbidden(*args: object, **kwargs: object) -> None:
    pytest.fail("persisted results must not rerun path labeling or evaluation")


def test_combined_path_exports_preserve_anchors_order_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, study, excursion = replay_study(tmp_path)
    source = study.outcome_source
    assert source is not None
    target = configured(source, "target")
    outcomes = (excursion, target)
    result = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=outcomes,
        output_root=tmp_path / "features",
        chunk_size=1,
    )
    assert len(result.rows) == 2
    assert len({row.candidate_id for row in result.rows}) == 2
    for row, decision in zip(
        result.rows, (instant(4, "10:00"), instant(4, "10:05")), strict=True
    ):
        values = row.to_primitive()
        assert values["decision_timestamp"] == decision.isoformat()
        for outcome in outcomes:
            prefix = f"outcome_{outcome.namespace}_"
            assert values[prefix + "decision_timestamp"] == decision.isoformat()
            assert values[prefix + "path_start_timestamp"] == decision.isoformat()
            assert (
                values[prefix + "path_end_timestamp"]
                == (decision + timedelta(hours=1)).isoformat()
            )
            assert values[prefix + "elapsed_duration_microseconds"] == 3_600_000_000
            assert values[prefix + "available"] is True
            assert values[prefix + "status"] == "available"
            assert (
                values[prefix + "evaluation_configuration_id"]
                == outcome.evaluator_configuration_id
            )
            assert values[
                prefix + "source_reference"
            ] == source.dataset_reference.to_primitive(include_feed_scope=True)
    assert all(
        field.category is SchemaFieldCategory.FUTURE_OUTCOME
        for field in result.schema.fields
        if any(
            field.name.startswith(f"outcome_{outcome.namespace}_")
            for outcome in outcomes
        )
    )
    path = tmp_path / "features" / result.dataset_id
    original_bytes = (path / "features.csv").read_bytes()
    with (path / "features.csv").open(newline="") as file:
        assert len(tuple(csv.DictReader(file))) == 2
    assert len(tuple((path / "rows").glob("*.json"))) == 2
    monkeypatch.setattr(IntradayPathOutcomeLabeler, "label_request", forbidden)
    monkeypatch.setattr(IntradayExcursionEvaluator, "evaluate", forbidden)
    monkeypatch.setattr(IntradayTargetStopEvaluator, "evaluate", forbidden)
    assert (
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=outcomes,
            output_root=tmp_path / "features",
            chunk_size=20,
        )
        == result
    )
    assert (path / "features.csv").read_bytes() == original_bytes
    assert (
        inspect_study(
            StudyType.FEATURE_DATASET, path, artifact_root=tmp_path
        ).provenance.producer_study_id
        == result.dataset_id
    )


@pytest.mark.parametrize("kind", ["excursion", "target"])
@pytest.mark.parametrize("truncate", [False, True])
def test_missing_path_exports_null_values_and_distinct_endpoint_status(
    tmp_path: Path, kind: str, truncate: bool
) -> None:
    dataset, template, _ = replay_study(tmp_path, kind)
    full = template.outcome_source
    assert full is not None
    source = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        full.dataset_reference,
        full.timeframe,
        tuple(
            bar
            for bar in full.bars
            if (
                bar.end_timestamp <= instant(4, "10:45")
                if truncate
                else bar.end_timestamp != instant(4, "10:20")
            )
        ),
        dataset_family_manifest_id=full.dataset_family_manifest_id,
    )
    outcome = configured(source, kind)
    study = PredictionStudy[SignalFeatureCandidate, IntradayPathValues, Any].create(
        template.strategy, outcome.labeler, outcome.evaluator, outcome_source=source
    )
    result = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(outcome,),
        output_root=tmp_path / "unavailable",
    )
    for row in result.rows:
        prefix = f"outcome_{outcome.namespace}_"
        values = row.to_primitive()
        assert values[prefix + "available"] is False
        assert values[prefix + "endpoint_available"] is (not truncate)
        assert values[prefix + "status"] == (
            "dataset_end" if truncate else "missing_required_observation"
        )
        assert (
            values[
                prefix
                + ("mfe_percentage" if kind == "excursion" else "event_timestamp")
            ]
            is None
        )
        if kind == "target":
            assert values[prefix + "label"] == "unavailable"
    path = tmp_path / "unavailable" / result.dataset_id
    inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize("change", ["target", "stop", "horizon", "source"])
def test_material_changes_reject_copied_feature_checkpoints(
    tmp_path: Path, change: str
) -> None:
    dataset, study, outcome = replay_study(tmp_path, "target")
    original = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(outcome,),
        output_root=tmp_path / "original",
    )
    source = study.outcome_source
    assert source is not None
    if change == "source":
        source = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
            replace(source.dataset_reference, dataset_id="different-source"),
            source.timeframe,
            source.bars,
            dataset_family_manifest_id=source.dataset_family_manifest_id,
        )
    modified = configured(
        source,
        "target",
        minutes=30 if change == "horizon" else 60,
        target="0.005" if change == "target" else "0.003",
        stop="0.003" if change == "stop" else "0.002",
    )
    changed_study = PredictionStudy[
        SignalFeatureCandidate, IntradayPathValues, Any
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
    assert modified.configuration_id != outcome.configuration_id
    shutil.copytree(
        tmp_path / "original" / original.dataset_id,
        tmp_path / "copied" / changed.dataset_id,
    )
    with pytest.raises(SignalFeaturePersistenceError):
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=changed_study,
            contextual_features=(),
            outcomes=(modified,),
            output_root=tmp_path / "copied",
        )


def test_qf29_parquet_and_feature_causality(tmp_path: Path) -> None:
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

    outcomes = (configured(source, "excursion"), configured(source, "target"))
    study = PredictionStudy[SignalFeatureCandidate, IntradayPathValues, Any].create(
        CausalRule(requirements),
        outcomes[0].labeler,
        outcomes[0].evaluator,
        outcome_source=source,
    )
    result = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=outcomes,
        output_root=tmp_path / "contextual",
        context_provider=provider,
        multi_timeframe_features=(
            MultiTimeframeFeatureRequest(
                "primary",
                source.timeframe,
                requirements.primary.indicators[0].alias,
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
    inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize("kind", ["excursion", "target"])
def test_qf9_manifest_preserves_path_configuration_without_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    dataset, study, outcome = replay_study(tmp_path, kind)
    result = run_prediction_study(dataset, study)
    path = tmp_path / "prediction.json"
    write_json(path, result.to_primitive())
    block_research(monkeypatch)
    monkeypatch.setattr(IntradayPathOutcomeLabeler, "label_request", forbidden)
    monkeypatch.setattr(IntradayExcursionEvaluator, "evaluate", forbidden)
    monkeypatch.setattr(IntradayTargetStopEvaluator, "evaluate", forbidden)
    bundle = inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    manifest = create_manifest(bundle, execution())
    saved = write_manifest(manifest, tmp_path / "manifests", artifact_root=tmp_path)
    assert (
        read_manifest(saved, artifact_root=tmp_path).manifest_id == manifest.manifest_id
    )
    configuration = mapping(
        manifest.provenance.configuration.to_primitive()["configuration"]
    )
    assert (
        mapping(configuration["outcome_labeler"])["configuration_id"]
        == outcome.labeler_configuration_id
    )
    assert (
        mapping(configuration["evaluator"])["configuration_id"]
        == outcome.evaluator_configuration_id
    )


class PathFactory(TimestampFactory):
    def __init__(
        self, source: TimeframeBarSeries, horizon: timedelta, kind: str, target: str
    ) -> None:
        super().__init__(source, horizon)
        self.kind, self.target = kind, target

    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        template = super().build(parameters)
        outcome = configured(
            self.source,
            self.kind,
            minutes=self.horizon // timedelta(minutes=1),
            target=self.target,
        )
        return PredictionStudy[Any, IntradayPathValues, Any].create(
            template.strategy,
            outcome.labeler,
            outcome.evaluator,
            outcome_source=self.source,
        )


def walk_forward_fixture(
    tmp_path: Path, kind: str, *, target: str = "0.003", minutes: int = 60
) -> tuple[WalkForwardConfig, PredictionEvaluator]:
    config, previous = timestamp_fixture(tmp_path, horizon=timedelta(minutes=minutes))
    factory = PathFactory(previous.series[0], timedelta(minutes=minutes), kind, target)
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
    # The original 30m fixture has no retained selection/test rows at 60m.
    # Widen its timestamp windows, preserving QF-48's full conservative purge.
    folds: list[ValidationFold] = []
    for day, fold in zip((4, 5), config.plan.folds, strict=True):
        assert fold.selection is not None
        folds.append(
            replace(
                fold,
                development=replace(
                    fold.development,
                    interval=replace(
                        fold.development.interval,
                        end=TimestampBoundary(instant(day, "11:25")),
                    ),
                ),
                selection=replace(
                    fold.selection,
                    interval=replace(
                        fold.selection.interval,
                        start=TimestampBoundary(instant(day, "11:30")),
                    ),
                ),
            )
        )
    holdout = config.plan.final_holdout
    plan = replace(
        config.plan,
        folds=tuple(folds),
        final_holdout=replace(
            holdout,
            window=replace(
                holdout.window,
                interval=replace(
                    holdout.window.interval,
                    start=TimestampBoundary(instant(5, "14:30")),
                ),
            ),
        ),
        environment=replace(config.plan.environment, outcomes=(outcome,)),
        purge_policy=PurgePolicy(
            outcome.future_horizon, TemporalOffset.duration(timedelta(0))
        ),
    )
    return replace(config, plan=plan), adapter


@pytest.mark.parametrize("kind", ["excursion", "target"])
def test_qf48_qf39_qf40_timestamp_artifacts_resume_and_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    config, adapter = walk_forward_fixture(tmp_path, kind)
    study = WalkForwardStudy(config, adapter, tmp_path / "walk-forward")
    result = study.run()
    assert all(f.status is FoldStatus.COMPLETED for f in result.folds), result.folds
    decisions_seen: list[str] = []
    stored_evaluations: list[PrimitiveMapping] = []
    for fold in result.folds:
        assert fold.artifact is not None
        for decision in records(fold.artifact.snapshot.to_primitive()["decisions"]):
            for row in records(mapping(decision["prediction_study"])["rows"]):
                values = mapping(mapping(row["evaluation"])["values"])
                assert values["decision_timestamp"] == decision["decision_timestamp"]
                assert values["available"] is True
                decisions_seen.append(str(values["decision_timestamp"]))
                stored_evaluations.append(values)
    assert len(decisions_seen) == len(set(decisions_seen)) > 2
    monkeypatch.setattr(IntradayPathOutcomeLabeler, "label_request", forbidden)
    monkeypatch.setattr(IntradayExcursionEvaluator, "evaluate", forbidden)
    monkeypatch.setattr(IntradayTargetStopEvaluator, "evaluate", forbidden)
    monkeypatch.setattr(PredictionEvaluator, "select", forbidden)
    monkeypatch.setattr(PredictionEvaluator, "evaluate", forbidden)
    assert study.resume() == result
    source = load_oos_source(config.plan, study.study_path)
    aggregate = aggregate_prediction(source)
    assert len(aggregate.observations) == len(decisions_seen)
    summary = aggregate.to_primitive()["summary"]
    assert isinstance(summary, dict)
    if kind == "excursion":
        for metric, field in (("mfe", "mfe_percentage"), ("mae", "mae_percentage")):
            expected = [Decimal(str(values[field])) for values in stored_evaluations]
            statistics = mapping(summary[metric])
            assert statistics["sample_count"] == len(expected)
            with localcontext() as context:
                context.prec = 34
                assert Decimal(str(statistics["mean"])) == sum(expected) / len(expected)
    else:
        from collections import Counter

        expected_counts = Counter(str(values["label"]) for values in stored_evaluations)
        assert mapping(summary["event_outcomes"])["counts"] == dict(expected_counts)
    assert (
        inspect_validation(
            source, study.study_path, artifact_root=tmp_path
        ).provenance.producer_study_id
        == study.study_id
    )
    # The same path with changed threshold must still change frozen study identity.
    incompatible, other = walk_forward_fixture(
        tmp_path, kind, target="0.005", minutes=30 if kind == "excursion" else 60
    )
    changed = WalkForwardStudy(incompatible, other, tmp_path / "changed-walk-forward")
    assert changed.study_id != study.study_id
    shutil.copytree(study.study_path, changed.study_path)
    with pytest.raises(WalkForwardPersistenceError):
        changed.resume()
    if kind == "excursion":
        with pytest.raises((ValueError, ManifestError)):
            load_oos_source(incompatible.plan, study.study_path)
