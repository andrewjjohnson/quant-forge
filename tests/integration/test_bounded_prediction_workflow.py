"""Canonical intraday inputs through real QF-39/QF-40/QF-9 producers."""

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data import (
    DatasetFamily,
    MarketDataset,
    TimeframeBarSeries,
    validate_market_dataset,
)
from quantforge.data.models import (
    BoundedPredictionProvenance,
    IntradayPredictionProvenance,
)
from quantforge.data.prediction_views import (
    bounded_prediction_view,
    validate_prediction_view_lineage,
)
from quantforge.experiments import inspect_validation
from quantforge.experiments._holdout_integrity import validate_holdout_artifact
from quantforge.oos import (
    HoldoutEvaluation,
    HoldoutLedger,
    HoldoutState,
    aggregate_prediction,
    load_oos_source,
)
from quantforge.oos._records import mapping, records
from quantforge.prediction import (
    PredictionContextRequirements,
    PredictionDecisionSchedule,
    PredictionStudy,
)
from quantforge.prediction.outcome_temporal import OutcomeTemporalConfiguration
from quantforge.validation import (
    ConfigurationReference,
    DatasetProvenance,
    IndicatorComponent,
    IndicatorProvenance,
    OutcomeProvenance,
    PartitionRole,
    PredictionMembershipSource,
    PurgePolicy,
    ResearchRuleProvenance,
    TemporalOffset,
    TimestampBoundary,
    ValidationWindow,
)
from quantforge.walk_forward import (
    FoldStatus,
    PredictionEvaluator,
    WalkForwardConfig,
    WalkForwardPersistenceError,
    WalkForwardStudy,
)
from quantforge.walk_forward.partitions import partition
from tests.integration.test_intraday_prediction_provenance import (
    DAILY,
    TWO_MINUTES,
    Fixture,
    cached_fixture,
)
from tests.unit.helpers import SESSIONS
from tests.unit.walk_forward.fixtures import StudyFactory, StudyRule
from tests.unit.walk_forward.timestamp_fixtures import (
    MetadataEvaluator,
    MetadataLabeler,
    timestamp_fixture,
)


class BoundedFactory(StudyFactory):
    def __init__(self, source: TimeframeBarSeries) -> None:
        self.source = source

    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        original = super().build(parameters)
        requirements = cast(
            PredictionContextRequirements,
            getattr(original.strategy, "context_requirements"),
        )
        requirements = replace(
            requirements,
            primary=replace(
                requirements.primary,
                timeframe=TWO_MINUTES,
                required_feed_scope=self.source.dataset_reference.feed_scope,
            ),
            contextual=tuple(
                replace(
                    item, required_feed_scope=self.source.dataset_reference.feed_scope
                )
                for item in requirements.contextual
            ),
        )
        return PredictionStudy[Any, Any, Any].create(
            StudyRule(requirements),
            MetadataLabeler(
                OutcomeTemporalConfiguration.elapsed_duration(
                    timedelta(minutes=10), TWO_MINUTES
                )
            ),
            MetadataEvaluator(),
            outcome_source=self.source,
        )


def workflow_fixture(
    root: Path, fixture: Fixture
) -> tuple[WalkForwardConfig, PredictionEvaluator]:
    base, legacy = timestamp_fixture(root)
    factory = BoundedFactory(fixture.primary)
    adapter = PredictionEvaluator(
        dataset=fixture.dataset,
        series=(fixture.primary, fixture.daily),
        primary_timeframe=TWO_MINUTES,
        study_factory=factory,
        analyzer=legacy.analyzer,
        indicator_backend=legacy.backend,
        grid_config=legacy.grid_config,
    )
    studies = [
        factory.build(c.parameters.to_primitive()) for c in adapter.universe.candidates
    ]
    indicators = {
        (
            req.timeframe.configuration_id,
            indicator.configuration_id,
        ): IndicatorProvenance.capture(
            cast(IndicatorComponent, indicator.indicator), req.timeframe
        )
        for study in studies
        for req in cast(
            PredictionContextRequirements,
            getattr(study.strategy, "context_requirements"),
        ).all_timeframes
        for indicator in req.indicators
    }
    provenance = fixture.dataset.metadata.intraday_provenance
    assert isinstance(provenance, IntradayPredictionProvenance)
    family = DatasetFamily.from_manifest(provenance.family_manifest.to_primitive())
    environment = replace(
        base.plan.environment,
        dataset=DatasetProvenance.from_dataset_family(
            family,
            (
                fixture.primary.dataset_reference.dataset_id,
                fixture.daily.dataset_reference.dataset_id,
            ),
        ),
        prediction_dataset=DatasetProvenance.from_market_dataset(fixture.dataset),
        timeframes=(TWO_MINUTES, DAILY),
        research_rule=ResearchRuleProvenance.capture_prediction(studies[0].strategy),
        indicators=tuple(indicators.values()),
        outcomes=(OutcomeProvenance.capture_timestamp(studies[0].outcome_labeler),),
        aggregation_policies=(
            ConfigurationReference.capture_aggregation_policy(
                family.aggregation_policy
            ),
        ),
    )

    def window(original: ValidationWindow) -> ValidationWindow:
        start = cast(TimestampBoundary, original.interval.start).timestamp
        duration = timedelta(
            minutes=20 if original.role is PartitionRole.FINAL_HOLDOUT else 4
        )
        return replace(
            original,
            interval=replace(
                original.interval, end=TimestampBoundary(start + duration)
            ),
            warm_up_by_timeframe=tuple(
                replace(item, timeframe=TWO_MINUTES)
                if item.timeframe != DAILY
                else item
                for item in original.warm_up_by_timeframe
            ),
        )

    folds = tuple(
        replace(
            fold,
            development=window(fold.development),
            selection=window(fold.selection) if fold.selection is not None else None,
            test=window(fold.test),
        )
        for fold in base.plan.folds
    )
    schedule = PredictionDecisionSchedule(
        TWO_MINUTES,
        fixture.primary.bars[0].end_timestamp,
        fixture.primary.bars[-1].end_timestamp,
    )
    plan = replace(
        base.plan,
        environment=environment,
        folds=folds,
        final_holdout=replace(
            base.plan.final_holdout, window=window(base.plan.final_holdout.window)
        ),
        purge_policy=PurgePolicy(
            environment.outcomes[0].future_horizon,
            TemporalOffset.duration(timedelta(0)),
        ),
        prediction_membership=PredictionMembershipSource.capture(
            schedule, fixture.primary
        ),
    )
    return replace(base, plan=plan), adapter


@dataclass(frozen=True)
class CompletedWorkflow:
    config: WalkForwardConfig
    adapter: PredictionEvaluator
    study: WalkForwardStudy


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory) -> CompletedWorkflow:
    root = tmp_path_factory.mktemp("bounded-workflow")
    fixture = cached_fixture(root / "cache", session_dates=SESSIONS[:7])
    config, adapter = workflow_fixture(root, fixture)
    study = WalkForwardStudy(config, adapter, root / "study")
    result = study.run()
    for fold in result.folds:
        if fold.status is not FoldStatus.COMPLETED:
            pytest.fail(str([failure.to_primitive() for failure in fold.failures]))
    return CompletedWorkflow(config, adapter, study)


def test_train_selection_test_membership_resume_and_offline_integrity(
    completed: CompletedWorkflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, adapter, study = completed.config, completed.adapter, completed.study
    result = study.resume()
    for index, fold in enumerate(result.folds):
        assert fold.selection is not None
        assert fold.artifact is not None
        frozen = fold.selection.to_primitive()
        for role, key in (
            (PartitionRole.DEVELOPMENT, "development"),
            (PartitionRole.SELECTION, "selection"),
            (PartitionRole.WALK_FORWARD_TEST, "test"),
        ):
            permitted = partition(
                adapter.dataset, config.plan, index, role, minimum_observations=1
            )
            assert validate_market_dataset(permitted.dataset) == ()
            provenance = permitted.dataset.metadata.intraday_provenance
            assert isinstance(provenance, BoundedPredictionProvenance)
            assert provenance.causal_cutoff == permitted.decision_timestamps[0]
            assert provenance.canonical_input_id == adapter.dataset.metadata.dataset_id
            assert mapping(frozen["membership"])[key] == permitted.to_primitive()
        payload = fold.artifact.snapshot.to_primitive()
        expected = mapping(mapping(frozen["membership"])["test"])[
            "evaluation_timestamps"
        ]
        assert [
            d["decision_timestamp"] for d in records(payload["decisions"])
        ] == expected
        market = mapping(mapping(payload["manifest"])["market_data"])
        validate_prediction_view_lineage(
            market, adapter.dataset.metadata, permitted.decision_timestamps[0]
        )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("compatible resume and inspection must not execute research")

    monkeypatch.setattr(PredictionEvaluator, "select", forbidden)
    monkeypatch.setattr(PredictionEvaluator, "evaluate", forbidden)
    assert study.resume() == result
    source = load_oos_source(config.plan, study.study_path)
    assert aggregate_prediction(source) is not None
    inspected = inspect_validation(
        source, study.study_path, artifact_root=study.study_path.parent
    )
    assert inspected is not None


def test_reserved_holdout_explicit_consumption_and_durable_retry(
    completed: CompletedWorkflow, tmp_path: Path
) -> None:
    source = load_oos_source(completed.config.plan, completed.study.study_path)
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    assert ledger.reserve(source).state is HoldoutState.RESERVED
    evaluation = HoldoutEvaluation.prepare(
        source, completed.adapter, selection_fold_id=source.folds[-1].fold_id
    )
    assert ledger.state(source).state is HoldoutState.RESERVED
    view = evaluation.permitted.dataset
    assert validate_market_dataset(view) == ()
    provenance = view.metadata.intraday_provenance
    assert isinstance(provenance, BoundedPredictionProvenance)
    assert provenance.causal_cutoff == evaluation.permitted.decision_timestamps[0]
    consumed = ledger.consume(evaluation, run_id="qf52-offline-fixture")
    assert consumed.state is HoldoutState.CONSUMED
    assert consumed.consumption is not None
    validate_holdout_artifact(
        source,
        consumed.consumption.to_primitive(),
        ledger.result(evaluation).to_primitive(),
    )
    assert ledger.consume(evaluation, run_id="retry") == consumed
    assert ledger.state(source).state is HoldoutState.CONSUMED


def test_resume_rejects_changed_bounded_cutoff(
    completed: CompletedWorkflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    def shifted(
        dataset: MarketDataset, cutoff: datetime, *, start: date | None = None
    ) -> MarketDataset:
        return bounded_prediction_view(
            dataset, cutoff + timedelta(minutes=1), start=start
        )

    monkeypatch.setattr(
        "quantforge.data.prepared_prediction_views.bounded_prediction_view", shifted
    )
    with pytest.raises(WalkForwardPersistenceError):
        completed.study.resume()
