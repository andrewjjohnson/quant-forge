"""QF-39 prediction orchestration using QF-42 windows and QF-32 grids."""

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.data import (
    MarketDataset,
    MultiTimeframeContext,
    TimeframeBarSeries,
    build_multi_timeframe_context,
)
from quantforge.prediction import (
    PredictionContextRequirements,
    PredictionDecisionSchedule,
    run_prediction_window,
)
from quantforge.prediction.grid import (
    PREDICTION_GRID_ENGINE_VERSION,
    PredictionContextEnvironment,
    PredictionGridCombination,
    PredictionGridConfig,
    PredictionGridStudy,
    PredictionIndicatorBackendEnvironment,
    PredictionStudyFactory,
    PredictionWindowAnalyzer,
    _trial_definition,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.study import (
    STUDY_ENGINE_VERSION,
    _capture_study_configuration,  # pyright: ignore[reportPrivateUsage]
    prepare_prediction_study_dataset,
)
from quantforge.prediction.window import (
    PREDICTION_WINDOW_ENGINE_VERSION,
    PredictionWindowContextProvider,
    _capture_window_identity,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.window_validation import validate_prediction_window_snapshot
from quantforge.timeframes import Timeframe, resolve_exchange_session
from quantforge.validation import (
    DatasetProvenance,
    OutcomeProvenance,
    PartitionRole,
    ResearchRuleProvenance,
    ResearchStudyType,
    TimestampBoundary,
    TimestampOutcomeComponent,
    ValidationPlan,
    select_prediction_context_observations,
)
from quantforge.walk_forward._adapters import (
    choose,
    frozen_candidate,
    membership,
    validate_fixed_backends,
    validate_rule,
    validate_search_parameters,
)
from quantforge.walk_forward.models import (
    CandidateConfiguration,
    CandidateUniverse,
    FrozenSelection,
    OOSArtifact,
    PredictionOOSArtifact,
    SelectionEvidence,
    WalkForwardConfig,
    WalkForwardError,
)
from quantforge.walk_forward.partitions import (
    EvaluationPartition,
    PermittedPartition,
    partition,
)


@dataclass(frozen=True)
class _PermittedContextProvider:
    plan: ValidationPlan
    permitted: EvaluationPartition
    series: tuple[TimeframeBarSeries, ...]
    schedule: PredictionDecisionSchedule

    def get_context_at(
        self, requirements: PredictionContextRequirements, *, as_of: datetime
    ) -> MultiTimeframeContext:
        if as_of not in self.schedule.decision_timestamps:
            raise WalkForwardError(
                "prediction requested a decision outside permitted membership"
            )
        selected: list[TimeframeBarSeries] = []
        required_ids = {
            item.timeframe.configuration_id for item in requirements.all_timeframes
        }
        for source in self.series:
            if source.timeframe.configuration_id not in required_ids:
                continue
            evidence = select_prediction_context_observations(
                self.plan,
                self.permitted.window,
                source=source,
                as_of=TimestampBoundary(as_of),
            ).source_selection
            keys = {*evidence.warm_up_context, *evidence.study_observations}
            # QF-8 verified this original artifact and exact membership. Preserve
            # its lineage while exposing only those original bars to QF-20/QF-28.
            selected.append(
                TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
                    source.dataset_reference,
                    source.timeframe,
                    tuple(
                        bar
                        for bar in source.bars
                        if TimestampBoundary(bar.end_timestamp) in keys
                    ),
                    dataset_family_manifest_id=source.dataset_family_manifest_id,
                    developing_source_evidence=source._developing_source_evidence,  # pyright: ignore[reportPrivateUsage]
                )
            )
        return build_multi_timeframe_context(
            as_of=as_of,
            primary_timeframe=requirements.primary.timeframe,
            required_timeframes=requirements.context_timeframe_requirements(),
            completion_policy=requirements.context_completion_policy,
            series=tuple(selected),
        )


def _logical_definition(
    candidate: PredictionGridCombination,
) -> PrimitiveMappingSnapshot:
    definition = candidate.trial_definition_snapshot.to_primitive()
    definition.pop("decision_schedule", None)
    return PrimitiveMappingSnapshot.capture(definition)


class PredictionEvaluator:
    """QF-8 session or explicit timestamp membership; QF-42 decisions."""

    def __init__(
        self,
        *,
        dataset: MarketDataset,
        series: tuple[TimeframeBarSeries, ...],
        primary_timeframe: Timeframe,
        study_factory: PredictionStudyFactory,
        analyzer: PredictionWindowAnalyzer,
        indicator_backend: PredictionIndicatorBackendEnvironment,
        grid_config: PredictionGridConfig,
    ) -> None:
        validate_search_parameters(grid_config.search_space.names)
        self.dataset = dataset
        self.series = tuple(
            sorted(series, key=lambda item: item.timeframe.configuration_id)
        )
        if len({item.timeframe.configuration_id for item in self.series}) != len(
            self.series
        ):
            raise WalkForwardError("prediction sources must have unique timeframes")
        self.primary_timeframe = primary_timeframe
        self.factory = deepcopy(study_factory)
        self.analyzer = deepcopy(analyzer)
        self.backend = indicator_backend
        self.grid_config = deepcopy(grid_config)
        self._universe = self._capture_universe()

    @property
    def universe(self) -> CandidateUniverse:
        return self._universe

    def _environment(
        self,
        plan: ValidationPlan | None = None,
        permitted: EvaluationPartition | None = None,
    ) -> PredictionContextEnvironment:
        return PredictionContextEnvironment.create(
            "qf39_permitted_local_context",
            "1",
            {
                "sources": [
                    {
                        "reference": source.dataset_reference.to_primitive(),
                        "manifest_id": source.dataset_family_manifest_id,
                    }
                    for source in self.series
                ],
                "plan_id": None if plan is None else plan.plan_id,
                "partition": None if permitted is None else permitted.to_primitive(),
            },
        )

    def _grid(
        self,
        dataset: MarketDataset,
        schedule: PredictionDecisionSchedule,
        provider: PredictionWindowContextProvider,
        environment: PredictionContextEnvironment,
        config: PredictionGridConfig,
    ) -> PredictionGridStudy:
        return PredictionGridStudy(
            dataset=dataset,
            dataset_family_fingerprint=self.series[0].dataset_reference.family_id,
            study_factory=deepcopy(self.factory),
            analyzer=deepcopy(self.analyzer),
            context_provider=provider,
            context_environment=environment,
            indicator_backend=self.backend,
            config=config,
            decision_schedule=schedule,
        )

    def _capture_universe(self) -> CandidateUniverse:
        primary = next(
            (s for s in self.series if s.timeframe == self.primary_timeframe), None
        )
        if primary is None or not primary.bars:
            raise WalkForwardError(
                "prediction adapter requires its declared primary series"
            )
        timestamp = primary.bars[0].end_timestamp
        schedule = PredictionDecisionSchedule(
            self.primary_timeframe, timestamp, timestamp
        )
        # Enumeration never resolves this provider. QF-32 owns candidate checks.
        grid = self._grid(
            self.dataset, schedule, self, self._environment(), self.grid_config
        )
        candidates = tuple(
            CandidateConfiguration(
                item.combination_id,
                item.parameters_snapshot,
                _logical_definition(item),
            )
            for item in grid.candidates
            if isinstance(item, PredictionGridCombination)
        )
        definition: PrimitiveMapping = {
            "label": self.grid_config.label,
            "factory": {
                "name": self.factory.name,
                "version": self.factory.version,
                "configuration": self.factory.configuration(),
            },
            "analyzer": {
                "name": self.analyzer.name,
                "version": self.analyzer.version,
                "configuration_id": self.analyzer.configuration_id,
                "configuration": self.analyzer.configuration(),
            },
            "search_space": self.grid_config.search_space.to_primitive(
                self.factory.parameter_order
            ),
            "constraints": [
                c.to_primitive() for c in self.grid_config.parameter_constraints
            ],
            "ranking": self.grid_config.ranking.to_primitive(),
            "stability": self.grid_config.stability.to_primitive(),
            "backend": self.backend.to_primitive(),
            "primary_timeframe": self.primary_timeframe.to_primitive(),
            "context": self._environment().to_primitive(),
        }
        return CandidateUniverse(
            ResearchStudyType.PREDICTION,
            candidates,
            PrimitiveMappingSnapshot.capture(definition),
        )

    def get_context_at(
        self, requirements: PredictionContextRequirements, *, as_of: datetime
    ) -> MultiTimeframeContext:
        raise WalkForwardError("unbounded prediction context cannot be executed")

    def configuration(self) -> PrimitiveMapping:
        return {
            "adapter": "qf39_prediction",
            "prediction_engine": STUDY_ENGINE_VERSION,
            "grid_engine": PREDICTION_GRID_ENGINE_VERSION,
            "window_engine": PREDICTION_WINDOW_ENGINE_VERSION,
            "version": "1",
            "universe": self._capture_universe().to_primitive(),
            "dataset": DatasetProvenance.from_market_dataset(
                self.dataset
            ).to_primitive(),
        }

    def validate(self, plan: ValidationPlan) -> None:
        if (
            plan.environment.study_type is not ResearchStudyType.PREDICTION
            or plan.environment.prediction_dataset
            != DatasetProvenance.from_market_dataset(self.dataset)
        ):
            raise WalkForwardError(
                "prediction adapter requires the plan's two-input dataset contract"
            )
        if {s.dataset_reference for s in self.series} != set(
            plan.environment.dataset.family_references
        ):
            raise WalkForwardError("context sources differ from the validation plan")
        if any(
            s.dataset_family_manifest_id != plan.environment.dataset.family_manifest_id
            for s in self.series
        ):
            raise WalkForwardError(
                "context family manifest differs from the validation plan"
            )
        if plan.prediction_membership is not None:
            primary = next(
                s for s in self.series if s.timeframe == self.primary_timeframe
            )
            plan.prediction_membership.validate_source(primary)
        validate_fixed_backends(plan)
        for candidate in self.universe.candidates:
            study = self.factory.build(candidate.parameters.to_primitive())
            validate_rule(
                plan, ResearchRuleProvenance.capture_prediction(study.strategy)
            )
            if plan.prediction_membership is None:
                from quantforge.validation import SessionOutcomeComponent

                outcome = OutcomeProvenance.capture_exchange_sessions(
                    cast(SessionOutcomeComponent, study.outcome_labeler)
                )
            else:
                outcome = OutcomeProvenance.capture_timestamp(
                    cast(TimestampOutcomeComponent, study.outcome_labeler)
                )
                if study.outcome_source is None or not any(
                    study.outcome_source == source for source in self.series
                ):
                    raise WalkForwardError(
                        "timestamp outcome source must match a selected "
                        "canonical artifact"
                    )
            if outcome not in plan.environment.outcomes:
                raise WalkForwardError(
                    "candidate outcome is outside the declared universe"
                )
            requirements = cast(
                PredictionContextRequirements,
                getattr(study.strategy, "context_requirements"),
            )
            if requirements.primary.timeframe != self.primary_timeframe:
                raise WalkForwardError(
                    "candidate changed the fixed primary decision timeframe"
                )

    def membership(
        self, config: WalkForwardConfig, fold_index: int
    ) -> PrimitiveMapping:
        return membership(self.dataset, config, fold_index)

    def _partition(
        self, config: WalkForwardConfig, fold_index: int, *, test: bool
    ) -> PermittedPartition:
        role = (
            PartitionRole.WALK_FORWARD_TEST
            if test
            else PartitionRole.SELECTION
            if config.plan.folds[fold_index].selection is not None
            else PartitionRole.DEVELOPMENT
        )
        return partition(
            self.dataset,
            config.plan,
            fold_index,
            role,
            minimum_observations=(
                config.minimum_test_observations
                if test
                else config.minimum_training_observations
            ),
        )

    def _schedule(self, permitted: EvaluationPartition) -> PredictionDecisionSchedule:
        timestamps = permitted.to_primitive().get("evaluation_timestamps")
        if timestamps is not None:
            values = cast(list[str], timestamps)
            schedule = PredictionDecisionSchedule(
                self.primary_timeframe,
                datetime.fromisoformat(values[0]),
                datetime.fromisoformat(values[-1]),
            )
            if [t.isoformat() for t in schedule.decision_timestamps] != values:
                raise WalkForwardError(
                    "QF-42 schedule differs from retained QF-8 membership"
                )
            return schedule
        policy = self.primary_timeframe.session_policy
        schedule = PredictionDecisionSchedule(
            self.primary_timeframe,
            resolve_exchange_session(permitted.sessions[0], policy).open_timestamp,
            resolve_exchange_session(permitted.sessions[-1], policy).close_timestamp,
        )
        # Missing daily outcome observations cannot be silently scheduled as
        # eligible decisions. The caller must choose compatible source coverage.
        if not set(schedule.decision_sessions).issubset(permitted.sessions):
            raise WalkForwardError(
                "prediction schedule includes a non-permitted outcome session"
            )
        return schedule

    def select(
        self, config: WalkForwardConfig, fold_index: int, output_root: Path
    ) -> SelectionEvidence:
        permitted = self._partition(config, fold_index, test=False)
        schedule = self._schedule(permitted)
        grid = self._grid(
            permitted.dataset,
            schedule,
            _PermittedContextProvider(config.plan, permitted, self.series, schedule),
            self._environment(config.plan, permitted),
            replace(self.grid_config, output_root=output_root),
        )
        for item in grid.candidates:
            if isinstance(item, PredictionGridCombination):
                if (
                    _logical_definition(item)
                    != self.universe.candidate(item.combination_id).definition
                ):
                    raise WalkForwardError("selection changed a candidate definition")
        exists = (output_root / grid.study_id / "manifest.json").exists()
        result = grid.resume() if exists else grid.run()
        return choose(self.universe, result, config.selection_policy)

    def evaluate(
        self,
        config: WalkForwardConfig,
        fold_index: int,
        selection: FrozenSelection,
        output_root: Path,
    ) -> PredictionOOSArtifact:
        return self.evaluate_partition(
            config.plan,
            self._partition(config, fold_index, test=True),
            selection,
            output_root,
        )

    def evaluate_partition(
        self,
        plan: ValidationPlan,
        permitted: EvaluationPartition,
        selection: FrozenSelection,
        output_root: Path,
    ) -> PredictionOOSArtifact:
        """Evaluate a prevalidated boundary with the existing frozen candidate."""
        candidate = frozen_candidate(self.universe, selection)
        schedule = self._schedule(permitted)
        study = deepcopy(self.factory).build(candidate.parameters.to_primitive())
        definition, _ = _trial_definition(study, self.backend)
        if PrimitiveMappingSnapshot.capture(definition) != candidate.definition:
            raise WalkForwardError("test configuration differs from frozen selection")
        result = run_prediction_window(
            permitted.dataset,
            study,
            schedule=schedule,
            context_provider=_PermittedContextProvider(
                plan, permitted, self.series, schedule
            ),
            dataset_family_fingerprint=self.series[0].dataset_reference.family_id,
            context_environment=self._environment(plan, permitted).to_primitive(),
            indicator_backend_environment=self.backend.to_primitive(),
        )
        if not any(
            getattr(row.signal, "disposition", None) != "rejected"
            for decision in result.decisions
            for row in decision.result.rows
        ):
            raise WalkForwardError(
                "prediction test window contains no valid predictions"
            )
        return PredictionOOSArtifact(
            selection.selection_id,
            result.window_result_id,
            PrimitiveMappingSnapshot.capture(result.to_primitive()),
        )

    def validate_artifact(
        self,
        config: WalkForwardConfig,
        fold_index: int,
        selection: FrozenSelection,
        artifact: OOSArtifact,
        output_root: Path,
    ) -> None:
        self.validate_partition_artifact(
            config.plan,
            self._partition(config, fold_index, test=True),
            selection,
            artifact,
            output_root,
        )

    def validate_partition_artifact(
        self,
        plan: ValidationPlan,
        permitted: EvaluationPartition,
        selection: FrozenSelection,
        artifact: OOSArtifact,
        output_root: Path,
    ) -> None:
        """Verify an existing artifact against its explicit evaluation boundary."""
        if (
            not isinstance(artifact, PredictionOOSArtifact)
            or artifact.selection_id != selection.selection_id
        ):
            raise WalkForwardError("incompatible prediction OOS artifact")
        candidate = frozen_candidate(self.universe, selection)
        schedule = self._schedule(permitted)
        study = deepcopy(self.factory).build(candidate.parameters.to_primitive())
        definition, _ = _trial_definition(study, self.backend)
        if PrimitiveMappingSnapshot.capture(definition) != candidate.definition:
            raise WalkForwardError("test configuration differs from frozen selection")
        expected = _capture_window_identity(
            prepare_prediction_study_dataset(permitted.dataset),
            _capture_study_configuration(study),
            schedule=schedule,
            dataset_family_fingerprint=self.series[0].dataset_reference.family_id,
            context_environment=self._environment(plan, permitted).to_primitive(),
            indicator_backend_environment=self.backend.to_primitive(),
        )
        snapshot = artifact.snapshot.to_primitive()
        validate_prediction_window_snapshot(
            snapshot,
            expected_identity=expected,
            schedule=schedule,
            outcome_sessions=tuple(b.session_date for b in permitted.dataset.bars),
            strategy_parameters=study.strategy.parameters.to_primitive(),
        )
        if (
            cast(PrimitiveMapping, snapshot["manifest"])["window_result_id"]
            != artifact.window_result_id
        ):
            raise WalkForwardError("prediction OOS result identity mismatch")
