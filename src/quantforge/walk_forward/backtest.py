"""QF-39 adapter over the unchanged QF-6 grid and QF-43 evaluation boundary."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

from quantforge.backtesting import EvaluationInterval, run_backtest
from quantforge.backtesting.export import (
    export_backtest_result,
    validate_backtest_result_artifact,
    validate_backtest_result_export,
)
from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import MarketDataset
from quantforge.optimization import GridSearchConfig, GridSearchStudy
from quantforge.optimization.combinations import ParameterCombination
from quantforge.optimization.factories import StrategyFactory
from quantforge.optimization.study import OPTIMIZATION_ENGINE_VERSION
from quantforge.validation import (
    BacktestProvenance,
    DatasetProvenance,
    PartitionRole,
    ResearchRuleProvenance,
    ResearchStudyType,
    ValidationPlan,
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
    BacktestOOSArtifact,
    CandidateConfiguration,
    CandidateUniverse,
    FrozenSelection,
    OOSArtifact,
    SelectionEvidence,
    WalkForwardConfig,
    WalkForwardError,
)
from quantforge.walk_forward.partitions import PermittedPartition, partition


class BacktestEvaluator:
    """Each selection/test run starts a fresh QF-43 account on bounded inputs."""

    def __init__(
        self,
        dataset: MarketDataset,
        strategy_factory: StrategyFactory,
        grid_config: GridSearchConfig,
    ) -> None:
        if grid_config.backtest.evaluation_interval is not None:
            raise WalkForwardError("QF-39 owns each fold's evaluation interval")
        validate_search_parameters(grid_config.search_space.names)
        self.dataset = dataset
        self.factory = deepcopy(strategy_factory)
        self.grid_config = deepcopy(grid_config)
        self._universe = self._capture_universe()

    def _capture_universe(self) -> CandidateUniverse:
        grid = GridSearchStudy(self.dataset, self.factory, self.grid_config)
        definition = self.grid_config.to_primitive(self.factory)
        definition.pop("persistence")
        definition.pop("execution")
        definition["factory"] = self.factory.configuration()
        definition["strategy_name"] = self.factory.strategy_name
        definition["strategy_version"] = self.factory.strategy_version
        candidates = tuple(
            CandidateConfiguration(
                item.combination_id,
                item.parameters_snapshot,
                PrimitiveMappingSnapshot.capture(
                    self.factory.build(item.parameters).configuration()
                ),
            )
            for item in grid.candidates
            if isinstance(item, ParameterCombination)
        )
        return CandidateUniverse(
            ResearchStudyType.TRADING_BACKTEST,
            candidates,
            PrimitiveMappingSnapshot.capture(definition),
        )

    @property
    def universe(self) -> CandidateUniverse:
        return self._universe

    def configuration(self) -> PrimitiveMapping:
        return {
            "adapter": "qf39_backtest",
            "grid_engine": OPTIMIZATION_ENGINE_VERSION,
            "version": "1",
            "universe": self._capture_universe().to_primitive(),
            "dataset": DatasetProvenance.from_market_dataset(
                self.dataset
            ).to_primitive(),
        }

    def validate(self, plan: ValidationPlan) -> None:
        if (
            plan.environment.study_type is not ResearchStudyType.TRADING_BACKTEST
            or plan.environment.outcome_dataset
            != DatasetProvenance.from_market_dataset(self.dataset)
            or plan.environment.execution
            != BacktestProvenance.capture(self.grid_config.backtest)
        ):
            raise WalkForwardError(
                "backtest environment differs from the validation plan"
            )
        validate_fixed_backends(plan)
        for item in self.universe.candidates:
            strategy = self.factory.build(item.parameters.to_primitive())
            if strategy.configuration() != item.definition.to_primitive():
                raise WalkForwardError("factory changed a frozen candidate definition")
            validate_rule(plan, ResearchRuleProvenance.capture_trading(strategy))

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

    def select(
        self, config: WalkForwardConfig, fold_index: int, output_root: Path
    ) -> SelectionEvidence:
        permitted = self._partition(config, fold_index, test=False)
        bounded = replace(
            self.grid_config,
            backtest=replace(
                self.grid_config.backtest,
                evaluation_interval=EvaluationInterval(
                    permitted.sessions[0], permitted.sessions[-1]
                ),
            ),
            persistence=replace(self.grid_config.persistence, output_root=output_root),
        )
        grid = GridSearchStudy(permitted.dataset, deepcopy(self.factory), bounded)
        for item in grid.candidates:
            if isinstance(item, ParameterCombination):
                candidate = self.universe.candidate(item.combination_id)
                if (
                    self.factory.build(item.parameters).configuration()
                    != candidate.definition.to_primitive()
                ):
                    raise WalkForwardError("selection changed candidate configuration")
        result = grid.resume() if grid.store.manifest_path.exists() else grid.run()
        return choose(self.universe, result, config.selection_policy)

    def evaluate(
        self,
        config: WalkForwardConfig,
        fold_index: int,
        selection: FrozenSelection,
        output_root: Path,
    ) -> BacktestOOSArtifact:
        candidate = frozen_candidate(self.universe, selection)
        strategy = deepcopy(self.factory).build(candidate.parameters.to_primitive())
        if strategy.configuration() != candidate.definition.to_primitive():
            raise WalkForwardError("test configuration differs from frozen selection")
        permitted = self._partition(config, fold_index, test=True)
        bounded = replace(
            self.grid_config.backtest,
            evaluation_interval=EvaluationInterval(
                permitted.sessions[0], permitted.sessions[-1]
            ),
        )
        result = run_backtest(permitted.dataset, strategy, bounded)
        if result.strategy_configuration != candidate.definition.to_primitive():
            raise WalkForwardError("test evaluator changed the frozen configuration")
        destination = output_root / result.run_id
        if destination.exists():
            validate_backtest_result_export(result, destination)
        else:
            export_backtest_result(result, output_root)
        integrity = (destination / "integrity.json").read_text()
        return BacktestOOSArtifact(
            selection.selection_id,
            result.run_id,
            PrimitiveMappingSnapshot.capture(result.to_primitive()),
            result.run_id,
            configuration_identity({"integrity": integrity}),
        )

    def validate_artifact(
        self,
        config: WalkForwardConfig,
        fold_index: int,
        selection: FrozenSelection,
        artifact: OOSArtifact,
        output_root: Path,
    ) -> None:
        if not isinstance(artifact, BacktestOOSArtifact):
            raise WalkForwardError("expected a typed backtest OOS artifact")
        candidate = frozen_candidate(self.universe, selection)
        permitted = self._partition(config, fold_index, test=True)
        record = artifact.snapshot.to_primitive()
        manifest = cast(PrimitiveMapping, record["manifest"])
        strategy = cast(PrimitiveMapping, manifest["strategy"])
        expected = replace(
            self.grid_config.backtest,
            evaluation_interval=EvaluationInterval(
                permitted.sessions[0], permitted.sessions[-1]
            ),
        ).to_primitive()
        if (
            artifact.selection_id != selection.selection_id
            or artifact.export_location != artifact.run_id
            or manifest["run_id"] != artifact.run_id
            or strategy["configuration"] != candidate.definition.to_primitive()
            or manifest["backtest_configuration"] != expected
        ):
            raise WalkForwardError("incompatible OOS backtest artifact")
        destination = output_root / artifact.export_location
        validate_backtest_result_artifact(destination)
        if artifact.export_fingerprint != configuration_identity(
            {"integrity": (destination / "integrity.json").read_text()}
        ):
            raise WalkForwardError("backtest export fingerprint mismatch")
