"""Failed smoke comparisons stop before freeze/OOS; sparse successes may select."""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.examples import spy_ema_runner
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.examples.spy_ema_plan import SmokePredictionEvaluator
from quantforge.experiments import ArtifactEntry, ArtifactRelationship
from quantforge.oos import HoldoutLedger
from quantforge.optimization import IntegerValues, ParameterSearchSpace, TrialStatus
from quantforge.prediction import PredictionRuleContext, PredictionStrategyOutput
from quantforge.prediction.grid import (
    PredictionGridTrialRecord,
    PredictionTrialAnalysis,
)
from quantforge.prediction.window_encoding import mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.validation import IndicatorComponent, IndicatorProvenance, PartitionRole
from quantforge.walk_forward import (
    FrozenSelection,
    PredictionEvaluator,
    PredictionOOSArtifact,
    WalkForwardConfig,
)
from quantforge.walk_forward.persistence import read_record
from tests.unit.experiments.test_contracts import execution
from tests.unit.prediction.test_incremental_prediction_grid import CompactWindowAnalyzer
from tests.unit.walk_forward.fixtures import StudyRule
from tests.unit.walk_forward.timestamp_fixtures import timestamp_fixture


class ComparisonInspectionReachedError(Exception):
    """Stop after native freeze/OOS, before comparison exports and publication."""


@pytest.mark.parametrize("scenario", ["failed_analysis", "empty", "unrankable"])
def test_comparison_completion_before_freeze_and_oos_on_run_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    scenario: str,
) -> None:
    # Small native windows exercise the real QF-32/QF-39 gate. The full EMA
    # acceptance fixture separately covers the frozen rule and report pipeline.
    original_config, original = timestamp_fixture(tmp_path)

    def adapter_for(windows: list[int]) -> SmokePredictionEvaluator:
        return SmokePredictionEvaluator(
            dataset=original.dataset,
            series=original.series,
            primary_timeframe=original.primary_timeframe,
            study_factory=original.factory,
            analyzer=CompactWindowAnalyzer(),
            indicator_backend=original.backend,
            grid_config=replace(
                original.grid_config,
                search_space=ParameterSearchSpace({"window": IntegerValues(windows)}),
                window_schema_version="2",
            ),
        )

    comparison, fixed_adapter = adapter_for([1, 2, 3]), adapter_for([2])
    studies = [
        comparison.factory.build(candidate.parameters.to_primitive())
        for candidate in comparison.universe.candidates
    ]
    indicators = {
        (
            requirement.timeframe.configuration_id,
            indicator.configuration_id,
        ): IndicatorProvenance.capture(
            cast(IndicatorComponent, indicator.indicator), requirement.timeframe
        )
        for study in studies
        for requirement in getattr(
            study.strategy, "context_requirements"
        ).all_timeframes
        for indicator in requirement.indicators
    }
    config = replace(
        original_config,
        plan=replace(
            original_config.plan,
            folds=original_config.plan.folds[:1],
            environment=replace(
                original_config.plan.environment, indicators=tuple(indicators.values())
            ),
        ),
        continue_on_failure=False,
    )

    def prepared(
        inputs: SmokeInputs, root: Path, *, fixed: bool = False
    ) -> tuple[WalkForwardConfig, PredictionEvaluator]:
        return config, fixed_adapter if fixed else comparison

    def inspection(
        inputs: SmokeInputs,
        config: WalkForwardConfig,
        adapter: PredictionEvaluator,
        phase_root: Path,
        *args: object,
        role: PartitionRole = PartitionRole.SELECTION,
    ) -> tuple[tuple[ArtifactEntry, ...], tuple[ArtifactRelationship, ...]]:
        if phase_root.name != "fixed":
            raise ComparisonInspectionReachedError
        return (), ()

    monkeypatch.setattr(spy_ema_runner, "prepare_walk_forward", prepared)
    monkeypatch.setattr(spy_ema_runner, "inspect_phase", inspection)
    original_analysis = CompactWindowAnalyzer.analyze_compact_window
    analysis_calls = 0

    def analysis(
        self: CompactWindowAnalyzer, reader: PredictionWindowReader
    ) -> PredictionTrialAnalysis:
        nonlocal analysis_calls
        analysis_calls += 1
        reader.verify_integrity()
        if analysis_calls == 2 and scenario == "failed_analysis":
            raise ValueError("injected comparison analysis failure after finalization")
        result = original_analysis(self, reader)
        if analysis_calls == 2 and scenario == "unrankable":
            return replace(
                result, metrics_snapshot=PrimitiveMappingSnapshot.capture({})
            )
        return result

    monkeypatch.setattr(CompactWindowAnalyzer, "analyze_compact_window", analysis)
    if scenario == "empty":
        empty_rule_id = comparison.factory.build(
            {"window": 1}
        ).strategy.configuration_id
        original_rule = StudyRule.generate_with_context

        def sparse_rule(
            self: StudyRule, context: PredictionRuleContext
        ) -> PredictionStrategyOutput:
            if self.configuration_id == empty_rule_id:
                return PredictionStrategyOutput(
                    self.name, self.configuration_id, context.prediction_dataset_id, ()
                )
            return original_rule(self, context)

        monkeypatch.setattr(StudyRule, "generate_with_context", sparse_rule)

    original_evaluate = PredictionEvaluator.evaluate
    oos_calls = 0

    def evaluate(
        self: PredictionEvaluator,
        config: WalkForwardConfig,
        fold_index: int,
        selection: FrozenSelection,
        output_root: Path,
    ) -> PredictionOOSArtifact:
        nonlocal oos_calls
        oos_calls += 1
        return original_evaluate(self, config, fold_index, selection, output_root)

    monkeypatch.setattr(PredictionEvaluator, "evaluate", evaluate)
    root = tmp_path / "run"
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    recorded: dict[Path, bytes] = {}
    for _ in range(2):
        expected = (
            ValueError
            if scenario == "failed_analysis"
            else ComparisonInspectionReachedError
        )
        with pytest.raises(expected):
            spy_ema_runner.run_pre_holdout(
                cast(SmokeInputs, object()), root, ledger, execution()
            )
        assert analysis_calls == 4  # Fixed + three trials; resume reruns none.
        paths = sorted(
            (root / "walk-forward").glob("*/folds/*/selection/*/trials/*.json")
        )
        trials = [
            PredictionGridTrialRecord.from_primitive(
                mapping(json.loads(path.read_text()))
            )
            for path in paths
        ]
        assert len(trials) == 3
        assert {cast(int, trial.parameters["window"]) for trial in trials} == {1, 2, 3}
        assert sum(trial.status is TrialStatus.FAILED for trial in trials) == (
            1 if scenario == "failed_analysis" else 0
        )
        affected = next(trial for trial in trials if trial.parameters["window"] == 1)
        if scenario == "failed_analysis":
            assert affected.failure_type == "ValueError"
            assert affected.failure_message
            assert oos_calls == 0
            assert not list((root / "walk-forward").glob("*/folds/*/selection.json"))
            assert not list((root / "walk-forward").glob("*/folds/*/test"))
            assert not list((root / "walk-forward").glob("*/folds/*/oos.json"))
            state_paths = list((root / "walk-forward").glob("*/folds/*/state.json"))
            state = read_record(state_paths[0])
            assert state["status"] == "failed"
            assert state["selection_id"] is None
            failure = mapping(cast(list[object], state["failures"])[0])
            assert failure["stage"] == "selection"
            assert failure["error_type"] == "WalkForwardError"
            assert failure["message"] == (
                "QF-45 requires every comparison trial to succeed before selection "
                "is frozen; preserve failure evidence"
            )
        else:
            assert all(trial.status is TrialStatus.SUCCEEDED for trial in trials)
            assert affected.analysis is not None
            assert (affected.analysis.prediction_count == 0) is (scenario == "empty")
            assert oos_calls == 1  # Completed OOS is verified on resume.
            frozen_paths = list(
                (root / "walk-forward").glob("*/folds/*/selection.json")
            )
            assert len(frozen_paths) == 1
            frozen = read_record(frozen_paths[0])
            assert mapping(mapping(frozen["candidate"])["parameters"])["window"] != 1
        windows = sorted(root.rglob("prediction-window.jsonl"))
        assert len(windows) == (4 if scenario == "failed_analysis" else 5)
        current = {path: path.read_bytes() for path in [*paths, *windows]}
        if recorded:
            assert current == recorded
        recorded = current
        assert "PRE_HOLDOUT_COMPLETE" not in capsys.readouterr().out
        assert not (root / "pre-holdout-complete.json").exists()
        assert not list((ledger.root / "exposures").iterdir())
