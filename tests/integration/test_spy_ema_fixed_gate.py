"""The fixed-run gate distinguishes native QF-32 failure from unrankability."""

import json
from dataclasses import replace
from pathlib import Path
from typing import NoReturn, cast

import pytest

from quantforge.examples import spy_ema_runner
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.oos import HoldoutLedger
from quantforge.optimization import IntegerValues, ParameterSearchSpace, TrialStatus
from quantforge.prediction import PredictionRuleContext, PredictionStrategyOutput
from quantforge.prediction.grid import PredictionGridTrialRecord
from quantforge.prediction.window_encoding import mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.walk_forward import (
    PredictionEvaluator,
    WalkForwardConfig,
    WalkForwardError,
)
from tests.unit.experiments.test_contracts import execution
from tests.unit.prediction.test_incremental_prediction_grid import CompactWindowAnalyzer
from tests.unit.walk_forward.fixtures import StudyRule
from tests.unit.walk_forward.timestamp_fixtures import timestamp_fixture


class FixedInspectionReachedError(Exception):
    """Stop after the gate, before any inspection/comparison/OOS work."""


@pytest.mark.parametrize("scenario", ["failed_analysis", "empty", "unrankable"])
def test_fixed_gate_uses_persisted_trial_status_on_run_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    scenario: str,
) -> None:
    # Small native QF-39/QF-32 fixtures isolate this orchestration gate. The
    # separate EMA end-to-end test covers the real rule, calendar and reports.
    config, original = timestamp_fixture(tmp_path)
    adapter = PredictionEvaluator(
        dataset=original.dataset,
        series=original.series,
        primary_timeframe=original.primary_timeframe,
        study_factory=original.factory,
        analyzer=CompactWindowAnalyzer(),
        indicator_backend=original.backend,
        grid_config=replace(
            original.grid_config,
            search_space=ParameterSearchSpace({"window": IntegerValues([2])}),
            ranking=replace(
                original.grid_config.ranking, minimum_prediction_count=1000
            ),
            window_schema_version="2",
        ),
    )

    def prepared(
        inputs: SmokeInputs, root: Path, *, fixed: bool = False
    ) -> tuple[WalkForwardConfig, PredictionEvaluator]:
        return config, adapter

    def inspection(*args: object, **kwargs: object) -> NoReturn:
        raise FixedInspectionReachedError

    monkeypatch.setattr(spy_ema_runner, "prepare_walk_forward", prepared)
    monkeypatch.setattr(spy_ema_runner, "inspect_phase", inspection)
    analysis_calls = 0

    if scenario == "failed_analysis":

        def failing_analysis(
            self: CompactWindowAnalyzer, reader: PredictionWindowReader
        ) -> NoReturn:
            nonlocal analysis_calls
            analysis_calls += 1
            assert reader.decision_count > 0
            reader.verify_integrity()
            raise ValueError("injected analysis failure after finalized window")

        monkeypatch.setattr(
            CompactWindowAnalyzer, "analyze_compact_window", failing_analysis
        )
    elif scenario == "empty":

        def empty_rule(
            self: StudyRule, context: PredictionRuleContext
        ) -> PredictionStrategyOutput:
            return PredictionStrategyOutput(
                self.name, self.configuration_id, context.prediction_dataset_id, ()
            )

        monkeypatch.setattr(StudyRule, "generate_with_context", empty_rule)

    root = tmp_path / "run"
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    recorded: dict[Path, bytes] = {}
    for _ in range(2):
        expected = (
            WalkForwardError
            if scenario == "failed_analysis"
            else FixedInspectionReachedError
        )
        with pytest.raises(expected) as caught:
            spy_ema_runner.run_pre_holdout(
                cast(SmokeInputs, object()), root, ledger, execution()
            )
        if scenario == "failed_analysis":
            assert (
                str(caught.value)
                == "no eligible candidate under the declared selection policy"
            )
            assert analysis_calls == 1  # Resume preserves the failed trial.
        paths = list((root / "fixed").glob("*/trials/*.json"))
        assert len(paths) == 1
        trial = PredictionGridTrialRecord.from_primitive(
            mapping(json.loads(paths[0].read_text()))
        )
        if scenario == "failed_analysis":
            assert trial.status is TrialStatus.FAILED
            assert trial.failure_type == "ValueError"
            assert trial.failure_message
        else:
            assert trial.status is TrialStatus.SUCCEEDED
            assert trial.analysis is not None
            assert (trial.analysis.prediction_count == 0) is (scenario == "empty")
        windows = list((root / "fixed").rglob("prediction-window.jsonl"))
        assert len(windows) == 1  # A finalized window alone never proves success.
        current = {path: path.read_bytes() for path in [*paths, *windows]}
        if recorded:
            assert current == recorded
        recorded = current
        output = capsys.readouterr().out
        assert "STAGE comparison" not in output
        assert "PRE_HOLDOUT_COMPLETE" not in output
        assert not (root / "pre-holdout-complete.json").exists()
        assert not list((ledger.root / "exposures").iterdir())
