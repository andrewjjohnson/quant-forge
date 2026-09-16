"""Files created after producer validation never acquire backtest provenance."""

from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import export as backtest_export
from quantforge.experiments import (
    StudyArtifacts,
    StudyType,
    inspect_study,
    inspect_validation,
    verify_artifacts,
)
from quantforge.experiments._json import mapping
from quantforge.oos import HoldoutEvaluation, HoldoutLedger
from quantforge.walk_forward.models import BacktestOOSArtifact
from tests.unit.backtesting.test_runner import configured_result
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.oos.conftest import complete_study


@pytest.mark.parametrize("family", ["standalone", "optimization", "fold", "holdout"])
@pytest.mark.parametrize("filename", ["extra.json", "extra.csv"])
def test_unowned_file_created_after_validation_is_not_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, filename: str
) -> None:
    completed = None
    ledger = None
    if family == "standalone":
        study_path = export = backtest_export.export_backtest_result(
            configured_result(), tmp_path / "backtests"
        )
    elif family == "optimization":
        study_path = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
        trial = read_record(trial_path(study_path))
        export = study_path / cast(str, trial["artifact_location"])
    else:
        completed = complete_study(tmp_path, prediction=False)
        study_path = completed.study.study_path
        if family == "holdout":
            ledger = HoldoutLedger.create(tmp_path / "ledger")
            ledger.reserve(completed.source)
            consumed = ledger.consume(
                HoldoutEvaluation.prepare(
                    completed.source,
                    completed.evaluator,
                    selection_fold_id=completed.source.folds[-1].fold_id,
                ),
                run_id="backtest-file-addition",
            )
            assert consumed.result_reference is not None
            result_path = ledger.root / cast(
                str, consumed.result_reference.to_primitive()["path"]
            )
            artifact = mapping(mapping(read_record(result_path)["payload"])["artifact"])
            export = (
                result_path.parent
                / "evaluation"
                / cast(str, artifact["export_location"])
            )
        else:
            fold = completed.source.folds[0]
            assert isinstance(fold.artifact, BacktestOOSArtifact)
            export = (
                study_path
                / "folds"
                / fold.fold_id
                / "test"
                / fold.artifact.export_location
            )
    block_research(monkeypatch)

    def inspect() -> StudyArtifacts:
        if completed is None:
            return inspect_study(
                StudyType.BACKTEST
                if family == "standalone"
                else StudyType.OPTIMIZATION,
                study_path,
                artifact_root=tmp_path,
            )
        return inspect_validation(
            completed.source, study_path, artifact_root=tmp_path, ledger=ledger
        )

    original = inspect()
    before = {path: path.read_bytes() for path in export.iterdir()}
    extra = export / filename
    original_validate = backtest_export.validate_backtest_result_artifact

    def validate_then_add(path: Path) -> Path:
        result = original_validate(path)
        if path == export:
            extra.write_text("{}\n" if extra.suffix == ".json" else "untrusted\n1\n")
        return result

    monkeypatch.setattr(
        backtest_export, "validate_backtest_result_artifact", validate_then_add
    )
    inspected = inspect()
    assert extra.is_file()
    assert inspected.index == original.index
    assert all(
        entry.path != extra.relative_to(tmp_path).as_posix()
        for entry in inspected.index.entries
    )
    assert verify_artifacts(inspected.index, tmp_path).valid
    assert {path: path.read_bytes() for path in before} == before
    if ledger is not None:
        assert completed is not None
        assert ledger.state(completed.source).state.value == "consumed"
