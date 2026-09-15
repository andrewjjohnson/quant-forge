"""Nested QF-5 indexes must retain the exact sidecar-validated file hashes."""

from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.backtesting import export as backtest_export
from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import (
    ManifestError,
    StudyType,
    _backtest_artifacts,
    inspect_study,
    inspect_validation,
    verify_artifacts,
)
from quantforge.experiments.artifacts import ArtifactEntry
from quantforge.oos import HoldoutEvaluation, HoldoutLedger
from quantforge.walk_forward.models import BacktestOOSArtifact
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.experiments.test_inspection_races import replace_bytes
from tests.unit.oos.conftest import complete_study


@pytest.mark.parametrize("family", ["optimization", "fold", "holdout"])
@pytest.mark.parametrize("filename", ["fills.csv", "equity.csv", "integrity.json"])
@pytest.mark.parametrize("phase", ["after_validation", "before_hash", "after_hash"])
@pytest.mark.parametrize("changed", [False, True], ids=["identical", "changed"])
def test_nested_backtest_replacements_are_bound_to_validated_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    filename: str,
    phase: str,
    changed: bool,
) -> None:
    ledger = None
    if family == "optimization":
        study_path = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
        trial = read_record(trial_path(study_path))
        export = study_path / cast(str, trial["artifact_location"])
        completed = None
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
                run_id="racing-backtest-files",
            )
            assert consumed.result_reference is not None
            result_path = ledger.root / cast(
                str, consumed.result_reference.to_primitive()["path"]
            )
            result = read_record(result_path)
            artifact = cast(
                PrimitiveMapping, cast(PrimitiveMapping, result["payload"])["artifact"]
            )
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
    target = export / filename
    replacement = target.read_bytes() + (b"\n" if changed else b"")
    updated = False

    def update() -> None:
        nonlocal updated
        replace_bytes(target, replacement)
        updated = True

    original_validate = backtest_export.validate_backtest_result_artifact

    def validate_then_replace(path: Path) -> Path:
        result = original_validate(path)
        if path == export and not updated:
            update()
        return result

    original_index = _backtest_artifacts.index_artifact

    def replace_then_index(root: Path, **arguments: Any) -> ArtifactEntry:
        matches = root / arguments["path"] == target and not updated
        if matches and phase == "before_hash":
            update()
        entry = original_index(root, **arguments)
        if matches and phase == "after_hash":
            update()
        return entry

    if phase == "after_validation":
        monkeypatch.setattr(
            backtest_export, "validate_backtest_result_artifact", validate_then_replace
        )
    else:
        monkeypatch.setattr(_backtest_artifacts, "index_artifact", replace_then_index)

    def inspect() -> None:
        if completed is None:
            result = inspect_study(
                StudyType.OPTIMIZATION, study_path, artifact_root=tmp_path
            )
        else:
            result = inspect_validation(
                completed.source, study_path, artifact_root=tmp_path, ledger=ledger
            )
        assert verify_artifacts(result.index, tmp_path).valid

    if changed:
        with pytest.raises(ManifestError):
            inspect()
    else:
        inspect()
    assert updated
    assert target.read_bytes() == replacement
    if ledger is not None:
        assert completed is not None
        assert ledger.state(completed.source).state.value == "consumed"
