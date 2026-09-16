"""Atomic producer updates cannot mix validated records with newer file hashes."""

from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import (
    ManifestError,
    StudyType,
    _grid_integrity,
    adapters,
    inspect_study,
    inspect_validation,
    validation,
)
from quantforge.experiments.artifacts import ArtifactEntry
from quantforge.oos import HoldoutEvaluation, HoldoutLedger
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.oos.conftest import complete_study


def replace_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(".replacement")
    temporary.write_bytes(content)
    temporary.replace(path)


@pytest.mark.parametrize(
    "study_type", [StudyType.PARAMETER_STUDY, StudyType.OPTIMIZATION]
)
@pytest.mark.parametrize(
    "complete_trial", [False, True], ids=["identical_bytes", "completed"]
)
def test_trial_completion_between_validation_and_hashing_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_type: StudyType,
    complete_trial: bool,
) -> None:
    root = build_grid_export(tmp_path, study_type, monkeypatch)
    for name in ("summary.json", "ranking.json", "stability.json"):
        (root / name).unlink(missing_ok=True)
    path = trial_path(root)
    completed = path.read_bytes()
    pending = read_record(path)
    pending["status"] = "pending"
    fields = (
        ("analysis", "artifact_fingerprint")
        if study_type is StudyType.PARAMETER_STUDY
        else ("metrics", "qf5_run_id")
    )
    for field in ("artifact_location", *fields):
        pending[field] = None
    if study_type is StudyType.OPTIMIZATION:
        # Native pending trials carry the study dataset; only a completed
        # backtest enriches that mapping with QF-5 market-data fields.
        configuration = cast(
            PrimitiveMapping, read_record(root / "manifest.json")["identity_inputs"]
        )
        pending["dataset"] = configuration["dataset"]
    write_json(path, pending)
    replacement = completed if complete_trial else path.read_bytes()
    inspect_study(study_type, root, artifact_root=tmp_path)
    original = _grid_integrity.validate_trial_status
    updated = False

    def finish_trial(kind: StudyType, trial: PrimitiveMapping) -> None:
        nonlocal updated
        original(kind, trial)
        if trial["trial_id"] == pending["trial_id"] and not updated:
            replace_bytes(path, replacement)
            updated = True

    monkeypatch.setattr(_grid_integrity, "validate_trial_status", finish_trial)
    if complete_trial:
        with pytest.raises(ManifestError, match="changed during indexing"):
            inspect_study(study_type, root, artifact_root=tmp_path)
    else:
        inspect_study(study_type, root, artifact_root=tmp_path)
    assert updated
    assert path.read_bytes() == replacement


@pytest.mark.parametrize(
    "study_type", [StudyType.PARAMETER_STUDY, StudyType.OPTIMIZATION]
)
def test_summary_change_after_validation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, study_type: StudyType
) -> None:
    root = build_grid_export(tmp_path, study_type, monkeypatch)
    path = root / "summary.json"
    original = _grid_integrity.validate_trial_counts
    changed = path.read_bytes() + b"\n"

    def replace_summary(
        kind: StudyType,
        manifest: PrimitiveMapping,
        configuration: PrimitiveMapping,
        trials: list[PrimitiveMapping],
        summary: PrimitiveMapping | None,
    ) -> None:
        original(kind, manifest, configuration, trials, summary)
        replace_bytes(path, changed)

    monkeypatch.setattr(_grid_integrity, "validate_trial_counts", replace_summary)
    with pytest.raises(ManifestError, match="changed during indexing"):
        inspect_study(study_type, root, artifact_root=tmp_path)
    assert path.read_bytes() == changed


@pytest.mark.parametrize(
    "study_type", [StudyType.PARAMETER_STUDY, StudyType.OPTIMIZATION]
)
def test_result_replacement_between_validation_and_hashing_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, study_type: StudyType
) -> None:
    root = build_grid_export(tmp_path, study_type, monkeypatch)
    trial = read_record(trial_path(root))
    path = root / cast(str, trial["artifact_location"])
    if study_type is StudyType.OPTIMIZATION:
        path /= "manifest.json"
        module, name = _grid_integrity, "validate_backtest_trial"
    else:
        module, name = adapters, "validate_prediction_trial_result"
    original = getattr(module, name)
    changed = path.read_bytes() + b"\n"

    def replace_result(*arguments: Any) -> None:
        original(*arguments)
        replace_bytes(path, changed)

    monkeypatch.setattr(module, name, replace_result)
    with pytest.raises(ManifestError, match="changed during indexing"):
        inspect_study(study_type, root, artifact_root=tmp_path)
    assert path.read_bytes() == changed


@pytest.mark.parametrize("target", ["fold_state", "fold_result", "holdout_result"])
def test_enveloped_validation_record_replacement_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    completed = complete_study(tmp_path, prediction=True)
    source = completed.source
    ledger = None
    if target == "holdout_result":
        ledger = HoldoutLedger.create(tmp_path / "ledger")
        ledger.reserve(source)
        consumed = ledger.consume(
            HoldoutEvaluation.prepare(
                source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
            ),
            run_id="racing-holdout",
        )
        assert consumed.result_reference is not None
        path = ledger.root / cast(str, consumed.result_reference.to_primitive()["path"])
    else:
        path = completed.study.study_path / "folds" / source.folds[0].fold_id
        path /= "state.json" if target == "fold_state" else "oos.json"
    original = validation.index_artifact
    changed = path.read_bytes() + b"\n"
    updated = False

    def replace_record(root: Path, **arguments: Any) -> ArtifactEntry:
        nonlocal updated
        if root / arguments["path"] == path and not updated:
            replace_bytes(path, changed)
            updated = True
        return original(root, **arguments)

    block_research(monkeypatch)
    monkeypatch.setattr(validation, "index_artifact", replace_record)
    with pytest.raises(ManifestError, match="changed during indexing"):
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )
    assert updated
    assert path.read_bytes() == changed
    if ledger is not None:
        assert ledger.state(source).state.value == "consumed"
