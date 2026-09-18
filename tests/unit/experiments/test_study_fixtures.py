"""Shared producer baselines must remain pristine after a test corrupts its copy."""

import shutil
from pathlib import Path

import pytest

from quantforge.oos import load_oos_source
from tests.unit.experiments.study_fixtures import CapturedStudy, copy_study
from tests.unit.experiments.study_fixtures import study_baselines as study_baselines
from tests.unit.experiments.test_adapters import block_research


def artifact_bytes(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("prediction", [False, True], ids=["backtest", "prediction"])
def test_corrupted_copy_cannot_change_baseline_or_another_copy(
    study_baselines: dict[bool, CapturedStudy],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prediction: bool,
) -> None:
    baseline = study_baselines[prediction]
    original = artifact_bytes(baseline.study_path)
    block_research(monkeypatch)
    first = copy_study(baseline, tmp_path / "first")
    second = copy_study(baseline, tmp_path / "second")
    assert artifact_bytes(first.study_path) == original
    assert artifact_bytes(second.study_path) == original
    assert first.source == second.source == baseline.source
    assert first.source is not second.source

    (first.study_path / "manifest.json").write_text("corrupted")
    shutil.rmtree(first.study_path / "folds" / first.source.folds[0].fold_id)
    detached = first.source.definition.to_primitive()
    detached.clear()

    assert artifact_bytes(baseline.study_path) == original
    assert artifact_bytes(second.study_path) == original
    assert load_oos_source(second.source.plan, second.study_path) == baseline.source
    third = copy_study(baseline, tmp_path / "third")
    assert artifact_bytes(third.study_path) == original
