"""An interrupted export must not turn a resumable grid into a corrupt study."""

from pathlib import Path

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import StudyType, inspect_study
from quantforge.optimization import GridSearchStudy, MovingAverageCrossoverFactory
from quantforge.optimization import export as export_module
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    selective_runner,
)
from tests.unit.optimization.test_study import (
    _dataset,  # pyright: ignore[reportPrivateUsage]
    _study_config,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize(
    "present",
    [(), ("ranking.json",), ("stability.json",), ("ranking.json", "stability.json")],
)
@pytest.mark.parametrize("malformed", [False, True])
def test_unfinished_derived_exports_are_omitted_until_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    present: tuple[str, ...],
    malformed: bool,
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    (root / "summary.json").unlink()
    for name in ("ranking.json", "stability.json"):
        if name not in present:
            (root / name).unlink()
        elif malformed:
            (root / name).write_text("unfinished")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    bundle = inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    indexed = {Path(entry.path).name for entry in bundle.index.entries}
    assert not {"ranking.json", "stability.json", "summary.json"} & indexed
    assert bundle.provenance.observations.to_primitive().get("trial_counts") is None
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "interrupted_file", ["ranking.json", "stability.json", "summary.json"]
)
def test_native_export_interruption_remains_inspectable_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_file: str,
) -> None:
    study = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path),
        backtest_runner=selective_runner,
    )
    write = export_module._atomic_json  # pyright: ignore[reportPrivateUsage]

    def interrupt(path: Path, record: PrimitiveMapping) -> None:
        if path.name == interrupted_file:
            raise KeyboardInterrupt("fixture export interruption")
        write(path, record)

    with monkeypatch.context() as interruption:
        interruption.setattr(export_module, "_atomic_json", interrupt)
        with pytest.raises(KeyboardInterrupt):
            study.run()
    with monkeypatch.context() as inspection:
        block_research(inspection)
        partial = inspect_study(
            StudyType.OPTIMIZATION, study.study_path, artifact_root=tmp_path
        )
        assert not any(
            Path(entry.path).name in {"ranking.json", "stability.json", "summary.json"}
            for entry in partial.index.entries
        )
    original_trials = {
        path: path.read_bytes() for path in (study.study_path / "trials").glob("*.json")
    }
    study.resume()
    assert {path: path.read_bytes() for path in original_trials} == original_trials
    block_research(monkeypatch)
    complete = inspect_study(
        StudyType.OPTIMIZATION, study.study_path, artifact_root=tmp_path
    )
    assert {"ranking.json", "stability.json", "summary.json"}.issubset(
        {Path(entry.path).name for entry in complete.index.entries}
    )
