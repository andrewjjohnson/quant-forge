"""An interrupted export must not turn a resumable grid into a corrupt study."""

from pathlib import Path

import pytest

from quantforge.backtesting import BacktestConfig, BacktestResult, run_backtest
from quantforge.configuration import PrimitiveMapping
from quantforge.data import MarketDataset
from quantforge.experiments import (
    ArtifactType,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.optimization import (
    ExecutionConfig,
    GridSearchStudy,
    MovingAverageCrossoverFactory,
)
from quantforge.optimization import export as export_module
from quantforge.strategies import Strategy
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    read_record,
    selective_runner,
    trial_path,
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


@pytest.mark.parametrize("malformed", [False, True])
def test_interrupted_optimization_retry_omits_all_stale_exports_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed: bool
) -> None:
    retrying = False
    interrupt = False

    def runner(
        dataset: MarketDataset, strategy: Strategy, config: BacktestConfig
    ) -> BacktestResult:
        if interrupt:
            raise KeyboardInterrupt("retry interrupted after running record")
        return (run_backtest if retrying else selective_runner)(
            dataset, strategy, config
        )

    study = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path, execution=ExecutionConfig(retry_failed=True)),
        backtest_runner=runner,
    )
    study.run()
    root = study.study_path
    path = trial_path(root, "failed")
    summary = (root / "summary.json").read_bytes()
    retrying = interrupt = True
    with pytest.raises(KeyboardInterrupt, match="retry interrupted"):
        study.resume()
    assert read_record(path)["status"] == "running"
    assert (root / "summary.json").read_bytes() == summary
    if malformed:
        for item in root.iterdir():
            if item.suffix == ".csv" or item.name in {
                "summary.json",
                "ranking.json",
                "stability.json",
            }:
                item.write_bytes(b"stale incomplete export")
    before = {item: item.read_bytes() for item in root.rglob("*") if item.is_file()}
    with monkeypatch.context() as inspection:
        block_research(inspection)
        partial = inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
        assert "trial_counts" not in partial.provenance.observations.to_primitive()
        assert not any(
            entry.artifact_type is ArtifactType.PARAMETER_SUMMARY
            for entry in partial.index.entries
        )
        assert verify_artifacts(partial.index, tmp_path).valid
    assert {item: item.read_bytes() for item in before} == before
    interrupt = False
    study.resume()
    assert read_record(path)["status"] == "succeeded"
    block_research(monkeypatch)
    complete = inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    assert "trial_counts" in complete.provenance.observations.to_primitive()
    assert {"ranking.json", "stability.json", "summary.json"}.issubset(
        {Path(entry.path).name for entry in complete.index.entries}
    )
    assert verify_artifacts(complete.index, tmp_path).valid
