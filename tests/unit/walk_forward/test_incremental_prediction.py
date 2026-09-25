"""QF-39 compact production without downstream QF-57 consumer migration."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.prediction.window_encoding import mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.walk_forward import FoldStatus, PredictionEvaluator, WalkForwardStudy
from tests.unit.prediction.test_incremental_prediction_grid import CompactWindowAnalyzer
from tests.unit.walk_forward.timestamp_fixtures import timestamp_fixture


def compact_adapter(adapter: PredictionEvaluator) -> PredictionEvaluator:
    return PredictionEvaluator(
        dataset=adapter.dataset,
        series=adapter.series,
        primary_timeframe=adapter.primary_timeframe,
        study_factory=adapter.factory,
        analyzer=CompactWindowAnalyzer(),
        indicator_backend=adapter.backend,
        grid_config=replace(adapter.grid_config, window_schema_version="2"),
    )


def test_timestamp_selection_test_membership_and_frozen_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, original = timestamp_fixture(tmp_path)
    legacy = WalkForwardStudy(config, original, tmp_path / "legacy").run()
    adapter = compact_adapter(original)
    study = WalkForwardStudy(config, adapter, tmp_path / "compact")
    result = study.run()
    assert all(f.status is FoldStatus.COMPLETED for f in result.folds)
    for fold, old in zip(result.folds, legacy.folds, strict=True):
        assert fold.selection is not None
        assert old.selection is not None
        frozen, previous = fold.selection.to_primitive(), old.selection.to_primitive()
        for key in ("candidate", "membership"):
            assert frozen[key] == previous[key]
        assert fold.artifact is not None
        assert old.artifact is not None
        reference = fold.artifact.snapshot.to_primitive()
        assert reference["schema_version"] == "2"
        paths = [
            p
            for p in study.study_path.rglob("prediction-window.jsonl")
            if p.parent.name == "test"
            and PredictionWindowReader.open(p).header() == reference["header"]
        ]
        assert len(paths) == 1
        reader = PredictionWindowReader.open(paths[0])
        reader.verify_integrity()
        decisions = list(reader.iterate_decisions())
        retained = mapping(mapping(frozen["membership"])["test"])[
            "evaluation_timestamps"
        ]
        assert [d.to_primitive()["decision_timestamp"] for d in decisions] == retained
        old_decisions = cast(
            list[PrimitiveMapping], old.artifact.snapshot.to_primitive()["decisions"]
        )
        for new, embedded in zip(decisions, old_decisions, strict=True):
            assert (
                mapping(new.to_primitive()["prediction_study"])["rows"]
                == mapping(embedded["prediction_study"])["rows"]
            )
    selection_paths = [
        p
        for p in study.study_path.rglob("prediction-window.jsonl")
        if "selection" in p.parts
    ]
    assert selection_paths
    for path in selection_paths:
        PredictionWindowReader.open(path).verify_integrity()

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("completed compact windows must be reused")

    monkeypatch.setattr(PredictionEvaluator, "select", forbidden)
    monkeypatch.setattr(PredictionEvaluator, "evaluate", forbidden)
    assert study.resume() == result


def test_interrupted_test_window_resumes_after_frozen_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import Any

    from quantforge.walk_forward.prediction import (
        _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
    )

    config, original_adapter = timestamp_fixture(tmp_path)
    adapter = compact_adapter(original_adapter)
    permitted = adapter._partition(config, 0, test=True)  # pyright: ignore[reportPrivateUsage]
    expected = adapter._schedule(permitted).decision_timestamps  # pyright: ignore[reportPrivateUsage]
    original = _PermittedContextProvider.get_context_at
    calls: list[object] = []
    interrupted = False

    def probe(provider: _PermittedContextProvider, *args: Any, **kwargs: Any) -> Any:
        nonlocal interrupted
        if provider.schedule.decision_timestamps == expected:
            if len(calls) == 2 and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            calls.append(kwargs["as_of"])
        return original(provider, *args, **kwargs)

    monkeypatch.setattr(_PermittedContextProvider, "get_context_at", probe)
    study = WalkForwardStudy(config, adapter, tmp_path / "studies")
    with pytest.raises(KeyboardInterrupt):
        study.run()
    frozen = {p: p.read_bytes() for p in study.study_path.rglob("selection.json")}
    assert frozen
    assert calls == list(expected[:2])
    result = study.resume()
    assert all(f.status is FoldStatus.COMPLETED for f in result.folds)
    assert calls == list(expected)
    assert all(path.read_bytes() == contents for path, contents in frozen.items())
