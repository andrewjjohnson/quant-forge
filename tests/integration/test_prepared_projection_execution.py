"""Count actual QF-52 work and compare unchanged QF-32/QF-39 artifacts."""

import json
import socket
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import quantforge.data.prediction_views as views
import quantforge.data.prepared_prediction_views as preparation
import quantforge.prediction.grid as grid_module
import quantforge.walk_forward.partitions as partitions
from quantforge.data import MarketDataset
from quantforge.data.prepared_prediction_views import PreparedProjectionRegistry
from quantforge.experiments import (
    create_manifest,
    inspect_validation,
    verify_artifacts,
    write_manifest,
)
from quantforge.oos import (
    HoldoutLedger,
    aggregate_prediction,
    export_oos_aggregate,
    load_oos_source,
)
from quantforge.prediction.window_incremental import IncrementalPredictionWindowWriter
from quantforge.reporting import build_research_report
from quantforge.walk_forward import FoldStatus, WalkForwardStudy
from tests.integration.test_bounded_prediction_workflow import workflow_fixture
from tests.integration.test_intraday_prediction_provenance import (
    Fixture,
    cached_fixture,
)
from tests.unit.experiments.test_contracts import execution
from tests.unit.helpers import SESSIONS
from tests.unit.walk_forward.test_incremental_prediction import compact_adapter


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return cached_fixture(
        tmp_path_factory.mktemp("prepared-trials"), session_dates=SESSIONS[:7]
    )


def windows(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("prediction-window.jsonl")
    }


def test_two_folds_two_trials_counts_identities_rankings_and_completed_resume(
    fixture: Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, original_adapter = workflow_fixture(tmp_path, fixture)
    original_project = preparation.bounded_prediction_view
    original_verify = PreparedProjectionRegistry.verify_lineage
    original_retain = PreparedProjectionRegistry._retain  # pyright: ignore[reportPrivateUsage]
    original_trial = grid_module.run_incremental_prediction_window_in_session
    counts: Counter[str] = Counter()
    reference = True

    def project(*args: Any, **kwargs: Any) -> MarketDataset:
        counts["projection_builds"] += 1
        return original_project(*args, **kwargs)

    def registry_project(
        self: PreparedProjectionRegistry,
        dataset: MarketDataset,
        cutoff: Any,
        *,
        scope: Any,
        start: Any = None,
    ) -> MarketDataset:
        return project(dataset, cutoff, start=start)

    def verify(self: PreparedProjectionRegistry, *args: Any, **kwargs: Any) -> None:
        counts["lineage_verifications"] += 1
        if reference:
            counts["lineage_reconstructions"] += 1
            kwargs.pop("scope")
            views.validate_prediction_view_lineage(*args, **kwargs)
        else:
            original_verify(self, *args, **kwargs)

    def retain(
        self: PreparedProjectionRegistry, *args: Any, **kwargs: Any
    ) -> MarketDataset:
        counts["prepared_views"] += 1
        return original_retain(self, *args, **kwargs)

    def trial(*args: Any, **kwargs: Any) -> Any:
        counts["trial_executions"] += 1
        return original_trial(*args, **kwargs)

    monkeypatch.setattr(preparation, "bounded_prediction_view", project)
    monkeypatch.setattr(views, "bounded_prediction_view", project)
    monkeypatch.setattr(partitions, "bounded_prediction_view", project)
    monkeypatch.setattr(PreparedProjectionRegistry, "verify_lineage", verify)
    monkeypatch.setattr(PreparedProjectionRegistry, "_retain", retain)
    monkeypatch.setattr(
        grid_module, "run_incremental_prediction_window_in_session", trial
    )
    with monkeypatch.context() as baseline:
        baseline.setattr(PreparedProjectionRegistry, "project", registry_project)
        adapter = compact_adapter(original_adapter)
        expected = [
            adapter.select(config, fold, tmp_path / "before" / str(fold))
            for fold in range(2)
        ]
    before = dict(counts)
    assert before == {
        "projection_builds": 10,
        "lineage_verifications": 8,
        "lineage_reconstructions": 8,
        "trial_executions": 4,
    }

    reference = False
    counts.clear()
    adapter = compact_adapter(original_adapter)
    with adapter.preparation_scope():
        actual = [
            adapter.select(config, fold, tmp_path / "after" / str(fold))
            for fold in range(2)
        ]
        assert actual == expected
        assert windows(tmp_path / "after") == windows(tmp_path / "before")
        after = dict(counts)
        assert after == {
            "projection_builds": 2,
            "prepared_views": 2,
            "trial_executions": 4,
            "lineage_verifications": 8,
        }
        counts.clear()
        assert adapter.select(config, 0, tmp_path / "after" / "0") == expected[0]
        assert counts == {"lineage_verifications": 4}
    assert adapter._projection_registry is None  # pyright: ignore[reportPrivateUsage]
    counts.clear()
    # A fresh invocation reconstructs once; both completed trials still verify.
    assert adapter.select(config, 0, tmp_path / "after" / "0") == expected[0]
    assert counts == {
        "projection_builds": 1,
        "prepared_views": 1,
        "lineage_verifications": 4,
    }
    assert windows(tmp_path / "after") == windows(tmp_path / "before")
    print({"before": before, "after": after, "fresh_completed_resume": dict(counts)})


def test_interrupted_prefix_and_preparation_lifecycle(
    fixture: Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, original = workflow_fixture(tmp_path, fixture)
    expected_adapter = compact_adapter(original)
    expected = expected_adapter.select(config, 0, tmp_path / "expected")
    adapter = compact_adapter(original)
    original_append = IncrementalPredictionWindowWriter.append
    sequences: list[int] = []

    def interrupt(writer: IncrementalPredictionWindowWriter, decision: Any) -> None:
        original_append(writer, decision)
        sequences.append(writer.completed_count - 1)
        if len(sequences) == 2:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(IncrementalPredictionWindowWriter, "append", interrupt)
        with pytest.raises(KeyboardInterrupt):
            adapter.select(config, 0, tmp_path / "resumed")
    assert adapter._projection_registry is None  # pyright: ignore[reportPrivateUsage]
    journal = next((tmp_path / "resumed").rglob("decisions.jsonl"))
    prefix = journal.read_bytes()
    assert len(prefix.splitlines()) == 2

    # Record the exact scheduled timestamps actually executed on resume.
    import quantforge.walk_forward.prediction as prediction

    original_context = prediction._PermittedContextProvider.get_context_at  # pyright: ignore[reportPrivateUsage]
    timestamps: list[datetime] = []

    def context(provider: Any, *args: Any, **kwargs: Any) -> Any:
        timestamps.append(kwargs["as_of"])
        return original_context(provider, *args, **kwargs)

    monkeypatch.setattr(prediction._PermittedContextProvider, "get_context_at", context)  # pyright: ignore[reportPrivateUsage]
    assert adapter.select(config, 0, tmp_path / "resumed") == expected
    finalized = journal.parent.parent / "prediction-window.jsonl"
    assert b"".join(finalized.read_bytes().splitlines(keepends=True)[2:4]) == prefix
    assert windows(tmp_path / "resumed") == windows(tmp_path / "expected")
    # Two trials of three decisions; only the incomplete suffix is executed.
    assert len(timestamps) == 4
    first_schedule = [
        json.loads(line)["decision_timestamp"] for line in prefix.splitlines()
    ]
    assert timestamps[0].isoformat() not in first_schedule
    timestamps.clear()
    assert adapter.select(config, 0, tmp_path / "resumed") == expected
    assert timestamps == []


def test_walk_forward_and_offline_consumers_need_no_registry(
    fixture: Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, original = workflow_fixture(tmp_path, fixture)
    adapter = compact_adapter(original)
    study = WalkForwardStudy(config, adapter, tmp_path / "studies")
    result = study.run()
    assert all(fold.status is FoldStatus.COMPLETED for fold in result.folds)
    assert adapter._projection_registry is None  # pyright: ignore[reportPrivateUsage]
    assert study.resume() == result
    source = load_oos_source(config.plan, study.study_path)
    ledger = HoldoutLedger.create(tmp_path / "holdout")
    reserved = ledger.reserve(source)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("offline replay must not use preparation, research, or network")

    monkeypatch.setattr(PreparedProjectionRegistry, "__init__", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(
        grid_module, "run_incremental_prediction_window_in_session", forbidden
    )
    source = load_oos_source(config.plan, study.study_path)
    aggregate_path = export_oos_aggregate(
        aggregate_prediction(source), tmp_path / "aggregate"
    )
    artifacts = inspect_validation(
        source,
        study.study_path,
        artifact_root=tmp_path,
        ledger=ledger,
        aggregate_path=aggregate_path,
    )
    verify_artifacts(artifacts.index, tmp_path).require_valid()
    manifest = write_manifest(
        create_manifest(artifacts, execution()),
        tmp_path / "manifests",
        artifact_root=tmp_path,
    )
    report = build_research_report(
        manifest, artifact_root=tmp_path, holdout_source=source, holdout_ledger=ledger
    )
    assert report.header.to_primitive()["holdout_state"] == "reserved_unconsumed"
    assert ledger.state(source) == reserved


def test_standalone_grid_rebuilds_once_per_run_resume_or_load(
    fixture: Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from quantforge.walk_forward.prediction import (
        _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
    )

    config, original = workflow_fixture(tmp_path, fixture)
    adapter = compact_adapter(original)
    permitted = adapter._partition(config, 0, test=False)  # pyright: ignore[reportPrivateUsage]
    schedule = adapter._schedule(permitted)  # pyright: ignore[reportPrivateUsage]
    grid = adapter._grid(  # pyright: ignore[reportPrivateUsage]
        permitted.dataset,
        schedule,
        _PermittedContextProvider(config.plan, permitted, adapter.series, schedule),
        adapter._environment(config.plan, permitted),  # pyright: ignore[reportPrivateUsage]
        replace(adapter.grid_config, output_root=tmp_path / "grid"),
    )
    counts: Counter[str] = Counter()
    original_project = preparation.bounded_prediction_view
    original_verify = PreparedProjectionRegistry.verify_lineage

    def project(*args: Any, **kwargs: Any) -> MarketDataset:
        counts["projection_builds"] += 1
        return original_project(*args, **kwargs)

    def verify(self: PreparedProjectionRegistry, *args: Any, **kwargs: Any) -> None:
        counts["lineage_verifications"] += 1
        original_verify(self, *args, **kwargs)

    monkeypatch.setattr(preparation, "bounded_prediction_view", project)
    monkeypatch.setattr(PreparedProjectionRegistry, "verify_lineage", verify)
    expected = grid.run()
    assert counts == {"projection_builds": 1, "lineage_verifications": 4}
    counts.clear()
    resumed = grid.resume()
    assert resumed.rankings == expected.rankings
    assert resumed.trials == expected.trials
    assert counts == {"projection_builds": 1, "lineage_verifications": 4}
    counts.clear()
    loaded = grid.load_result()
    assert loaded.rankings == expected.rankings
    assert loaded.trials == expected.trials
    assert counts == {"projection_builds": 1, "lineage_verifications": 2}
    assert grid._projection_registry is None  # pyright: ignore[reportPrivateUsage]
