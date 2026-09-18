"""Standalone and non-window elapsed artifacts remain readable without research."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data import MarketDataset
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._json import mapping
from quantforge.prediction import (
    PredictionContextEnvironment,
    PredictionContextRequirements,
    PredictionGridStudy,
    PredictionStudy,
    run_prediction_study,
)
from quantforge.validation import PartitionRole
from quantforge.walk_forward.partitions import partition
from quantforge.walk_forward.prediction import (
    _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import FixtureTrialAnalyzer
from tests.unit.experiments.test_prediction_session_integrity import rehash_row
from tests.unit.prediction.test_multi_timeframe_study import FixtureContextProvider
from tests.unit.prediction.test_timestamp_replay import population
from tests.unit.walk_forward.timestamp_fixtures import instant, timestamp_fixture


def contextual_study(
    tmp_path: Path,
) -> tuple[MarketDataset, PredictionStudy[Any, Any, Any], FixtureContextProvider]:
    config, adapter = timestamp_fixture(tmp_path)
    study = adapter.factory.build({"window": 2})
    requirements = cast(
        PredictionContextRequirements, getattr(study.strategy, "context_requirements")
    )
    permitted = partition(
        adapter.dataset,
        config.plan,
        0,
        PartitionRole.DEVELOPMENT,
        minimum_observations=1,
    )
    source = config.plan.prediction_membership
    assert source is not None
    provider = _PermittedContextProvider(
        config.plan, permitted, adapter.series, source.schedule
    )
    context = provider.get_context_at(requirements, as_of=instant(4, "10:00"))
    return permitted.dataset, study, FixtureContextProvider(context)


@pytest.mark.parametrize("contextual", [False, True])
def test_direct_elapsed_artifact_is_readable_without_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contextual: bool
) -> None:
    if contextual:
        dataset, study, provider = contextual_study(tmp_path)
    else:
        dataset, _, _, study = population(tmp_path)
        provider = None
    result = run_prediction_study(dataset, study, context_provider=provider)
    path = tmp_path / "elapsed.json"
    write_json(path, result.to_primitive())
    block_research(monkeypatch)
    bundle = inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    assert bundle.provenance.producer_study_id == result.study_id
    if not contextual:
        assert len(result.rows) == 2
        assert (
            result.rows[0].signal.signal_session == result.rows[1].signal.signal_session
        )


@pytest.mark.parametrize(
    "change", ["anchor", "source", "target", "availability", "order", "omit"]
)
def test_direct_elapsed_reader_rejects_rehashed_temporal_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    dataset, _, _, study = population(tmp_path)
    snapshot = run_prediction_study(dataset, study).to_primitive()
    rows = cast(list[PrimitiveMapping], snapshot["rows"])
    if change == "order":
        rows.reverse()
    elif change == "omit":
        rows.pop()
        counts = mapping(mapping(snapshot["manifest"])["record_counts"])
        counts["labeled_rows"] = 1
        counts["unavailable_outcomes"] = 1
    else:
        row = rows[0]
        resolution = mapping(mapping(row["outcome"])["temporal_resolution"])
        request = mapping(resolution["request"])
        if change == "anchor":
            mapping(request["anchor"])["decision_timestamp"] = instant(
                4, "10:05"
            ).isoformat()
        elif change == "source":
            mapping(request["source_reference"])["dataset_id"] = "foreign"
        elif change == "target":
            resolution["requested_target_timestamp"] = instant(4, "11:00").isoformat()
        else:
            resolution["available"] = False
        rehash_row(row)
    path = tmp_path / "elapsed.json"
    write_json(path, snapshot)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=r"elapsed|ordered unique"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


def test_non_window_elapsed_grid_artifacts_are_readable_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, adapter = timestamp_fixture(tmp_path)
    dataset, _, provider = contextual_study(tmp_path)
    source = adapter.series[0]
    grid = PredictionGridStudy(
        dataset=dataset,
        dataset_family_fingerprint=source.dataset_reference.family_id,
        study_factory=adapter.factory,
        analyzer=FixtureTrialAnalyzer(),
        context_provider=provider,
        context_environment=PredictionContextEnvironment.create(
            "elapsed_fixture_context",
            "1",
            {"family": source.dataset_reference.family_id},
        ),
        indicator_backend=adapter.backend,
        config=replace(adapter.grid_config, output_root=tmp_path / "grid"),
    )
    result = grid.run()
    assert any(str(trial.status) == "succeeded" for trial in result.trials)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("completed elapsed trial must not rerun prediction")

    monkeypatch.setattr(
        "quantforge.prediction.grid.run_prediction_study_in_session", forbidden
    )
    resumed = grid.resume()
    assert resumed.trials == result.trials
    block_research(monkeypatch)
    bundle = inspect_study(
        StudyType.PARAMETER_STUDY,
        tmp_path / "grid" / result.study_id,
        artifact_root=tmp_path,
    )
    assert bundle.provenance.producer_study_id == result.study_id
