from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import FeedScope
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._json import mapping
from quantforge.prediction import PredictionContextFailurePolicy, PredictionStudy
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    FixtureTrialAnalyzer,
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.experiments.test_standalone_provenance import rehash_prediction
from tests.unit.prediction import test_multi_timeframe_study as fixtures
from tests.unit.prediction.test_prediction_grid import (
    FixtureStudyFactory,
    _grid,  # pyright: ignore[reportPrivateUsage]
)


class SkipContextFactory(FixtureStudyFactory):
    def __init__(self, feed_scope: FeedScope) -> None:
        self.feed_scope = feed_scope

    def configuration(self) -> PrimitiveMapping:
        return {**super().configuration(), "feed_scope": self.feed_scope.to_primitive()}

    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        study = super().build(parameters)
        requirements = fixtures._requirements(  # pyright: ignore[reportPrivateUsage]
            window=cast(int, parameters["window"]),
            failure_policy=PredictionContextFailurePolicy.SKIP,
            feed_scope=self.feed_scope,
        )
        return replace(study, strategy=fixtures.FixtureMultiTimeframeRule(requirements))


@pytest.mark.parametrize(
    "change", ["skipped_predictions", "future_bar", "selected_bars", "source_id"]
)
def test_nested_qf32_context_cannot_bypass_standalone_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    record_path = trial_path(root)
    trial = read_record(record_path)
    artifact_path = root / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    prediction = mapping(artifact["prediction_study"])
    manifest = mapping(prediction["manifest"])
    context = mapping(manifest["prediction_context"])
    source = mapping(context["source_context"])
    if change == "skipped_predictions":
        context["status"] = "skipped"
        context["reason"] = "rejected context"
    elif change == "future_bar":
        aligned = cast(list[PrimitiveMapping], source["timeframes"])
        boundary = datetime.fromisoformat(
            cast(str, aligned[0]["latest_completed_bar_timestamp"])
        )
        future = boundary + timedelta(minutes=1)
        aligned[1]["latest_completed_bar_timestamp"] = future.isoformat()
        aligned[1]["age_microseconds"] = (
            datetime.fromisoformat(cast(str, source["as_of"])) - future
        ) // timedelta(microseconds=1)
    elif change == "selected_bars":
        cast(list[PrimitiveMapping], context["timeframes"])[0]["visible_bar_ids"] = [
            "foreign-bar"
        ]
    if change == "source_id":
        source["context_id"] = "0" * 64
    else:
        source["context_id"] = configuration_identity(
            {key: value for key, value in source.items() if key != "context_id"}
        )
    rehash_prediction(prediction)
    artifact["prediction_study_id"] = manifest["study_id"]
    artifact["artifact_fingerprint"] = trial["artifact_fingerprint"] = (
        configuration_identity(
            {
                key: value
                for key, value in artifact.items()
                if key != "artifact_fingerprint"
            }
        )
    )
    write_json(artifact_path, artifact)
    write_json(record_path, trial)
    before = {path: path.read_bytes() for path in (artifact_path, record_path)}
    with pytest.raises(ManifestError, match="context"):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("incompatible_feed", [False, True])
def test_nested_context_accepts_available_and_skipped_producer_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, incompatible_feed: bool
) -> None:
    grid = _grid(
        tmp_path,
        factory=SkipContextFactory(
            FeedScope.iex_only() if incompatible_feed else FeedScope.consolidated()
        ),
        analyzer=FixtureTrialAnalyzer(),
    )
    result = grid.run()
    root = tmp_path / result.study_id
    trial = read_record(trial_path(root))
    artifact = read_record(root / cast(str, trial["artifact_location"]))
    manifest = mapping(mapping(artifact["prediction_study"])["manifest"])
    context = mapping(manifest["prediction_context"])
    assert context["status"] == ("skipped" if incompatible_feed else "available")
    if incompatible_feed:
        assert mapping(manifest["record_counts"])["generated_predictions"] == 0
    block_research(monkeypatch)
    inspected = inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert inspected.provenance.producer_study_id == result.study_id
