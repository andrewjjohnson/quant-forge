"""Empty artifacts still require the sources implied by their declarations."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import (
    PredictionRuleContext,
    PredictionStudy,
    SignalFeatureCandidate,
    SignalFeatureCandidateOutput,
    SignalFeatureDatasetResult,
    build_signal_feature_dataset,
    intraday_forward_return_outcome,
    run_prediction_study,
)
from tests.integration.test_intraday_prediction_feed_integrity import (
    _rehash_prediction,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_manifest_integrity import (
    _write_rehashed_feature,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_provenance import Fixture, study_inputs
from tests.integration.test_intraday_prediction_provenance import fixture as fixture
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _FixtureCandidateRule,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture(scope="module")
def empty_artifacts(
    fixture: Fixture, tmp_path_factory: pytest.TempPathFactory
) -> tuple[PrimitiveMapping, SignalFeatureDatasetResult]:
    original = _FixtureCandidateRule.generate_with_context

    def generate_empty(
        rule: _FixtureCandidateRule, context: PredictionRuleContext
    ) -> SignalFeatureCandidateOutput:
        return replace(original(rule, context), signals=())

    rule, provider = study_inputs(fixture)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary)
    study = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
        rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_FixtureCandidateRule, "generate_with_context", generate_empty)
        prediction = run_prediction_study(
            fixture.dataset, study, context_provider=provider
        )
        features = build_signal_feature_dataset(
            dataset=fixture.dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(outcome,),
            context_provider=provider,
            output_root=tmp_path_factory.mktemp("empty-source-requirements"),
        )
    assert not prediction.rows
    assert not prediction.signals
    assert not features.rows
    return prediction.manifest_primitive(), features


def _remove_source(owner: PrimitiveMapping, change: str) -> None:
    if change == "missing":
        del owner["outcome_source"]
    elif change == "null":
        owner["outcome_source"] = None


@pytest.mark.parametrize("change", ["same", "missing", "null"])
def test_empty_elapsed_prediction_requires_its_outcome_source(
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    manifest = deepcopy(empty_artifacts[0])
    configuration = cast(PrimitiveMapping, manifest["configuration"])
    _remove_source(cast(PrimitiveMapping, configuration["outcome_labeler"]), change)
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    if change == "same":
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match=r"elapsed.*outcome source"):
            inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("change", ["same", "missing", "null"])
@pytest.mark.parametrize("source", ["template", "outcome"])
@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_empty_elapsed_features_require_every_outcome_source(
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    source: str,
    directory: bool,
) -> None:
    result = empty_artifacts[1]
    configuration = result.configuration
    if source == "template":
        _remove_source(configuration, change)
    else:
        outcome = cast(list[PrimitiveMapping], configuration["outcomes"])[0]
        component = cast(PrimitiveMapping, outcome["component_configuration"])
        _remove_source(component, change)
        outcome["configuration_id"] = configuration_identity(component)
    path = _write_rehashed_feature(result, configuration, tmp_path, directory)
    block_research(monkeypatch)
    if change == "same":
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match=r"elapsed.*outcome source"):
            inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "change", ["same", "swapped_reference", "spoofed_requirement_id"]
)
@pytest.mark.parametrize("target", [0, 1], ids=["primary", "contextual"])
@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_context_member_must_match_its_aligned_timeframe(
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    target: int,
    directory: bool,
) -> None:
    result = empty_artifacts[1]
    configuration = result.configuration
    context = cast(PrimitiveMapping, configuration["prediction_context"])
    captured = cast(PrimitiveMapping, context["source_context"])
    aligned = cast(list[PrimitiveMapping], captured["timeframes"])
    if change != "same":
        replacement = deepcopy(
            cast(PrimitiveMapping, aligned[1 - target]["dataset_reference"])
        )
        aligned[target].update(
            dataset_reference=replacement, dataset_id=replacement["dataset_id"]
        )
        if change == "spoofed_requirement_id":
            requirement = cast(PrimitiveMapping, aligned[target]["requirement"])
            cast(PrimitiveMapping, requirement["timeframe"])["configuration_id"] = (
                replacement["timeframe_configuration_id"]
            )
    captured["context_id"] = configuration_identity(
        {key: value for key, value in captured.items() if key != "context_id"}
    )
    path = _write_rehashed_feature(result, configuration, tmp_path, directory)
    block_research(monkeypatch)
    if change == "same":
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    else:
        with pytest.raises(
            ManifestError, match=r"source lineage timeframe|source timeframe"
        ):
            inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize("change", ["same", "missing", "null"])
@pytest.mark.parametrize("target", [0, 1], ids=["primary", "contextual"])
@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_empty_available_context_requires_every_timeframe_reference(
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    target: int,
    directory: bool,
) -> None:
    result = empty_artifacts[1]
    configuration = result.configuration
    context = cast(PrimitiveMapping, configuration["prediction_context"])
    captured = cast(PrimitiveMapping, context["source_context"])
    aligned = cast(list[PrimitiveMapping], captured["timeframes"])[target]
    assert context["status"] == "available"
    assert aligned["availability"] == "available"
    assert aligned["visible_bar_ids"]
    for field in ("dataset_reference", "dataset_id"):
        if change == "missing":
            del aligned[field]
        elif change == "null":
            aligned[field] = None
    captured["context_id"] = configuration_identity(
        {key: value for key, value in captured.items() if key != "context_id"}
    )
    path = _write_rehashed_feature(result, configuration, tmp_path, directory)
    block_research(monkeypatch)
    if change == "same":
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match=r"context timeframe.*provenance"):
            inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
