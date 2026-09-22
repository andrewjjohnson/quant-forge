"""Saved available contexts must retain their exact validated family evidence."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import (
    PredictionContextFailurePolicy,
    PredictionRuleContext,
    PredictionStudy,
    SignalFeatureCandidate,
    SignalFeatureCandidateOutput,
    SignalFeatureDatasetResult,
    build_signal_feature_dataset,
    intraday_forward_return_outcome,
)
from tests.integration.test_intraday_prediction_feed_integrity import (
    _rehash_prediction,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_feed_integrity import (
    prediction_manifest as prediction_manifest,
)
from tests.integration.test_intraday_prediction_manifest_integrity import (
    _write_rehashed_feature,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_manifest_integrity import (
    feature_result as feature_result,
)
from tests.integration.test_intraday_prediction_projection_boundaries import (
    _alternate_family_context,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_provenance import Fixture, study_inputs
from tests.integration.test_intraday_prediction_provenance import fixture as fixture
from tests.integration.test_intraday_prediction_source_requirements import (
    empty_artifacts as empty_artifacts,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _FixtureCandidateRule,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_multi_timeframe_study import InvalidContextProvider


def test_context_serialization_and_identity_bind_the_exact_family(
    fixture: Fixture, prediction_manifest: PrimitiveMapping
) -> None:
    _, provider = study_inputs(fixture)
    context = provider.context
    alternate = _alternate_family_context(fixture)
    assert context.source_consistency == alternate.source_consistency
    assert context.timeframes == alternate.timeframes
    for captured in (context, alternate):
        primitive = captured.to_primitive()
        assert primitive["dataset_family_manifest_id"] == (
            captured.dataset_family_manifest_id
        )
        assert primitive["context_id"] == configuration_identity(
            {key: value for key, value in primitive.items() if key != "context_id"}
        )
    assert context.context_id != alternate.context_id
    source = cast(
        PrimitiveMapping,
        cast(PrimitiveMapping, prediction_manifest["prediction_context"])[
            "source_context"
        ],
    )
    assert source == context.to_primitive()
    unbound = _alternate_family_context(fixture, mixed_graphs=True).to_primitive()
    assert "dataset_family_manifest_id" not in unbound
    legacy = context.to_primitive()
    del legacy["dataset_family_manifest_id"]
    legacy["context_id"] = configuration_identity(
        {key: value for key, value in legacy.items() if key != "context_id"}
    )
    assert unbound == legacy


def _change_context_family(
    context: PrimitiveMapping, fixture: Fixture, change: str
) -> None:
    source = cast(PrimitiveMapping, context["source_context"])
    if change == "different":
        # A valid graph with the same family and referenced members still has
        # a different exact manifest; per-timeframe checks cannot detect it.
        context["source_context"] = _alternate_family_context(fixture).to_primitive()
        return
    if change == "missing":
        source.pop("dataset_family_manifest_id", None)
    elif change == "null":
        source["dataset_family_manifest_id"] = None
    source["context_id"] = configuration_identity(
        {key: value for key, value in source.items() if key != "context_id"}
    )


@pytest.mark.parametrize("change", ["same", "different", "missing", "null"])
def test_prediction_context_requires_exact_retained_manifest(
    fixture: Fixture,
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    manifest = deepcopy(prediction_manifest)
    _change_context_family(
        cast(PrimitiveMapping, manifest["prediction_context"]), fixture, change
    )
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    if change == "same":
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match="context family manifest"):
            inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("change", ["same", "different", "missing", "null"])
@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_feature_context_requires_exact_retained_manifest(
    fixture: Fixture,
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    directory: bool,
) -> None:
    configuration = feature_result.configuration
    _change_context_family(
        cast(PrimitiveMapping, configuration["prediction_context"]), fixture, change
    )
    path = _write_rehashed_feature(feature_result, configuration, tmp_path, directory)
    block_research(monkeypatch)
    if change == "same":
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match="context family manifest"):
            inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize("change", ["same", "missing", "null", "skipped", "status"])
@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_empty_feature_context_requires_source_evidence_unless_skipped(
    fixture: Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    directory: bool,
) -> None:
    original = _FixtureCandidateRule.generate_with_context

    def generate_empty(
        rule: _FixtureCandidateRule, context: PredictionRuleContext
    ) -> SignalFeatureCandidateOutput:
        return replace(original(rule, context), signals=())

    rule, provider = study_inputs(fixture)
    if change == "skipped":
        rule.context_requirements = replace(
            rule.context_requirements,
            failure_policy=PredictionContextFailurePolicy.SKIP,
        )
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary)
    with monkeypatch.context() as patch:
        patch.setattr(_FixtureCandidateRule, "generate_with_context", generate_empty)
        result = build_signal_feature_dataset(
            dataset=fixture.dataset,
            prediction_study=PredictionStudy[SignalFeatureCandidate, Any, Any].create(
                rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
            ),
            contextual_features=(),
            outcomes=(outcome,),
            context_provider=(
                InvalidContextProvider() if change == "skipped" else provider
            ),
            output_root=tmp_path / "original",
        )
    assert not result.rows
    configuration = result.configuration
    context = cast(PrimitiveMapping, configuration["prediction_context"])
    assert context["status"] == ("skipped" if change == "skipped" else "available")
    if change == "skipped":
        assert context["source_context"] is None
    elif change == "missing":
        del context["source_context"]
    elif change == "null":
        context["source_context"] = None
    elif change == "status":
        context.update(status="unknown", source_context=None)
    path = _write_rehashed_feature(result, configuration, tmp_path, directory)
    block_research(monkeypatch)
    if change in {"same", "skipped"}:
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match="prediction source context"):
            inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize("change", ["same", "missing", "null", "altered", "stale"])
@pytest.mark.parametrize("empty", [False, True], ids=["candidates", "no_candidates"])
@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_feature_source_context_requires_its_own_identity(
    feature_result: SignalFeatureDatasetResult,
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    empty: bool,
    directory: bool,
) -> None:
    result = empty_artifacts[1] if empty else feature_result
    configuration = result.configuration
    context = cast(PrimitiveMapping, configuration["prediction_context"])
    source = cast(PrimitiveMapping, context["source_context"])
    if change == "missing":
        del source["context_id"]
    elif change == "null":
        source["context_id"] = None
    elif change == "altered":
        source["context_id"] = "0" * 64
    elif change == "stale":
        source["completion_policy"] = "developing_bar_as_of"
    path = _write_rehashed_feature(result, configuration, tmp_path, directory)
    block_research(monkeypatch)
    if change == "same":
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match="source context identity"):
            inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
