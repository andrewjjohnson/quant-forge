"""Persisted intraday sources cannot omit or contradict their feed scope."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import FeedScope
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._prediction_context_integrity import (
    validate_prediction_context,
)
from quantforge.prediction import (
    PredictionContextFailurePolicy,
    PredictionStudy,
    SignalFeatureCandidate,
    SignalFeatureDatasetResult,
    build_signal_feature_dataset,
    intraday_forward_return_outcome,
    run_prediction_study,
)
from tests.integration.test_intraday_prediction_manifest_integrity import (
    _write_rehashed_feature,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_manifest_integrity import (
    feature_result as feature_result,
)
from tests.integration.test_intraday_prediction_provenance import (
    Fixture,
    study_inputs,
)
from tests.integration.test_intraday_prediction_provenance import (
    fixture as fixture,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json


@pytest.fixture(scope="module")
def prediction_manifest(fixture: Fixture) -> PrimitiveMapping:
    rule, provider = study_inputs(fixture)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary)
    return run_prediction_study(
        fixture.dataset,
        PredictionStudy[SignalFeatureCandidate, Any, Any].create(
            rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
        ),
        context_provider=provider,
    ).manifest_primitive()


def _rehash_prediction(manifest: PrimitiveMapping) -> None:
    manifest["study_id"] = configuration_identity(
        {
            "component": manifest["component"],
            "engine_version": manifest["engine_version"],
            "market_data": manifest["market_data"],
            "study_configuration": manifest["configuration"],
            "prediction_context": manifest["prediction_context"],
        }
    )


def test_prediction_outcome_source_requires_explicit_feed(
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = deepcopy(prediction_manifest)
    configuration = cast(PrimitiveMapping, manifest["configuration"])
    labeler = cast(PrimitiveMapping, configuration["outcome_labeler"])
    source = cast(PrimitiveMapping, labeler["outcome_source"])
    del cast(PrimitiveMapping, source["source_reference"])["feed_scope"]
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="source lineage"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
@pytest.mark.parametrize("source", ["template", "outcome"])
def test_feature_outcome_sources_require_explicit_feed(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
    source: str,
) -> None:
    configuration = feature_result.configuration
    outcome: PrimitiveMapping | None = None
    owner = configuration
    if source == "outcome":
        outcome = cast(list[PrimitiveMapping], configuration["outcomes"])[0]
        owner = cast(PrimitiveMapping, outcome["component_configuration"])
    reference = cast(
        PrimitiveMapping,
        cast(PrimitiveMapping, owner["outcome_source"])["source_reference"],
    )
    del reference["feed_scope"]
    if outcome is not None:
        outcome["configuration_id"] = configuration_identity(owner)
    path = _write_rehashed_feature(feature_result, configuration, tmp_path, directory)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="source lineage"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


def _change_context_feed(context: PrimitiveMapping, target: str) -> None:
    selected = cast(list[PrimitiveMapping], context["timeframes"])
    selected_index = 0 if target.endswith("primary") else 1
    cast(PrimitiveMapping, selected[selected_index]["requirement"])["feed_scope"] = (
        FeedScope.iex_only().to_primitive()
    )
    if target.startswith("selected_"):
        return
    requirements = cast(PrimitiveMapping, context["requirements"])
    requirement = (
        cast(PrimitiveMapping, requirements["primary"])
        if target == "primary"
        else cast(list[PrimitiveMapping], requirements["contextual"])[0]
    )
    requirement["feed_scope"] = FeedScope.iex_only().to_primitive()


@pytest.mark.parametrize("target", ["primary", "contextual"])
def test_prediction_context_requirements_must_match_input_feed(
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    manifest = deepcopy(prediction_manifest)
    context = cast(PrimitiveMapping, manifest["prediction_context"])
    _change_context_feed(context, target)
    configuration = cast(PrimitiveMapping, manifest["configuration"])
    configuration["prediction_context_requirements"] = deepcopy(context["requirements"])
    rule = cast(PrimitiveMapping, configuration["prediction_rule"])
    definition = cast(PrimitiveMapping, rule["configuration"])
    definition["context_requirements"] = deepcopy(context["requirements"])
    rule["configuration_id"] = configuration_identity(definition)
    _rehash_prediction(manifest)
    # The compressed source references omit feed scope by schema. All captured
    # requirement copies agree, but contradict the input's consolidated feed.
    validate_prediction_context(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="feed scope"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
@pytest.mark.parametrize(
    "target", ["primary", "contextual", "selected_primary", "selected_contextual"]
)
def test_feature_context_requirements_must_match_input_feed(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
    target: str,
) -> None:
    configuration = feature_result.configuration
    _change_context_feed(
        cast(PrimitiveMapping, configuration["prediction_context"]), target
    )
    path = _write_rehashed_feature(feature_result, configuration, tmp_path, directory)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="feed scope"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "study_type", [StudyType.PREDICTION, StudyType.FEATURE_DATASET]
)
def test_skipped_feed_mismatch_remains_readable(
    fixture: Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_type: StudyType,
) -> None:
    rule, provider = study_inputs(fixture)
    requirements = rule.context_requirements
    rule.context_requirements = replace(
        requirements,
        primary=replace(requirements.primary, required_feed_scope=FeedScope.iex_only()),
        contextual=tuple(
            replace(item, required_feed_scope=FeedScope.iex_only())
            for item in requirements.contextual
        ),
        failure_policy=PredictionContextFailurePolicy.SKIP,
    )
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary)
    study = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
        rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
    )
    path = tmp_path / "skipped.json"
    if study_type is StudyType.PREDICTION:
        prediction = run_prediction_study(
            fixture.dataset, study, context_provider=provider
        )
        assert not prediction.rows
        context = cast(
            PrimitiveMapping, prediction.manifest_primitive()["prediction_context"]
        )
        write_json(path, prediction.to_primitive())
        expected_id = prediction.study_id
    else:
        feature = build_signal_feature_dataset(
            dataset=fixture.dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(outcome,),
            context_provider=provider,
            output_root=tmp_path / "features",
        )
        assert not feature.rows
        context = cast(PrimitiveMapping, feature.configuration["prediction_context"])
        write_json(path, feature.to_primitive())
        expected_id = feature.dataset_id
    assert context["status"] == "skipped"
    block_research(monkeypatch)
    assert (
        inspect_study(
            study_type, path, artifact_root=tmp_path
        ).provenance.producer_study_id
        == expected_id
    )
