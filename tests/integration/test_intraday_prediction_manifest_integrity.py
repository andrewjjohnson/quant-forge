"""Rehashed intraday artifacts must retain their original source semantics."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._prediction_context_integrity import (
    validate_prediction_context,
)
from quantforge.prediction import (
    PredictionContextFailurePolicy,
    PredictionStudy,
    SignalFeatureCandidate,
    SignalFeatureDatasetResult,
    SignalFeatureRow,
    build_signal_feature_dataset,
    intraday_forward_return_outcome,
    run_prediction_study,
)
from quantforge.prediction.feature_dataset import (
    _finalize_export,  # pyright: ignore[reportPrivateUsage]
    _row_id,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.timeframes import ExchangeSessionPolicy
from tests.integration.test_intraday_prediction_provenance import (
    DAILY,
    TWO_MINUTES,
    Fixture,
    cached_fixture,
    study_inputs,
)
from tests.integration.test_intraday_prediction_provenance import (
    fixture as fixture,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json


@pytest.fixture(scope="module")
def feature_result(
    fixture: Fixture, tmp_path_factory: pytest.TempPathFactory
) -> SignalFeatureDatasetResult:
    rule, provider = study_inputs(fixture)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary)
    return build_signal_feature_dataset(
        dataset=fixture.dataset,
        prediction_study=PredictionStudy[SignalFeatureCandidate, Any, Any].create(
            rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
        ),
        contextual_features=(),
        outcomes=(outcome,),
        context_provider=provider,
        output_root=tmp_path_factory.mktemp("intraday-manifest-features"),
    )


def _write_rehashed_feature(
    result: SignalFeatureDatasetResult,
    configuration: PrimitiveMapping,
    root: Path,
    directory: bool,
) -> Path:
    dataset_id = configuration_identity(configuration)
    rows: list[SignalFeatureRow] = []
    for original in result.rows:
        row = original.to_primitive()
        row["study_id"] = dataset_id
        row["row_id"] = _row_id(dataset_id, row)
        rows.append(SignalFeatureRow.capture(row))
    altered = replace(
        result,
        dataset_id=dataset_id,
        configuration_snapshot=PrimitiveMappingSnapshot.capture(configuration),
        rows=tuple(rows),
    )
    if directory:
        path = root / "feature"
        (path / "rows").mkdir(parents=True)
        for row in altered.rows:
            write_json(path / "rows" / f"{row.row_id}.json", row.to_primitive())
        _finalize_export(path, altered)
    else:
        path = root / "feature.json"
        write_json(path, altered.to_primitive())
    return path


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
@pytest.mark.parametrize("source", ["template", "outcome", "context"])
@pytest.mark.parametrize("field", ["family_id", "canonical_source_snapshot_id"])
def test_rehashed_feature_sources_must_match_intraday_provenance(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
    source: str,
    field: str,
) -> None:
    configuration = feature_result.configuration
    if source == "context":
        context = cast(PrimitiveMapping, configuration["prediction_context"])
        captured = cast(PrimitiveMapping, context["source_context"])
        aligned = cast(list[PrimitiveMapping], captured["timeframes"])
        for timeframe in aligned:
            cast(PrimitiveMapping, timeframe["dataset_reference"])[field] = "0" * 64
        if field == "family_id":
            cast(PrimitiveMapping, captured["source_consistency"])[field] = "0" * 64
        captured["context_id"] = configuration_identity(
            {key: value for key, value in captured.items() if key != "context_id"}
        )
    else:
        owner = configuration
        outcome: PrimitiveMapping | None = None
        if source == "outcome":
            outcome = cast(list[PrimitiveMapping], configuration["outcomes"])[0]
            owner = cast(PrimitiveMapping, outcome["component_configuration"])
        reference = cast(
            PrimitiveMapping,
            cast(PrimitiveMapping, owner["outcome_source"])["source_reference"],
        )
        reference[field] = "0" * 64
        if outcome is not None:
            outcome["configuration_id"] = configuration_identity(owner)
    path = _write_rehashed_feature(feature_result, configuration, tmp_path, directory)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="source lineage"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


def _change_context_session_policy(
    context: PrimitiveMapping, policy: ExchangeSessionPolicy
) -> None:
    replacements: dict[str, PrimitiveMapping] = {}
    for original in (TWO_MINUTES, DAILY):
        changed = replace(original, session_policy=policy)
        replacements[original.configuration_id] = {
            "configuration_id": changed.configuration_id,
            "configuration": changed.to_primitive(),
        }

    def rewrite(record: Primitive) -> None:
        if isinstance(record, list):
            for item in record:
                rewrite(item)
        elif isinstance(record, dict):
            identity = record.get("configuration_id")
            if isinstance(identity, str) and identity in replacements:
                record.clear()
                record.update(deepcopy(replacements[identity]))
                return
            timeframe_id = record.get("timeframe_configuration_id")
            if isinstance(timeframe_id, str) and timeframe_id in replacements:
                record["timeframe_configuration_id"] = replacements[timeframe_id][
                    "configuration_id"
                ]
            for item in record.values():
                rewrite(item)

    rewrite(context)
    captured = cast(PrimitiveMapping, context["source_context"])
    captured["context_id"] = configuration_identity(
        {key: value for key, value in captured.items() if key != "context_id"}
    )


@pytest.mark.parametrize(
    ("calendar", "timezone"), [("XNAS", "America/New_York"), ("XLON", "Europe/London")]
)
def test_rehashed_prediction_context_must_match_intraday_session_policy(
    fixture: Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calendar: str,
    timezone: str,
) -> None:
    rule, provider = study_inputs(fixture)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary)
    result = run_prediction_study(
        fixture.dataset,
        PredictionStudy[SignalFeatureCandidate, Any, Any].create(
            rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
        ),
        context_provider=provider,
    )
    manifest = result.manifest_primitive()
    context = cast(PrimitiveMapping, manifest["prediction_context"])
    _change_context_session_policy(context, ExchangeSessionPolicy(calendar, timezone))
    configuration = cast(PrimitiveMapping, manifest["configuration"])
    configuration["prediction_context_requirements"] = deepcopy(context["requirements"])
    rule_record = cast(PrimitiveMapping, configuration["prediction_rule"])
    rule_configuration = cast(PrimitiveMapping, rule_record["configuration"])
    rule_configuration["context_requirements"] = deepcopy(context["requirements"])
    rule_record["configuration_id"] = configuration_identity(rule_configuration)
    manifest["study_id"] = configuration_identity(
        {
            "component": manifest["component"],
            "engine_version": manifest["engine_version"],
            "market_data": manifest["market_data"],
            "study_configuration": configuration,
            "prediction_context": context,
        }
    )
    # The altered context is internally canonical. Its only inconsistency is
    # with the session policy retained in the intraday input provenance.
    validate_prediction_context(manifest)
    path = tmp_path / "prediction-manifest.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="session policy"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_rehashed_feature_context_must_match_intraday_session_policy(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
) -> None:
    configuration = feature_result.configuration
    _change_context_session_policy(
        cast(PrimitiveMapping, configuration["prediction_context"]),
        ExchangeSessionPolicy("XLON", "Europe/London"),
    )
    path = _write_rehashed_feature(feature_result, configuration, tmp_path, directory)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="session policy"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_matching_intraday_feature_provenance_is_read_without_research(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
) -> None:
    path = _write_rehashed_feature(
        feature_result, feature_result.configuration, tmp_path, directory
    )
    block_research(monkeypatch)
    inspected = inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    assert inspected.provenance.producer_study_id == feature_result.dataset_id


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_skipped_feature_context_preserves_incompatible_source_evidence(
    fixture: Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
) -> None:
    other = cached_fixture(tmp_path / "other", "other-provider")
    rule, _ = study_inputs(fixture)
    rule.context_requirements = replace(
        rule.context_requirements, failure_policy=PredictionContextFailurePolicy.SKIP
    )
    _, provider = study_inputs(other)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary)
    result = build_signal_feature_dataset(
        dataset=fixture.dataset,
        prediction_study=PredictionStudy[SignalFeatureCandidate, Any, Any].create(
            rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
        ),
        contextual_features=(),
        outcomes=(outcome,),
        context_provider=provider,
        output_root=tmp_path / "original",
    )
    assert not result.rows
    context = cast(PrimitiveMapping, result.configuration["prediction_context"])
    assert context["status"] == "skipped"
    captured = cast(PrimitiveMapping, context["source_context"])
    assert (
        cast(PrimitiveMapping, captured["source_consistency"])["family_id"]
        == other.primary.dataset_reference.family_id
    )
    test_matching_intraday_feature_provenance_is_read_without_research(
        result, tmp_path, monkeypatch, directory
    )
