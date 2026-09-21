"""Intraday projections and references must name recorded family members."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
)
from quantforge.data import (
    MarketDataset,
    dataset_identity_matches,
    validate_market_dataset,
)
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import calculate_dataset_id
from quantforge.data.models import IntradayPredictionProvenance
from quantforge.data.prediction_inputs import validate_prediction_provenance
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._prediction_context_integrity import (
    validate_prediction_context,
)
from quantforge.prediction import SignalFeatureDatasetResult
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
from tests.integration.test_intraday_prediction_provenance import Fixture
from tests.integration.test_intraday_prediction_provenance import fixture as fixture
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json


def _replace_session_reference(provenance: PrimitiveMapping, change: str) -> None:
    if change == "intraday_member":
        evidence = cast(PrimitiveMapping, provenance["family_manifest"])
        source = cast(PrimitiveMapping, evidence["canonical_source"])
        timeframe = cast(PrimitiveMapping, source["timeframe"])
        provenance["session_dataset_id"] = provenance["source_dataset_id"]
        provenance["session_timeframe_configuration_id"] = timeframe["configuration_id"]
    else:
        provenance[change] = "0" * 64


def _rehash_dataset(dataset: MarketDataset) -> MarketDataset:
    metadata = dataset.metadata
    values = asdict(metadata)
    for field in (
        "raw_location",
        "normalized_location",
        "raw_sha256",
        "data_sha256",
        "dataset_id",
        "schema_version",
        "corporate_actions_location",
    ):
        del values[field]
    dataset_id = calculate_dataset_id(
        values,
        raw_sha256=metadata.raw_sha256,
        data_sha256=metadata.data_sha256,
        schema_version=metadata.schema_version,
    )
    return replace(
        dataset,
        metadata=replace(
            metadata,
            dataset_id=dataset_id,
            normalized_location=f"datasets/{dataset_id}/bars.csv",
            corporate_actions_location=f"datasets/{dataset_id}/corporate_actions.json",
        ),
    )


@pytest.mark.parametrize("source_name", ["outcome", "context"])
@pytest.mark.parametrize("change", ["unknown", "other_timeframe"])
def test_prediction_references_require_matching_family_member(
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
    change: str,
) -> None:
    manifest = deepcopy(prediction_manifest)
    market = cast(PrimitiveMapping, manifest["market_data"])
    provenance = cast(PrimitiveMapping, market["intraday_provenance"])
    replacement = "0" * 64 if change == "unknown" else provenance["source_dataset_id"]
    if source_name == "outcome":
        configuration = cast(PrimitiveMapping, manifest["configuration"])
        labeler = cast(PrimitiveMapping, configuration["outcome_labeler"])
        source = cast(PrimitiveMapping, labeler["outcome_source"])
        cast(PrimitiveMapping, source["source_reference"])["dataset_id"] = replacement
    else:
        context = cast(PrimitiveMapping, manifest["prediction_context"])
        captured = cast(PrimitiveMapping, context["source_context"])
        for timeframe in cast(list[PrimitiveMapping], captured["timeframes"]):
            timeframe["dataset_id"] = replacement
            cast(PrimitiveMapping, timeframe["dataset_reference"])["dataset_id"] = (
                replacement
            )
        captured["context_id"] = configuration_identity(
            {key: value for key, value in captured.items() if key != "context_id"}
        )
        validate_prediction_context(manifest)
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="source lineage"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "session_dataset_id",
        "session_timeframe_configuration_id",
        "intraday_member",
    ],
)
def test_rehashed_dataset_requires_recorded_session_artifact(
    fixture: Fixture,
    change: str,
) -> None:
    original = fixture.dataset.metadata.intraday_provenance
    assert original is not None
    provenance = original.to_primitive()
    _replace_session_reference(provenance, change)
    altered = _rehash_dataset(
        replace(
            fixture.dataset,
            metadata=replace(
                fixture.dataset.metadata,
                intraday_provenance=IntradayPredictionProvenance.from_primitive(
                    provenance
                ),
            ),
        )
    )
    assert dataset_identity_matches(altered)
    with pytest.raises(ValidationError, match=r"session.*lineage"):
        validate_market_dataset(altered)


@pytest.mark.parametrize(
    "change",
    [
        "session_dataset_id",
        "session_timeframe_configuration_id",
        "intraday_member",
    ],
)
def test_rehashed_prediction_requires_recorded_session_artifact(
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    manifest = deepcopy(prediction_manifest)
    market = cast(PrimitiveMapping, manifest["market_data"])
    _replace_session_reference(
        cast(PrimitiveMapping, market["intraday_provenance"]), change
    )
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=r"session.*lineage"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
@pytest.mark.parametrize(
    "change",
    [
        "session_dataset_id",
        "session_timeframe_configuration_id",
        "intraday_member",
    ],
)
def test_rehashed_feature_requires_recorded_session_artifact(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
    change: str,
) -> None:
    configuration = feature_result.configuration
    market = cast(PrimitiveMapping, configuration["source_data"])
    provenance = cast(PrimitiveMapping, market["intraday_provenance"])
    _replace_session_reference(provenance, change)
    altered = replace(
        feature_result,
        market_data=replace(
            feature_result.market_data,
            intraday_provenance=IntradayPredictionProvenance.from_primitive(provenance),
        ),
    )
    path = _write_rehashed_feature(altered, configuration, tmp_path, directory)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=r"session.*lineage"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "change", ["duplicate", "timeframe_definition", "source", "role", "session_policy"]
)
def test_session_lineage_must_be_unambiguous_and_consistent(
    prediction_manifest: PrimitiveMapping,
    change: str,
) -> None:
    market = deepcopy(cast(PrimitiveMapping, prediction_manifest["market_data"]))
    provenance = cast(PrimitiveMapping, market["intraday_provenance"])
    evidence = cast(PrimitiveMapping, provenance["family_manifest"])
    lineage = cast(list[PrimitiveMapping], evidence["lineage"])
    session = next(
        entry
        for entry in lineage
        if entry["dataset_id"] == provenance["session_dataset_id"]
    )
    if change == "duplicate":
        lineage.append(deepcopy(session))
    elif change == "timeframe_definition":
        timeframe = cast(PrimitiveMapping, session["timeframe"])
        configuration = cast(PrimitiveMapping, timeframe["configuration"])
        configuration["interval"] = {}
    elif change == "source":
        session["canonical_source_snapshot_id"] = "0" * 64
    elif change == "role":
        session["role"] = "canonical_source_snapshot"
    else:
        provenance["session_policy_id"] = "0" * 64
    evidence["manifest_id"] = configuration_identity(
        {key: value for key, value in evidence.items() if key != "manifest_id"}
    )
    with pytest.raises(ValidationError, match=r"session.*lineage"):
        validate_prediction_provenance(market)
