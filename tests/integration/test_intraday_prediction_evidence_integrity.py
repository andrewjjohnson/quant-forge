"""Raw source evidence and the complete family DAG survive observational checks."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data import dataset_identity_matches, validate_market_dataset
from quantforge.data.exceptions import ValidationError
from quantforge.data.intraday_ingestion import IntradayMarketDataCache
from quantforge.data.models import IntradayPredictionProvenance
from quantforge.data.prediction_inputs import validate_prediction_provenance
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import PredictionMarketData, SignalFeatureDatasetResult
from tests.integration.test_intraday_prediction_feed_integrity import (
    _rehash_prediction,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_feed_integrity import (
    prediction_manifest as prediction_manifest,
)
from tests.integration.test_intraday_prediction_lineage_integrity import (
    _rehash_dataset,  # pyright: ignore[reportPrivateUsage]
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

CHANGES = (
    "request",
    "raw_snapshots",
    "both_raw_ids",
    "parent_missing",
    "parent_unknown",
    "children_disagree",
    "cycle",
    "orphan_member",
)


def _alter_projection_metadata(
    market: PredictionMarketData, change: str
) -> PredictionMarketData:
    if change == "calendar":
        return replace(market, calendar="XNAS")
    if change == "timezone":
        return replace(market, provider_timezone="UTC")
    if change == "session_policy":
        return replace(market, calendar="XNAS", provider_timezone="UTC")
    return replace(market, retrieved_at=market.retrieved_at + timedelta(hours=1))


@pytest.mark.parametrize(
    "change", ["calendar", "timezone", "session_policy", "retrieval"]
)
def test_dataset_projection_metadata_must_match_retained_evidence(
    fixture: Fixture, change: str
) -> None:
    market = _alter_projection_metadata(
        PredictionMarketData.from_qf3(fixture.dataset.metadata), change
    )
    altered = _rehash_dataset(
        replace(
            fixture.dataset,
            metadata=replace(
                fixture.dataset.metadata,
                calendar=market.calendar,
                provider_timezone=market.provider_timezone,
                retrieved_at=market.retrieved_at,
            ),
        )
    )
    assert dataset_identity_matches(altered)
    with pytest.raises(
        ValidationError,
        match="retrieval" if change == "retrieval" else "session policy",
    ):
        validate_market_dataset(altered)


@pytest.mark.parametrize(
    "change", ["calendar", "timezone", "session_policy", "retrieval"]
)
def test_prediction_projection_metadata_must_match_retained_evidence(
    prediction_manifest: PrimitiveMapping,
    fixture: Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    manifest = deepcopy(prediction_manifest)
    manifest["market_data"] = _alter_projection_metadata(
        PredictionMarketData.from_qf3(fixture.dataset.metadata), change
    ).to_primitive()
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(
        ManifestError, match="retrieval" if change == "retrieval" else "session policy"
    ):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
@pytest.mark.parametrize(
    "change", ["calendar", "timezone", "session_policy", "retrieval"]
)
def test_feature_projection_metadata_must_match_retained_evidence(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
    change: str,
) -> None:
    altered = replace(
        feature_result,
        market_data=_alter_projection_metadata(feature_result.market_data, change),
    )
    configuration = altered.configuration
    configuration["source_data"] = altered.market_data.to_primitive()
    path = _write_rehashed_feature(altered, configuration, tmp_path, directory)
    block_research(monkeypatch)
    with pytest.raises(
        ManifestError, match="retrieval" if change == "retrieval" else "session policy"
    ):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "timestamp",
    [None, 123, "not-a-timestamp", "2024-01-04T00:00:00", "0001-01-01T00:00:00+01:00"],
)
def test_projection_retrieval_timestamp_requires_an_aware_instant(
    prediction_manifest: PrimitiveMapping, timestamp: Primitive
) -> None:
    market = deepcopy(cast(PrimitiveMapping, prediction_manifest["market_data"]))
    market["retrieved_at"] = timestamp
    with pytest.raises(ValidationError, match="retrieval timestamp"):
        validate_prediction_provenance(market)


@pytest.mark.parametrize("representation", ["utc_z", "offset"])
def test_equivalent_retrieval_timestamp_representation_is_accepted(
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    representation: str,
) -> None:
    manifest = deepcopy(prediction_manifest)
    market = cast(PrimitiveMapping, manifest["market_data"])
    original = cast(str, market["retrieved_at"])
    market["retrieved_at"] = (
        original.replace("+00:00", "Z")
        if representation == "utc_z"
        else datetime.fromisoformat(original)
        .astimezone(timezone(timedelta(hours=-5)))
        .isoformat()
    )
    assert market["retrieved_at"] != original
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    assert (
        inspect_study(
            StudyType.PREDICTION, path, artifact_root=tmp_path
        ).provenance.producer_study_id
        == manifest["study_id"]
    )


def _alter_evidence(provenance: PrimitiveMapping, change: str) -> str:
    if change in {"request", "raw_snapshots", "both_raw_ids"}:
        if change != "raw_snapshots":
            provenance["source_request_id"] = "0" * 64
        if change != "request":
            provenance["source_raw_snapshot_ids"] = ["1" * 64]
        return "source manifest"
    evidence = cast(PrimitiveMapping, provenance["family_manifest"])
    lineage = cast(list[PrimitiveMapping], evidence["lineage"])
    session = next(
        entry
        for entry in lineage
        if entry["dataset_id"] == provenance["session_dataset_id"]
    )
    source = next(
        entry
        for entry in lineage
        if entry["dataset_id"] == provenance["source_dataset_id"]
    )
    if change == "parent_missing":
        session["parent_dataset_id"] = None
    elif change == "parent_unknown":
        session["parent_dataset_id"] = "0" * 64
    elif change == "children_disagree":
        source["child_dataset_ids"] = []
    elif change == "cycle":
        sibling = next(
            entry for entry in lineage if entry is not session and entry is not source
        )
        session["parent_dataset_id"] = sibling["dataset_id"]
        sibling["parent_dataset_id"] = session["dataset_id"]
        session["child_dataset_ids"] = [sibling["dataset_id"]]
        sibling["child_dataset_ids"] = [session["dataset_id"]]
        source["child_dataset_ids"] = []
    else:
        orphan = deepcopy(session)
        orphan["dataset_id"] = "0" * 64
        orphan["parent_dataset_id"] = None
        orphan["child_dataset_ids"] = []
        lineage.append(orphan)
    evidence["manifest_id"] = configuration_identity(
        {key: value for key, value in evidence.items() if key != "manifest_id"}
    )
    return "family.*lineage|lineage.*family"


def _align_outcome_manifest_ids(record: Primitive, manifest_id: Primitive) -> None:
    if isinstance(record, dict):
        if "family_manifest_id" in record:
            record["family_manifest_id"] = manifest_id
        for value in record.values():
            _align_outcome_manifest_ids(value, manifest_id)
    elif isinstance(record, list):
        for value in record:
            _align_outcome_manifest_ids(value, manifest_id)


def test_source_manifest_is_exact_cached_evidence_and_immutable(
    fixture: Fixture,
) -> None:
    provenance = fixture.dataset.metadata.intraday_provenance
    assert provenance is not None
    manifest = provenance.source_manifest.to_primitive()
    cache = IntradayMarketDataCache(fixture.cache.root)
    assert manifest == cache.read_manifest(fixture.source.metadata.dataset_id)
    assert manifest["dataset_id"] == provenance.source_dataset_id
    assert (
        cast(PrimitiveMapping, manifest["request"])["request_id"]
        == provenance.source_request_id
    )
    assert (
        tuple(
            chunk["raw_snapshot_id"]
            for chunk in cast(list[PrimitiveMapping], manifest["chunks"])
        )
        == provenance.source_raw_snapshot_ids
    )
    manifest["dataset_id"] = "0" * 64
    assert (
        provenance.source_manifest.to_primitive()["dataset_id"]
        == provenance.source_dataset_id
    )


@pytest.mark.parametrize("change", ["request", "raw_snapshots"])
@pytest.mark.parametrize("rehash_source", [False, True])
def test_changed_source_evidence_cannot_retain_the_family_snapshot(
    prediction_manifest: PrimitiveMapping,
    change: str,
    rehash_source: bool,
) -> None:
    market = deepcopy(cast(PrimitiveMapping, prediction_manifest["market_data"]))
    provenance = cast(PrimitiveMapping, market["intraday_provenance"])
    manifest = cast(PrimitiveMapping, provenance["source_manifest"])
    request = cast(PrimitiveMapping, manifest["request"])
    chunks = cast(list[PrimitiveMapping], manifest["chunks"])
    if change == "request":
        configuration = cast(PrimitiveMapping, request["configuration"])
        configuration["end_timestamp"] = "2024-02-01T00:00:00+00:00"
        request["request_id"] = configuration_identity(configuration)
        provenance["source_request_id"] = request["request_id"]
    else:
        chunks[0]["raw_snapshot_id"] = "0" * 64
        chunks[0]["raw_sha256"] = "0" * 64
        chunks[0]["raw_location"] = f"intraday/raw/{'0' * 64}.json"
        provenance["source_raw_snapshot_ids"] = [
            chunk["raw_snapshot_id"] for chunk in chunks
        ]
    if rehash_source:
        identity = {
            key: value
            for key, value in manifest.items()
            if key not in {"dataset_id", "normalized_location"}
        }
        identity["chunks"] = [
            {key: value for key, value in chunk.items() if key != "raw_location"}
            for chunk in chunks
        ]
        manifest["dataset_id"] = configuration_identity(identity)
        manifest["normalized_location"] = (
            f"intraday/datasets/{manifest['dataset_id']}/bars.json"
        )
    with pytest.raises(ValidationError, match="source manifest"):
        validate_prediction_provenance(market)


@pytest.mark.parametrize("change", CHANGES)
def test_dataset_rejects_rehashed_source_or_graph(
    fixture: Fixture,
    change: str,
) -> None:
    original = fixture.dataset.metadata.intraday_provenance
    assert original is not None
    provenance = original.to_primitive()
    expected_error = _alter_evidence(provenance, change)
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
    with pytest.raises(ValidationError, match=expected_error):
        validate_market_dataset(altered)


@pytest.mark.parametrize("change", CHANGES)
def test_prediction_rejects_rehashed_source_or_graph(
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    manifest = deepcopy(prediction_manifest)
    market = cast(PrimitiveMapping, manifest["market_data"])
    provenance = cast(PrimitiveMapping, market["intraday_provenance"])
    expected_error = _alter_evidence(provenance, change)
    evidence = cast(PrimitiveMapping, provenance["family_manifest"])
    _align_outcome_manifest_ids(manifest["configuration"], evidence["manifest_id"])
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=expected_error):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("change", CHANGES)
@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_feature_rejects_rehashed_source_or_graph(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    directory: bool,
) -> None:
    configuration = feature_result.configuration
    market = cast(PrimitiveMapping, configuration["source_data"])
    provenance = cast(PrimitiveMapping, market["intraday_provenance"])
    expected_error = _alter_evidence(provenance, change)
    evidence = cast(PrimitiveMapping, provenance["family_manifest"])
    _align_outcome_manifest_ids(configuration, evidence["manifest_id"])
    for outcome in cast(list[PrimitiveMapping], configuration["outcomes"]):
        outcome["configuration_id"] = configuration_identity(
            cast(PrimitiveMapping, outcome["component_configuration"])
        )
    altered = replace(
        feature_result,
        market_data=replace(
            feature_result.market_data,
            intraday_provenance=IntradayPredictionProvenance.from_primitive(provenance),
        ),
    )
    path = _write_rehashed_feature(altered, configuration, tmp_path, directory)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=expected_error):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
