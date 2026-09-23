"""Cache reload must bind manifest lineage to the retained raw snapshot bodies."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data import (
    CacheError,
    IntradayBarBatch,
    IntradayDataset,
    IntradayFetchResult,
    IntradayMarketDataCache,
    IntradayRawSnapshot,
    IntradayValidationMode,
    MarketDataCache,
    aggregate_session_dataset,
    validate_intraday_coverage,
)
from quantforge.data.identity import canonical_json_bytes, sha256_hex
from quantforge.data.models import IntradayPredictionProvenance
from quantforge.data.multi_timeframe import MultiTimeframeContextValidationError
from quantforge.data.prediction_inputs import prediction_dataset_from_intraday
from tests.integration.test_intraday_prediction_provenance import DAILY, Fixture
from tests.integration.test_intraday_prediction_provenance import fixture as fixture


@pytest.fixture(scope="module", params=[1, 2], ids=["one-chunk", "two-chunks"])
def source_cache(
    request: pytest.FixtureRequest,
    fixture: Fixture,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[IntradayDataset, IntradayMarketDataCache]:
    source = fixture.source
    count = cast(int, request.param)
    boundary = datetime.fromisoformat("2024-01-03T00:00:00+00:00")
    snapshots = tuple(
        IntradayRawSnapshot(
            source.metadata.provider_name,
            source.metadata.provider_symbol,
            source.metadata.adapter_version,
            "offline-fixture",
            source.request.request_id,
            source.request.start_timestamp if index == 0 else boundary,
            source.request.end_timestamp if index == count - 1 else boundary,
            source.metadata.retrieved_at + timedelta(hours=index),
            (("symbol", "SPY"), ("interval", "1min")),
            ({"chunk": index, "nested": [1.5, True, None, {"key": "value"}]},),
        )
        for index in range(count)
    )
    bars = tuple(
        replace(
            bar,
            provenance=replace(
                bar.provenance,
                source_snapshot_id=snapshot.snapshot_id,
                retrieved_at=snapshot.retrieved_at,
            ),
        )
        for bar in source.bars
        for snapshot in snapshots
        if snapshot.chunk_start_timestamp
        <= bar.start_timestamp
        < snapshot.chunk_end_timestamp
    )
    result = IntradayFetchResult(
        IntradayBarBatch(source.request, bars),
        snapshots,
        source.metadata.capabilities_configuration_id,
    )
    cache = IntradayMarketDataCache(tmp_path_factory.mktemp("raw-evidence"))
    dataset = cache.persist(result)
    assert cache.load(dataset.metadata.dataset_id, dataset.request) == dataset
    for snapshot, location in zip(
        snapshots, dataset.metadata.raw_locations, strict=True
    ):
        assert (cache.root / location).read_bytes() == snapshot.serialize()
    return dataset, cache


def _write_rehashed_source(
    source_cache: tuple[IntradayDataset, IntradayMarketDataCache],
    manifest: PrimitiveMapping,
    root: Path,
    *,
    raw_override: bytes | None = None,
) -> tuple[IntradayDataset, IntradayMarketDataCache]:
    source, original_cache = source_cache
    chunks = cast(list[PrimitiveMapping], manifest["chunks"])
    bars = source.bars
    if raw_override is not None:
        snapshot_id = sha256_hex(raw_override)
        old_id = source.metadata.raw_snapshot_ids[0]
        chunks[0].update(
            raw_snapshot_id=snapshot_id,
            raw_sha256=snapshot_id,
            raw_location=f"intraday/raw/{snapshot_id}.json",
        )
        bars = tuple(
            replace(
                bar, provenance=replace(bar.provenance, source_snapshot_id=snapshot_id)
            )
            if bar.provenance.source_snapshot_id == old_id
            else bar
            for bar in bars
        )
    batch = IntradayBarBatch(source.request, bars)
    normalized_bytes = batch.serialize()
    report = validate_intraday_coverage(batch, mode=IntradayValidationMode.DIAGNOSTIC)
    manifest.update(
        batch_id=batch.batch_id,
        data_sha256=sha256_hex(normalized_bytes),
        quality_report={"report_id": report.report_id, "report": report.to_primitive()},
    )
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"dataset_id", "normalized_location"}
    }
    identity["chunks"] = [
        {key: value for key, value in chunk.items() if key != "raw_location"}
        for chunk in chunks
    ]
    dataset_id = configuration_identity(identity)
    location = f"intraday/datasets/{dataset_id}/bars.json"
    manifest.update(dataset_id=dataset_id, normalized_location=location)
    directory = root / "intraday" / "datasets" / dataset_id
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    (root / location).write_bytes(normalized_bytes)
    for index, chunk in enumerate(chunks):
        raw_path = root / cast(str, chunk["raw_location"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_bytes = (
            raw_override
            if index == 0 and raw_override is not None
            else (
                original_cache.root / source.metadata.raw_locations[index]
            ).read_bytes()
        )
        assert sha256_hex(raw_bytes) == chunk["raw_snapshot_id"]
        raw_path.write_bytes(raw_bytes)
    altered = replace(
        source,
        bars=bars,
        metadata=replace(
            source.metadata,
            dataset_id=dataset_id,
            normalized_location=location,
            batch_id=batch.batch_id,
            data_sha256=sha256_hex(normalized_bytes),
            quality_report=report,
            raw_snapshot_ids=tuple(
                cast(str, chunk["raw_snapshot_id"]) for chunk in chunks
            ),
            raw_locations=tuple(cast(str, chunk["raw_location"]) for chunk in chunks),
        ),
    )
    return altered, IntradayMarketDataCache(root)


@pytest.mark.parametrize(
    "field",
    [
        "endpoint",
        "provider_name",
        "provider_symbol",
        "adapter_version",
        "retrieved_at",
        "chunk_start_timestamp",
        "chunk_end_timestamp",
    ],
)
def test_cache_rejects_rehashed_manifest_metadata_contradicting_raw_snapshots(
    source_cache: tuple[IntradayDataset, IntradayMarketDataCache],
    tmp_path: Path,
    field: str,
) -> None:
    source, cache = source_cache
    manifest = cache.read_manifest(source.metadata.dataset_id)
    chunks = cast(list[PrimitiveMapping], manifest["chunks"])
    if field == "endpoint":
        for chunk in chunks:
            chunk[field] = "forged-common-endpoint"
    elif field in {"provider_name", "provider_symbol", "adapter_version"}:
        manifest[field] = "forged"
    elif field == "retrieved_at":
        manifest[field] = "2024-01-05T00:00:00+00:00"
        for chunk in chunks:
            chunk[field] = manifest[field]
    else:
        chunk = chunks[0] if field == "chunk_start_timestamp" else chunks[-1]
        chunk[field] = (
            datetime.fromisoformat(cast(str, chunk[field])) + timedelta(minutes=1)
        ).isoformat()
    altered, forged_cache = _write_rehashed_source(source_cache, manifest, tmp_path)
    with pytest.raises(CacheError, match=r"raw.*snapshot"):
        forged_cache.load(altered.metadata.dataset_id, altered.request)


def test_projection_rejects_a_common_endpoint_contradicting_unchanged_raw_files(
    source_cache: tuple[IntradayDataset, IntradayMarketDataCache], tmp_path: Path
) -> None:
    source, cache = source_cache
    manifest = cache.read_manifest(source.metadata.dataset_id)
    for chunk in cast(list[PrimitiveMapping], manifest["chunks"]):
        chunk["endpoint"] = "forged-common-endpoint"
    altered, forged_cache = _write_rehashed_source(
        source_cache, manifest, tmp_path / "source"
    )
    # Standalone metadata is internally consistent; only loading the retained
    # raw bytes can establish that the claimed endpoint contradicts them.
    assert forged_cache.read_manifest(altered.metadata.dataset_id) == manifest
    sessions = aggregate_session_dataset(altered, DAILY)
    projection_cache = MarketDataCache(tmp_path / "projection")
    with pytest.raises(MultiTimeframeContextValidationError, match="immutable cache"):
        prediction_dataset_from_intraday(
            altered, sessions, cache=projection_cache, intraday_cache=forged_cache
        )
    assert not projection_cache.root.exists()


def test_valid_raw_snapshots_remain_reloadable_and_projectable(
    source_cache: tuple[IntradayDataset, IntradayMarketDataCache], tmp_path: Path
) -> None:
    source, cache = source_cache
    projection_cache = MarketDataCache(tmp_path)
    projected = prediction_dataset_from_intraday(
        source,
        aggregate_session_dataset(source, DAILY),
        cache=projection_cache,
        intraday_cache=cache,
    )
    provenance = projected.metadata.intraday_provenance
    assert isinstance(provenance, IntradayPredictionProvenance)
    assert provenance.source_manifest.to_primitive() == cache.read_manifest(
        source.metadata.dataset_id
    )
    assert projection_cache.load(projected.metadata.dataset_id) == projected
    assert cache.load(source.metadata.dataset_id, source.request) == source


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("endpoint", "forged-raw-endpoint"),
        ("endpoint", None),
        ("schema_version", "0"),
        ("artifact_type", "other"),
        ("source_request_id", "other-request"),
        ("retrieved_at", "2024-01-04T00:00:00"),
        ("retrieved_at", "0001-01-01T00:00:00+01:00"),
        ("request_parameters", {"symbol": 123}),
        ("records", [123]),
        ("extra_field", "not-in-the-producer-schema"),
    ],
)
def test_cache_parses_rehashed_raw_snapshots_before_accepting_lineage(
    source_cache: tuple[IntradayDataset, IntradayMarketDataCache],
    tmp_path: Path,
    field: str,
    replacement: Primitive,
) -> None:
    source, cache = source_cache
    manifest = cache.read_manifest(source.metadata.dataset_id)
    raw = cast(
        PrimitiveMapping,
        json.loads((cache.root / source.metadata.raw_locations[0]).read_bytes()),
    )
    raw[field] = deepcopy(replacement)
    altered, forged_cache = _write_rehashed_source(
        source_cache, manifest, tmp_path, raw_override=canonical_json_bytes(raw)
    )
    with pytest.raises(CacheError):
        forged_cache.load(altered.metadata.dataset_id, altered.request)
