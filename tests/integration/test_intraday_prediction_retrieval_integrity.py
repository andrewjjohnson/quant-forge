"""Projection acquisition times must agree with the retained raw chunks."""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from shutil import copy2, copytree
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.data import (
    MarketDataCache,
    dataset_identity_matches,
    validate_market_dataset,
)
from quantforge.data.exceptions import CacheError, ValidationError
from quantforge.data.identity import serialize_metadata_values
from quantforge.data.models import IntradayPredictionProvenance
from quantforge.data.prediction_inputs import validate_prediction_provenance
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import (
    PredictionMarketData,
    SignalFeatureDatasetResult,
    SignalFeatureRow,
)
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
from tests.integration.test_intraday_prediction_request_integrity import (
    _alter_bounds,  # pyright: ignore[reportPrivateUsage]
    _rehash_nested,  # pyright: ignore[reportPrivateUsage]
    _rehash_source_evidence,  # pyright: ignore[reportPrivateUsage]
    _replace_ids,  # pyright: ignore[reportPrivateUsage]
    bind_source_bars_to_chunks,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json


def _changed_retrieval(
    original: PrimitiveMapping, hours: int
) -> tuple[PrimitiveMapping, dict[str, str], datetime]:
    provenance = deepcopy(original)
    source = cast(PrimitiveMapping, provenance["source_manifest"])
    retrieved_at = datetime.fromisoformat(
        cast(str, source["retrieved_at"])
    ) + timedelta(hours=hours)
    source["retrieved_at"] = retrieved_at.isoformat()
    provenance, replacements = _rehash_source_evidence(provenance, original)
    assert (
        source["chunks"]
        == cast(PrimitiveMapping, original["source_manifest"])["chunks"]
    )
    assert provenance["source_raw_snapshot_ids"] == original["source_raw_snapshot_ids"]
    return provenance, replacements, retrieved_at


@pytest.mark.parametrize("hours", [-1, 1])
@pytest.mark.parametrize("boundary", ["dataset", "cache"])
def test_rehashed_projection_retrieval_must_match_chunks(
    fixture: Fixture, tmp_path: Path, hours: int, boundary: str
) -> None:
    original = fixture.dataset.metadata.intraday_provenance
    assert original is not None
    provenance, _, retrieved_at = _changed_retrieval(original.to_primitive(), hours)
    altered = _rehash_dataset(
        replace(
            fixture.dataset,
            metadata=replace(
                fixture.dataset.metadata,
                retrieved_at=retrieved_at,
                intraday_provenance=IntradayPredictionProvenance.from_primitive(
                    provenance
                ),
            ),
        )
    )
    assert dataset_identity_matches(altered)
    if boundary == "dataset":
        with pytest.raises(ValidationError, match="retrieval timestamp"):
            validate_market_dataset(altered)
        return
    cache = MarketDataCache(tmp_path / "cache")
    directory = cache.root / "datasets" / altered.metadata.dataset_id
    copytree(
        fixture.cache.root / "datasets" / fixture.dataset.metadata.dataset_id, directory
    )
    raw_path = cache.root / altered.metadata.raw_location
    raw_path.parent.mkdir(parents=True)
    copy2(fixture.cache.root / altered.metadata.raw_location, raw_path)
    write_json(
        directory / "manifest.json",
        cast(PrimitiveMapping, serialize_metadata_values(asdict(altered.metadata))),
    )
    with pytest.raises(CacheError, match="retrieval timestamp"):
        cache.load(altered.metadata.dataset_id)


@pytest.mark.parametrize("hours", [-1, 1])
def test_prediction_rejects_rehashed_retrieval(
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hours: int,
) -> None:
    manifest = deepcopy(prediction_manifest)
    market = cast(PrimitiveMapping, manifest["market_data"])
    provenance, replacements, retrieved_at = _changed_retrieval(
        cast(PrimitiveMapping, market["intraday_provenance"]), hours
    )
    manifest = cast(PrimitiveMapping, _replace_ids(manifest, replacements))
    cast(PrimitiveMapping, manifest["market_data"]).update(
        intraday_provenance=provenance, retrieved_at=retrieved_at.isoformat()
    )
    _rehash_nested(manifest)
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="retrieval timestamp"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("hours", [-1, 1])
@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_feature_rejects_rehashed_retrieval(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hours: int,
    directory: bool,
) -> None:
    configuration = feature_result.configuration
    market = cast(PrimitiveMapping, configuration["source_data"])
    provenance, replacements, retrieved_at = _changed_retrieval(
        cast(PrimitiveMapping, market["intraday_provenance"]), hours
    )
    configuration = cast(PrimitiveMapping, _replace_ids(configuration, replacements))
    cast(PrimitiveMapping, configuration["source_data"]).update(
        intraday_provenance=provenance, retrieved_at=retrieved_at.isoformat()
    )
    _rehash_nested(configuration)
    altered = replace(
        feature_result,
        market_data=replace(
            feature_result.market_data,
            retrieved_at=retrieved_at,
            intraday_provenance=IntradayPredictionProvenance.from_primitive(provenance),
        ),
        rows=tuple(
            SignalFeatureRow.capture(
                cast(PrimitiveMapping, _replace_ids(row.to_primitive(), replacements))
            )
            for row in feature_result.rows
        ),
    )
    path = _write_rehashed_feature(altered, configuration, tmp_path, directory)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="retrieval timestamp"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "timestamp",
    [None, 123, "invalid", "2024-01-04T00:00:00", "0001-01-01T00:00:00+01:00"],
)
def test_every_chunk_requires_an_aware_retrieval_timestamp(
    fixture: Fixture, timestamp: Primitive
) -> None:
    original = fixture.dataset.metadata.intraday_provenance
    assert original is not None
    provenance, _ = _alter_bounds(original.to_primitive(), "contiguous")
    source = cast(PrimitiveMapping, provenance["source_manifest"])
    cast(list[PrimitiveMapping], source["chunks"])[0]["retrieved_at"] = timestamp
    provenance, _ = _rehash_source_evidence(provenance, original.to_primitive())
    market = PredictionMarketData.from_qf3(fixture.dataset.metadata).to_primitive()
    market["intraday_provenance"] = provenance
    with pytest.raises(ValidationError, match="retrieval timestamp"):
        validate_prediction_provenance(market)


@pytest.mark.parametrize("latest_chunk", [0, 1])
@pytest.mark.parametrize("wrong_maximum", [False, True], ids=["latest", "earliest"])
def test_retrieval_uses_the_latest_chunk_instant(
    fixture: Fixture, latest_chunk: int, wrong_maximum: bool
) -> None:
    original = fixture.dataset.metadata.intraday_provenance
    assert original is not None
    provenance, _ = _alter_bounds(original.to_primitive(), "contiguous")
    source = cast(PrimitiveMapping, provenance["source_manifest"])
    chunks = cast(list[PrimitiveMapping], source["chunks"])
    chunks[latest_chunk]["retrieved_at"] = "2024-01-03T20:00:00-05:00"
    chunks[1 - latest_chunk]["retrieved_at"] = "2024-01-04T00:00:00Z"
    source["retrieved_at"] = (
        "2024-01-04T00:00:00+00:00" if wrong_maximum else "2024-01-04T01:00:00+00:00"
    )
    bind_source_bars_to_chunks(provenance)
    provenance, _ = _rehash_source_evidence(provenance, original.to_primitive())
    market = PredictionMarketData.from_qf3(fixture.dataset.metadata).to_primitive()
    market.update(intraday_provenance=provenance, retrieved_at=source["retrieved_at"])
    if wrong_maximum:
        with pytest.raises(ValidationError, match="retrieval timestamp"):
            validate_prediction_provenance(market)
    else:
        assert validate_prediction_provenance(market) is not None
