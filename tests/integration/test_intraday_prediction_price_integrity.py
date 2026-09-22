"""Projected OHLCV remains bound to its named session artifact after rehashing."""

from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path
from shutil import copy2, copytree
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data import (
    MarketDataCache,
    dataset_identity_matches,
    validate_market_dataset,
)
from quantforge.data.exceptions import CacheError, ValidationError
from quantforge.data.identity import (
    serialize_bars_csv,
    serialize_metadata_values,
    sha256_hex,
)
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import PredictionMarketData, SignalFeatureDatasetResult
from tests.integration.test_intraday_prediction_feed_integrity import (
    _rehash_prediction,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_lineage_integrity import (
    _rehash_dataset,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_manifest_integrity import (
    _write_rehashed_feature,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_provenance import Fixture
from tests.integration.test_intraday_prediction_provenance import fixture as fixture
from tests.integration.test_intraday_prediction_request_integrity import (
    _rehash_nested,  # pyright: ignore[reportPrivateUsage]
    _replace_ids,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_source_requirements import (
    empty_artifacts as empty_artifacts,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json


@pytest.mark.parametrize("field", ["same", "open", "high", "low", "close", "volume"])
@pytest.mark.parametrize(
    "boundary", ["dataset", "cache", "prediction", "feature", "feature_directory"]
)
def test_rehashed_projection_prices_must_match_the_session_artifact(
    fixture: Fixture,
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    boundary: str,
) -> None:
    original = fixture.dataset
    bars = original.bars
    if field != "same":
        increment = Decimal("-0.05" if field == "low" else "0.05")
        bars = (
            replace(bars[0], **{field: getattr(bars[0], field) + increment}),
            *bars[1:],
        )
    altered = _rehash_dataset(
        replace(
            original,
            bars=bars,
            metadata=replace(
                original.metadata, data_sha256=sha256_hex(serialize_bars_csv(bars))
            ),
        )
    )
    assert dataset_identity_matches(altered)
    block_research(monkeypatch)
    if boundary == "dataset":
        if field == "same":
            validate_market_dataset(altered)
        else:
            with pytest.raises(ValidationError, match=r"session.*bars"):
                validate_market_dataset(altered)
        return
    if boundary == "cache":
        cache = MarketDataCache(tmp_path / "cache")
        directory = cache.root / "datasets" / altered.metadata.dataset_id
        copytree(
            fixture.cache.root / "datasets" / original.metadata.dataset_id, directory
        )
        raw_path = cache.root / altered.metadata.raw_location
        raw_path.parent.mkdir(parents=True)
        copy2(fixture.cache.root / altered.metadata.raw_location, raw_path)
        (directory / "bars.csv").write_bytes(serialize_bars_csv(bars))
        write_json(
            directory / "manifest.json",
            cast(PrimitiveMapping, serialize_metadata_values(asdict(altered.metadata))),
        )
        if field == "same":
            assert cache.load(altered.metadata.dataset_id) == altered
        else:
            with pytest.raises(CacheError, match=r"session.*bars"):
                cache.load(altered.metadata.dataset_id)
        return
    market = PredictionMarketData.from_qf3(altered.metadata)
    replacements = {
        original.metadata.dataset_id: altered.metadata.dataset_id,
        original.metadata.data_sha256: altered.metadata.data_sha256,
    }
    if boundary == "prediction":
        manifest = cast(
            PrimitiveMapping, _replace_ids(deepcopy(empty_artifacts[0]), replacements)
        )
        manifest["market_data"] = market.to_primitive()
        _rehash_nested(manifest)
        _rehash_prediction(manifest)
        path = tmp_path / "prediction.json"
        write_json(path, manifest)
        study_type = StudyType.PREDICTION
    else:
        result = empty_artifacts[1]
        configuration = cast(
            PrimitiveMapping, _replace_ids(result.configuration, replacements)
        )
        configuration["source_data"] = market.to_primitive()
        _rehash_nested(configuration)
        path = _write_rehashed_feature(
            replace(result, market_data=market),
            configuration,
            tmp_path,
            boundary == "feature_directory",
        )
        study_type = StudyType.FEATURE_DATASET
    if field == "same":
        inspect_study(study_type, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match=r"session.*bars"):
            inspect_study(study_type, path, artifact_root=tmp_path)
