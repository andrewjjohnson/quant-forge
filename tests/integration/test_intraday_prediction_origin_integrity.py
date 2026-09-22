"""Projection origin survives stripped provenance and rehashed outer artifacts."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from shutil import copy2, copytree
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data import (
    MarketDataCache,
    MarketDataset,
    dataset_identity_matches,
    validate_market_dataset,
)
from quantforge.data.exceptions import CacheError, ValidationError
from quantforge.data.identity import (
    INTRADAY_PREDICTION_DATASET_PREFIX,
    serialize_metadata_values,
)
from quantforge.data.prediction_inputs import DAILY_CORPORATE_ACTION_POLICY
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


def inspect_projection(
    fixture: Fixture,
    altered: MarketDataset,
    artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    root: Path,
    boundary: str,
    replacements: dict[str, str] | None = None,
) -> None:
    """Rewrite actual cache or experiment artifacts with consistent outer IDs."""
    if boundary == "dataset":
        validate_market_dataset(altered)
        return
    if boundary == "cache":
        cache = MarketDataCache(root / "cache")
        directory = cache.root / "datasets" / altered.metadata.dataset_id
        copytree(
            fixture.cache.root / "datasets" / fixture.dataset.metadata.dataset_id,
            directory,
        )
        raw_path = cache.root / altered.metadata.raw_location
        raw_path.parent.mkdir(parents=True)
        copy2(fixture.cache.root / altered.metadata.raw_location, raw_path)
        write_json(
            directory / "manifest.json",
            cast(PrimitiveMapping, serialize_metadata_values(asdict(altered.metadata))),
        )
        cache.load(altered.metadata.dataset_id)
        return
    identities = {
        **(replacements or {}),
        fixture.dataset.metadata.dataset_id: altered.metadata.dataset_id,
        fixture.dataset.metadata.corporate_action_policy: (
            altered.metadata.corporate_action_policy
        ),
    }
    market = PredictionMarketData.from_qf3(altered.metadata)
    if boundary == "prediction":
        manifest = cast(
            PrimitiveMapping, _replace_ids(deepcopy(artifacts[0]), identities)
        )
        manifest["market_data"] = market.to_primitive()
        _rehash_nested(manifest)
        _rehash_prediction(manifest)
        path = root / "prediction.json"
        write_json(path, manifest)
        study_type = StudyType.PREDICTION
    else:
        result = artifacts[1]
        configuration = cast(
            PrimitiveMapping, _replace_ids(result.configuration, identities)
        )
        configuration["source_data"] = market.to_primitive()
        _rehash_nested(configuration)
        path = _write_rehashed_feature(
            replace(result, market_data=market),
            configuration,
            root,
            boundary == "feature_directory",
        )
        study_type = StudyType.FEATURE_DATASET
    inspect_study(study_type, path, artifact_root=root)


@pytest.mark.parametrize("stripped", [False, True], ids=["same", "stripped"])
@pytest.mark.parametrize(
    "boundary", ["dataset", "cache", "prediction", "feature", "feature_directory"]
)
def test_projection_cannot_downgrade_to_daily_policy(
    fixture: Fixture,
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stripped: bool,
    boundary: str,
) -> None:
    altered = fixture.dataset
    if stripped:
        altered = _rehash_dataset(
            replace(
                altered,
                metadata=replace(
                    altered.metadata,
                    intraday_provenance=None,
                    corporate_action_policy=DAILY_CORPORATE_ACTION_POLICY,
                ),
            )
        )
    assert altered.metadata.dataset_id.startswith(INTRADAY_PREDICTION_DATASET_PREFIX)
    assert dataset_identity_matches(altered)
    block_research(monkeypatch)
    if stripped:
        with pytest.raises(
            (ValidationError, CacheError, ManifestError),
            match=r"projection.*provenance",
        ):
            inspect_projection(fixture, altered, empty_artifacts, tmp_path, boundary)
    else:
        inspect_projection(fixture, altered, empty_artifacts, tmp_path, boundary)


def test_raw_projection_cannot_hide_its_adapter_origin(
    fixture: Fixture,
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
) -> None:
    altered = _rehash_dataset(
        replace(
            fixture.dataset,
            metadata=replace(
                fixture.dataset.metadata,
                intraday_provenance=None,
                corporate_action_policy=DAILY_CORPORATE_ACTION_POLICY,
                adapter_version="legacy",
            ),
        )
    )
    assert not altered.metadata.dataset_id.startswith(
        INTRADAY_PREDICTION_DATASET_PREFIX
    )
    assert dataset_identity_matches(altered)
    with pytest.raises(CacheError, match=r"raw artifact.*provenance"):
        inspect_projection(fixture, altered, empty_artifacts, tmp_path, "cache")
