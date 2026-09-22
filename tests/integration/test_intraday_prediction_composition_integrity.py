"""Composed prediction families must bind the retained QF-19 artifact family."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from shutil import copy2, copytree
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.data import (
    AggregationPolicy,
    DatasetFamily,
    MarketDataCache,
    MarketDataset,
    TimeframeBarSeries,
    aggregate_session_dataset,
    dataset_identity_matches,
    validate_market_dataset,
)
from quantforge.data.exceptions import CacheError, ValidationError
from quantforge.data.identity import serialize_metadata_values
from quantforge.data.multi_timeframe import MultiTimeframeContextValidationError
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
from tests.integration.test_intraday_prediction_provenance import DAILY, Fixture
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


def _with_family(fixture: Fixture, family: DatasetFamily) -> MarketDataset:
    provenance = fixture.dataset.metadata.intraday_provenance
    assert provenance is not None
    return _rehash_dataset(
        replace(
            fixture.dataset,
            metadata=replace(
                fixture.dataset.metadata,
                intraday_provenance=replace(
                    provenance,
                    family_id=family.family_id,
                    family_manifest=PrimitiveMappingSnapshot.capture(
                        family.to_manifest()
                    ),
                ),
            ),
        )
    )


@pytest.mark.parametrize(
    "change", ["same", "artifact_id", "missing", "null", "name", "version"]
)
@pytest.mark.parametrize(
    "boundary", ["dataset", "cache", "prediction", "feature", "feature_directory"]
)
def test_rehashed_composed_family_requires_its_session_artifact(
    fixture: Fixture,
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    boundary: str,
) -> None:
    provenance = fixture.dataset.metadata.intraday_provenance
    assert provenance is not None
    original_family = DatasetFamily.from_manifest(
        provenance.family_manifest.to_primitive()
    )
    sessions = aggregate_session_dataset(fixture.source, DAILY)
    policy = original_family.aggregation_policy
    configuration = cast(PrimitiveMapping, policy.to_primitive()["configuration"])
    if change == "artifact_id":
        configuration["artifact_family_manifest_ids"] = [
            "0" * 64 if identity == sessions.dataset_family.manifest_id else identity
            for identity in cast(
                list[str], configuration["artifact_family_manifest_ids"]
            )
        ]
    elif change == "missing":
        del configuration["artifact_family_manifest_ids"]
    elif change == "null":
        configuration["artifact_family_manifest_ids"] = None
    family = replace(
        original_family,
        aggregation_policy=AggregationPolicy(
            "unbound_policy" if change == "name" else policy.policy_name,
            "unknown" if change == "version" else policy.policy_version,
            configuration,
        ),
    )
    # All generic graph/hash checks pass, but the artifact-binding API must not.
    assert DatasetFamily.from_manifest(family.to_manifest()) == family
    if change == "same":
        TimeframeBarSeries.from_aggregated_session_dataset(sessions, family=family)
    else:
        with pytest.raises(MultiTimeframeContextValidationError, match="bind"):
            TimeframeBarSeries.from_aggregated_session_dataset(sessions, family=family)
    altered = _with_family(fixture, family)
    assert dataset_identity_matches(altered)
    block_research(monkeypatch)
    if boundary == "dataset":
        if change == "same":
            validate_market_dataset(altered)
        else:
            with pytest.raises(ValidationError, match="artifact family manifest"):
                validate_market_dataset(altered)
        return
    if boundary == "cache":
        cache = MarketDataCache(tmp_path / "cache")
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
        if change == "same":
            assert cache.load(altered.metadata.dataset_id) == altered
        else:
            with pytest.raises(CacheError, match="artifact family manifest"):
                cache.load(altered.metadata.dataset_id)
        return
    replacements = {
        original_family.family_id: family.family_id,
        original_family.manifest_id: family.manifest_id,
        fixture.dataset.metadata.dataset_id: altered.metadata.dataset_id,
    }
    market = PredictionMarketData.from_qf3(altered.metadata)
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
    if change == "same":
        inspect_study(study_type, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match="artifact family manifest"):
            inspect_study(study_type, path, artifact_root=tmp_path)


@pytest.mark.parametrize("changed", [False, True], ids=["same", "different"])
def test_uncomposed_projection_requires_the_exact_session_family(
    fixture: Fixture, changed: bool
) -> None:
    family = aggregate_session_dataset(fixture.source, DAILY).dataset_family
    if changed:
        family = replace(
            family, aggregation_policy=AggregationPolicy("unbound_policy", "1", {})
        )
    assert DatasetFamily.from_manifest(family.to_manifest()) == family
    altered = _with_family(fixture, family)
    assert dataset_identity_matches(altered)
    if changed:
        with pytest.raises(ValidationError, match="artifact family manifest"):
            validate_market_dataset(altered)
    else:
        validate_market_dataset(altered)
