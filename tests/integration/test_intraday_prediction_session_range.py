"""Intraday projections retain the exact completed-session range they emit."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.data import (
    MarketDataset,
    dataset_identity_matches,
    validate_market_dataset,
)
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import serialize_bars_csv, sha256_hex
from quantforge.data.prediction_inputs import validate_prediction_provenance
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

CHANGES = (
    "same",
    "leading_gap",
    "trailing_gap",
    "leading_bound",
    "trailing_bound",
    "missing_only",
    "leading_subset",
    "trailing_subset",
)


def _alter_projection(fixture: Fixture, change: str) -> MarketDataset:
    original = fixture.dataset
    bars = original.bars
    if change.startswith("leading"):
        bars = bars[1:]
    elif change.startswith("trailing"):
        bars = bars[:-1]
    missing = (
        tuple(bar.session_date for bar in original.bars if bar not in bars)
        if change.endswith("gap")
        else (original.metadata.requested_end + timedelta(days=1),)
        if change == "missing_only"
        else ()
    )
    altered = _rehash_dataset(
        replace(
            original,
            bars=bars,
            metadata=replace(
                original.metadata,
                actual_first_session=bars[0].session_date,
                actual_last_session=bars[-1].session_date,
                requested_start=(
                    bars[0].session_date
                    if change.endswith("subset")
                    else original.metadata.requested_start
                ),
                requested_end=(
                    bars[-1].session_date
                    if change.endswith("subset")
                    else original.metadata.requested_end
                ),
                bar_count=len(bars),
                missing_sessions=missing,
                data_sha256=sha256_hex(serialize_bars_csv(bars)),
            ),
        )
    )
    assert dataset_identity_matches(altered)
    return altered


@pytest.mark.parametrize(
    "change",
    ["same", "leading_gap", "trailing_gap", "leading_subset", "trailing_subset"],
)
def test_dataset_projection_cannot_relabel_edge_sessions_as_missing(
    fixture: Fixture, change: str
) -> None:
    altered = _alter_projection(fixture, change)
    if change == "same":
        assert validate_market_dataset(altered) == ()
    else:
        with pytest.raises(ValidationError, match="projection session range"):
            validate_market_dataset(altered)


@pytest.mark.parametrize("change", CHANGES)
@pytest.mark.parametrize("artifact", ["prediction", "feature", "feature_directory"])
def test_projection_session_range_survives_rehashed_artifacts(
    fixture: Fixture,
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    artifact: str,
) -> None:
    altered = _alter_projection(fixture, change)
    market = PredictionMarketData.from_qf3(altered.metadata)
    replacements = {
        fixture.dataset.metadata.dataset_id: altered.metadata.dataset_id,
        fixture.dataset.metadata.data_sha256: altered.metadata.data_sha256,
    }
    if artifact == "prediction":
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
            artifact == "feature_directory",
        )
        study_type = StudyType.FEATURE_DATASET
    block_research(monkeypatch)
    if change == "same":
        inspect_study(study_type, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match="projection session range"):
            inspect_study(study_type, path, artifact_root=tmp_path)


@pytest.mark.parametrize("bar_count", [1, 3, True, None])
def test_projection_count_matches_retained_full_sessions(
    fixture: Fixture, bar_count: Primitive
) -> None:
    market = PredictionMarketData.from_qf3(fixture.dataset.metadata).to_primitive()
    market["bar_count"] = bar_count
    with pytest.raises(ValidationError, match="projection session range"):
        validate_prediction_provenance(market)
