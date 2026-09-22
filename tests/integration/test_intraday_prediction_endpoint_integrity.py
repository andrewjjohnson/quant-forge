"""Retained raw chunks must use one valid provider endpoint revision."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.data.intraday_ingestion import validate_intraday_manifest_identity
from quantforge.data.models import IntradayPredictionProvenance
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import SignalFeatureDatasetResult, SignalFeatureRow
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
from tests.integration.test_intraday_prediction_request_integrity import (
    _alter_bounds,  # pyright: ignore[reportPrivateUsage]
    _rehash_nested,  # pyright: ignore[reportPrivateUsage]
    _rehash_source_evidence,  # pyright: ignore[reportPrivateUsage]
    _replace_ids,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json


def _endpoint_evidence(
    original: PrimitiveMapping, mixed: bool
) -> tuple[PrimitiveMapping, dict[str, str]]:
    provenance, _ = _alter_bounds(original, "contiguous")
    source = cast(PrimitiveMapping, provenance["source_manifest"])
    chunks = cast(list[PrimitiveMapping], source["chunks"])
    chunks[-1]["endpoint"] = "another-provider-revision"
    if not mixed:
        chunks[0]["endpoint"] = chunks[-1]["endpoint"]
    return _rehash_source_evidence(provenance, original)


@pytest.mark.parametrize("position", [0, 1, "all"])
@pytest.mark.parametrize("endpoint", ["missing", None, 123, "", " ", " padded "])
def test_retained_chunk_endpoints_require_nonempty_trimmed_strings(
    fixture: Fixture, endpoint: Primitive, position: int | str
) -> None:
    original = fixture.dataset.metadata.intraday_provenance
    assert original is not None
    provenance, _ = _alter_bounds(original.to_primitive(), "contiguous")
    source = cast(PrimitiveMapping, provenance["source_manifest"])
    chunks = cast(list[PrimitiveMapping], source["chunks"])
    for index, chunk in enumerate(chunks):
        if position == "all" or index == position:
            if endpoint == "missing":
                del chunk["endpoint"]
            else:
                chunk["endpoint"] = endpoint
    provenance, _ = _rehash_source_evidence(provenance, original.to_primitive())
    with pytest.raises(ValueError, match="endpoint"):
        validate_intraday_manifest_identity(
            cast(PrimitiveMapping, provenance["source_manifest"])
        )


@pytest.mark.parametrize("mixed", [False, True], ids=["same", "mixed"])
def test_prediction_requires_one_endpoint_after_rehashing(
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mixed: bool,
) -> None:
    manifest = deepcopy(prediction_manifest)
    market = cast(PrimitiveMapping, manifest["market_data"])
    provenance, replacements = _endpoint_evidence(
        cast(PrimitiveMapping, market["intraday_provenance"]), mixed
    )
    manifest = cast(PrimitiveMapping, _replace_ids(manifest, replacements))
    cast(PrimitiveMapping, manifest["market_data"])["intraday_provenance"] = provenance
    _rehash_nested(manifest)
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    if mixed:
        with pytest.raises(ManifestError, match="one provider endpoint revision"):
            inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    else:
        assert (
            inspect_study(
                StudyType.PREDICTION, path, artifact_root=tmp_path
            ).provenance.producer_study_id
            == manifest["study_id"]
        )


@pytest.mark.parametrize("mixed", [False, True], ids=["same", "mixed"])
@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_feature_requires_one_endpoint_after_rehashing(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mixed: bool,
    directory: bool,
) -> None:
    configuration = feature_result.configuration
    market = cast(PrimitiveMapping, configuration["source_data"])
    provenance, replacements = _endpoint_evidence(
        cast(PrimitiveMapping, market["intraday_provenance"]), mixed
    )
    configuration = cast(PrimitiveMapping, _replace_ids(configuration, replacements))
    cast(PrimitiveMapping, configuration["source_data"])["intraday_provenance"] = (
        provenance
    )
    _rehash_nested(configuration)
    altered = replace(
        feature_result,
        market_data=replace(
            feature_result.market_data,
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
    if mixed:
        with pytest.raises(ManifestError, match="one provider endpoint revision"):
            inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    else:
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
