"""Retained source observations must agree with raw chunks and coverage facts."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import dataset_identity_matches
from quantforge.data.exceptions import CacheError, ValidationError
from quantforge.data.models import IntradayPredictionProvenance
from quantforge.experiments import ManifestError
from quantforge.prediction import SignalFeatureDatasetResult
from tests.integration.test_intraday_prediction_lineage_integrity import (
    _rehash_dataset,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_origin_integrity import (
    inspect_projection,
)
from tests.integration.test_intraday_prediction_provenance import Fixture
from tests.integration.test_intraday_prediction_provenance import fixture as fixture
from tests.integration.test_intraday_prediction_request_integrity import (
    _alter_bounds,  # pyright: ignore[reportPrivateUsage]
    _rehash_source_evidence,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_source_requirements import (
    empty_artifacts as empty_artifacts,
)
from tests.unit.experiments.test_adapters import block_research

BOUNDARIES = ("dataset", "cache", "prediction", "feature", "feature_directory")


@pytest.mark.parametrize("change", ["same", "unused", "duplicate", "reordered"])
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_rehashed_source_templates_require_canonical_encoding(
    fixture: Fixture,
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    boundary: str,
) -> None:
    original = fixture.dataset.metadata.intraday_provenance
    assert original is not None
    primitive, replacements = _alter_bounds(original.to_primitive(), "contiguous")
    source = cast(PrimitiveMapping, primitive["source_manifest"])
    evidence = cast(PrimitiveMapping, primitive["source_bar_evidence"])
    templates = cast(list[PrimitiveMapping], evidence["templates"])
    observations = cast(list[PrimitiveMapping], evidence["observations"])
    assert len(templates) == 2
    if change == "unused":
        unused = deepcopy(templates[0])
        cast(PrimitiveMapping, unused["provenance"])["provider_name"] = "unused"
        templates.append(unused)
    elif change == "duplicate":
        templates.append(deepcopy(templates[0]))
        observations[0]["template_index"] = 2
    elif change == "reordered":
        templates.reverse()
        for observation in observations:
            observation["template_index"] = 1 - cast(int, observation["template_index"])
    expanded = [
        {
            **templates[cast(int, observation["template_index"])],
            **{
                key: value
                for key, value in observation.items()
                if key != "template_index"
            },
        }
        for observation in observations
    ]
    # Every variant retains the exact source batch and both source digests.
    assert (
        configuration_identity(
            {
                "schema_version": "1",
                "contract_type": "intraday_bar_batch",
                "request": source["request"],
                "bars": [
                    {"bar_id": configuration_identity(bar), "bar": bar}
                    for bar in expanded
                ],
            }
        )
        == source["batch_id"]
        == source["data_sha256"]
    )
    altered = _rehash_dataset(
        replace(
            fixture.dataset,
            metadata=replace(
                fixture.dataset.metadata,
                intraday_provenance=IntradayPredictionProvenance.from_primitive(
                    primitive
                ),
            ),
        )
    )
    assert dataset_identity_matches(altered)
    block_research(monkeypatch)
    if change == "same":
        inspect_projection(
            fixture, altered, empty_artifacts, tmp_path, boundary, replacements
        )
    else:
        with pytest.raises(
            (ValidationError, CacheError, ManifestError), match="canonical template"
        ):
            inspect_projection(
                fixture, altered, empty_artifacts, tmp_path, boundary, replacements
            )


@pytest.mark.parametrize(
    "change",
    [
        "same",
        "multi_chunk",
        "source_snapshot_id",
        "provider_name",
        "provider_symbol",
        "adapter_version",
        "retrieved_at",
        "wrong_chunk",
    ],
)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_rehashed_source_bars_must_match_their_raw_chunks(
    fixture: Fixture,
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    boundary: str,
) -> None:
    original = fixture.dataset.metadata.intraday_provenance
    assert original is not None
    primitive = original.to_primitive()
    if change in {"multi_chunk", "wrong_chunk"}:
        primitive, _ = _alter_bounds(primitive, "contiguous")
    evidence = cast(PrimitiveMapping, primitive["source_bar_evidence"])
    templates = cast(list[PrimitiveMapping], evidence["templates"])
    retained_provenance = cast(PrimitiveMapping, templates[0]["provenance"])
    if change == "retrieved_at":
        retained_provenance[change] = "2024-01-05T00:00:00+00:00"
    elif change == "wrong_chunk":
        retained_provenance["source_snapshot_id"] = cast(
            PrimitiveMapping, templates[1]["provenance"]
        )["source_snapshot_id"]
    elif change not in {"same", "multi_chunk"}:
        retained_provenance[change] = "unrelated-raw-source"
    primitive, replacements = _rehash_source_evidence(
        primitive, original.to_primitive()
    )
    altered = _rehash_dataset(
        replace(
            fixture.dataset,
            metadata=replace(
                fixture.dataset.metadata,
                intraday_provenance=IntradayPredictionProvenance.from_primitive(
                    primitive
                ),
            ),
        )
    )
    assert dataset_identity_matches(altered)
    block_research(monkeypatch)
    if change in {"same", "multi_chunk"}:
        inspect_projection(
            fixture, altered, empty_artifacts, tmp_path, boundary, replacements
        )
    else:
        with pytest.raises(
            (ValidationError, CacheError, ManifestError), match="raw snapshot"
        ):
            inspect_projection(
                fixture, altered, empty_artifacts, tmp_path, boundary, replacements
            )


@pytest.mark.parametrize("change", ["same", "missing", "extra", "misplaced"])
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_rehashed_coverage_must_describe_the_retained_observations(
    fixture: Fixture,
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    boundary: str,
) -> None:
    original = fixture.dataset.metadata.intraday_provenance
    assert original is not None
    primitive = original.to_primitive()
    evidence = cast(PrimitiveMapping, primitive["source_bar_evidence"])
    observations = cast(list[PrimitiveMapping], evidence["observations"])
    if change != "extra":
        # Preserve the session total, so an incorrect coverage report is the
        # only contradiction; the existing OHLCV reduction check still passes.
        observations[0]["volume"] = "0"
        observations[1]["volume"] = "2000"
    source = cast(PrimitiveMapping, primitive["source_manifest"])
    quality = cast(PrimitiveMapping, source["quality_report"])
    report = cast(PrimitiveMapping, quality["report"])
    if change != "missing":
        observation = observations[1 if change == "misplaced" else 0]
        interval: PrimitiveMapping = {
            "session_date": observation["session_identifier"],
            "start_timestamp": observation["start_timestamp"],
            "end_timestamp": observation["end_timestamp"],
            "completion": observation["completion"],
        }
        report["zero_volume_intervals"] = [interval]
        cast(list[PrimitiveMapping], report["sessions"])[0]["zero_volume_intervals"] = [
            deepcopy(interval)
        ]
        report["has_warnings"] = True
    quality["report_id"] = configuration_identity(report)
    primitive, replacements = _rehash_source_evidence(
        primitive, original.to_primitive()
    )
    altered = _rehash_dataset(
        replace(
            fixture.dataset,
            metadata=replace(
                fixture.dataset.metadata,
                intraday_provenance=IntradayPredictionProvenance.from_primitive(
                    primitive
                ),
            ),
        )
    )
    assert dataset_identity_matches(altered)
    assert altered.bars == fixture.dataset.bars
    block_research(monkeypatch)
    if change == "same":
        inspect_projection(
            fixture, altered, empty_artifacts, tmp_path, boundary, replacements
        )
    else:
        with pytest.raises(
            (ValidationError, CacheError, ManifestError), match="source coverage report"
        ):
            inspect_projection(
                fixture, altered, empty_artifacts, tmp_path, boundary, replacements
            )
