"""Retained session constituents must belong to the named canonical source batch."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import dataset_identity_matches
from quantforge.data.exceptions import CacheError, ValidationError
from quantforge.data.models import IntradayPredictionProvenance
from quantforge.data.prediction_source_bars import (
    capture_source_bar_evidence,
    validate_source_bar_evidence,
)
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
    _rehash_source_evidence,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_source_requirements import (
    empty_artifacts as empty_artifacts,
)
from tests.unit.experiments.test_adapters import block_research


def test_compact_source_evidence_reproduces_the_immutable_batch(
    fixture: Fixture,
) -> None:
    provenance = fixture.dataset.metadata.intraday_provenance
    assert provenance is not None
    evidence = provenance.source_bar_evidence.to_primitive()
    assert evidence == capture_source_bar_evidence(fixture.source.bars)
    restored = IntradayPredictionProvenance.from_primitive(provenance.to_primitive())
    assert restored == provenance
    batch = validate_source_bar_evidence(
        evidence, provenance.source_manifest.to_primitive()
    )
    assert batch.bars == fixture.source.bars
    assert batch.batch_id == fixture.source.metadata.batch_id
    cast(list[PrimitiveMapping], evidence["observations"])[0]["open"] = "999"
    assert provenance.source_bar_evidence == restored.source_bar_evidence
    with pytest.raises(ValueError, match="canonical source batch"):
        validate_source_bar_evidence(
            evidence, provenance.source_manifest.to_primitive()
        )


@pytest.mark.parametrize(
    "change", ["same", "arbitrary", "swap_sessions", "reversed", "forged_source"]
)
@pytest.mark.parametrize(
    "boundary", ["dataset", "cache", "prediction", "feature", "feature_directory"]
)
def test_rehashed_session_constituents_require_canonical_source_membership(
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
    evidence = cast(PrimitiveMapping, primitive["session_evidence"])
    bars = [
        cast(PrimitiveMapping, entry["bar"])
        for entry in cast(
            list[PrimitiveMapping], cast(PrimitiveMapping, evidence["bars"])["bars"]
        )
    ]
    if change == "arbitrary":
        for index, bar in enumerate(bars):
            bar["source_bar_ids"] = [
                configuration_identity({"session": index, "fake": offset})
                for offset in range(len(cast(list[str], bar["source_bar_ids"])))
            ]
    elif change == "swap_sessions":
        bars[0]["source_bar_ids"], bars[1]["source_bar_ids"] = (
            bars[1]["source_bar_ids"],
            bars[0]["source_bar_ids"],
        )
    elif change == "reversed":
        for bar in bars:
            cast(list[str], bar["source_bar_ids"]).reverse()
    primitive, replacements = _rehash_source_evidence(deepcopy(primitive), primitive)
    if change == "forged_source":
        source_evidence = cast(PrimitiveMapping, primitive["source_bar_evidence"])
        cast(list[PrimitiveMapping], source_evidence["observations"])[0]["volume"] = (
            "1234"
        )
    assert primitive["source_dataset_id"] == original.source_dataset_id
    assert primitive["source_manifest"] == original.source_manifest.to_primitive()
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
            (ValidationError, CacheError, ManifestError),
            match=r"(constituent IDs|source bar evidence)",
        ):
            inspect_projection(
                fixture, altered, empty_artifacts, tmp_path, boundary, replacements
            )


@pytest.mark.parametrize(
    "change",
    ["nonfinite", "impossible_ohlc", "timestamp", "extra_field", "duplicate", "order"],
)
@pytest.mark.parametrize(
    "boundary", ["dataset", "cache", "prediction", "feature", "feature_directory"]
)
def test_rehashed_source_evidence_requires_valid_typed_bars(
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
    if change == "nonfinite":
        observations[0]["high"] = "NaN"
    elif change == "impossible_ohlc":
        observations[0]["high"] = "99"
    elif change == "timestamp":
        observations[0]["start_timestamp"] = "not-a-timestamp"
    elif change == "extra_field":
        cast(list[PrimitiveMapping], evidence["templates"])[0]["uncommitted"] = True
    elif change == "duplicate":
        observations[1] = deepcopy(observations[0])
    else:
        observations[0], observations[1] = observations[1], observations[0]
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
    assert primitive["source_bar_evidence"] == evidence
    block_research(monkeypatch)
    with pytest.raises(
        (ValidationError, CacheError, ManifestError), match="source bar evidence"
    ):
        inspect_projection(
            fixture, altered, empty_artifacts, tmp_path, boundary, replacements
        )


@pytest.mark.parametrize(
    "change", ["open", "high", "low", "close", "volume", "same_ohlcv"]
)
@pytest.mark.parametrize(
    "boundary", ["dataset", "cache", "prediction", "feature", "feature_directory"]
)
def test_rehashed_source_prices_must_reproduce_the_retained_session_ohlcv(
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
    if change == "close":
        observations[-1]["close"] = "100.1"
    elif change == "same_ohlcv":
        # A changed interior open leaves the complete session's OHLCV unchanged.
        # Consistent new identities remain valid: hashes are not authenticity.
        observations[1]["open"] = "100.1"
    else:
        observations[0][change] = {
            "open": "100.1",
            "high": "101",
            "low": "99",
            "volume": "1001",
        }[change]
    primitive, replacements = _rehash_source_evidence(
        primitive, original.to_primitive()
    )
    evidence = cast(PrimitiveMapping, primitive["session_evidence"])
    batch = validate_source_bar_evidence(
        cast(PrimitiveMapping, primitive["source_bar_evidence"]),
        cast(PrimitiveMapping, primitive["source_manifest"]),
    )
    for entry in cast(
        list[PrimitiveMapping], cast(PrimitiveMapping, evidence["bars"])["bars"]
    ):
        bar = cast(PrimitiveMapping, entry["bar"])
        assert bar["source_bar_ids"] == [
            item.bar_id
            for item in batch.bars
            if item.session_identifier == cast(list[str], bar["session_dates"])[0]
        ]
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
    if change == "same_ohlcv":
        inspect_projection(
            fixture, altered, empty_artifacts, tmp_path, boundary, replacements
        )
    else:
        with pytest.raises(
            (ValidationError, CacheError, ManifestError), match="session OHLCV"
        ):
            inspect_projection(
                fixture, altered, empty_artifacts, tmp_path, boundary, replacements
            )
