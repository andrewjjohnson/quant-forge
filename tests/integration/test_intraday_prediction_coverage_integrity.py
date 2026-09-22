"""Coverage-report evidence survives rehashing across cache and QF-9 readers."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from shutil import copy2, copytree
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import (
    DatasetFamily,
    MarketDataCache,
    dataset_identity_matches,
    validate_market_dataset,
)
from quantforge.data.exceptions import CacheError, ValidationError
from quantforge.data.identity import serialize_metadata_values
from quantforge.data.intraday_coverage_evidence import validate_retained_coverage_report
from quantforge.data.intraday_ingestion import (
    IntradayMarketDataCache,
    validate_intraday_manifest_identity,
)
from quantforge.data.models import IntradayPredictionProvenance
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import SignalFeatureDatasetResult, SignalFeatureRow
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
    _rehash_nested,  # pyright: ignore[reportPrivateUsage]
    _rehash_source_evidence,  # pyright: ignore[reportPrivateUsage]
    _replace_ids,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json

CHANGES = (
    "report_id",
    "stale_hash",
    "request_id",
    "batch_id",
    "timeframe_configuration_id",
    "requested_start_timestamp",
    "requested_end_timestamp",
    "source_interval",
    "feed_scope",
    "session_scope",
    "observed_bar_count",
    "expected_completed_interval_count",
    "validation_mode",
    "schema_version",
    "unknown_field",
    "false_complete",
    "incomplete",
    "missing_summary",
    "session_count",
    "session_open",
    "full_session_flag",
    "session_missing",
    "session_duplicate",
    "session_order",
    "warning_flag",
    "zero_volume_unknown",
)


def _changed_coverage(
    original: PrimitiveMapping, fixture: Fixture, change: str
) -> tuple[PrimitiveMapping, dict[str, str]]:
    provenance = deepcopy(original)
    manifest = cast(PrimitiveMapping, provenance["source_manifest"])
    quality = cast(PrimitiveMapping, manifest["quality_report"])
    report = cast(PrimitiveMapping, quality["report"])
    sessions = cast(list[PrimitiveMapping], report["sessions"])
    first = sessions[0]
    bar = fixture.source.bars[0]
    interval: PrimitiveMapping = {
        "session_date": bar.session_date.isoformat(),
        "start_timestamp": bar.start_timestamp.isoformat(),
        "end_timestamp": bar.end_timestamp.isoformat(),
        "completion": bar.completion.value,
    }
    if change == "report_id":
        quality["report_id"] = "0" * 64
    elif change in {
        "stale_hash",
        "observed_bar_count",
        "expected_completed_interval_count",
    }:
        key = "observed_bar_count" if change == "stale_hash" else change
        report[key] = cast(int, report[key]) + 1
    elif change in {
        "request_id",
        "batch_id",
        "timeframe_configuration_id",
        "schema_version",
    }:
        report[change] = "different"
    elif change in {"requested_start_timestamp", "requested_end_timestamp"}:
        report[change] = "2024-01-01T00:00:00+00:00"
    elif change == "source_interval":
        cast(PrimitiveMapping, report[change])["nominal_duration_microseconds"] = (
            120_000_000
        )
    elif change == "feed_scope":
        cast(PrimitiveMapping, report[change])["coverage"] = "unknown"
    elif change == "session_scope":
        report[change] = "extended_hours"
    elif change == "validation_mode":
        report[change] = "strict"
    elif change == "unknown_field":
        report[change] = True
    elif change in {
        "false_complete",
        "incomplete",
        "missing_summary",
        "zero_volume_unknown",
    }:
        first["missing_intervals"] = [interval]
        first["observed_completed_interval_count"] = (
            cast(int, first["observed_completed_interval_count"]) - 1
        )
        first["is_complete"] = False
        report["missing_intervals"] = [interval]
        report["observed_bar_count"] = cast(int, report["observed_bar_count"]) - 1
        manifest["bar_count"] = report["observed_bar_count"]
        report["incomplete_sessions"] = [first["session_date"]]
        report["status"] = "incomplete"
        report["is_complete"] = False
        if change == "false_complete":
            report["status"] = "complete"
            report["is_complete"] = True
        elif change == "missing_summary":
            report["missing_intervals"] = []
        elif change == "zero_volume_unknown":
            first["zero_volume_intervals"] = [interval]
            report["zero_volume_intervals"] = [interval]
            report["has_warnings"] = True
    elif change == "session_count":
        first["expected_interval_count"] = 1
    elif change == "session_open":
        first["session_open_timestamp"] = "2024-01-02T15:00:00+00:00"
    elif change == "full_session_flag":
        first["request_covers_full_session"] = False
    elif change == "session_missing":
        sessions.pop()
    elif change == "session_duplicate":
        sessions.append(deepcopy(first))
    elif change == "session_order":
        sessions.reverse()
    elif change == "warning_flag":
        report["has_warnings"] = True
    if change not in {"report_id", "stale_hash"}:
        quality["report_id"] = configuration_identity(report)
    return _rehash_source_evidence(provenance, original)


def test_retained_report_matches_the_cache_report(fixture: Fixture) -> None:
    manifest = IntradayMarketDataCache(fixture.cache.root).read_manifest(
        fixture.source.metadata.dataset_id
    )
    assert validate_retained_coverage_report(manifest) == fixture.source.quality_report


@pytest.mark.parametrize("change", CHANGES)
def test_rehashed_projection_rejects_invalid_coverage(
    fixture: Fixture, change: str
) -> None:
    original = fixture.dataset.metadata.intraday_provenance
    assert original is not None
    provenance, _ = _changed_coverage(original.to_primitive(), fixture, change)
    # Both enclosing source and family hashes are coherent; coverage must still
    # be checked against its report, request, and calendar rather than trusted.
    DatasetFamily.from_manifest(cast(PrimitiveMapping, provenance["family_manifest"]))
    if change == "incomplete":
        source = cast(PrimitiveMapping, provenance["source_manifest"])
        validate_intraday_manifest_identity(source)
        assert not validate_retained_coverage_report(source).is_complete
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
    with pytest.raises(ValidationError, match="coverage report"):
        validate_market_dataset(altered)


@pytest.mark.parametrize("change", ["report_id", "request_id", "false_complete"])
def test_projection_cache_rejects_rehashed_coverage(
    fixture: Fixture, tmp_path: Path, change: str
) -> None:
    original = fixture.dataset.metadata.intraday_provenance
    assert original is not None
    provenance, _ = _changed_coverage(original.to_primitive(), fixture, change)
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
    with pytest.raises(CacheError, match="coverage report"):
        cache.load(altered.metadata.dataset_id)


@pytest.mark.parametrize(
    "change", ["report_id", "request_id", "false_complete", "incomplete"]
)
def test_prediction_rejects_rehashed_coverage(
    prediction_manifest: PrimitiveMapping,
    fixture: Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    manifest = deepcopy(prediction_manifest)
    market = cast(PrimitiveMapping, manifest["market_data"])
    provenance, replacements = _changed_coverage(
        cast(PrimitiveMapping, market["intraday_provenance"]), fixture, change
    )
    manifest = cast(PrimitiveMapping, _replace_ids(manifest, replacements))
    cast(PrimitiveMapping, manifest["market_data"])["intraday_provenance"] = provenance
    _rehash_nested(manifest)
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="coverage report"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
@pytest.mark.parametrize(
    "change", ["report_id", "request_id", "false_complete", "incomplete"]
)
def test_feature_rejects_rehashed_coverage(
    feature_result: SignalFeatureDatasetResult,
    fixture: Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    directory: bool,
) -> None:
    configuration = feature_result.configuration
    market = cast(PrimitiveMapping, configuration["source_data"])
    provenance, replacements = _changed_coverage(
        cast(PrimitiveMapping, market["intraday_provenance"]), fixture, change
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
    with pytest.raises(ManifestError, match="coverage report"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
