"""Rehashed source evidence must still describe possible request coverage."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data import (
    DatasetFamily,
    dataset_identity_matches,
    validate_market_dataset,
)
from quantforge.data.exceptions import ValidationError
from quantforge.data.models import IntradayPredictionProvenance
from quantforge.data.prediction_inputs import validate_prediction_provenance
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
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json

INVALID_BOUNDS = (
    "missing_start",
    "missing_end",
    "malformed",
    "naive",
    "noncanonical",
    "overflow",
    "reversed",
    "empty",
    "outside",
    "late_start",
    "early_end",
    "chunk_start",
    "chunk_end",
    "chunk_reversed",
    "chunk_naive",
    "chunk_gap",
    "chunk_overlap",
)


def _replace_ids(record: Primitive, replacements: dict[str, str]) -> Primitive:
    if isinstance(record, dict):
        return {key: _replace_ids(value, replacements) for key, value in record.items()}
    if isinstance(record, list):
        return [_replace_ids(value, replacements) for value in record]
    return replacements.get(record, record) if isinstance(record, str) else record


def _alter_bounds(
    original: PrimitiveMapping, change: str
) -> tuple[PrimitiveMapping, dict[str, str]]:
    provenance = deepcopy(original)
    manifest = cast(PrimitiveMapping, provenance["source_manifest"])
    request = cast(PrimitiveMapping, manifest["request"])
    configuration = cast(PrimitiveMapping, request["configuration"])
    chunks = cast(list[PrimitiveMapping], manifest["chunks"])
    if change.startswith("missing_"):
        del configuration[f"{change.removeprefix('missing_')}_timestamp"]
    elif change in {"malformed", "naive", "noncanonical", "overflow"}:
        configuration["start_timestamp"] = {
            "malformed": "invalid",
            "naive": "2024-01-02T14:30:00",
            "noncanonical": "2024-01-02T09:30:00-05:00",
            "overflow": "0001-01-01T00:00:00+01:00",
        }[change]
    elif change == "reversed":
        configuration["start_timestamp"], configuration["end_timestamp"] = (
            configuration["end_timestamp"],
            configuration["start_timestamp"],
        )
    elif change == "empty":
        configuration["end_timestamp"] = configuration["start_timestamp"]
    elif change == "outside":
        configuration["start_timestamp"] = "2024-02-01T14:30:00+00:00"
        configuration["end_timestamp"] = "2024-02-02T21:00:00+00:00"
    elif change == "late_start":
        configuration["start_timestamp"] = "2024-01-02T14:31:00+00:00"
    elif change == "early_end":
        configuration["end_timestamp"] = "2024-01-03T20:59:00+00:00"
    elif change == "enclosing":
        configuration["start_timestamp"] = "2024-01-02T00:00:00+00:00"
        configuration["end_timestamp"] = "2024-01-04T00:00:00+00:00"
    # Keep raw-chunk bounds aligned when only the request is changed, ensuring
    # the projection coverage check cannot rely on an unrelated chunk mismatch.
    chunks[0]["chunk_start_timestamp"] = configuration.get("start_timestamp")
    chunks[-1]["chunk_end_timestamp"] = configuration.get("end_timestamp")
    if change in {"chunk_start", "chunk_end", "chunk_reversed", "chunk_naive"}:
        key = (
            "chunk_end_timestamp" if change == "chunk_end" else "chunk_start_timestamp"
        )
        chunks[0][key] = {
            "chunk_start": "2024-01-02T14:31:00+00:00",
            "chunk_end": "2024-01-03T20:59:00+00:00",
            "chunk_reversed": "2024-01-04T00:00:00+00:00",
            "chunk_naive": "2024-01-02T14:30:00",
        }[change]
    elif change in {"chunk_gap", "chunk_overlap", "contiguous"}:
        second = deepcopy(chunks[0])
        chunks[0]["chunk_end_timestamp"] = "2024-01-03T00:00:00+00:00"
        second.update(
            chunk_index=1,
            chunk_start_timestamp={
                "chunk_gap": "2024-01-03T00:01:00+00:00",
                "chunk_overlap": "2024-01-02T23:59:00+00:00",
                "contiguous": "2024-01-03T00:00:00+00:00",
            }[change],
            raw_snapshot_id="1" * 64,
            raw_sha256="1" * 64,
            raw_location=f"intraday/raw/{'1' * 64}.json",
        )
        chunks.append(second)
    request["request_id"] = configuration_identity(configuration)
    provenance["source_request_id"] = request["request_id"]
    provenance["source_raw_snapshot_ids"] = [
        chunk["raw_snapshot_id"] for chunk in chunks
    ]
    quality = cast(PrimitiveMapping, manifest["quality_report"])
    report = cast(PrimitiveMapping, quality["report"])
    report.update(
        request_id=request["request_id"],
        requested_start_timestamp=configuration.get("start_timestamp"),
        requested_end_timestamp=configuration.get("end_timestamp"),
    )
    quality["report_id"] = configuration_identity(report)
    return _rehash_source_evidence(provenance, original)


def _rehash_session_evidence(
    provenance: PrimitiveMapping,
    original: PrimitiveMapping,
    replacements: dict[str, str],
) -> None:
    """Rebind retained QF-19 evidence when a test changes its canonical source."""
    source = cast(PrimitiveMapping, provenance["source_manifest"])
    evidence = cast(
        PrimitiveMapping, _replace_ids(original["session_evidence"], replacements)
    )
    manifest = cast(PrimitiveMapping, evidence["manifest"])
    serialized = cast(PrimitiveMapping, evidence["bars"])
    old_session_id = cast(str, manifest["dataset_id"])
    old_digest = cast(str, manifest["data_sha256"])
    for entry in cast(list[PrimitiveMapping], serialized["bars"]):
        previous = cast(str, entry["bar_id"])
        entry["bar_id"] = configuration_identity(cast(PrimitiveMapping, entry["bar"]))
        replacements[previous] = entry["bar_id"]
    manifest = cast(PrimitiveMapping, _replace_ids(manifest, replacements))
    quality = cast(PrimitiveMapping, source["quality_report"])
    manifest["source_dataset"] = {
        "dataset_id": source["dataset_id"],
        "request_id": cast(PrimitiveMapping, source["request"])["request_id"],
        "batch_id": source["batch_id"],
        "data_sha256": source["data_sha256"],
        "raw_snapshot_ids": [
            chunk["raw_snapshot_id"]
            for chunk in cast(list[PrimitiveMapping], source["chunks"])
        ],
        "quality_report": deepcopy(quality),
        "provider_name": source["provider_name"],
        "provider_symbol": source["provider_symbol"],
    }
    aggregation = cast(PrimitiveMapping, manifest["aggregation_report"])
    report = cast(PrimitiveMapping, aggregation["report"])
    report["source_quality_report_id"] = quality["report_id"]
    report["source_batch_id"] = source["batch_id"]
    aggregation["report_id"] = configuration_identity(report)
    family = cast(PrimitiveMapping, manifest["dataset_family"])
    old_family_id = cast(str, family["family_id"])
    old_manifest_id = cast(str, family["manifest_id"])
    family["family_id"] = configuration_identity(
        {
            key: value
            for key, value in family.items()
            if key not in {"family_id", "manifest_id", "lineage"}
        }
    )
    cast(PrimitiveMapping, manifest["source_dataset_family"])["family_id"] = family[
        "family_id"
    ]
    manifest["data_sha256"] = configuration_identity(serialized)
    session_id = configuration_identity(
        {
            key: value
            for key, value in manifest.items()
            if key
            not in {
                "dataset_id",
                "normalized_location",
                "manifest_location",
                "dataset_family",
            }
        }
    )
    replacements[old_session_id] = session_id
    replacements[old_digest] = manifest["data_sha256"]
    family = cast(PrimitiveMapping, _replace_ids(family, {old_session_id: session_id}))
    cast(list[PrimitiveMapping], family["lineage"]).sort(
        key=lambda entry: cast(str, entry["dataset_id"])
    )
    family["manifest_id"] = configuration_identity(
        {key: value for key, value in family.items() if key != "manifest_id"}
    )
    replacements[old_family_id] = cast(str, family["family_id"])
    replacements[old_manifest_id] = family["manifest_id"]
    manifest.update(
        dataset_id=session_id,
        normalized_location=f"session/derived/{session_id}/bars.json",
        manifest_location=f"session/derived/{session_id}/manifest.json",
        dataset_family=family,
    )
    provenance["session_evidence"] = {"manifest": manifest, "bars": serialized}
    provenance["session_dataset_id"] = session_id


def _rehash_source_evidence(
    provenance: PrimitiveMapping, original: PrimitiveMapping
) -> tuple[PrimitiveMapping, dict[str, str]]:
    manifest = cast(PrimitiveMapping, provenance["source_manifest"])
    request = cast(PrimitiveMapping, manifest["request"])
    chunks = cast(list[PrimitiveMapping], manifest["chunks"])
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"dataset_id", "normalized_location"}
    }
    identity["chunks"] = [
        {key: value for key, value in chunk.items() if key != "raw_location"}
        for chunk in chunks
    ]
    source_id = configuration_identity(identity)
    manifest["dataset_id"] = source_id
    manifest["normalized_location"] = f"intraday/datasets/{source_id}/bars.json"
    provenance["source_dataset_id"] = source_id
    assert manifest["dataset_id"] == configuration_identity(identity)
    replacements = {
        cast(str, original["source_dataset_id"]): source_id,
        cast(str, original["source_request_id"]): cast(str, request["request_id"]),
    }
    _rehash_session_evidence(provenance, original, replacements)
    original_family = cast(PrimitiveMapping, original["family_manifest"])
    family = cast(PrimitiveMapping, _replace_ids(original_family, replacements))
    _rehash_nested(family)
    family["family_id"] = configuration_identity(
        {
            key: value
            for key, value in family.items()
            if key not in {"family_id", "manifest_id", "lineage"}
        }
    )
    # Source ID changes can change canonical ordering of the lineage entries.
    for entry in cast(list[PrimitiveMapping], family["lineage"]):
        cast(list[str], entry["child_dataset_ids"]).sort()
    cast(list[PrimitiveMapping], family["lineage"]).sort(
        key=lambda entry: cast(str, entry["dataset_id"])
    )
    family["manifest_id"] = configuration_identity(
        {key: value for key, value in family.items() if key != "manifest_id"}
    )
    assert DatasetFamily.from_manifest(family).to_manifest() == family
    replacements.update(
        {
            cast(str, original["family_id"]): family["family_id"],
            cast(str, original_family["manifest_id"]): family["manifest_id"],
        }
    )
    provenance["family_manifest"] = family
    provenance["family_id"] = family["family_id"]
    return provenance, replacements


def _rehash_nested(record: Primitive) -> None:
    if isinstance(record, list):
        for item in record:
            _rehash_nested(item)
    elif isinstance(record, dict):
        for value in record.values():
            _rehash_nested(value)
        for definition_key in ("configuration", "component_configuration"):
            if "configuration_id" in record and isinstance(
                record.get(definition_key), dict
            ):
                record["configuration_id"] = configuration_identity(
                    cast(PrimitiveMapping, record[definition_key])
                )
        if "context_id" in record:
            record["context_id"] = configuration_identity(
                {key: value for key, value in record.items() if key != "context_id"}
            )


@pytest.mark.parametrize("change", INVALID_BOUNDS)
def test_rehashed_dataset_rejects_invalid_request_bounds(
    fixture: Fixture, change: str
) -> None:
    original = fixture.dataset.metadata.intraday_provenance
    assert original is not None
    provenance, _ = _alter_bounds(original.to_primitive(), change)
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
    with pytest.raises(ValidationError, match="source request"):
        validate_market_dataset(altered)


@pytest.mark.parametrize(
    "change", ["reversed", "outside", "chunk_gap", "enclosing", "contiguous"]
)
def test_prediction_request_bounds_survive_rehashing(
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    manifest = deepcopy(prediction_manifest)
    market = cast(PrimitiveMapping, manifest["market_data"])
    provenance, replacements = _alter_bounds(
        cast(PrimitiveMapping, market["intraday_provenance"]), change
    )
    manifest = cast(PrimitiveMapping, _replace_ids(manifest, replacements))
    market = cast(PrimitiveMapping, manifest["market_data"])
    market["intraday_provenance"] = provenance
    _rehash_nested(manifest)
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    if change in {"enclosing", "contiguous"}:
        assert (
            inspect_study(
                StudyType.PREDICTION, path, artifact_root=tmp_path
            ).provenance.producer_study_id
            == manifest["study_id"]
        )
        assert validate_prediction_provenance(market) is not None
    else:
        with pytest.raises(ManifestError, match="source request"):
            inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
@pytest.mark.parametrize("change", ["reversed", "outside", "chunk_gap", "enclosing"])
def test_feature_request_bounds_survive_rehashing(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    directory: bool,
) -> None:
    configuration = feature_result.configuration
    market = cast(PrimitiveMapping, configuration["source_data"])
    provenance, replacements = _alter_bounds(
        cast(PrimitiveMapping, market["intraday_provenance"]), change
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
    if change == "enclosing":
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match="source request"):
            inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
