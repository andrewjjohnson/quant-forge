"""Price semantics and outcome graph identities stay bound to source evidence."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import (
    AdjustmentMode,
    AggregationPolicy,
    TimeframeBarSeries,
    aggregate_intraday_dataset,
    dataset_identity_matches,
    validate_market_dataset,
)
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import calculate_dataset_id
from quantforge.data.models import IntradayPredictionProvenance
from quantforge.data.prediction_inputs import (
    validate_prediction_provenance,
    validate_prediction_source,
)
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
from tests.integration.test_intraday_prediction_provenance import TWO_MINUTES, Fixture
from tests.integration.test_intraday_prediction_provenance import fixture as fixture
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json


def _change_basis(record: PrimitiveMapping) -> None:
    record.update(
        adjustment_mode="split_adjusted",
        ohlc_basis="split_adjusted",
        volume_basis="split_adjusted",
        adjusted_fields_used=True,
    )


def test_family_evidence_roundtrip_is_immutable(fixture: Fixture) -> None:
    provenance = fixture.dataset.metadata.intraday_provenance
    assert provenance is not None
    serialized = provenance.to_primitive()
    assert serialized["schema_version"] == "5"
    restored = IntradayPredictionProvenance.from_primitive(serialized)
    evidence = cast(PrimitiveMapping, serialized["family_manifest"])
    assert evidence["manifest_id"] == fixture.primary.dataset_family_manifest_id
    assert restored == provenance
    evidence["manifest_id"] = "0" * 64
    assert restored.to_primitive() == provenance.to_primitive()
    assert restored.to_primitive() != serialized


@pytest.mark.parametrize("change", ["basis", "manifest_id", "lineage", "family_id"])
def test_family_evidence_must_reproduce_its_identities(
    prediction_manifest: PrimitiveMapping,
    change: str,
) -> None:
    market = deepcopy(cast(PrimitiveMapping, prediction_manifest["market_data"]))
    provenance = cast(PrimitiveMapping, market["intraday_provenance"])
    evidence = cast(PrimitiveMapping, provenance["family_manifest"])
    if change == "basis":
        _change_basis(market)
        source = cast(PrimitiveMapping, evidence["canonical_source"])
        _change_basis(cast(PrimitiveMapping, source["adjustment_basis"]))
        # Even a self-consistent replacement manifest cannot retain the raw family ID.
        evidence["manifest_id"] = configuration_identity(
            {key: value for key, value in evidence.items() if key != "manifest_id"}
        )
    elif change == "lineage":
        cast(list[PrimitiveMapping], evidence["lineage"]).pop()
    else:
        evidence[change] = "0" * 64
    with pytest.raises(ValidationError, match="family manifest"):
        validate_prediction_provenance(market)


@pytest.mark.parametrize(
    "change",
    [
        "version1",
        "version2",
        "version3",
        "version4",
        "missing_evidence",
        "missing_source_evidence",
        "missing_session_evidence",
        "missing_source_bar_evidence",
    ],
)
def test_old_or_incomplete_intraday_provenance_is_rejected(
    prediction_manifest: PrimitiveMapping,
    change: str,
) -> None:
    market = deepcopy(cast(PrimitiveMapping, prediction_manifest["market_data"]))
    provenance = cast(PrimitiveMapping, market["intraday_provenance"])
    if change.startswith("version"):
        provenance["schema_version"] = change[-1]
    elif change == "missing_source_evidence":
        del provenance["source_manifest"]
    elif change == "missing_session_evidence":
        del provenance["session_evidence"]
    elif change == "missing_source_bar_evidence":
        del provenance["source_bar_evidence"]
    else:
        del provenance["family_manifest"]
    with pytest.raises(ValueError, match="provenance"):
        validate_prediction_provenance(market)


def test_producer_rejects_outcome_with_another_family_manifest(
    fixture: Fixture,
) -> None:
    derived = aggregate_intraday_dataset(fixture.source, TWO_MINUTES)
    provenance = fixture.dataset.metadata.intraday_provenance
    assert provenance is not None
    evidence = provenance.family_manifest.to_primitive()
    canonical_source = cast(PrimitiveMapping, evidence["canonical_source"])
    aggregation = cast(PrimitiveMapping, canonical_source["aggregation_policy"])
    policy = cast(PrimitiveMapping, aggregation["configuration"])
    family = replace(
        derived.dataset_family,
        aggregation_policy=AggregationPolicy(
            cast(str, policy["policy_name"]),
            cast(str, policy["policy_version"]),
            cast(PrimitiveMapping, policy["configuration"]),
        ),
    )
    source = TimeframeBarSeries.from_aggregated_intraday_dataset(derived, family=family)
    assert source.dataset_reference == fixture.primary.dataset_reference
    assert (
        source.dataset_family_manifest_id != fixture.primary.dataset_family_manifest_id
    )
    with pytest.raises(ValidationError, match="family manifest"):
        validate_prediction_source(fixture.dataset, source)


def test_rehashed_dataset_basis_must_match_source_family(fixture: Fixture) -> None:
    metadata = replace(
        fixture.dataset.metadata,
        adjustment_mode=AdjustmentMode.SPLIT_ADJUSTED,
        ohlc_basis="split_adjusted",
        volume_basis="split_adjusted",
        adjusted_fields_used=True,
    )
    values = asdict(metadata)
    for field in (
        "raw_location",
        "normalized_location",
        "raw_sha256",
        "data_sha256",
        "dataset_id",
        "schema_version",
        "corporate_actions_location",
    ):
        del values[field]
    dataset_id = calculate_dataset_id(
        values,
        raw_sha256=metadata.raw_sha256,
        data_sha256=metadata.data_sha256,
        schema_version=metadata.schema_version,
    )
    altered = replace(
        fixture.dataset,
        metadata=replace(
            metadata,
            dataset_id=dataset_id,
            normalized_location=f"datasets/{dataset_id}/bars.csv",
            corporate_actions_location=f"datasets/{dataset_id}/corporate_actions.json",
        ),
    )
    assert dataset_identity_matches(altered)
    with pytest.raises(ValidationError, match="adjustment basis"):
        validate_market_dataset(altered)


def test_rehashed_prediction_basis_must_match_source_family(
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = deepcopy(prediction_manifest)
    _change_basis(cast(PrimitiveMapping, manifest["market_data"]))
    context = cast(PrimitiveMapping, manifest["prediction_context"])
    _change_basis(cast(PrimitiveMapping, context["adjustment_basis"]))
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="adjustment basis"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_rehashed_feature_basis_must_match_source_family(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
) -> None:
    configuration = feature_result.configuration
    _change_basis(cast(PrimitiveMapping, configuration["source_data"]))
    context = cast(PrimitiveMapping, configuration["prediction_context"])
    _change_basis(cast(PrimitiveMapping, context["adjustment_basis"]))
    rows: list[SignalFeatureRow] = []
    for original in feature_result.rows:
        row = original.to_primitive()
        for name in ("adjustment_mode", "ohlc_basis", "volume_basis"):
            row[name] = "split_adjusted"
        rows.append(SignalFeatureRow.capture(row))
    altered = replace(
        feature_result,
        market_data=replace(
            feature_result.market_data,
            adjustment_mode="split_adjusted",
            ohlc_basis="split_adjusted",
            volume_basis="split_adjusted",
            adjusted_fields_used=True,
        ),
        rows=tuple(rows),
    )
    path = _write_rehashed_feature(altered, configuration, tmp_path, directory)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="adjustment basis"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize("missing", [False, True], ids=["changed", "missing"])
def test_prediction_outcome_requires_exact_family_manifest(
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: bool,
) -> None:
    manifest = deepcopy(prediction_manifest)
    configuration = cast(PrimitiveMapping, manifest["configuration"])
    labeler = cast(PrimitiveMapping, configuration["outcome_labeler"])
    source = cast(PrimitiveMapping, labeler["outcome_source"])
    if missing:
        del source["family_manifest_id"]
    else:
        source["family_manifest_id"] = "0" * 64
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="family manifest"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


def test_rehashed_family_graph_must_match_outcome_manifest(
    prediction_manifest: PrimitiveMapping,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = deepcopy(prediction_manifest)
    market = cast(PrimitiveMapping, manifest["market_data"])
    provenance = cast(PrimitiveMapping, market["intraday_provenance"])
    evidence = cast(PrimitiveMapping, provenance["family_manifest"])
    # A smaller valid graph retains the source's family ID, but not its manifest ID.
    configuration = cast(PrimitiveMapping, manifest["configuration"])
    labeler = cast(PrimitiveMapping, configuration["outcome_labeler"])
    source = cast(PrimitiveMapping, labeler["outcome_source"])
    reference = cast(PrimitiveMapping, source["source_reference"])
    removed_id = reference["dataset_id"]
    lineage = cast(list[PrimitiveMapping], evidence["lineage"])
    evidence["lineage"] = [
        entry for entry in lineage if entry["dataset_id"] != removed_id
    ]
    for entry in cast(list[PrimitiveMapping], evidence["lineage"]):
        entry["child_dataset_ids"] = [
            child
            for child in cast(list[str], entry["child_dataset_ids"])
            if child != removed_id
        ]
    evidence["manifest_id"] = configuration_identity(
        {key: value for key, value in evidence.items() if key != "manifest_id"}
    )
    _rehash_prediction(manifest)
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="family manifest"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
@pytest.mark.parametrize("source_name", ["template", "outcome"])
@pytest.mark.parametrize("missing", [False, True], ids=["changed", "missing"])
def test_feature_outcomes_require_exact_family_manifest(
    feature_result: SignalFeatureDatasetResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
    source_name: str,
    missing: bool,
) -> None:
    configuration = feature_result.configuration
    owner = configuration
    outcome: PrimitiveMapping | None = None
    if source_name == "outcome":
        outcome = cast(list[PrimitiveMapping], configuration["outcomes"])[0]
        owner = cast(PrimitiveMapping, outcome["component_configuration"])
    source = cast(PrimitiveMapping, owner["outcome_source"])
    if missing:
        del source["family_manifest_id"]
    else:
        source["family_manifest_id"] = "0" * 64
    if outcome is not None:
        outcome["configuration_id"] = configuration_identity(owner)
    path = _write_rehashed_feature(feature_result, configuration, tmp_path, directory)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="family manifest"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
