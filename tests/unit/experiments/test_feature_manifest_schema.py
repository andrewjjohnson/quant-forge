"""QF-7/QF-29 indexing requires complete producer research declarations."""

from pathlib import Path

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.experiments._json import mapping
from quantforge.prediction import SignalFeatureDatasetResult
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_feature_integrity import (
    feature_result as feature_result,
)


@pytest.fixture(params=[False, True], ids=["direct", "directory"])
def directory(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


def inspect_manifest(
    tmp_path: Path,
    result: SignalFeatureDatasetResult,
    document: PrimitiveMapping,
    *,
    directory: bool,
    reject: bool,
) -> None:
    source = (
        tmp_path / "features" / result.dataset_id
        if directory
        else tmp_path / "feature.json"
    )
    write_json(
        source / "manifest.json" if directory else source,
        mapping(document["manifest"]) if directory else document,
    )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    if reject:
        with pytest.raises(ManifestError):
            inspect_study(StudyType.FEATURE_DATASET, source, artifact_root=tmp_path)
    else:
        bundle = inspect_study(
            StudyType.FEATURE_DATASET, source, artifact_root=tmp_path
        )
        assert bundle.provenance.producer_study_id == result.dataset_id
        assert verify_artifacts(bundle.index, tmp_path).valid
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "field",
    [
        "component",
        "configuration",
        "dataset_id",
        "engine_version",
        "feature_outcome_boundary",
        "limitations",
        "market_data",
        "prediction_study_ids",
        "record_counts",
        "status",
        "undeclared",
    ],
)
def test_feature_manifest_requires_exact_producer_fields(
    tmp_path: Path,
    feature_result: SignalFeatureDatasetResult,
    directory: bool,
    field: str,
) -> None:
    document = feature_result.to_primitive()
    manifest = mapping(document["manifest"])
    if field == "undeclared":
        manifest[field] = "extra"
    else:
        del manifest[field]
    inspect_manifest(
        tmp_path, feature_result, document, directory=directory, reject=True
    )


@pytest.mark.parametrize(
    "invalid",
    [None, "text", {}, False, 1, [None], [False], [1], [{}], [[]], ["valid", False]],
)
def test_feature_limitations_require_an_array_of_strings(
    tmp_path: Path,
    feature_result: SignalFeatureDatasetResult,
    directory: bool,
    invalid: Primitive,
) -> None:
    document = feature_result.to_primitive()
    mapping(document["manifest"])["limitations"] = invalid
    inspect_manifest(
        tmp_path, feature_result, document, directory=directory, reject=True
    )


@pytest.mark.parametrize("invalid", [None, "", False, 1, [], {}, "outcomes first"])
def test_feature_boundary_requires_the_exact_producer_declaration(
    tmp_path: Path,
    feature_result: SignalFeatureDatasetResult,
    directory: bool,
    invalid: Primitive,
) -> None:
    document = feature_result.to_primitive()
    mapping(document["manifest"])["feature_outcome_boundary"] = invalid
    inspect_manifest(
        tmp_path, feature_result, document, directory=directory, reject=True
    )


@pytest.mark.parametrize("limitations", [None, [], ["second", "", "first", "second"]])
def test_valid_feature_disclosures_are_preserved_without_research(
    tmp_path: Path,
    feature_result: SignalFeatureDatasetResult,
    directory: bool,
    limitations: list[str] | None,
) -> None:
    document = feature_result.to_primitive()
    if limitations is not None:
        mapping(document["manifest"])["limitations"] = list(limitations)
    inspect_manifest(
        tmp_path, feature_result, document, directory=directory, reject=False
    )
