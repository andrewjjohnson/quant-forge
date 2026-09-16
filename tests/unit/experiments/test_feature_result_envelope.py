"""A direct feature result must retain its complete producer envelope."""

from pathlib import Path

import pytest

from quantforge.experiments import (
    ArtifactType,
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.prediction import SignalFeatureDatasetResult
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_feature_integrity import (
    feature_result as feature_result,
)


@pytest.mark.parametrize("field", ["manifest", "rows", "schema", "summary"])
def test_direct_feature_result_requires_every_envelope_field(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult, field: str
) -> None:
    document = feature_result.to_primitive()
    del document[field]
    path = tmp_path / "feature.json"
    write_json(path, document)
    before = path.read_bytes()
    with pytest.raises(ManifestError):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["rows", "schema", "summary"])
def test_direct_feature_result_rejects_null_payloads(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult, field: str
) -> None:
    document = feature_result.to_primitive()
    document[field] = None
    path = tmp_path / "feature.json"
    write_json(path, document)
    before = path.read_bytes()
    with pytest.raises(ManifestError):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    assert path.read_bytes() == before


def test_complete_direct_feature_result_indexes_its_rows(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult
) -> None:
    path = tmp_path / "feature.json"
    write_json(path, feature_result.to_primitive())
    before = path.read_bytes()
    result = inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    assert any(
        entry.artifact_type is ArtifactType.FEATURE_DATASET
        and entry.json_pointer == "/rows"
        for entry in result.index.entries
    )
    assert verify_artifacts(result.index, tmp_path).valid
    assert path.read_bytes() == before
