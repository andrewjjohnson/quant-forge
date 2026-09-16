"""Refresh outer feature identities without masking contradictory outcome IDs."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.prediction import SignalFeatureDatasetResult, SignalFeatureRow
from quantforge.prediction.feature_dataset import (
    _finalize_export,  # pyright: ignore[reportPrivateUsage]
    _row_id,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_feature_integrity import (
    feature_result as feature_result,
)


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
@pytest.mark.parametrize(
    "change",
    [
        "wrong_id",
        "missing_id",
        "null_id",
        "component",
        "missing_component",
        "non_mapping_component",
    ],
)
def test_feature_outcome_identity_must_match_saved_configuration(
    tmp_path: Path,
    feature_result: SignalFeatureDatasetResult,
    directory: bool,
    change: str,
) -> None:
    configuration = feature_result.configuration
    outcomes = cast(list[PrimitiveMapping], configuration["outcomes"])
    outcome = outcomes[-1]
    assert outcome["configuration_id"] == configuration_identity(
        cast(PrimitiveMapping, outcome["component_configuration"])
    )
    if change == "wrong_id":
        outcome["configuration_id"] = "0" * 64
    elif change == "missing_id":
        del outcome["configuration_id"]
    elif change == "null_id":
        outcome["configuration_id"] = None
    elif change == "component":
        cast(PrimitiveMapping, outcome["component_configuration"])["changed"] = True
    elif change == "missing_component":
        del outcome["component_configuration"]
    else:
        outcome["component_configuration"] = True
    dataset_id = configuration_identity(configuration)
    rows: list[SignalFeatureRow] = []
    for original in feature_result.rows:
        row = original.to_primitive()
        row["study_id"] = dataset_id
        row["row_id"] = _row_id(dataset_id, row)
        rows.append(SignalFeatureRow.capture(row))
    altered = replace(
        feature_result,
        dataset_id=dataset_id,
        configuration_snapshot=PrimitiveMappingSnapshot.capture(configuration),
        rows=tuple(rows),
    )
    if directory:
        path = tmp_path / "altered"
        (path / "rows").mkdir(parents=True)
        for row in altered.rows:
            write_json(path / "rows" / f"{row.row_id}.json", row.to_primitive())
        _finalize_export(path, altered)
        paths = [item for item in path.rglob("*") if item.is_file()]
    else:
        path = tmp_path / "altered.json"
        write_json(path, altered.to_primitive())
        paths = [path]
    before = {item: item.read_bytes() for item in paths}
    with pytest.raises(ManifestError, match="feature outcome configuration"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    assert {item: item.read_bytes() for item in paths} == before


@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
@pytest.mark.parametrize("duplicate", [False, True], ids=["unique", "duplicate"])
def test_feature_outcomes_require_unique_namespaces_even_with_disjoint_fields(
    tmp_path: Path,
    feature_result: SignalFeatureDatasetResult,
    directory: bool,
    duplicate: bool,
) -> None:
    configuration = feature_result.configuration
    outcomes = cast(list[PrimitiveMapping], configuration["outcomes"])
    original = outcomes[-1]
    namespace = str(original["namespace"])
    added_namespace = namespace if duplicate else namespace + "_copy"
    added = deepcopy(original)
    added["namespace"] = added_namespace
    field = deepcopy(cast(list[PrimitiveMapping], original["fields"])[0])
    original_name = "outcome_" + namespace + "_" + str(field["field_name"])
    field["field_name"] = str(field["field_name"]) + "_copy"
    added_name = "outcome_" + added_namespace + "_" + str(field["field_name"])
    added["fields"] = [field]
    outcomes.append(added)
    original_field = next(
        field for field in feature_result.schema.fields if field.name == original_name
    )
    schema = replace(
        feature_result.schema,
        fields=(
            *feature_result.schema.fields,
            replace(original_field, name=added_name),
        ),
    )
    dataset_id = configuration_identity(configuration)
    rows: list[SignalFeatureRow] = []
    for original_row in feature_result.rows:
        row = original_row.to_primitive()
        row["study_id"] = dataset_id
        row[added_name] = row[original_name]
        studies = cast(PrimitiveMapping, row["prediction_study_ids"])
        studies[added_namespace] = studies[namespace]
        row["row_id"] = _row_id(dataset_id, row)
        rows.append(SignalFeatureRow.capture(row))
    altered = replace(
        feature_result,
        dataset_id=dataset_id,
        configuration_snapshot=PrimitiveMappingSnapshot.capture(configuration),
        schema=schema,
        rows=tuple(rows),
        prediction_study_ids=(
            *feature_result.prediction_study_ids,
            feature_result.prediction_study_ids[-1],
        ),
    )
    if directory:
        path = tmp_path / "altered_namespaces"
        (path / "rows").mkdir(parents=True)
        for row in altered.rows:
            write_json(path / "rows" / f"{row.row_id}.json", row.to_primitive())
        _finalize_export(path, altered)
        paths = [item for item in path.rglob("*") if item.is_file()]
    else:
        path = tmp_path / "altered_namespaces.json"
        write_json(path, altered.to_primitive())
        paths = [path]
    before = {item: item.read_bytes() for item in paths}
    if duplicate:
        with pytest.raises(ManifestError, match="outcome namespaces must be unique"):
            inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    else:
        inspected = inspect_study(
            StudyType.FEATURE_DATASET, path, artifact_root=tmp_path
        )
        assert verify_artifacts(inspected.index, tmp_path).valid
    assert {item: item.read_bytes() for item in paths} == before
