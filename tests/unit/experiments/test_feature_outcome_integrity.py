"""Refresh outer feature identities without masking contradictory outcome IDs."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.experiments import ManifestError, StudyType, inspect_study
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
