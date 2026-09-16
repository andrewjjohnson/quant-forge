from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import (
    AlwaysUpParameters,
    AlwaysUpPredictionStrategy,
    create_overnight_gap_prediction_study,
    run_prediction_study,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.helpers import make_dataset


@pytest.fixture
def prediction_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PrimitiveMapping, PrimitiveMapping]:
    study = create_overnight_gap_prediction_study(
        AlwaysUpPredictionStrategy(AlwaysUpParameters())
    )
    first = run_prediction_study(
        make_dataset(("100", "101", "102", "103", "104")), study
    ).to_primitive()
    second = run_prediction_study(
        make_dataset(("110", "111", "112", "113", "114")), study
    ).to_primitive()
    block_research(monkeypatch)
    return first, second


def test_same_count_foreign_prediction_rows_are_rejected(
    tmp_path: Path, prediction_pair: tuple[PrimitiveMapping, PrimitiveMapping]
) -> None:
    target, foreign = prediction_pair
    assert len(cast(list[Primitive], target["rows"])) == len(
        cast(list[Primitive], foreign["rows"])
    )
    target["rows"] = foreign["rows"]
    path = tmp_path / "prediction.json"
    write_json(path, target)
    with pytest.raises(ManifestError, match="row provenance"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "study_id",
        "dataset_id",
        "dataset_fingerprint",
        "row_id",
        "features",
        "prediction",
        "outcome",
        "evaluation",
    ],
)
def test_row_provenance_and_deterministic_payload_identities_are_checked(
    tmp_path: Path,
    prediction_pair: tuple[PrimitiveMapping, PrimitiveMapping],
    field: str,
) -> None:
    result, _ = prediction_pair
    row = cast(list[PrimitiveMapping], result["rows"])[0]
    if field in {"outcome", "evaluation"}:
        cast(PrimitiveMapping, cast(PrimitiveMapping, row[field])["values"])[
            "changed"
        ] = True
    elif field == "prediction":
        cast(PrimitiveMapping, cast(PrimitiveMapping, row[field])["values"])[
            "changed"
        ] = True
    elif field == "features":
        cast(PrimitiveMapping, row[field])["changed"] = True
    else:
        row[field] = "0" * 64
    path = tmp_path / "prediction.json"
    write_json(path, result)
    with pytest.raises(ManifestError, match=r"prediction (row|outcome|evaluation)"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "field", ["dataset_id", "dataset_fingerprint", "outcome_configuration_id"]
)
def test_rehashed_outcome_still_belongs_to_manifest(
    tmp_path: Path,
    prediction_pair: tuple[PrimitiveMapping, PrimitiveMapping],
    field: str,
) -> None:
    result, _ = prediction_pair
    row = cast(list[PrimitiveMapping], result["rows"])[0]
    outcome = cast(PrimitiveMapping, row["outcome"])
    outcome[field] = "0" * 64
    outcome["outcome_id"] = configuration_identity(
        {
            "record_type": "prediction_outcome",
            **{key: value for key, value in outcome.items() if key != "outcome_id"},
        }
    )
    path = tmp_path / "prediction.json"
    write_json(path, result)
    with pytest.raises(ManifestError, match="row provenance"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


def test_duplicate_row_cannot_replace_another_labeled_row(
    tmp_path: Path, prediction_pair: tuple[PrimitiveMapping, PrimitiveMapping]
) -> None:
    result, _ = prediction_pair
    rows = cast(list[PrimitiveMapping], result["rows"])
    rows[1] = rows[0]
    path = tmp_path / "prediction.json"
    write_json(path, result)
    with pytest.raises(ManifestError, match="duplicate prediction row"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
