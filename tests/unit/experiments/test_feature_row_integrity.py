from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import SignalDisposition, SignalFeatureDatasetResult
from quantforge.prediction.feature_dataset import (
    _row_id,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_feature_integrity import (
    feature_result as feature_result,
)
from tests.unit.helpers import make_dataset
from tests.unit.prediction.test_feature_dataset import (
    FixtureCandidateRule,
    _build_fixture,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _build,  # pyright: ignore[reportPrivateUsage]
    _dataset,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize(
    "field",
    [
        "study_id",
        "source_dataset_id",
        "dataset_fingerprint",
        "symbol",
        "provider_name",
        "adjustment_mode",
        "ohlc_basis",
        "volume_basis",
        "candidate_rule_id",
        "candidate_rule_configuration_id",
        "feature_schema_version",
        "outcome_schema_version",
        "prediction_study_ids",
        "strategy_parameters",
        "strategy_parameters_id",
        "candidate_id",
    ],
)
def test_rehashed_feature_row_keeps_its_dataset_provenance(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult, field: str
) -> None:
    result = feature_result.to_primitive()
    row = cast(list[PrimitiveMapping], result["rows"])[0]
    row[field] = (
        {"foreign": "0" * 64}
        if field in {"prediction_study_ids", "strategy_parameters"}
        else "0" * 64
    )
    row["row_id"] = _row_id(feature_result.dataset_id, row)
    path = tmp_path / "features.json"
    write_json(path, result)
    with pytest.raises(ManifestError, match=r"feature.*(provenance|identity)"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "missing_field",
        "extra_field",
        "field_type",
        "row_id",
        "schema",
        "study_references",
        "reason_codes",
        "selected_reason",
        "signal_session",
        "reorder",
    ],
)
def test_feature_row_contract_and_schema_are_validated_together(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult, change: str
) -> None:
    result = feature_result.to_primitive()
    rows = cast(list[PrimitiveMapping], result["rows"])
    row = rows[0]
    if change == "missing_field":
        row.pop("symbol")
    elif change == "extra_field":
        row["unexpected"] = True
    elif change == "field_type":
        field = next(
            name
            for name in row
            if name.startswith("feature_") and name != "feature_schema_version"
        )
        row[field] = []
    elif change == "row_id":
        row["row_id"] = "0" * 64
    elif change == "schema":
        schema = cast(PrimitiveMapping, result["schema"])
        cast(list[PrimitiveMapping], schema["fields"])[0]["data_type"] = "object"
    elif change == "study_references":
        cast(PrimitiveMapping, result["manifest"])["prediction_study_ids"] = ["0" * 64]
    elif change == "reason_codes":
        row["disposition_reason_codes"] = []
    elif change == "selected_reason":
        row["selected_rule_reason"] = None
    elif change == "signal_session":
        row["signal_session"] = "2024-07-04"
    else:
        if len(rows) == 1:
            rows.append(dict(row))
            counts = cast(
                PrimitiveMapping,
                cast(PrimitiveMapping, result["manifest"])["record_counts"],
            )
            counts["candidate_count"] = 2
            counts["accepted_count"] = 2
            result["summary"] = dict(counts)
        else:
            rows.reverse()
    if change != "row_id":
        row["row_id"] = _row_id(feature_result.dataset_id, row)
    path = tmp_path / "features.json"
    write_json(path, result)
    with pytest.raises(ManifestError):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize("contextual", [False, True])
def test_same_count_feature_rows_cannot_come_from_another_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contextual: bool
) -> None:
    if contextual:
        original = _build(tmp_path / "original")[0]
        dataset = _dataset()
        foreign_dataset = make_dataset(
            tuple(str(bar.close + 10) for bar in dataset.bars),
            sessions=tuple(bar.session_date for bar in dataset.bars),
        )
        foreign = _build(tmp_path / "foreign", dataset=foreign_dataset)[0]
    else:
        original = _build_fixture(
            make_dataset(("100", "101", "102", "103")),
            FixtureCandidateRule(tuple(SignalDisposition)),
            tmp_path / "original",
        )
        foreign = _build_fixture(
            make_dataset(("110", "111", "112", "113")),
            FixtureCandidateRule(tuple(SignalDisposition)),
            tmp_path / "foreign",
        )
    result = original.to_primitive()
    assert original.summary == foreign.summary
    result["rows"] = foreign.to_primitive()["rows"]
    block_research(monkeypatch)
    path = tmp_path / "features.json"
    write_json(path, result)
    with pytest.raises(ManifestError, match="feature row provenance"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
