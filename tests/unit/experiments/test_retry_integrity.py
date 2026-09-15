from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import ManifestError, StudyType, inspect_study
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    grid_export as grid_export,
)
from tests.unit.experiments.test_grid_integrity import (
    read_record,
    trial_path,
)


@pytest.mark.parametrize("status", ["succeeded", "failed", "excluded"])
@pytest.mark.parametrize(
    "change",
    [
        None,
        "string",
        "non_record",
        "missing_type",
        "missing_message",
        "wrong_type",
        "timestamp",
        "producer_field",
    ],
)
def test_archived_failures_follow_producer_schema_for_every_trial_status(
    tmp_path: Path, grid_export: tuple[StudyType, Path], status: str, change: str | None
) -> None:
    study_type, root = grid_export
    path = trial_path(root, status)
    trial = read_record(path)
    attempt: PrimitiveMapping = {
        "failure_type": "FixtureFailure",
        "failure_message": "fixture diagnostic",
        "started_at": "2024-07-01T12:00:00+00:00",
        "finished_at": "2024-07-01T12:01:00+00:00",
    }
    if study_type is StudyType.OPTIMIZATION:
        attempt.update(attempt_number=1, status="failed", failure_category="execution")
    trial["failed_attempts"] = [attempt]
    if change == "string":
        trial["failed_attempts"] = "corrupt history"
    elif change == "non_record":
        trial["failed_attempts"] = [123]
    elif change == "missing_type":
        attempt.pop("failure_type")
    elif change == "missing_message":
        attempt.pop("failure_message")
    elif change == "wrong_type":
        attempt["failure_type"] = 123
    elif change == "timestamp":
        attempt["finished_at"] = 123
    elif change == "producer_field":
        if study_type is StudyType.OPTIMIZATION:
            attempt["attempt_number"] = True
        else:
            attempt.pop("started_at")
    write_json(path, trial)
    if change is None:
        bundle = inspect_study(study_type, root, artifact_root=tmp_path)
        assert any(
            entry.path == path.relative_to(tmp_path).as_posix()
            for entry in bundle.index.entries
        )
        assert cast(list[PrimitiveMapping], read_record(path)["failed_attempts"]) == [
            attempt
        ]
    else:
        with pytest.raises(ManifestError):
            inspect_study(study_type, root, artifact_root=tmp_path)
