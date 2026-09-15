import csv
import json
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import ManifestError, StudyType, inspect_study
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    FixtureTrialAnalyzer,
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.experiments.test_grid_integrity import (
    grid_export as grid_export,
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
        if study_type is StudyType.OPTIMIZATION:
            # Keep the valid retry fixture's CSV projections consistent with JSON.
            for filename in ("trials.csv", "failures.csv", "exclusions.csv"):
                csv_path = root / filename
                with csv_path.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    headers = reader.fieldnames
                    rows = list(reader)
                assert headers is not None
                for row in rows:
                    if row["trial_id"] == trial["trial_id"]:
                        row["failed_attempts"] = json.dumps(
                            [attempt], sort_keys=True, separators=(",", ":")
                        )
                with csv_path.open("w", newline="") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=headers, lineterminator="\n"
                    )
                    writer.writeheader()
                    writer.writerows(rows)
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


@pytest.mark.parametrize("complete", [False, True], ids=["resumable", "completed"])
@pytest.mark.parametrize("field", ["started_at", "finished_at"])
@pytest.mark.parametrize("invalid", ["missing", None, "", "  ", 123, True])
def test_failed_prediction_trial_requires_retry_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
    field: str,
    invalid: Primitive,
) -> None:
    root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    if not complete:
        (root / "summary.json").unlink()
    path = trial_path(root, "failed")
    trial = read_record(path)
    assert isinstance(trial[field], str)
    assert trial[field]
    if invalid == "missing":
        del trial[field]
    else:
        trial[field] = invalid
    write_json(path, trial)
    before = path.read_bytes()
    with pytest.raises(ManifestError, match="retry timestamps"):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("complete", [False, True], ids=["resumable", "completed"])
def test_valid_failed_prediction_trial_remains_retryable_after_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    from tests.unit.prediction.test_prediction_grid import (
        _grid,  # pyright: ignore[reportPrivateUsage]
    )

    grid = _grid(
        tmp_path / "prediction", analyzer=FixtureTrialAnalyzer(), retry_failed=True
    )
    result = grid.run()
    root = tmp_path / "prediction" / result.study_id
    if not complete:
        (root / "summary.json").unlink()
    path = trial_path(root, "failed")
    failed = read_record(path)
    before = path.read_bytes()
    with monkeypatch.context() as inspection:
        block_research(inspection)
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert path.read_bytes() == before
    grid.resume()
    retried = read_record(path)
    assert retried["status"] == "succeeded"
    attempt = cast(list[PrimitiveMapping], retried["failed_attempts"])[0]
    assert {name: attempt[name] for name in ("started_at", "finished_at")} == {
        name: failed[name] for name in ("started_at", "finished_at")
    }
    block_research(monkeypatch)
    inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
