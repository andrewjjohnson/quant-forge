"""Required nullable trial fields remain required in every producer status."""

from pathlib import Path

import pytest

from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.optimization.errors import StudyPersistenceError
from quantforge.optimization.models import TrialRecord
from quantforge.prediction.grid import (
    PredictionGridPersistenceError,
    PredictionGridTrialRecord,
)
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import grid_export as grid_export
from tests.unit.experiments.test_grid_integrity import (
    read_record,
    trial_path,
)


@pytest.mark.parametrize(
    "status", ["pending", "running", "succeeded", "failed", "excluded"]
)
def test_required_nullable_fields_cannot_be_omitted(
    tmp_path: Path, grid_export: tuple[StudyType, Path], status: str
) -> None:
    study_type, root = grid_export
    path = trial_path(root, "failed" if status in {"pending", "running"} else status)
    trial = read_record(path)
    if status in {"pending", "running"}:
        trial.update(
            status=status, failure_type=None, failure_message=None, finished_at=None
        )
        if study_type is StudyType.OPTIMIZATION:
            trial["failure_category"] = None
        (root / "summary.json").unlink()
    write_json(path, trial)
    inspect_study(study_type, root, artifact_root=tmp_path)
    reader = (
        TrialRecord.from_primitive
        if study_type is StudyType.OPTIMIZATION
        else PredictionGridTrialRecord.from_primitive
    )
    # QF-32 explicitly permits an absent fingerprint on legacy non-successes.
    required_nulls = [
        name
        for name, value in trial.items()
        if value is None and name != "artifact_fingerprint"
    ]
    assert required_nulls
    for name in required_nulls:
        missing = {key: value for key, value in trial.items() if key != name}
        with pytest.raises((StudyPersistenceError, PredictionGridPersistenceError)):
            reader(missing)
        write_json(path, missing)
        before = path.read_bytes()
        with pytest.raises(ManifestError, match="trial record"):
            inspect_study(study_type, root, artifact_root=tmp_path)
        assert path.read_bytes() == before
    write_json(path, trial)
    trial.pop("failed_attempts")
    if study_type is StudyType.PARAMETER_STUDY and status != "succeeded":
        trial.pop("artifact_fingerprint")
    write_json(path, trial)
    reader(trial)
    inspect_study(study_type, root, artifact_root=tmp_path)
