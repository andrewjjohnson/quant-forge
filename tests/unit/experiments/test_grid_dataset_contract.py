"""QF-6 dataset validation agrees with the producer's resume contract."""

from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.optimization import (
    GridSearchStudy,
    MovingAverageCrossoverFactory,
    StudyPersistenceError,
)
from quantforge.optimization.models import TrialRecord
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    read_record,
    selective_runner,
    trial_path,
)
from tests.unit.optimization.test_study import (
    _dataset,  # pyright: ignore[reportPrivateUsage]
    _study_config,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize("extra_field", [False, True], ids=["native", "extra-field"])
@pytest.mark.parametrize(
    ("status", "complete"),
    [
        ("pending", False),
        ("running", False),
        ("failed", False),
        ("failed", True),
        ("excluded", False),
        ("excluded", True),
    ],
)
def test_non_success_trial_dataset_matches_resume_contract_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    complete: bool,
    extra_field: bool,
) -> None:
    study = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        _study_config(tmp_path),
        backtest_runner=selective_runner,
    )
    study.run()
    root = study.study_path
    path = trial_path(root, "failed" if status in {"pending", "running"} else status)
    trial = read_record(path)
    if status in {"pending", "running"}:
        trial.update(
            status=status,
            failure_category=None,
            failure_type=None,
            failure_message=None,
            finished_at=None,
        )
        if status == "pending":
            trial["started_at"] = None
    if not complete:
        (root / "summary.json").unlink()
    dataset = cast(PrimitiveMapping, trial["dataset"])
    configuration = cast(
        PrimitiveMapping, read_record(root / "manifest.json")["identity_inputs"]
    )
    assert dataset == configuration["dataset"]
    if extra_field:
        dataset["unexpected_metadata"] = {"source": "foreign"}
    # The pure record deserializer preserves extra dataset fields; the native
    # resume path additionally compares the complete mapping to the study.
    assert TrialRecord.from_primitive(trial).dataset == dataset
    write_json(path, trial)
    before = {item: item.read_bytes() for item in root.rglob("*") if item.is_file()}
    with monkeypatch.context() as inspection:
        block_research(inspection)
        if extra_field:
            with pytest.raises(ManifestError, match=r"trial.*coordinates"):
                inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
        else:
            bundle = inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
            assert verify_artifacts(bundle.index, tmp_path).valid
            # Successful trials retain the enriched QF-5 market-data mapping.
            successful_dataset = cast(
                PrimitiveMapping, read_record(trial_path(root))["dataset"]
            )
            assert set(successful_dataset) > set(dataset)
    assert {item: item.read_bytes() for item in before} == before
    if extra_field:
        with pytest.raises(StudyPersistenceError, match="persisted trial contents"):
            study.resume()
    else:
        study.resume()
