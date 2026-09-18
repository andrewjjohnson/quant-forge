"""Real producer baselines, copied before integrity tests corrupt their artifacts."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from quantforge.oos import OOSSource, load_oos_source
from tests.unit.oos.conftest import complete_study


@dataclass(frozen=True)
class CapturedStudy:
    study_path: Path
    source: OOSSource


@pytest.fixture(scope="module")
def study_baselines(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[bool, CapturedStudy]:
    baselines: dict[bool, CapturedStudy] = {}
    for prediction in (False, True):
        completed = complete_study(
            tmp_path_factory.mktemp("study-baseline"), prediction=prediction
        )
        baselines[prediction] = CapturedStudy(
            completed.study.study_path, completed.source
        )
    return baselines


def copy_study(baseline: CapturedStudy, root: Path) -> CapturedStudy:
    """Copy bytes, retaining producer identities and independently loaded records."""
    study_path = root / "walk-forward" / baseline.source.study_id
    shutil.copytree(baseline.study_path, study_path)
    source = load_oos_source(baseline.source.plan, study_path)
    assert source == baseline.source
    return CapturedStudy(study_path, source)
