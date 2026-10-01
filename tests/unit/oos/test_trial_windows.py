"""QF-67 offline verification of frozen-fold QF-32 trial windows (QF-40 reader)."""

import json
import shutil
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.oos import (
    OOSIntegrityError,
    load_oos_source,
    load_prediction_trial_window,
)
from quantforge.validation import PartitionRole
from tests.unit.oos.conftest import CompletedStudy


def candidates(study: CompletedStudy) -> list[str]:
    adapter = cast(PrimitiveMapping, study.source.definition.to_primitive()["adapter"])
    universe = cast(PrimitiveMapping, adapter["universe"])
    return [
        cast(str, item["combination_id"])
        for item in cast(list[PrimitiveMapping], universe["candidates"])
    ]


def test_every_candidate_trial_of_every_frozen_fold_verifies(
    prediction_study: CompletedStudy,
) -> None:
    source, path = prediction_study.source, prediction_study.study.study_path
    for index, fold in enumerate(source.folds):
        assert fold.selection is not None
        frozen = fold.selection.snapshot.to_primitive()
        plan_fold = source.plan.folds[index]
        expected_role = (
            PartitionRole.SELECTION
            if plan_fold.selection is not None
            else PartitionRole.DEVELOPMENT
        )
        for combination in candidates(prediction_study):
            trial = load_prediction_trial_window(
                source, path, fold_id=fold.fold_id, combination_id=combination
            )
            assert trial.role is expected_role
            assert trial.fold_index == index
            assert trial.candidate.combination_id == combination
            assert trial.grid_study_id == frozen["selection_grid_study_id"]
            if (
                combination
                == cast(PrimitiveMapping, frozen["candidate"])["combination_id"]
            ):
                assert trial.trial_id == frozen["selected_trial_id"]
            assert trial.window_result_id == trial.reader.header()["window_result_id"]


def test_trial_windows_fail_closed_on_foreign_or_corrupt_evidence(
    prediction_study: CompletedStudy, tmp_path: Path
) -> None:
    source, path = prediction_study.source, prediction_study.study.study_path
    fold = source.folds[0]
    combination = candidates(prediction_study)[0]
    with pytest.raises(OOSIntegrityError, match="universe"):
        load_prediction_trial_window(
            source, path, fold_id=fold.fold_id, combination_id="0" * 64
        )
    with pytest.raises(OOSIntegrityError, match="plan"):
        load_prediction_trial_window(
            source, path, fold_id="not-a-fold", combination_id=combination
        )
    trial = load_prediction_trial_window(
        source, path, fold_id=fold.fold_id, combination_id=combination
    )

    def corrupted(relative: str, edit: str) -> Path:
        copy = tmp_path / edit / path.name
        shutil.copytree(path, copy)
        target = copy / relative
        record = json.loads(target.read_text())
        if edit == "status":
            record["status"] = "failed"
        elif edit == "analysis":
            record["analysis"]["prediction_count"] = 999
        else:
            record["prediction_window_id"] = "0" * 64
        target.write_text(json.dumps(record, indent=2, sort_keys=True))
        return copy

    base = f"folds/{fold.fold_id}/selection/{trial.grid_study_id}"
    for relative, edit in (
        (f"{base}/trials/{trial.trial_id}.json", "status"),
        (f"{base}/artifacts/{trial.trial_id}/prediction-window.json", "analysis"),
        (f"{base}/artifacts/{trial.trial_id}/prediction-window.json", "window"),
    ):
        copy = corrupted(relative, edit)
        reloaded = load_oos_source(source.plan, copy)
        with pytest.raises(OOSIntegrityError):
            load_prediction_trial_window(
                reloaded, copy, fold_id=fold.fold_id, combination_id=combination
            )
    # A source verified from another directory cannot vouch for this path.
    other = tmp_path / "moved" / path.name
    shutil.copytree(path, other)
    (other / "manifest.json").write_text("{}")
    with pytest.raises(OOSIntegrityError, match="manifest is invalid"):
        load_prediction_trial_window(
            source, other, fold_id=fold.fold_id, combination_id=combination
        )
