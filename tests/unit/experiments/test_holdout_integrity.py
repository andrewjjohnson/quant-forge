from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, inspect_validation
from quantforge.oos import HoldoutEvaluation, HoldoutLedger
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_adapters import block_research
from tests.unit.oos.conftest import complete_study


@pytest.mark.parametrize("prediction", [False, True], ids=["backtest", "prediction"])
@pytest.mark.parametrize("change", ["selection_id", "result_id", "manifest"])
def test_consumed_result_preserves_frozen_selection_and_result_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction: bool, change: str
) -> None:
    completed = complete_study(tmp_path, prediction=prediction)
    source = completed.source
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    consumed = ledger.consume(
        HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        ),
        run_id="fixture-holdout",
    )
    assert consumed.result_reference is not None
    path = ledger.root / cast(str, consumed.result_reference.to_primitive()["path"])
    result = read_record(path)
    artifact = cast(PrimitiveMapping, result["artifact"])
    if change == "manifest":
        manifest = cast(
            PrimitiveMapping, cast(PrimitiveMapping, artifact["result"])["manifest"]
        )
        if prediction:
            manifest["window_result_id"] = "0" * 64
        else:
            manifest["initiated_at"] = "2000-01-01T00:00:00+00:00"
    else:
        artifact[change] = "0" * 64
    result["artifact_sha256"] = configuration_identity(artifact)
    write_record(path, result)
    assert ledger.state(source).state.value == "consumed"
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=r"(holdout|window)"):
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )


@pytest.mark.parametrize(
    "change",
    ["foreign_selection", "foreign_partition", "window_identity", "window_counts"],
)
def test_consumed_prediction_rejects_rehashed_foreign_or_inconsistent_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    completed = complete_study(tmp_path, prediction=True)
    source = completed.source
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    consumed = ledger.consume(
        HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        ),
        run_id="fixture-holdout",
    )
    assert consumed.result_reference is not None
    path = ledger.root / cast(str, consumed.result_reference.to_primitive()["path"])
    result = read_record(path)
    original = cast(PrimitiveMapping, result["artifact"])
    if change.startswith("foreign"):
        fold_artifact = source.folds[0].artifact
        assert fold_artifact is not None
        artifact = deepcopy(fold_artifact.to_primitive())
        if change == "foreign_partition":
            artifact["selection_id"] = original["selection_id"]
        result["artifact"] = artifact
    else:
        artifact = original
        manifest = cast(
            PrimitiveMapping, cast(PrimitiveMapping, artifact["result"])["manifest"]
        )
        if change == "window_identity":
            manifest["window_result_id"] = "0" * 64
        else:
            cast(PrimitiveMapping, manifest["record_counts"])["scheduled_decisions"] = (
                999
            )
    result["artifact_sha256"] = configuration_identity(artifact)
    write_record(path, result)
    assert ledger.state(source).state.value == "consumed"
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=r"(holdout|window)"):
        inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )
