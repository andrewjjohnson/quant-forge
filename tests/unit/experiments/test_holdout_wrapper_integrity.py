"""Rehashed holdouts cannot extend a frozen version-two component wrapper."""

from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, inspect_validation, verify_artifacts
from quantforge.experiments._json import mapping, text
from quantforge.experiments._window_integrity import validate_window_snapshot
from quantforge.oos import HoldoutEvaluation, HoldoutLedger
from quantforge.prediction import PredictionStudy
from quantforge.prediction import grid as grid_module
from quantforge.walk_forward import prediction as walk_forward_prediction
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_generic_trial_contract import legacy_definition
from tests.unit.experiments.test_standalone_provenance import rehash_prediction
from tests.unit.experiments.test_window_session_integrity import refresh_window_manifest
from tests.unit.oos.conftest import complete_study


@pytest.mark.parametrize("legacy", [False, True], ids=["v2", "legacy-v1"])
@pytest.mark.parametrize(
    "component", ["prediction_rule", "outcome_labeler", "evaluator"]
)
@pytest.mark.parametrize("extra", [None, "undeclared"], ids=["null", "text"])
def test_holdout_component_wrappers_follow_the_frozen_contract_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
    component: str,
    extra: Primitive,
) -> None:
    if legacy:
        capture = grid_module._trial_definition  # pyright: ignore[reportPrivateUsage]

        def old_capture(
            study: PredictionStudy[Any, Any, Any], backend: Any
        ) -> tuple[PrimitiveMapping, tuple[str, ...]]:
            definition, ids = capture(study, backend)
            return legacy_definition(definition), ids

        monkeypatch.setattr(grid_module, "_trial_definition", old_capture)
        monkeypatch.setattr(grid_module, "_TRIAL_DEFINITION_VERSION", "1")
        monkeypatch.setattr(walk_forward_prediction, "_trial_definition", old_capture)
    completed = complete_study(tmp_path, prediction=True)
    source = completed.source
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    consumed = ledger.consume(
        HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        ),
        run_id="wrapper-holdout",
    )
    assert consumed.result_reference is not None
    path = ledger.root / text(consumed.result_reference.to_primitive()["path"])
    block_research(monkeypatch)
    inspect_validation(
        source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
    )
    result = read_record(path)
    artifact = mapping(result["artifact"])
    window = mapping(artifact["result"])
    manifest = mapping(window["manifest"])
    mapping(mapping(manifest["configuration"])[component])["undeclared"] = extra
    for decision in cast(list[PrimitiveMapping], window["decisions"]):
        prediction = mapping(decision["prediction_study"])
        configuration = mapping(mapping(prediction["manifest"])["configuration"])
        mapping(configuration[component])["undeclared"] = extra
        rehash_prediction(prediction)
        decision["prediction_study_id"] = mapping(prediction["manifest"])["study_id"]
    refresh_window_manifest(window)
    manifest["window_result_id"] = configuration_identity(
        {"window_id": manifest["window_id"], "decisions": window["decisions"]}
    )
    artifact["result_id"] = manifest["window_result_id"]
    mapping(artifact["holdout_summary"])["window_result_id"] = manifest[
        "window_result_id"
    ]
    result["artifact_sha256"] = configuration_identity(artifact)
    write_record(path, result)
    validate_window_snapshot(window)
    before = {item: item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    if legacy:
        inspected = inspect_validation(
            source, completed.study.study_path, artifact_root=tmp_path, ledger=ledger
        )
        assert verify_artifacts(inspected.index, tmp_path).valid
    else:
        with pytest.raises(
            ManifestError, match=r"holdout prediction.*frozen candidate"
        ):
            inspect_validation(
                source,
                completed.study.study_path,
                artifact_root=tmp_path,
                ledger=ledger,
            )
    assert {item: item.read_bytes() for item in before} == before
    assert ledger.state(source).state.value == "consumed"
