"""Captured QF-39 wrappers cannot substitute arbitrary QF-5 run identities."""

from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import validate_backtest_result_artifact
from quantforge.configuration import configuration_identity
from quantforge.experiments import ManifestError, inspect_validation, verify_artifacts
from quantforge.experiments._json import mapping
from quantforge.oos import load_oos_source
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record as read_json
from tests.unit.oos.conftest import complete_study


@pytest.mark.parametrize("fold_index", [0, 1])
def test_fold_rejects_forged_run_id_with_refreshed_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fold_index: int
) -> None:
    completed = complete_study(tmp_path, prediction=False)
    source = completed.source
    block_research(monkeypatch)
    original = inspect_validation(
        source, completed.study.study_path, artifact_root=tmp_path
    )
    assert verify_artifacts(original.index, tmp_path).valid
    fold_root = completed.study.study_path / "folds" / source.folds[fold_index].fold_id
    path = fold_root / "oos.json"
    artifact = read_record(path)
    export = fold_root / "test" / cast(str, artifact["export_location"])
    manifest = read_json(export / "manifest.json")
    forged_id = "f" * 64
    original_id = cast(str, manifest["run_id"])
    assert forged_id != original_id
    manifest["run_id"] = forged_id
    benchmark = mapping(manifest["benchmark"])
    benchmark["benchmark_id"] = configuration_identity(
        {
            "run_id": forged_id,
            "record_type": "benchmark",
            "configuration": benchmark["configuration"],
        }
    )
    write_json(export / "manifest.json", manifest)
    integrity = read_json(export / "integrity.json")
    for name in mapping(integrity["files"]):
        file = export / name
        if file.suffix == ".csv":
            file.write_text(file.read_text().replace(original_id, forged_id))
        mapping(integrity["files"])[name] = sha256(file.read_bytes()).hexdigest()
    write_json(export / "integrity.json", integrity)
    renamed = export.with_name(forged_id)
    export.rename(renamed)
    assert validate_backtest_result_artifact(renamed) == renamed
    artifact["result_id"] = forged_id
    artifact["export_location"] = renamed.relative_to(fold_root / "test").as_posix()
    mapping(artifact["result"])["manifest"] = manifest
    artifact["export_fingerprint"] = configuration_identity(
        {"integrity": (renamed / "integrity.json").read_text()}
    )
    write_record(path, artifact)
    state_path = fold_root / "state.json"
    state = read_record(state_path)
    state["artifact_id"] = configuration_identity(artifact)
    write_record(state_path, state)
    source = load_oos_source(source.plan, completed.study.study_path)
    before = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    with pytest.raises(ManifestError, match="backtest run identity"):
        inspect_validation(source, completed.study.study_path, artifact_root=tmp_path)
    assert {file: file.read_bytes() for file in before} == before
