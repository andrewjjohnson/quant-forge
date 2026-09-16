"""Unhashed causal declarations are required throughout QF-11 inspection."""

from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.experiments._json import mapping
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.experiments.test_prediction_row_integrity import (
    prediction_pair as prediction_pair,
)
from tests.unit.prediction.test_prediction_window import (
    WindowProvider,
    _rewrite_window_checksums,  # pyright: ignore[reportPrivateUsage]
    grid,
)


@pytest.fixture(params=["missing", None, "", False, 1, [], {}, "outcomes first"])
def invalid(request: pytest.FixtureRequest) -> Primitive:
    return cast(Primitive, request.param)


def change_boundary(manifest: PrimitiveMapping, invalid: Primitive) -> None:
    original_id = manifest["study_id"]
    if invalid == "missing":
        del manifest["feature_outcome_boundary"]
    else:
        manifest["feature_outcome_boundary"] = invalid
    assert manifest["study_id"] == original_id


@pytest.mark.parametrize("manifest_only", [False, True], ids=["result", "manifest"])
def test_plain_prediction_requires_boundary_even_without_rows(
    tmp_path: Path,
    prediction_pair: tuple[PrimitiveMapping, PrimitiveMapping],
    manifest_only: bool,
    invalid: Primitive,
) -> None:
    result, _ = prediction_pair
    manifest = mapping(result["manifest"])
    path = tmp_path / "prediction.json"
    document = manifest if manifest_only else result
    write_json(path, document)
    intact = inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    assert verify_artifacts(intact.index, tmp_path).valid
    change_boundary(manifest, invalid)
    write_json(path, document)
    before = path.read_bytes()
    with pytest.raises(ManifestError, match="prediction feature/outcome boundary"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("nested", [False, True], ids=["standalone", "grid"])
@pytest.mark.parametrize("window", [False, True], ids=["contextual", "window"])
def test_nested_prediction_boundary_cannot_be_removed_by_rehashing_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
    window: bool,
    invalid: Primitive,
) -> None:
    if window:
        study = grid(tmp_path / "grid", WindowProvider())
        study.run()
        root = tmp_path / "grid" / study.study_id
        block_research(monkeypatch)
    else:
        root = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    intact = inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert verify_artifacts(intact.index, tmp_path).valid
    record_path = trial_path(root)
    trial = read_record(record_path)
    artifact_path = root / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    snapshot = mapping(artifact["prediction_window" if window else "prediction_study"])
    if window:
        decision = mapping(cast(list[Primitive], snapshot["decisions"])[-1])
        change_boundary(
            mapping(mapping(decision["prediction_study"])["manifest"]), invalid
        )
        _rewrite_window_checksums(artifact_path, record_path, artifact)
    else:
        change_boundary(mapping(snapshot["manifest"]), invalid)
        artifact["artifact_fingerprint"] = trial["artifact_fingerprint"] = (
            configuration_identity(
                {
                    key: value
                    for key, value in artifact.items()
                    if key != "artifact_fingerprint"
                }
            )
        )
        write_json(artifact_path, artifact)
        write_json(record_path, trial)
    if nested:
        source, kind = root, StudyType.PARAMETER_STUDY
    else:
        source = tmp_path / "prediction.json"
        kind = StudyType.PREDICTION_WINDOW if window else StudyType.PREDICTION
        write_json(source, snapshot)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ManifestError, match="prediction feature/outcome boundary"):
        inspect_study(kind, source, artifact_root=tmp_path)
    assert {path: path.read_bytes() for path in before} == before
