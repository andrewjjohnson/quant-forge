"""Only identify native producer payloads and owned summary exports as such."""

from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import (
    ArtifactType,
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.prediction.test_prediction_window import (
    WindowProvider,
    grid,
    run_window,
)


@pytest.mark.parametrize("nested", [False, True], ids=["standalone", "grid"])
@pytest.mark.parametrize("component", ["foreign", "", None, 1, True, "missing"])
def test_rehashed_window_requires_its_producer_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
    component: Primitive,
) -> None:
    def alter_window(window: PrimitiveMapping) -> None:
        manifest = cast(PrimitiveMapping, window["manifest"])
        if component == "missing":
            manifest.pop("component")
        else:
            manifest["component"] = component
        manifest["window_id"] = configuration_identity(
            {
                key: value
                for key, value in manifest.items()
                if key
                not in {"window_id", "window_result_id", "schedule_id", "record_counts"}
            }
        )
        manifest["window_result_id"] = configuration_identity(
            {"window_id": manifest["window_id"], "decisions": window["decisions"]}
        )

    if nested:
        study = grid(tmp_path / "grid", WindowProvider())
        result = study.run()
        source = tmp_path / "grid" / result.study_id
        record_path = trial_path(source)
        trial = read_record(record_path)
        artifact_path = source / cast(str, trial["artifact_location"])
        artifact = read_record(artifact_path)
        window = cast(PrimitiveMapping, artifact["prediction_window"])
        alter_window(window)
        study_type = StudyType.PARAMETER_STUDY
        artifact["prediction_window_id"] = cast(PrimitiveMapping, window["manifest"])[
            "window_result_id"
        ]
        artifact["artifact_fingerprint"] = configuration_identity(
            {
                key: value
                for key, value in artifact.items()
                if key != "artifact_fingerprint"
            }
        )
        trial["artifact_fingerprint"] = artifact["artifact_fingerprint"]
        write_json(artifact_path, artifact)
        write_json(record_path, trial)
    else:
        window = run_window().to_primitive()
        alter_window(window)
        source = tmp_path / "window.json"
        write_json(source, window)
        study_type = StudyType.PREDICTION_WINDOW
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="prediction window component"):
        inspect_study(study_type, source, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "study_type", [StudyType.PARAMETER_STUDY, StudyType.OPTIMIZATION]
)
@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("malformed", [False, True])
def test_grid_summaries_only_include_native_export_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_type: StudyType,
    completed: bool,
    malformed: bool,
) -> None:
    root = build_grid_export(tmp_path, study_type, monkeypatch)
    if not completed:
        (root / "summary.json").unlink()
    original = inspect_study(study_type, root, artifact_root=tmp_path)
    optimization_names = {
        "trials.csv",
        "failures.csv",
        "exclusions.csv",
        "eligible_rankings.csv",
        "ineligible_trials.csv",
        "stability.csv",
        "parameter_summary.csv",
        "ranking.json",
        "stability.json",
    }
    native_names = {"summary.json"}
    if study_type is StudyType.OPTIMIZATION:
        native_names |= optimization_names
    unrelated = {
        "notes.csv",
        "result.json",
        "summary.json.bak",
        "summary.csv",
        *optimization_names,
    } - native_names
    for name in unrelated:
        if malformed:
            (root / name).write_text("unrelated,not producer output\n")
        elif name.endswith(".json"):
            write_json(root / name, {"study_id": original.provenance.producer_study_id})
        else:
            (root / name).write_text("trial_id,objective_value\nforeign-trial,999\n")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    inspected = inspect_study(study_type, root, artifact_root=tmp_path)
    assert inspected == original
    summary_names = {
        Path(entry.path).name
        for entry in inspected.index.entries
        if entry.artifact_type is ArtifactType.PARAMETER_SUMMARY
    }
    assert summary_names == (native_names if completed else set())
    assert verify_artifacts(inspected.index, tmp_path).valid
    assert {path: path.read_bytes() for path in before} == before
