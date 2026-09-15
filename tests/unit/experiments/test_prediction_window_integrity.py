from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import (
    ManifestError,
    StudyType,
    create_manifest,
    inspect_study,
    read_manifest,
    write_manifest,
)
from quantforge.prediction.window import (
    _window_record_counts,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import execution, write_json
from tests.unit.experiments.test_grid_integrity import read_record, trial_path
from tests.unit.prediction.test_prediction_window import (
    START,
    WindowProvider,
    grid,
    run_window,
    schedule,
)


@pytest.fixture
def window_snapshot(monkeypatch: pytest.MonkeyPatch) -> PrimitiveMapping:
    snapshot = run_window().to_primitive()
    block_research(monkeypatch)
    return snapshot


@pytest.mark.parametrize("change", ["truncate", "edit", "reorder"])
def test_changed_window_decisions_cannot_keep_old_result_identity(
    tmp_path: Path, window_snapshot: PrimitiveMapping, change: str
) -> None:
    decisions = cast(list[PrimitiveMapping], window_snapshot["decisions"])
    if change == "truncate":
        decisions.pop()
    elif change == "reorder":
        decisions.reverse()
    else:
        decisions[0]["context_id"] = "changed-context"
    path = tmp_path / "window.json"
    write_json(path, window_snapshot)
    with pytest.raises(ManifestError, match="window result identity"):
        inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "scheduled_decisions",
        "valid_decisions",
        "skipped_decisions",
        "no_prediction_decisions",
        "generated_predictions",
        "unavailable_outcomes",
        "signal_dispositions",
    ],
)
def test_window_record_counts_must_match_stored_decisions(
    tmp_path: Path, window_snapshot: PrimitiveMapping, field: str
) -> None:
    manifest = cast(PrimitiveMapping, window_snapshot["manifest"])
    counts = cast(PrimitiveMapping, manifest["record_counts"])
    counts[field] = {"altered": 999} if field == "signal_dispositions" else 999
    path = tmp_path / "window.json"
    write_json(path, window_snapshot)
    with pytest.raises(ManifestError, match="window record counts"):
        inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)


@pytest.mark.parametrize("empty", [False, True])
def test_complete_and_empty_windows_round_trip_without_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty: bool
) -> None:
    result = run_window(
        decision_schedule=schedule(
            START + timedelta(seconds=1), START + timedelta(seconds=2)
        )
        if empty
        else schedule()
    )
    path = tmp_path / "window.json"
    write_json(path, result.to_primitive())
    block_research(monkeypatch)
    bundle = inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)
    manifest = create_manifest(bundle, execution())
    exported = write_manifest(
        manifest, tmp_path / "experiments", artifact_root=tmp_path
    )
    assert (
        read_manifest(exported, artifact_root=tmp_path).serialize()
        == manifest.serialize()
    )
    assert (
        bundle.provenance.observations.to_primitive()["window_result_id"]
        == result.window_result_id
    )


@pytest.mark.parametrize("rehash_window", [False, True])
def test_nested_grid_window_cannot_hide_missing_scheduled_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rehash_window: bool
) -> None:
    study = grid(tmp_path / "grid", WindowProvider())
    result = study.run()
    root = tmp_path / "grid" / result.study_id
    record_path = trial_path(root)
    trial = read_record(record_path)
    artifact_path = root / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    window = cast(PrimitiveMapping, artifact["prediction_window"])
    cast(list[Primitive], window["decisions"]).pop()
    if rehash_window:
        manifest = cast(PrimitiveMapping, window["manifest"])
        manifest["window_result_id"] = configuration_identity(
            {"window_id": manifest["window_id"], "decisions": window["decisions"]}
        )
        manifest["record_counts"] = _window_record_counts(
            cast(list[PrimitiveMapping], window["decisions"])
        )
    artifact["artifact_fingerprint"] = configuration_identity(
        {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
    )
    trial["artifact_fingerprint"] = artifact["artifact_fingerprint"]
    write_json(artifact_path, artifact)
    write_json(record_path, trial)
    block_research(monkeypatch)
    with pytest.raises(
        ManifestError,
        match="declared schedule" if rehash_window else "window result identity",
    ):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)


@pytest.mark.parametrize("missing", ["manifest", "decisions"])
def test_window_requires_complete_result_snapshot(
    tmp_path: Path, window_snapshot: PrimitiveMapping, missing: str
) -> None:
    window_snapshot.pop(missing)
    path = tmp_path / "incomplete-window.json"
    write_json(path, window_snapshot)
    with pytest.raises(ManifestError):
        inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)


@pytest.mark.parametrize("change", ["truncate", "reorder", "duplicate", "timestamp"])
def test_rehashed_window_decisions_must_match_declared_schedule(
    tmp_path: Path, window_snapshot: PrimitiveMapping, change: str
) -> None:
    decisions = cast(list[PrimitiveMapping], window_snapshot["decisions"])
    if change == "truncate":
        decisions.pop()
    elif change == "reorder":
        decisions.reverse()
    elif change == "duplicate":
        decisions.append(decisions[0])
    else:
        decisions[0]["decision_timestamp"] = "2000-01-01T00:00:00+00:00"
    manifest = cast(PrimitiveMapping, window_snapshot["manifest"])
    manifest["window_result_id"] = configuration_identity(
        {"window_id": manifest["window_id"], "decisions": window_snapshot["decisions"]}
    )
    manifest["record_counts"] = _window_record_counts(decisions)
    path = tmp_path / "window.json"
    write_json(path, window_snapshot)
    with pytest.raises(ManifestError, match=r"decisions.*schedule"):
        inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)


def change_decision_provenance(
    window: PrimitiveMapping, field: str
) -> PrimitiveMapping:
    decision = cast(list[PrimitiveMapping], window["decisions"])[0]
    study = cast(PrimitiveMapping, decision["prediction_study"])
    manifest = cast(PrimitiveMapping, study["manifest"])
    if field == "prediction_study_id":
        decision[field] = "unrelated-study"
    else:
        if field == "engine_version":
            manifest[field] = "historical-different-engine"
        else:
            cast(PrimitiveMapping, manifest[field])["changed"] = True
        manifest["study_id"] = configuration_identity(
            {
                "component": "quantforge_prediction_study",
                "engine_version": manifest["engine_version"],
                "market_data": manifest["market_data"],
                "study_configuration": manifest["configuration"],
                "prediction_context": manifest["prediction_context"],
            }
        )
        decision["prediction_study_id"] = manifest["study_id"]
        for row in cast(list[PrimitiveMapping], study["rows"]):
            row["study_id"] = manifest["study_id"]
            row["row_id"] = configuration_identity(
                {
                    "record_type": "prediction_study_row",
                    "study_id": manifest["study_id"],
                    "evaluation_id": cast(PrimitiveMapping, row["evaluation"])[
                        "evaluation_id"
                    ],
                    "outcome_id": cast(PrimitiveMapping, row["outcome"])["outcome_id"],
                    "signal": {
                        "features": row["features"],
                        "prediction": row["prediction"],
                    },
                }
            )
    outer = cast(PrimitiveMapping, window["manifest"])
    outer["window_result_id"] = configuration_identity(
        {"window_id": outer["window_id"], "decisions": window["decisions"]}
    )
    outer["record_counts"] = _window_record_counts(
        cast(list[PrimitiveMapping], window["decisions"])
    )
    return study


@pytest.mark.parametrize(
    "field", ["configuration", "market_data", "engine_version", "prediction_study_id"]
)
def test_self_consistent_decision_must_belong_to_its_window(
    tmp_path: Path, window_snapshot: PrimitiveMapping, field: str
) -> None:
    study = change_decision_provenance(window_snapshot, field)
    standalone = tmp_path / "standalone.json"
    write_json(standalone, study)
    inspect_study(StudyType.PREDICTION, standalone, artifact_root=tmp_path)
    path = tmp_path / "window.json"
    write_json(path, window_snapshot)
    with pytest.raises(ManifestError, match=r"decision.*window"):
        inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)


def test_nested_grid_window_rejects_rehashed_unrelated_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = grid(tmp_path / "grid", WindowProvider())
    result = study.run()
    root = tmp_path / "grid" / result.study_id
    record_path = trial_path(root)
    trial = read_record(record_path)
    artifact_path = root / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    change_decision_provenance(
        cast(PrimitiveMapping, artifact["prediction_window"]), "configuration"
    )
    artifact["artifact_fingerprint"] = configuration_identity(
        {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
    )
    trial["artifact_fingerprint"] = artifact["artifact_fingerprint"]
    write_json(artifact_path, artifact)
    write_json(record_path, trial)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match=r"decision.*window"):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
