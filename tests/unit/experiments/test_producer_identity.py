from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import ArtifactType, ManifestError, StudyType, inspect_study
from quantforge.prediction import (
    AlwaysUpParameters,
    AlwaysUpPredictionStrategy,
    create_overnight_gap_prediction_study,
    run_prediction_study,
)
from tests.unit.backtesting.test_runner import configured_result
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    build_grid_export,
    read_record,
    trial_path,
)
from tests.unit.helpers import make_dataset
from tests.unit.prediction.test_prediction_window import run_window


@pytest.fixture
def prediction_snapshot(monkeypatch: pytest.MonkeyPatch) -> PrimitiveMapping:
    result = run_prediction_study(
        make_dataset(("100", "101", "102", "103", "104")),
        create_overnight_gap_prediction_study(
            AlwaysUpPredictionStrategy(AlwaysUpParameters())
        ),
    )
    block_research(monkeypatch)
    return result.to_primitive()


@pytest.mark.parametrize("manifest_only", [False, True])
@pytest.mark.parametrize(
    "field", ["engine_version", "market_data", "configuration", "prediction_context"]
)
def test_prediction_provenance_cannot_retain_stale_study_id(
    tmp_path: Path,
    prediction_snapshot: PrimitiveMapping,
    field: str,
    manifest_only: bool,
) -> None:
    manifest = cast(PrimitiveMapping, prediction_snapshot["manifest"])
    manifest[field] = "changed" if field == "engine_version" else {"changed": True}
    path = tmp_path / "prediction.json"
    write_json(path, manifest if manifest_only else prediction_snapshot)
    with pytest.raises(ManifestError, match="prediction study identity"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("change", "manifest_only"),
    [
        ("truncate", False),
        ("generated", False),
        ("unavailable", False),
        ("generated", True),
        ("unavailable", True),
    ],
)
def test_prediction_rows_must_agree_with_record_counts(
    tmp_path: Path,
    prediction_snapshot: PrimitiveMapping,
    change: str,
    manifest_only: bool,
) -> None:
    manifest = cast(PrimitiveMapping, prediction_snapshot["manifest"])
    counts = cast(PrimitiveMapping, manifest["record_counts"])
    if change == "truncate":
        cast(list[Primitive], prediction_snapshot["rows"]).pop()
    else:
        field = (
            "generated_predictions" if change == "generated" else "unavailable_outcomes"
        )
        counts[field] = cast(int, counts[field]) + 1
    path = tmp_path / "prediction.json"
    write_json(path, manifest if manifest_only else prediction_snapshot)
    with pytest.raises(ManifestError, match="prediction record counts"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize("manifest_only", [False, True])
@pytest.mark.parametrize(
    "field", ["generated_predictions", "labeled_rows", "unavailable_outcomes"]
)
@pytest.mark.parametrize("invalid", [None, -1, True, "4", 4.0])
def test_prediction_counts_are_nonnegative_integers(
    tmp_path: Path,
    prediction_snapshot: PrimitiveMapping,
    invalid: Primitive,
    field: str,
    manifest_only: bool,
) -> None:
    manifest = cast(PrimitiveMapping, prediction_snapshot["manifest"])
    cast(PrimitiveMapping, manifest["record_counts"])[field] = invalid
    path = tmp_path / "prediction.json"
    write_json(path, manifest if manifest_only else prediction_snapshot)
    with pytest.raises(ManifestError, match="prediction record counts"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize(("labeled", "unavailable"), [(0, 0), (0, 2), (3, 1)])
def test_prediction_manifest_retains_consistent_counts_without_claiming_rows(
    tmp_path: Path,
    prediction_snapshot: PrimitiveMapping,
    labeled: int,
    unavailable: int,
) -> None:
    manifest = cast(PrimitiveMapping, prediction_snapshot["manifest"])
    counts: PrimitiveMapping = {
        "generated_predictions": labeled + unavailable,
        "labeled_rows": labeled,
        "unavailable_outcomes": unavailable,
    }
    manifest["record_counts"] = counts
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    study = inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    assert study.provenance.observations.to_primitive()["record_counts"] == counts


@pytest.mark.parametrize(
    "field", ["generated_predictions", "labeled_rows", "unavailable_outcomes"]
)
def test_prediction_manifest_requires_each_declared_count(
    tmp_path: Path,
    prediction_snapshot: PrimitiveMapping,
    field: str,
) -> None:
    manifest = cast(PrimitiveMapping, prediction_snapshot["manifest"])
    del cast(PrimitiveMapping, manifest["record_counts"])[field]
    path = tmp_path / "prediction.json"
    write_json(path, manifest)
    with pytest.raises(ManifestError, match="prediction record counts"):
        inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "nested_field"),
    [
        ("engine_version", None),
        ("result_schema_version", None),
        ("market_data", "dataset_id"),
        ("market_data", "schema_version"),
        ("market_data", "adjustment_mode"),
        ("market_data", "calendar"),
        ("market_data", "corporate_action_snapshot_id"),
        ("market_data", "bars_fingerprint"),
        ("strategy", "strategy_id"),
        ("strategy", "strategy_implementation_version"),
        ("strategy", "strategy_configuration_id"),
        ("strategy", "configuration"),
        ("backtest_configuration", "initial_capital"),
    ],
)
def test_backtest_manifest_cannot_retain_stale_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    nested_field: str | None,
) -> None:
    manifest = configured_result().manifest_primitive()
    if nested_field is None:
        manifest[field] = "changed"
    else:
        nested = cast(PrimitiveMapping, manifest[field])
        if isinstance(nested[nested_field], dict):
            cast(PrimitiveMapping, nested[nested_field])["changed"] = True
        else:
            nested[nested_field] = "changed"
    path = tmp_path / "backtest.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="backtest run identity"):
        inspect_study(StudyType.BACKTEST, path, artifact_root=tmp_path)


def test_backtest_diagnostic_metadata_is_not_a_run_identity_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = configured_result()
    manifest = result.manifest_primitive()
    manifest["initiated_at"] = "2020-01-01T00:00:00+00:00"
    cast(PrimitiveMapping, manifest["strategy"])["warm_up_observations"] = 100
    path = tmp_path / "backtest.json"
    write_json(path, manifest)
    block_research(monkeypatch)
    bundle = inspect_study(StudyType.BACKTEST, path, artifact_root=tmp_path)
    assert bundle.provenance.producer_study_id == result.run_id


@pytest.mark.parametrize("manifest_only", [False, True])
def test_contextual_prediction_snapshot_preserves_original_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_only: bool
) -> None:
    snapshot = run_window().decisions[0].result.to_primitive()
    manifest = cast(PrimitiveMapping, snapshot["manifest"])
    assert "prediction_context" in manifest
    path = tmp_path / "prediction.json"
    write_json(path, manifest if manifest_only else snapshot)
    block_research(monkeypatch)
    bundle = inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    assert bundle.provenance.producer_study_id == manifest["study_id"]


def test_historical_prediction_engine_identity_uses_recorded_version(
    tmp_path: Path, prediction_snapshot: PrimitiveMapping
) -> None:
    manifest = cast(PrimitiveMapping, prediction_snapshot["manifest"])
    manifest["engine_version"] = "historical-version"
    manifest["study_id"] = configuration_identity(
        {
            "component": manifest["component"],
            "engine_version": manifest["engine_version"],
            "market_data": manifest["market_data"],
            "study_configuration": manifest["configuration"],
        }
    )
    path = tmp_path / "historical-prediction-manifest.json"
    write_json(path, manifest)
    bundle = inspect_study(StudyType.PREDICTION, path, artifact_root=tmp_path)
    assert bundle.provenance.producer_study_id == manifest["study_id"]
    assert (
        bundle.provenance.configuration.to_primitive()["engine_version"]
        == "historical-version"
    )


def test_rehashed_window_cannot_hide_stale_nested_prediction_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = run_window().to_primitive()
    decisions = cast(list[PrimitiveMapping], snapshot["decisions"])
    prediction = cast(PrimitiveMapping, decisions[0]["prediction_study"])
    cast(PrimitiveMapping, prediction["manifest"])["engine_version"] = "changed"
    manifest = cast(PrimitiveMapping, snapshot["manifest"])
    manifest["window_result_id"] = configuration_identity(
        {
            "window_id": manifest["window_id"],
            "decisions": snapshot["decisions"],
        }
    )
    path = tmp_path / "window.json"
    write_json(path, snapshot)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="prediction study identity"):
        inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)


def test_rehashed_grid_cannot_hide_stale_nested_prediction_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)
    record_path = trial_path(path)
    trial = read_record(record_path)
    artifact_path = path / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    prediction = cast(PrimitiveMapping, artifact["prediction_study"])
    cast(PrimitiveMapping, prediction["manifest"])["engine_version"] = "changed"
    artifact["artifact_fingerprint"] = configuration_identity(
        {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
    )
    trial["artifact_fingerprint"] = artifact["artifact_fingerprint"]
    write_json(record_path, trial)
    write_json(artifact_path, artifact)
    with pytest.raises(ManifestError, match="prediction study identity"):
        inspect_study(StudyType.PARAMETER_STUDY, path, artifact_root=tmp_path)


@pytest.mark.parametrize("manifest_only", [False, True], ids=["directory", "manifest"])
@pytest.mark.parametrize("change", ["version", "missing", "null", "numeric"])
def test_optimization_schema_must_match_hashed_identity_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_only: bool,
    change: str,
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    path = root / "manifest.json"
    source = path if manifest_only else root
    document = read_record(path)
    configuration = cast(PrimitiveMapping, document["identity_inputs"])
    schema = configuration["study_schema_version"]
    bundle = inspect_study(StudyType.OPTIMIZATION, source, artifact_root=tmp_path)
    manifest_entry = next(
        entry
        for entry in bundle.index.entries
        if entry.artifact_type is ArtifactType.CONFIGURATION
    )
    assert manifest_entry.schema_version == schema
    assert manifest_entry.bindings.to_primitive()["/study_schema_version"] == schema
    if change == "missing":
        del document["study_schema_version"]
    else:
        document["study_schema_version"] = {
            "version": "foreign-version",
            "null": None,
            "numeric": 1,
        }[change]
    write_json(path, document)
    assert configuration_identity(configuration) == document["study_id"]
    with pytest.raises(ManifestError, match="optimization schema"):
        inspect_study(StudyType.OPTIMIZATION, source, artifact_root=tmp_path)
