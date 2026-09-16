"""Generic component wrappers must remain bound to the frozen grid candidate."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._json import mapping
from quantforge.experiments._prediction_trial_integrity import (
    frozen_prediction_components,
)
from quantforge.prediction import PredictionStudy
from quantforge.prediction import grid as grid_module
from quantforge.prediction.outcomes.overnight_gap import (
    NextSessionOpenGapOutcomeLabeler,
    OvernightGapDirectionEvaluator,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import (
    FixtureTrialAnalyzer,
    read_record,
    trial_path,
)
from tests.unit.experiments.test_outcome_contract_integrity import (
    change_prediction,
    change_wrapper,
)
from tests.unit.experiments.test_standalone_provenance import rehash_prediction
from tests.unit.experiments.test_window_session_integrity import refresh_window_manifest
from tests.unit.prediction.test_prediction_grid import (
    FixtureStudyFactory,
    _backend_environment,  # pyright: ignore[reportPrivateUsage]
    _grid,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_window import (
    WindowProvider,
    WindowRule,
    _rewrite_window_checksums,  # pyright: ignore[reportPrivateUsage]
    grid,
)

WRAPPER_FIELDS = (
    ("prediction_rule", "warm_up_observations"),
    ("outcome_labeler", "required_future_sessions"),
    ("outcome_labeler", "required_market_fields"),
    ("outcome_labeler", "result_schema_version"),
    ("evaluator", "result_schema_version"),
)


class GenericLabeler(NextSessionOpenGapOutcomeLabeler):
    def configuration(self) -> PrimitiveMapping:
        return {
            key: value
            for key, value in super().configuration().items()
            if key
            not in {"parameters", "required_market_fields", "result_schema_version"}
        }


class GenericEvaluator(OvernightGapDirectionEvaluator):
    def configuration(self) -> PrimitiveMapping:
        return {
            key: value
            for key, value in super().configuration().items()
            if key != "result_schema_version"
        }


class GenericRule(WindowRule):
    def configuration(self) -> PrimitiveMapping:
        return {
            key: value
            for key, value in super().configuration().items()
            if key != "warm_up_observations"
        }


class GenericFactory(FixtureStudyFactory):
    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        original = super().build(parameters)
        rule = cast(WindowRule, original.strategy)
        return replace(
            original,
            strategy=GenericRule(rule.context_requirements),
            outcome_labeler=GenericLabeler(),
            evaluator=GenericEvaluator(),
        )


def export_generic_grid(tmp_path: Path, *, window: bool) -> Path:
    study = (
        grid(tmp_path / "grid", WindowProvider(), factory=GenericFactory())
        if window
        else _grid(
            tmp_path / "grid", factory=GenericFactory(), analyzer=FixtureTrialAnalyzer()
        )
    )
    study.run()
    return tmp_path / "grid" / study.study_id


def alter_prediction(snapshot: PrimitiveMapping, field: str) -> None:
    if field.startswith("extra_"):
        configuration = mapping(mapping(snapshot["manifest"])["configuration"])
        mapping(configuration[field.removeprefix("extra_")])["undeclared"] = None
        rehash_prediction(snapshot)
    elif field == "warm_up_observations":
        mapping(
            mapping(mapping(snapshot["manifest"])["configuration"])["prediction_rule"]
        )[field] = 1
        rehash_prediction(snapshot)
    else:
        change_prediction(snapshot, field)


@pytest.mark.parametrize("window", [False, True], ids=["plain", "window"])
@pytest.mark.parametrize(
    "field",
    [
        "warm_up_observations",
        "required_future_sessions",
        "required_market_fields",
        "outcome_schema",
        "evaluation_schema",
        "extra_prediction_rule",
        "extra_outcome_labeler",
        "extra_evaluator",
    ],
)
def test_generic_wrapper_changes_cannot_rewrite_a_frozen_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: bool,
    field: str,
) -> None:
    root = export_generic_grid(tmp_path, window=window)
    block_research(monkeypatch)
    inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    record_path = trial_path(root)
    trial = read_record(record_path)
    original_definition = trial["trial_definition"]
    artifact_path = root / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    snapshot = mapping(artifact["prediction_window" if window else "prediction_study"])
    if window:
        configuration = mapping(mapping(snapshot["manifest"])["configuration"])
        if field.startswith("extra_"):
            mapping(configuration[field.removeprefix("extra_")])["undeclared"] = None
        elif field == "warm_up_observations":
            mapping(configuration["prediction_rule"])[field] = 1
        else:
            change_wrapper(configuration, field)
        for decision in cast(list[PrimitiveMapping], snapshot["decisions"]):
            prediction = mapping(decision["prediction_study"])
            alter_prediction(prediction, field)
            decision["prediction_study_id"] = mapping(prediction["manifest"])[
                "study_id"
            ]
        refresh_window_manifest(snapshot)
        _rewrite_window_checksums(artifact_path, record_path, artifact)
    else:
        alter_prediction(snapshot, field)
        artifact["prediction_study_id"] = mapping(snapshot["manifest"])["study_id"]
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
    assert read_record(record_path)["trial_definition"] == original_definition
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ManifestError, match="trial definition"):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(("component", "field"), WRAPPER_FIELDS)
def test_version_two_requires_every_wrapper_even_with_duplicate_declarations(
    component: str,
    field: str,
) -> None:
    definition, _ = grid_module._trial_definition(  # pyright: ignore[reportPrivateUsage]
        FixtureStudyFactory().build({"window": 2}), _backend_environment()
    )
    mapping(definition[component]).pop(field)
    with pytest.raises(ManifestError, match="contract metadata is unavailable"):
        frozen_prediction_components(definition, {"trial_definition_version": "2"})


@pytest.mark.parametrize("location", ["definition", "study"])
@pytest.mark.parametrize("version", [None, "1", "3", 2, [], {}])
def test_unknown_or_mismatched_contract_versions_are_rejected(
    location: str,
    version: Primitive,
) -> None:
    definition, _ = grid_module._trial_definition(  # pyright: ignore[reportPrivateUsage]
        GenericFactory().build({"window": 2}), _backend_environment()
    )
    study: PrimitiveMapping = {"trial_definition_version": "2"}
    if location == "definition":
        definition["contract_version"] = version
    else:
        study["trial_definition_version"] = version
    with pytest.raises(ManifestError, match="trial definition version"):
        frozen_prediction_components(definition, study)


@pytest.mark.parametrize(
    ("component", "field", "replacement"),
    [
        (GenericRule, "warm_up_observations", 1),
        (GenericLabeler, "required_future_sessions", 2),
        (GenericLabeler, "required_market_fields", ("close",)),
        (GenericLabeler, "result_schema_version", "2"),
        (GenericEvaluator, "result_schema_version", "2"),
    ],
)
def test_native_capture_identity_includes_opaque_component_wrappers(
    monkeypatch: pytest.MonkeyPatch,
    component: type[object],
    field: str,
    replacement: object,
) -> None:
    study = GenericFactory().build({"window": 2})
    original, _ = grid_module._trial_definition(study, _backend_environment())  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(component, field, replacement)
    changed, _ = grid_module._trial_definition(study, _backend_environment())  # pyright: ignore[reportPrivateUsage]
    assert configuration_identity(original) != configuration_identity(changed)
    for name in ("prediction_rule", "outcome_labeler", "evaluator"):
        assert (
            mapping(original[name])["configuration_id"]
            == mapping(changed[name])["configuration_id"]
        )


def legacy_definition(definition: PrimitiveMapping) -> PrimitiveMapping:
    definition.pop("contract_version")
    for name, field in WRAPPER_FIELDS:
        mapping(definition[name]).pop(field)
    return definition


def test_unversioned_legacy_definitions_use_saved_declarations() -> None:
    definition, _ = grid_module._trial_definition(  # pyright: ignore[reportPrivateUsage]
        FixtureStudyFactory().build({"window": 2}), _backend_environment()
    )
    expected = frozen_prediction_components(
        definition, {"trial_definition_version": "2"}
    )
    original = legacy_definition(definition)
    assert frozen_prediction_components(original, {}) == expected
    assert "contract_version" not in original
    for name, field in WRAPPER_FIELDS:
        assert field not in mapping(original[name])


@pytest.mark.parametrize("window", [False, True], ids=["plain", "window"])
@pytest.mark.parametrize("generic", [False, True])
def test_legacy_grid_inspection_requires_recorded_wrapper_declarations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: bool,
    generic: bool,
) -> None:
    capture = grid_module._trial_definition  # pyright: ignore[reportPrivateUsage]

    def old_capture(
        study: PredictionStudy[Any, Any, Any], backend: Any
    ) -> tuple[PrimitiveMapping, tuple[str, ...]]:
        definition, ids = capture(study, backend)
        return legacy_definition(definition), ids

    monkeypatch.setattr(grid_module, "_trial_definition", old_capture)
    monkeypatch.setattr(grid_module, "_TRIAL_DEFINITION_VERSION", "1")
    if generic:
        root = export_generic_grid(tmp_path, window=window)
    else:
        study = (
            grid(tmp_path / "grid", WindowProvider())
            if window
            else _grid(tmp_path / "grid", analyzer=FixtureTrialAnalyzer())
        )
        study.run()
        root = tmp_path / "grid" / study.study_id
    block_research(monkeypatch)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    if generic:
        with pytest.raises(ManifestError, match="contract metadata is unavailable"):
            inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    else:
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("window", [False, True], ids=["plain", "window"])
def test_legacy_results_keep_projection_for_uncaptured_wrapper_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: bool
) -> None:
    capture = grid_module._trial_definition  # pyright: ignore[reportPrivateUsage]

    def old_capture(
        study: PredictionStudy[Any, Any, Any], backend: Any
    ) -> tuple[PrimitiveMapping, tuple[str, ...]]:
        definition, ids = capture(study, backend)
        return legacy_definition(definition), ids

    monkeypatch.setattr(grid_module, "_trial_definition", old_capture)
    monkeypatch.setattr(grid_module, "_TRIAL_DEFINITION_VERSION", "1")
    study = (
        grid(tmp_path / "grid", WindowProvider())
        if window
        else _grid(tmp_path / "grid", analyzer=FixtureTrialAnalyzer())
    )
    study.run()
    root = tmp_path / "grid" / study.study_id
    record_path = trial_path(root)
    trial = read_record(record_path)
    artifact_path = root / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    snapshot = mapping(artifact["prediction_window" if window else "prediction_study"])
    for name in ("prediction_rule", "outcome_labeler", "evaluator"):
        if window:
            configuration = mapping(mapping(snapshot["manifest"])["configuration"])
            mapping(configuration[name])["undeclared"] = None
            for decision in cast(list[PrimitiveMapping], snapshot["decisions"]):
                prediction = mapping(decision["prediction_study"])
                alter_prediction(prediction, f"extra_{name}")
                decision["prediction_study_id"] = mapping(prediction["manifest"])[
                    "study_id"
                ]
        else:
            alter_prediction(snapshot, f"extra_{name}")
    if window:
        refresh_window_manifest(snapshot)
        _rewrite_window_checksums(artifact_path, record_path, artifact)
    else:
        artifact["prediction_study_id"] = mapping(snapshot["manifest"])["study_id"]
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
    block_research(monkeypatch)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)
    assert {path: path.read_bytes() for path in before} == before
