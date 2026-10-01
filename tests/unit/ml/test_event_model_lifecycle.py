"""QF-68 freeze, inference, prediction artifacts, corruption and holdout refusal.

Synthetic QF-67 datasets bound to the real QF-8 timestamp plan fixture. The
consumed-holdout success path needs a real QF-40 ledger consumption and is in
``tests/integration/test_event_model_lifecycle.py``.
"""

import ast
import hashlib
import json
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import quantforge.ml.modeling as modeling
from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.ml import NonAuthoritativeSourceError
from quantforge.ml.modeling import (
    MODEL_FILE,
    PREDICTIONS_FILE,
    FittedEventModel,
    FrozenEventModel,
    MissingFeaturePolicy,
    ModelArtifactIntegrityError,
    ModelBindingError,
    ModelFreezeError,
    ModelHoldoutError,
    ModelLeakageError,
    PredictionIntegrityError,
    PredictionSet,
    PreprocessingConfiguration,
    ScoreStatus,
    TrainingStatus,
    export_prediction_set,
    fit_event_model,
    freeze_event_model,
    predict_final_holdout,
    predict_out_of_sample,
    predict_selection,
    read_event_model,
    read_prediction_set,
    reproduce_prediction_set,
)
from quantforge.prediction.window_encoding import canonical
from quantforge.validation import PartitionRole, ValidationPlan
from tests.unit.ml.model_fixtures import (
    REVERSED_SCHEMA,
    change,
    configuration,
    plan_fixture,
    standard_events,
    write_dataset,
)


@dataclass(frozen=True)
class Frozen:
    plan: ValidationPlan
    dataset: Path
    fitted: FittedEventModel
    model: FrozenEventModel
    root: Path


@pytest.fixture(scope="module")
def plan(tmp_path_factory: pytest.TempPathFactory) -> ValidationPlan:
    return plan_fixture(tmp_path_factory.mktemp("qf68-plan"))


def train(plan: ValidationPlan, dataset: Path, **options: object) -> FittedEventModel:
    result = fit_event_model(
        dataset,
        plan=plan,
        fold_id=plan.folds[0].fold_id,
        configuration=configuration(**options),
    )
    assert result.status is TrainingStatus.FITTED, result.detail
    assert result.model is not None
    return result.model


@pytest.fixture(scope="module")
def frozen(plan: ValidationPlan, tmp_path_factory: pytest.TempPathFactory) -> Frozen:
    root = tmp_path_factory.mktemp("qf68-lifecycle")
    dataset = write_dataset(plan, standard_events(0, holdout=True), root / "datasets")
    fitted = train(plan, dataset)
    path = freeze_event_model(fitted, dataset=dataset, output_root=root / "models")
    return Frozen(plan, dataset, fitted, read_event_model(path), root)


def envelope(path: Path) -> dict[str, Any]:
    return json.loads((path / MODEL_FILE).read_bytes())


def rewrite_model(path: Path, payload: PrimitiveMapping) -> None:
    (path / MODEL_FILE).write_bytes(
        canonical({"payload": payload, "fingerprint": configuration_identity(payload)})
        + b"\n"
    )


def rewrite_predictions(
    path: Path, lines: list[PrimitiveMapping], **edits: Any
) -> None:
    """A fully rehashed forgery: records, file hash and manifest fingerprint."""
    content = b"".join(canonical(line) + b"\n" for line in lines)
    (path / PREDICTIONS_FILE).write_bytes(content)
    manifest = json.loads((path / "manifest.json").read_bytes())
    payload = manifest["payload"]
    payload.update(edits)
    payload["files"] = {
        PREDICTIONS_FILE: {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    }
    (path / "manifest.json").write_bytes(
        canonical({"payload": payload, "fingerprint": configuration_identity(payload)})
        + b"\n"
    )


def records(path: Path) -> list[PrimitiveMapping]:
    return [
        json.loads(line) for line in (path / PREDICTIONS_FILE).read_bytes().splitlines()
    ]


def test_out_of_sample_inference_is_refused_before_freezing(frozen: Frozen) -> None:
    with pytest.raises(ModelFreezeError, match="frozen model"):
        predict_out_of_sample(cast(FrozenEventModel, frozen.fitted), frozen.dataset)
    with pytest.raises(ModelFreezeError, match="read_event_model"):
        FrozenEventModel(
            frozen.model.path, frozen.model.state, frozen.model.selection, "x"
        )
    with pytest.raises(ModelFreezeError):
        predict_out_of_sample(cast(FrozenEventModel, object()), frozen.dataset)
    # A tampered in-memory copy no longer equals its persisted envelope.
    other = replace(frozen.model, fingerprint="0" * 64)
    with pytest.raises(ModelFreezeError, match="differs from its persisted"):
        predict_out_of_sample(other, frozen.dataset)
    with pytest.raises(ModelBindingError, match="fit_event_model"):
        FittedEventModel(frozen.fitted.state)
    with pytest.raises(ModelFreezeError):
        freeze_event_model(
            cast(FittedEventModel, frozen.model),
            dataset=frozen.dataset,
            output_root=frozen.root / "x",
        )


def test_frozen_envelope_round_trips_and_binds_selection(frozen: Frozen) -> None:
    model = frozen.model
    assert model.state == frozen.fitted.state
    assert model.path.name == model.model_id
    payload = envelope(model.path)["payload"]
    assert payload["lifecycle"]["state"] == "frozen"
    estimator = payload["model"]["estimator"]["state"]
    assert [item["feature"] for item in estimator["coefficients"]] == [
        "momentum",
        "range_width",
        "bar_count",
        "gap_up",
    ]
    assert payload["model"]["training_membership"]["fitting_observation_count"] == 34
    before = predict_selection(frozen.fitted, frozen.dataset)
    after = predict_selection(model, frozen.dataset)
    assert before == after
    assert after.prediction_set_id == model.selection_prediction_set_id
    assert payload["selection"]["metrics"] == after.metrics()
    # Freezing again is idempotent and reuses the verified directory.
    again = freeze_event_model(
        frozen.fitted, dataset=frozen.dataset, output_root=model.path.parent
    )
    assert again == model.path
    # The only persisted format is JSON; no library object is deserialized.
    assert sorted(item.name for item in model.path.iterdir()) == [MODEL_FILE]


def test_out_of_sample_predictions_use_frozen_state_only(frozen: Frozen) -> None:
    oos = predict_out_of_sample(frozen.model, frozen.dataset)
    assert oos.role is PartitionRole.WALK_FORWARD_TEST
    assert [record.partition_role for record in oos.records] == [
        PartitionRole.WALK_FORWARD_TEST
    ] * 10
    assert all(record.fold_id == frozen.plan.folds[0].fold_id for record in oos.records)
    assert [record.row_index for record in oos.records] == sorted(
        record.row_index for record in oos.records
    )
    # Holdout rows in the same dataset are never returned without authority.
    assert not any(
        record.partition_role is PartitionRole.FINAL_HOLDOUT for record in oos.records
    )
    bindings = oos.bindings.to_primitive()
    assert bindings["model_id"] == frozen.model.model_id
    assert bindings["fitted_state_id"] == frozen.model.state.fitted_state_id
    summary = oos.summary()
    assert summary["scored"] == summary["labeled_scored"] == 10
    baseline = oos.baseline_metrics()
    assert baseline["labeled_observations"] == 10
    assert oos.baseline_probability == frozen.model.state.baseline_probability
    other = write_dataset(frozen.plan, standard_events(0), frozen.root / "other")
    with pytest.raises(ModelBindingError, match="training dataset"):
        predict_out_of_sample(frozen.model, other)


def test_prediction_artifacts_round_trip_and_reproduce_offline(
    frozen: Frozen, tmp_path: Path
) -> None:
    oos = predict_out_of_sample(frozen.model, frozen.dataset)
    path = export_prediction_set(oos, tmp_path)
    assert path.name == oos.prediction_set_id
    loaded = read_prediction_set(path)
    assert loaded == oos
    assert [r.probability for r in loaded.records] == [
        r.probability for r in oos.records
    ]
    report = reproduce_prediction_set(path, model=frozen.model, dataset=frozen.dataset)
    assert report.exact
    assert report.records == 10
    selection = export_prediction_set(
        predict_selection(frozen.model, frozen.dataset), tmp_path
    )
    assert reproduce_prediction_set(
        selection, model=frozen.model, dataset=frozen.dataset
    ).exact
    assert export_prediction_set(oos, tmp_path) == path  # immutable reuse


def test_prediction_corruption_duplicates_and_missing_rows_fail_closed(
    frozen: Frozen, tmp_path: Path
) -> None:
    oos = predict_out_of_sample(frozen.model, frozen.dataset)

    def copy(name: str) -> Path:
        return export_prediction_set(oos, tmp_path / name)

    edited = copy("edited")
    (edited / PREDICTIONS_FILE).write_bytes(
        (edited / PREDICTIONS_FILE).read_bytes().replace(b'"scored"', b'"SCORED"', 1)
    )
    with pytest.raises(PredictionIntegrityError):
        read_prediction_set(edited)
    lines = records(copy("base"))
    duplicate = copy("duplicate")
    rewrite_predictions(duplicate, [lines[0], *lines])
    with pytest.raises(PredictionIntegrityError, match="duplicate"):
        read_prediction_set(duplicate)
    wrong_role = copy("role")
    rewrite_predictions(
        wrong_role, [{**lines[0], "partition_role": "validation_selection"}, *lines[1:]]
    )
    with pytest.raises(PredictionIntegrityError, match="roles"):
        read_prediction_set(wrong_role)
    # A complete forgery with one record removed and every identity recomputed
    # is structurally valid, but regeneration exposes the missing prediction.
    missing = replace(oos, records=oos.records[1:])
    forged = export_prediction_set(missing, tmp_path / "forged")
    assert read_prediction_set(forged) == missing
    with pytest.raises(PredictionIntegrityError, match="1 missing"):
        reproduce_prediction_set(forged, model=frozen.model, dataset=frozen.dataset)
    shifted = replace(
        oos,
        records=(
            replace(
                oos.records[0],
                probability=cast(float, oos.records[0].probability) + 1e-9,
            ),
            *oos.records[1:],
        ),
    )
    with pytest.raises(PredictionIntegrityError, match="probabilities differ"):
        reproduce_prediction_set(
            export_prediction_set(shifted, tmp_path / "shifted"),
            model=frozen.model,
            dataset=frozen.dataset,
        )
    # Within the documented cross-platform tolerance it reproduces, inexactly.
    nudged = replace(
        oos,
        records=(
            replace(
                oos.records[0],
                probability=cast(float, oos.records[0].probability) + 1e-14,
            ),
            *oos.records[1:],
        ),
    )
    report = reproduce_prediction_set(
        export_prediction_set(nudged, tmp_path / "nudged"),
        model=frozen.model,
        dataset=frozen.dataset,
    )
    assert not report.exact
    assert 0 < report.maximum_absolute_difference <= 1e-12
    relabeled = replace(
        oos,
        records=(
            replace(oos.records[0], target=not oos.records[0].target),
            *oos.records[1:],
        ),
    )
    with pytest.raises(PredictionIntegrityError, match="differs"):
        reproduce_prediction_set(
            export_prediction_set(relabeled, tmp_path / "relabeled"),
            model=frozen.model,
            dataset=frozen.dataset,
        )


def test_predictions_bound_to_another_model_or_dataset_are_rejected(
    frozen: Frozen, tmp_path: Path
) -> None:
    oos = export_prediction_set(
        predict_out_of_sample(frozen.model, frozen.dataset), tmp_path / "p"
    )
    other_fitted = train(frozen.plan, frozen.dataset, class_threshold=Decimal("0.5"))
    other = read_event_model(
        freeze_event_model(
            other_fitted, dataset=frozen.dataset, output_root=tmp_path / "m"
        )
    )
    with pytest.raises(PredictionIntegrityError, match="another model"):
        reproduce_prediction_set(oos, model=other, dataset=frozen.dataset)
    rebound = replace(
        predict_out_of_sample(frozen.model, frozen.dataset),
        bindings=predict_selection(frozen.model, frozen.dataset).bindings,
    )
    with pytest.raises(PredictionIntegrityError):
        reproduce_prediction_set(
            export_prediction_set(rebound, tmp_path / "rebound"),
            model=frozen.model,
            dataset=frozen.dataset,
        )


def test_model_envelope_corruption_fails_closed(frozen: Frozen, tmp_path: Path) -> None:
    def copy(name: str) -> Path:
        target = tmp_path / name / frozen.model.model_id
        target.mkdir(parents=True)
        (target / MODEL_FILE).write_bytes((frozen.model.path / MODEL_FILE).read_bytes())
        return target

    flipped = copy("flipped")
    raw = (flipped / MODEL_FILE).read_bytes()
    (flipped / MODEL_FILE).write_bytes(
        raw.replace(b'"iterations":', b'"iterationz":', 1)
    )
    with pytest.raises(ModelArtifactIntegrityError):
        read_event_model(flipped)
    # A rehashed coefficient edit no longer matches the model identity.
    forged = copy("forged")
    payload = envelope(forged)["payload"]
    payload["model"]["estimator"]["state"]["coefficients"][0]["coefficient"] = 9.0
    rewrite_model(forged, payload)
    with pytest.raises(ModelArtifactIntegrityError):
        read_event_model(forged)
    renamed = copy("renamed").rename(tmp_path / "renamed" / ("0" * 64))
    with pytest.raises(ModelArtifactIntegrityError, match="directory"):
        read_event_model(renamed)
    # Training rows whose reach crosses the cutoff are refused even if rehashed.
    leaky = copy("leaky")
    payload = envelope(leaky)["payload"]
    observation = payload["model"]["training_membership"]["fitting_observations"][-1]
    observation["decision_timestamp"] = "2024-07-08T15:30:00+00:00"
    rewrite_model(leaky, payload)
    with pytest.raises(ModelLeakageError, match="purge rule"):
        read_event_model(leaky)
    # Deleting the envelope after reading revokes out-of-sample inference.
    revoked_dir = copy("revoked")
    revoked = read_event_model(revoked_dir)
    (revoked_dir / MODEL_FILE).unlink()
    with pytest.raises(ModelFreezeError, match="missing or no longer valid"):
        predict_out_of_sample(revoked, frozen.dataset)


def test_holdout_inference_requires_explicit_consumed_authorization(
    frozen: Frozen, tmp_path: Path
) -> None:
    # The dataset physically holds holdout rows, but presence is not authority.
    with pytest.raises(ModelHoldoutError, match="HoldoutEvaluation"):
        predict_final_holdout(
            frozen.model,
            frozen.dataset,
            workspace=tmp_path,
            evaluation=cast(Any, SimpleNamespace(source=None)),
        )
    with pytest.raises(NonAuthoritativeSourceError):
        predict_final_holdout(
            frozen.model,
            frozen.dataset,
            workspace=tmp_path,
            evaluation=cast(Any, SimpleNamespace(authoritative=False)),
        )
    with pytest.raises(ModelFreezeError):
        predict_final_holdout(
            cast(FrozenEventModel, frozen.fitted),
            frozen.dataset,
            workspace=tmp_path,
            evaluation=cast(Any, None),
        )
    # Holdout scoring cannot be reached through the internal scorer either.
    from quantforge.ml.modeling.predictions import score_partition
    from quantforge.ml.modeling.training import load_event_dataset

    with pytest.raises(ModelBindingError, match="authorization"):
        score_partition(
            frozen.model.state,
            load_event_dataset(frozen.dataset),
            PartitionRole.FINAL_HOLDOUT,
        )
    with pytest.raises(ModelBindingError, match="never scored"):
        score_partition(
            frozen.model.state,
            load_event_dataset(frozen.dataset),
            PartitionRole.DEVELOPMENT,
        )
    # A holdout prediction set without authority is rejected when read.
    test_set = predict_out_of_sample(frozen.model, frozen.dataset)
    bindings = test_set.bindings.to_primitive()
    bindings["partition_role"] = PartitionRole.FINAL_HOLDOUT.value
    unauthorized = replace(
        test_set,
        bindings=type(test_set.bindings).capture(bindings),
        records=tuple(
            replace(r, partition_role=PartitionRole.FINAL_HOLDOUT, fold_id=None)
            for r in test_set.records
        ),
    )
    with pytest.raises(PredictionIntegrityError, match="authorization"):
        read_prediction_set(export_prediction_set(unauthorized, tmp_path / "h"))
    # Refusals never create a ledger or write into the workspace.
    assert not any(tmp_path.glob("**/holdout-ledger"))


def test_unscored_rows_and_thresholds_are_explicit(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    items = change(standard_events(0), PartitionRole.WALK_FORWARD_TEST, label=None)
    items = [
        replace(item, features=(None, *item.features[1:]))
        if item.role is PartitionRole.WALK_FORWARD_TEST and item.clock == "13:05"
        else item
        for item in items
    ]
    dataset = write_dataset(plan, items, tmp_path / "d")
    fitted = train(plan, dataset, class_threshold=Decimal("0.5"))
    model = read_event_model(
        freeze_event_model(fitted, dataset=dataset, output_root=tmp_path / "m")
    )
    oos = predict_out_of_sample(model, dataset)
    statuses = [record.score_status for record in oos.records]
    assert statuses.count(ScoreStatus.UNSCORED_NULL_FEATURE) == 2
    unscored = [r for r in oos.records if r.score_status is not ScoreStatus.SCORED]
    assert all(r.probability is None and r.predicted_class is None for r in unscored)
    scored = [r for r in oos.records if r.score_status is ScoreStatus.SCORED]
    assert all(r.predicted_class == (cast(float, r.probability) >= 0.5) for r in scored)
    # Unavailable labels do not prevent prediction; they are reported apart.
    assert oos.summary() == {
        "rows": 10,
        "scored": 8,
        "unscored_by_reason": {"unscored_null_feature": 2},
        "labeled_scored": 0,
        "unlabeled_scored_by_status": {"session_overflow": 8},
        "imputed_feature_values": 0,
    }
    assert oos.metrics()["log_loss"] == {
        "status": "undefined",
        "reason": "no_labeled_observations",
    }
    # A class contradicting its probability and the frozen threshold is refused,
    # both when exported and when read after a fully rehashed file edit.
    flip = next(
        index
        for index, r in enumerate(oos.records)
        if r.score_status is ScoreStatus.SCORED
    )
    flipped = replace(
        oos,
        records=tuple(
            replace(r, predicted_class=not r.predicted_class) if index == flip else r
            for index, r in enumerate(oos.records)
        ),
    )
    with pytest.raises(PredictionIntegrityError, match="predicted classes"):
        export_prediction_set(flipped, tmp_path / "flipped-export")
    edited = export_prediction_set(oos, tmp_path / "flipped-file")
    lines = records(edited)
    lines[flip] = {**lines[flip], "predicted_class": not lines[flip]["predicted_class"]}
    rewrite_predictions(edited, lines)
    with pytest.raises(PredictionIntegrityError, match="predicted classes"):
        read_prediction_set(edited)
    imputing = PreprocessingConfiguration(MissingFeaturePolicy.IMPUTE_TRAINING_MEAN)
    imputed = read_event_model(
        freeze_event_model(
            train(plan, dataset, preprocessing=imputing),
            dataset=dataset,
            output_root=tmp_path / "i",
        )
    )
    filled = predict_out_of_sample(imputed, dataset)
    assert filled.summary()["scored"] == 10
    assert filled.summary()["imputed_feature_values"] == 2


def test_feature_order_and_schema_are_bound(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    base = write_dataset(plan, standard_events(0), tmp_path / "a")
    reordered = write_dataset(
        plan,
        [
            replace(item, features=tuple(reversed(item.features)))
            for item in standard_events(0)
        ],
        tmp_path / "b",
        schema=REVERSED_SCHEMA,
    )
    model = read_event_model(
        freeze_event_model(train(plan, base), dataset=base, output_root=tmp_path / "m")
    )
    with pytest.raises(ModelBindingError):
        predict_out_of_sample(model, reordered)
    # A holdout extension must keep schema, order, target and population.
    from quantforge.ml.modeling.chronology import DatasetBinding
    from quantforge.ml.modeling.training import load_event_dataset

    binding = DatasetBinding.capture(load_event_dataset(reordered))
    with pytest.raises(ModelBindingError, match="feature schema"):
        model.state.bound.dataset.require_compatible(binding)
    renamed = write_dataset(
        plan, standard_events(0), tmp_path / "c", population="qf68-population-b"
    )
    with pytest.raises(ModelBindingError, match="population"):
        model.state.bound.dataset.require_compatible(
            DatasetBinding.capture(load_event_dataset(renamed))
        )


def test_model_layer_is_generic_and_strategy_independent() -> None:
    """No strategy, example or prediction-window internals are imported."""
    forbidden = (
        "quantforge.examples",
        "quantforge.strategies",
        "quantforge.rapid",
        "quantforge.prediction.window_reader",
        "quantforge.prediction.window_coverage",
        "pickle",
        "joblib",
    )
    package = Path(modeling.__file__).parent
    for source in sorted(package.glob("*.py")):
        tree = ast.parse(source.read_text())
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not {name for name in imported if name.startswith(forbidden)}, source


def test_exports_refuse_untyped_sets_ledgers_and_exploratory_paths(
    frozen: Frozen, tmp_path: Path
) -> None:
    with pytest.raises(PredictionIntegrityError):
        export_prediction_set(cast(PredictionSet, object()), tmp_path)
    oos = predict_out_of_sample(frozen.model, frozen.dataset)
    with pytest.raises(NonAuthoritativeSourceError):
        export_prediction_set(oos, tmp_path / "out.rapid.json")
    with pytest.raises(ModelBindingError, match="holdout ledger"):
        export_prediction_set(oos, tmp_path / "reports" / "holdout-ledger" / "p")
    with pytest.raises(ModelBindingError, match="holdout ledger"):
        freeze_event_model(
            frozen.fitted,
            dataset=frozen.dataset,
            output_root=tmp_path / "reports" / "holdout-ledger" / "m",
        )
    assert not (tmp_path / "reports").exists()
