"""QF-68 lifecycle on a real QF-39 study with development rows, offline.

Fixture (``event_model_fixtures``): the synthetic QF-45 composition, fixed
8/48, with a fold that declares no selection window, so QF-39 runs and freezes
its trial on development and QF-67 emits development (training) rows. The
final holdout is consumed only through the explicit QF-40 ledger workflow in a
dedicated fixture workspace. Nothing here is QF-45 research evidence.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples.spy_ema_ml_dataset import EMA_EVENT_FEATURES
from quantforge.ml import export_event_dataset, read_event_dataset
from quantforge.ml.modeling import (
    FrozenEventModel,
    ModelBindingError,
    ModelConfiguration,
    ModelHoldoutError,
    ScoreStatus,
    TrainingResult,
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
from quantforge.oos import HoldoutEvaluation, HoldoutLedger, HoldoutState
from quantforge.validation import PartitionRole
from tests.integration.event_dataset_fixtures import EventStudy
from tests.integration.event_model_fixtures import (
    development_inputs,
    run_development_study,
)

CONFIGURATION = ModelConfiguration(
    "qf68_integration_logistic", "1", minimum_training_observations=4
)


@pytest.fixture(scope="module", autouse=True)
def prepared() -> Iterator[None]:
    """One QF-65 load session for direct source loads (operational only)."""
    with canonical_preparation():
        yield


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory, prepared: None) -> EventStudy:
    del prepared
    root = tmp_path_factory.mktemp("qf68-event-model")
    return run_development_study(root, development_inputs(root))


@dataclass(frozen=True)
class Lifecycle:
    dataset: Path
    result: TrainingResult
    model: FrozenEventModel


def train(study: EventStudy, dataset: Path) -> TrainingResult:
    plan = study.config.plan
    return fit_event_model(
        dataset, plan=plan, fold_id=plan.folds[0].fold_id, configuration=CONFIGURATION
    )


@pytest.fixture(scope="module")
def lifecycle(study: EventStudy) -> Lifecycle:
    """Train and freeze before any holdout is reserved or consumed."""
    built = study.build(workspace=study.workspace("holdout"))
    dataset = export_event_dataset(built, study.root / "datasets")
    result = train(study, dataset)
    assert result.model is not None, result.detail
    path = freeze_event_model(
        result.model, dataset=dataset, output_root=study.root / "models"
    )
    return Lifecycle(dataset, result, read_event_model(path))


def ledger_files(ledger: HoldoutLedger) -> dict[str, bytes]:
    return {
        str(path.relative_to(ledger.root)): path.read_bytes()
        for path in sorted(ledger.root.rglob("*"))
        if path.is_file()
    }


def test_real_development_rows_train_freeze_and_score_out_of_sample(
    study: EventStudy, lifecycle: Lifecycle, tmp_path: Path
) -> None:
    result = lifecycle.result
    assert result.status is TrainingStatus.FITTED
    loaded = read_event_dataset(lifecycle.dataset)
    training = loaded.rows_for(PartitionRole.DEVELOPMENT)
    assert len(training) == 7
    assert result.membership.fitting_observation_ids == tuple(
        loaded.rows[index].source_observation_id for index in training
    )
    assert (result.membership.positives, result.membership.negatives) == (3, 4)
    # The generic layer uses the dataset's declared columns, in order.
    assert result.bound.dataset.feature_columns == tuple(
        name for name, _, _ in EMA_EVENT_FEATURES
    )
    # The fold declares no selection window: an explicit, empty selection set.
    selection = predict_selection(lifecycle.model, lifecycle.dataset)
    assert selection.records == ()
    assert selection.metrics()["log_loss"] == {
        "status": "undefined",
        "reason": "no_labeled_observations",
    }
    assert selection.prediction_set_id == lifecycle.model.selection_prediction_set_id
    oos = predict_out_of_sample(lifecycle.model, lifecycle.dataset)
    assert [record.score_status for record in oos.records] == [ScoreStatus.SCORED] * 4
    assert [record.target for record in oos.records] == [True, None, False, False]
    assert oos.summary()["unlabeled_scored_by_status"] == {"session_overflow": 1}
    metrics = oos.metrics()
    assert metrics["labeled_observations"] == 3
    assert cast(dict[str, object], metrics["roc_auc"])["status"] == "defined"
    path = export_prediction_set(oos, tmp_path)
    assert read_prediction_set(path) == oos
    report = reproduce_prediction_set(
        path, model=lifecycle.model, dataset=lifecycle.dataset
    )
    assert report.exact
    # Deterministic retraining reproduces the frozen identity.
    again = train(study, lifecycle.dataset)
    assert again.model is not None
    assert again.model.model_id == lifecycle.model.model_id
    # Training and inference never touched the permanent ledger.
    ledger = study.ledger(study.workspace("holdout"))
    assert not any((ledger.root / "lineages").iterdir())


def test_consumed_holdout_authorizes_frozen_model_inference(
    study: EventStudy, lifecycle: Lifecycle, tmp_path: Path
) -> None:
    workspace = study.workspace("holdout")
    ledger = study.ledger(workspace)
    source = study.source()
    assert ledger.reserve(source).state is HoldoutState.RESERVED
    evaluation = HoldoutEvaluation.prepare(
        source, study.adapter, selection_fold_id=source.folds[0].fold_id
    )
    # Before consumption the training dataset holds no holdout rows at all.
    with pytest.raises(ModelHoldoutError, match="no explicitly consumed"):
        predict_final_holdout(
            lifecycle.model,
            lifecycle.dataset,
            workspace=workspace,
            evaluation=evaluation,
        )
    assert ledger.state(source).state is HoldoutState.RESERVED
    consumed = ledger.consume(evaluation, run_id="qf68-fixture-explicit-consumption")
    assert consumed.state is HoldoutState.CONSUMED
    extended = export_event_dataset(
        study.build(workspace=workspace, final_holdout=evaluation),
        study.root / "datasets",
    )
    before = ledger_files(ledger)
    holdout = predict_final_holdout(
        lifecycle.model, extended, workspace=workspace, evaluation=evaluation
    )
    assert [record.partition_role for record in holdout.records] == [
        PartitionRole.FINAL_HOLDOUT
    ] * 2
    assert [record.fold_id for record in holdout.records] == [None, None]
    assert [record.target for record in holdout.records] == [True, False]
    authorization = holdout.authorization
    assert authorization is not None
    assert (
        authorization.to_primitive()["consumption_request_id"]
        == (
            consumed.to_primitive()["consumption"]["request_id"]  # pyright: ignore[reportIndexIssue,reportOptionalSubscript,reportCallIssue,reportArgumentType]
        )
    )
    path = export_prediction_set(holdout, tmp_path)
    assert reproduce_prediction_set(
        path,
        model=lifecycle.model,
        dataset=extended,
        workspace=workspace,
        evaluation=evaluation,
    ).exact
    with pytest.raises(ModelHoldoutError, match="same consumed holdout"):
        reproduce_prediction_set(path, model=lifecycle.model, dataset=extended)
    # Inference and reproduction only read the ledger.
    assert ledger_files(ledger) == before
    assert ledger.state(source).state is HoldoutState.CONSUMED
    # Out-of-sample inference stays bound to the exact training dataset.
    with pytest.raises(ModelBindingError, match="training dataset"):
        predict_out_of_sample(lifecycle.model, extended)
    # Physical holdout rows are not authority in a ledger that never consumed.
    reserved_only = study.workspace("reserved-only")
    other = study.ledger(reserved_only)
    other.reserve(source)
    with pytest.raises(ModelHoldoutError, match="permanent ledger"):
        predict_final_holdout(
            lifecycle.model, extended, workspace=reserved_only, evaluation=evaluation
        )
    assert other.state(source).state is HoldoutState.RESERVED
    assert not any((other.root / "exposures").iterdir())
    missing = study.root / "ledgerless-workspace"
    with pytest.raises(ModelHoldoutError, match="never created"):
        predict_final_holdout(
            lifecycle.model, extended, workspace=missing, evaluation=evaluation
        )
    assert not (missing / "reports").exists()
    # Training on the extended dataset never uses the consumed holdout rows.
    retrained = train(study, extended)
    assert retrained.model is not None
    assert retrained.model.state.fitted_state_id == (
        lifecycle.model.state.fitted_state_id
    )
    assert retrained.model.model_id != lifecycle.model.model_id
