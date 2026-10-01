"""Selection, frozen out-of-sample and authorized holdout inference (QF-68).

- ``predict_selection``: the fold's selection rows; a fitted (unfrozen) or a
  frozen model. Nothing is selected from them: hyperparameters and any class
  threshold are fixed in the configuration before fitting.
- ``predict_out_of_sample``: the fold's walk-forward test rows; a
  ``FrozenEventModel`` only, re-validated from its persisted envelope.
- ``predict_final_holdout``: the plan's final-holdout rows; a frozen model
  plus the existing QF-40 authorization. The permanent ledger must already
  record the exact ``HoldoutEvaluation`` as consumed (``HoldoutLedger.result``
  is only read; nothing is reserved, consumed or reset) and the dataset's
  holdout rows must come from that consumption. A row's role or physical
  presence in a dataset is never authorization.
- ``reproduce_prediction_set``: offline regeneration of a persisted set from
  the frozen model and dataset, compared within the documented tolerance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping
from quantforge.ml.errors import EventHoldoutError
from quantforge.ml.modeling.artifact import FrozenEventModel, require_frozen
from quantforge.ml.modeling.errors import (
    ModelFreezeError,
    ModelHoldoutError,
    PredictionIntegrityError,
)
from quantforge.ml.modeling.predictions import (
    PredictionSet,
    compare_prediction_sets,
    read_prediction_set,
    score_partition,
)
from quantforge.ml.modeling.training import FittedEventModel, load_event_dataset
from quantforge.ml.sources import (
    WORKSPACE_HOLDOUT_LEDGER,
    reject_non_authoritative,
    workspace_ledger,
)
from quantforge.oos import HoldoutEvaluation, OOSIntegrityError
from quantforge.validation import PartitionRole
from quantforge.walk_forward.models import WalkForwardError


def predict_selection(
    model: FittedEventModel | FrozenEventModel, dataset: Path
) -> PredictionSet:
    """Score the fold's selection rows (empty when the fold has none)."""
    if type(cast(object, model)) is FrozenEventModel:
        state = require_frozen(model).state
    elif type(cast(object, model)) is FittedEventModel:
        state = model.state
    else:
        raise ModelFreezeError("selection inference requires a fitted event model")
    return score_partition(state, load_event_dataset(dataset), PartitionRole.SELECTION)


def predict_out_of_sample(model: FrozenEventModel, dataset: Path) -> PredictionSet:
    """Score the fold's walk-forward test rows with the frozen state only."""
    state = require_frozen(model).state
    return score_partition(
        state, load_event_dataset(dataset), PartitionRole.WALK_FORWARD_TEST
    )


def predict_final_holdout(
    model: FrozenEventModel,
    dataset: Path,
    *,
    workspace: Path,
    evaluation: HoldoutEvaluation,
) -> PredictionSet:
    """Score explicitly consumed final-holdout rows with the frozen state.

    ``dataset`` may be the model's training dataset rebuilt after consumption
    (QF-67 ``final_holdout=evaluation``): every row outside the holdout must be
    identical to the training dataset's.
    """
    reject_non_authoritative(evaluation, label="holdout evaluation")
    state = require_frozen(model).state
    if type(cast(object, evaluation)) is not HoldoutEvaluation:
        raise ModelHoldoutError("holdout inference requires a HoldoutEvaluation")
    loaded = load_event_dataset(dataset)
    scientific = loaded.scientific.to_primitive()
    plan = cast(PrimitiveMapping, scientific["partition_plan"])
    sources = [
        item
        for item in cast(list[PrimitiveMapping], plan["sources"])
        if item.get("role") == PartitionRole.FINAL_HOLDOUT.value
    ]
    if len(sources) != 1:
        raise ModelHoldoutError(
            "the dataset holds no explicitly consumed final-holdout rows"
        )
    membership = cast(PrimitiveMapping, sources[0]["membership"])
    request_id = membership.get("consumption_request_id")
    chronology = state.chronology
    if (
        evaluation.source.plan.plan_id != chronology.plan_id
        or membership.get("holdout_id") != chronology.holdout_id
        or not isinstance(request_id, str)
    ):
        raise ModelHoldoutError("holdout evaluation belongs to another plan or holdout")
    try:
        ledger = workspace_ledger(workspace)
        result = ledger.result(evaluation).to_primitive()
    except (EventHoldoutError, OOSIntegrityError, WalkForwardError) as error:
        raise ModelHoldoutError(
            "holdout inference requires a holdout already consumed through the "
            f"workspace's permanent ledger: {error}"
        ) from error
    if result.get("request_id") != request_id:
        raise ModelHoldoutError(
            "dataset holdout rows were not produced by this ledger consumption"
        )
    population = cast(PrimitiveMapping, scientific["population"])
    frozen = evaluation.selection.snapshot.to_primitive()
    if frozen.get("candidate") != population.get("candidate"):
        raise ModelHoldoutError("the consumed holdout evaluated another population")
    authorization: PrimitiveMapping = {
        "authority": "qf40_permanent_holdout_ledger_result",
        "ledger": WORKSPACE_HOLDOUT_LEDGER.as_posix(),
        "final_holdout_id": chronology.holdout_id,
        "consumption_request_id": request_id,
        "lineage_id": evaluation.source.lineage_id,
        "ledger_effect": "read_only_never_reserve_consume_or_reset",
    }
    return score_partition(
        state, loaded, PartitionRole.FINAL_HOLDOUT, authorization=authorization
    )


@dataclass(frozen=True, slots=True)
class PredictionReproduction:
    """The outcome of an offline regeneration of one persisted prediction set."""

    prediction_set_id: str
    records: int
    maximum_absolute_difference: float

    @property
    def exact(self) -> bool:
        return self.maximum_absolute_difference == 0.0


def reproduce_prediction_set(
    path: Path,
    *,
    model: FrozenEventModel,
    dataset: Path,
    workspace: Path | None = None,
    evaluation: HoldoutEvaluation | None = None,
) -> PredictionReproduction:
    """Validate a persisted prediction set by regenerating it offline.

    Membership (no missing, duplicate, extra or reassigned rows), bindings,
    statuses, classes and labels must be identical and probabilities within
    ``PROBABILITY_TOLERANCE``. Holdout sets need the same authorization as
    ``predict_final_holdout``; reproduction never changes the ledger.
    """
    stored = read_prediction_set(path)
    if stored.model_id != require_frozen(model).model_id:
        raise PredictionIntegrityError("prediction set is bound to another model")
    if stored.role is PartitionRole.SELECTION:
        regenerated = predict_selection(model, dataset)
    elif stored.role is PartitionRole.WALK_FORWARD_TEST:
        regenerated = predict_out_of_sample(model, dataset)
    else:
        if workspace is None or evaluation is None:
            raise ModelHoldoutError(
                "reproducing holdout predictions requires the same consumed "
                "holdout authorization"
            )
        regenerated = predict_final_holdout(
            model, dataset, workspace=workspace, evaluation=evaluation
        )
    difference = compare_prediction_sets(stored, regenerated)
    return PredictionReproduction(
        stored.prediction_set_id, len(stored.records), difference
    )


__all__ = [
    "PredictionReproduction",
    "predict_final_holdout",
    "predict_out_of_sample",
    "predict_selection",
    "reproduce_prediction_set",
]
