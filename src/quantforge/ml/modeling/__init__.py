"""Leakage-safe chronological training and evaluation of event models (QF-68).

Lifecycle over one persisted QF-67 dataset and one fold of its QF-8 plan:

1. ``fit_event_model`` validates the dataset offline, binds it to the fold and
   fits preprocessing and one fixed L2 logistic regression on the fold's
   development rows only. Data limitations return an explicit status.
2. ``predict_selection`` scores the fold's selection rows.
3. ``freeze_event_model`` persists the inspectable model envelope and its
   selection evidence; ``read_event_model`` validates it.
4. ``predict_out_of_sample`` scores test rows with the frozen state only.
5. ``predict_final_holdout`` requires an explicitly consumed holdout.
6. ``export_prediction_set`` persists predictions apart from fitted state;
   ``reproduce_prediction_set`` regenerates them offline.

No hyperparameter or threshold search, no strategy-specific parsing, and no
claim about profitability. See ``docs/event-ml-models.md``.
"""

from quantforge.ml.modeling.artifact import (
    MODEL_FILE,
    FrozenEventModel,
    freeze_event_model,
    read_event_model,
)
from quantforge.ml.modeling.chronology import DatasetBinding, FoldChronology
from quantforge.ml.modeling.configuration import (
    LOG_LOSS_EPSILON,
    LogisticRegressionConfiguration,
    MissingFeaturePolicy,
    ModelConfiguration,
    PreprocessingConfiguration,
)
from quantforge.ml.modeling.errors import (
    EventModelError,
    ModelArtifactIntegrityError,
    ModelBindingError,
    ModelConfigurationError,
    ModelFreezeError,
    ModelHoldoutError,
    ModelInputError,
    ModelLeakageError,
    PredictionIntegrityError,
)
from quantforge.ml.modeling.inference import (
    PredictionReproduction,
    predict_final_holdout,
    predict_out_of_sample,
    predict_selection,
    reproduce_prediction_set,
)
from quantforge.ml.modeling.predictions import (
    PREDICTIONS_FILE,
    PROBABILITY_TOLERANCE,
    PredictionRecord,
    PredictionSet,
    ScoreStatus,
    export_prediction_set,
    read_prediction_set,
)
from quantforge.ml.modeling.status import TrainingStatus
from quantforge.ml.modeling.training import (
    EventModelState,
    FittedEventModel,
    TrainingMembership,
    TrainingResult,
    fit_event_model,
)

__all__ = [
    "LOG_LOSS_EPSILON",
    "MODEL_FILE",
    "PREDICTIONS_FILE",
    "PROBABILITY_TOLERANCE",
    "DatasetBinding",
    "EventModelError",
    "EventModelState",
    "FittedEventModel",
    "FoldChronology",
    "FrozenEventModel",
    "LogisticRegressionConfiguration",
    "MissingFeaturePolicy",
    "ModelArtifactIntegrityError",
    "ModelBindingError",
    "ModelConfiguration",
    "ModelConfigurationError",
    "ModelFreezeError",
    "ModelHoldoutError",
    "ModelInputError",
    "ModelLeakageError",
    "PredictionIntegrityError",
    "PredictionRecord",
    "PredictionReproduction",
    "PredictionSet",
    "PreprocessingConfiguration",
    "ScoreStatus",
    "TrainingMembership",
    "TrainingResult",
    "TrainingStatus",
    "export_prediction_set",
    "fit_event_model",
    "freeze_event_model",
    "predict_final_holdout",
    "predict_out_of_sample",
    "predict_selection",
    "read_event_model",
    "read_prediction_set",
    "reproduce_prediction_set",
]
