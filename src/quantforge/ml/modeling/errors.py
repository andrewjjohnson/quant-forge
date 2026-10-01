"""Fail-closed error family for chronological event models (QF-68).

Data limitations (no training rows, one class, a non-converging estimator) are
returned as an explicit ``TrainingStatus``; these exceptions are reserved for
contract violations that must never produce a model or a prediction.
"""


class EventModelError(ValueError):
    """An event model cannot be trained, frozen, applied or trusted."""


class ModelConfigurationError(EventModelError):
    """A model configuration is invalid or unsupported."""


class ModelBindingError(EventModelError):
    """A dataset, plan, schema, target, population or model binding differs."""


class ModelLeakageError(EventModelError):
    """Membership crosses a chronological, outcome-reach or partition boundary."""


class ModelHoldoutError(ModelLeakageError):
    """Final-holdout rows were offered without explicit consumed authorization."""


class ModelInputError(EventModelError):
    """A model input is not representable as a finite binary64 value."""


class ModelFreezeError(EventModelError):
    """Out-of-sample inference was requested without a validated frozen model."""


class ModelArtifactIntegrityError(EventModelError):
    """A persisted model or prediction artifact is corrupt or inconsistent."""


class PredictionIntegrityError(ModelArtifactIntegrityError):
    """Predictions are duplicated, missing, misassigned or wrongly bound."""


__all__ = [
    "EventModelError",
    "ModelArtifactIntegrityError",
    "ModelBindingError",
    "ModelConfigurationError",
    "ModelFreezeError",
    "ModelHoldoutError",
    "ModelInputError",
    "ModelLeakageError",
    "PredictionIntegrityError",
]
