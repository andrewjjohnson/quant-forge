"""Explicit training outcomes for data limitations (QF-68).

A dataset that cannot support the configured estimator yields a
``TrainingResult`` with one of these statuses instead of a model. The estimator
type is never changed and no fallback model is fitted.
"""

from enum import StrEnum


class TrainingStatus(StrEnum):
    """Why a training attempt did or did not produce a fitted model."""

    FITTED = "fitted"
    # The fold has no development rows (for example QF-39 executed its trials
    # on the selection window, so development was never run).
    NO_TRAINING_OBSERVATIONS = "no_training_observations"
    INSUFFICIENT_TRAINING_OBSERVATIONS = "insufficient_training_observations"
    SINGLE_CLASS_TRAINING_LABELS = "single_class_training_labels"
    FEATURE_MISSING_IN_TRAINING = "feature_entirely_missing_in_training"
    ALL_FEATURES_CONSTANT = "all_features_constant_in_training"
    ESTIMATOR_DID_NOT_CONVERGE = "estimator_did_not_converge"
    ESTIMATOR_FAILED = "estimator_failed"


class TrainingLimitationError(Exception):
    """Internal control flow for a data limitation; never escapes the API."""

    def __init__(self, status: TrainingStatus, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


__all__ = ["TrainingLimitationError", "TrainingStatus"]
