"""Prediction-analysis domain errors."""


class PredictionAnalysisError(Exception):
    """Base error for deterministic forward-label analysis."""


class InvalidPredictionConfigurationError(PredictionAnalysisError, ValueError):
    """Prediction strategy parameters or identity are invalid."""


class InvalidPredictionDataError(PredictionAnalysisError, ValueError):
    """Market data cannot support trustworthy prediction-study labels."""


class InvalidPredictionOutputError(PredictionAnalysisError, ValueError):
    """A prediction strategy emitted invalid or inconsistent signals."""


class PredictionExportError(PredictionAnalysisError):
    """A prediction result could not be exported or loaded."""


class SignalFeatureDatasetError(PredictionAnalysisError):
    """A signal-feature dataset could not be generated or resumed safely."""


class SignalFeaturePersistenceError(SignalFeatureDatasetError):
    """Persisted signal-feature state is missing, corrupt, or incompatible."""
