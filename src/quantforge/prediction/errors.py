"""Prediction-analysis domain errors."""


class PredictionAnalysisError(Exception):
    """Base error for deterministic forward-label analysis."""


class InvalidPredictionConfigurationError(PredictionAnalysisError, ValueError):
    """Prediction strategy parameters or identity are invalid."""


class InvalidPredictionDataError(PredictionAnalysisError, ValueError):
    """Market data cannot support trustworthy next-session labels."""


class InvalidPredictionOutputError(PredictionAnalysisError, ValueError):
    """A prediction strategy emitted invalid or inconsistent signals."""


class PredictionExportError(PredictionAnalysisError):
    """A prediction result could not be exported or loaded."""
