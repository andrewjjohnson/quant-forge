"""Optimization-domain errors."""


class OptimizationError(Exception):
    """Base class for deterministic grid-search failures."""


class InvalidSearchSpaceError(OptimizationError):
    """Raised when a parameter search space is malformed or unsupported."""


class InvalidStudyConfigurationError(OptimizationError):
    """Raised when a study definition is internally inconsistent."""


class CombinationLimitExceededError(InvalidStudyConfigurationError):
    """Raised before execution when a Cartesian grid exceeds its safeguard."""


class StudyPersistenceError(OptimizationError):
    """Raised for corrupt, incompatible, or unwritable study artifacts."""


class InvalidTrialTransitionError(StudyPersistenceError):
    """Raised when a persisted trial would make an invalid state transition."""


class StudyExecutionError(OptimizationError):
    """Raised when study-level execution cannot continue safely."""


class RankingError(OptimizationError):
    """Raised when ranking inputs or metric values are invalid."""
