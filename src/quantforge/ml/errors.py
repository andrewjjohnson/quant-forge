"""Fail-closed error family for conditional event ML datasets (QF-67)."""


class EventDatasetError(ValueError):
    """An event dataset cannot be built, read or trusted."""


class EventFeatureSchemaError(EventDatasetError):
    """A feature schema is unsafe or does not match persisted causal values."""


class EventTargetError(EventDatasetError):
    """A target is incompatible with the population, plan or label evidence."""


class EventPopulationError(EventDatasetError):
    """A source or observation is outside the declared strategy population."""


class EventPartitionError(EventDatasetError):
    """Partition membership is duplicated, conflicting or outside its window."""


class EventHoldoutError(EventPartitionError):
    """Rows would reach a reserved or unconsumed final holdout."""


class NonAuthoritativeSourceError(EventDatasetError):
    """Exploratory (QF-72 rapid) evidence offered as an authoritative source."""


class EventDatasetIntegrityError(EventDatasetError):
    """Source evidence or a dataset artifact is corrupt or inconsistent."""


__all__ = [
    "EventDatasetError",
    "EventDatasetIntegrityError",
    "EventFeatureSchemaError",
    "EventHoldoutError",
    "EventPartitionError",
    "EventPopulationError",
    "EventTargetError",
    "NonAuthoritativeSourceError",
]
