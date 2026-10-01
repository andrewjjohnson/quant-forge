"""Leakage-safe conditional event ML datasets (QF-67).

``build_event_dataset`` assembles one frozen strategy population's verified,
persisted QF-39/QF-40 event observations into a deterministic dataset: an
explicit causal feature schema, an explicit versioned target over existing
outcomes, and verified chronological partition membership. Only generated
signals become rows; no-trigger decisions are never negatives. Final-holdout
rows require an explicitly consumed holdout in the permanent ledger. Nothing
here trains, tunes or selects features. See ``docs/event-ml-datasets.md``.
"""

from quantforge.ml.artifact import (
    MANIFEST_FILE,
    ROWS_CSV,
    ROWS_PARQUET,
    export_event_dataset,
    read_event_dataset,
    validate_event_dataset,
)
from quantforge.ml.dataset import (
    EVENT_DATASET_SCHEMA_VERSION,
    ROW_ORDERING,
    EventDataset,
    EventPartitionMembership,
    EventRow,
    EventTargetColumn,
    assemble_event_dataset,
    build_event_dataset,
)
from quantforge.ml.errors import (
    EventDatasetError,
    EventDatasetIntegrityError,
    EventFeatureSchemaError,
    EventHoldoutError,
    EventPartitionError,
    EventPopulationError,
    EventTargetError,
    NonAuthoritativeSourceError,
)
from quantforge.ml.features import (
    FORBIDDEN_FEATURE_NAMES,
    CausalAvailability,
    EventFeatureDefinition,
    EventFeatureSchema,
    FeatureSource,
    FeatureValueType,
    MissingValuePolicy,
)
from quantforge.ml.sources import (
    WORKSPACE_HOLDOUT_LEDGER,
    DispositionPolicy,
    EventPopulation,
    EventSourceWindow,
    HoldoutIsolation,
    load_study_event_sources,
)
from quantforge.ml.targets import (
    BoundEventTarget,
    ForwardReturnBinaryTarget,
    TargetKind,
    TargetLabel,
)

__all__ = [
    "EVENT_DATASET_SCHEMA_VERSION",
    "FORBIDDEN_FEATURE_NAMES",
    "MANIFEST_FILE",
    "ROWS_CSV",
    "ROWS_PARQUET",
    "ROW_ORDERING",
    "WORKSPACE_HOLDOUT_LEDGER",
    "BoundEventTarget",
    "CausalAvailability",
    "DispositionPolicy",
    "EventDataset",
    "EventDatasetError",
    "EventDatasetIntegrityError",
    "EventFeatureDefinition",
    "EventFeatureSchema",
    "EventFeatureSchemaError",
    "EventHoldoutError",
    "EventPartitionError",
    "EventPartitionMembership",
    "EventPopulation",
    "EventPopulationError",
    "EventRow",
    "EventSourceWindow",
    "EventTargetColumn",
    "EventTargetError",
    "FeatureSource",
    "FeatureValueType",
    "ForwardReturnBinaryTarget",
    "HoldoutIsolation",
    "MissingValuePolicy",
    "NonAuthoritativeSourceError",
    "TargetKind",
    "TargetLabel",
    "assemble_event_dataset",
    "build_event_dataset",
    "export_event_dataset",
    "load_study_event_sources",
    "read_event_dataset",
    "validate_event_dataset",
]
