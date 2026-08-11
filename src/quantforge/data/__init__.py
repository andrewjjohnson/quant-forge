"""Provider-agnostic daily market data and typed corporate actions."""

from quantforge.data.cache import MarketDataCache
from quantforge.data.exceptions import (
    CacheError,
    MarketDataError,
    ProviderError,
    RequestError,
    ValidationError,
)
from quantforge.data.identity import dataset_identity_matches
from quantforge.data.lineage import (
    DATASET_FAMILY_SCHEMA_VERSION,
    AdjustmentBasis,
    AggregationPolicy,
    DatasetFamily,
    DatasetFamilyReference,
    DatasetFamilyValidationError,
    DatasetLineage,
    ExternalBarValidationPolicy,
    FeedCoverage,
    FeedScope,
    MixedDatasetFamilyError,
    SourceConsistencyMode,
    SourceConsistencyValidation,
    validate_source_consistency,
)
from quantforge.data.models import (
    AdjustmentMode,
    CashDividend,
    CorporateAction,
    CorporateActionType,
    DailyBar,
    DatasetMetadata,
    MarketDataset,
    StockSplit,
)
from quantforge.data.service import MarketDataService
from quantforge.data.validate import validate_market_dataset

__all__ = [
    "DATASET_FAMILY_SCHEMA_VERSION",
    "AdjustmentBasis",
    "AdjustmentMode",
    "AggregationPolicy",
    "CacheError",
    "CashDividend",
    "CorporateAction",
    "CorporateActionType",
    "DailyBar",
    "DatasetFamily",
    "DatasetFamilyReference",
    "DatasetFamilyValidationError",
    "DatasetLineage",
    "DatasetMetadata",
    "ExternalBarValidationPolicy",
    "FeedCoverage",
    "FeedScope",
    "MarketDataCache",
    "MarketDataError",
    "MarketDataService",
    "MarketDataset",
    "MixedDatasetFamilyError",
    "ProviderError",
    "RequestError",
    "SourceConsistencyMode",
    "SourceConsistencyValidation",
    "StockSplit",
    "ValidationError",
    "dataset_identity_matches",
    "validate_market_dataset",
    "validate_source_consistency",
]
