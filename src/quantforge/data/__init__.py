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
    "AdjustmentMode",
    "CacheError",
    "CashDividend",
    "CorporateAction",
    "CorporateActionType",
    "DailyBar",
    "DatasetMetadata",
    "MarketDataCache",
    "MarketDataError",
    "MarketDataService",
    "MarketDataset",
    "ProviderError",
    "RequestError",
    "StockSplit",
    "ValidationError",
    "dataset_identity_matches",
    "validate_market_dataset",
]
