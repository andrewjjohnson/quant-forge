"""Compatibility checks for already validated QF-3 dataset metadata."""

from quantforge.backtesting.config import DividendPolicy
from quantforge.backtesting.errors import InvalidMarketDataError
from quantforge.data.models import AdjustmentMode, DatasetMetadata


def validate_backtest_dataset_metadata(
    metadata: DatasetMetadata,
    *,
    dividend_policy: DividendPolicy,
) -> None:
    """Reject metadata incompatible with the existing raw-price backtest engine.

    The caller must first validate the complete MarketDataset with QF-3's
    validate_market_dataset(). That binds this immutable metadata to the bars
    and action records, and verifies missing_sessions against the calendar.
    This check does not replace dataset validation or execute a backtest.
    """
    if metadata.adjustment_mode is not AdjustmentMode.UNADJUSTED:
        raise InvalidMarketDataError(
            "adjusted market data is unsupported by raw-price explicit "
            "corporate-action accounting"
        )
    internal_missing_sessions = tuple(
        missing_session
        for missing_session in metadata.missing_sessions
        if metadata.actual_first_session
        <= missing_session
        <= metadata.actual_last_session
    )
    if internal_missing_sessions:
        rendered = ", ".join(
            missing_session.isoformat() for missing_session in internal_missing_sessions
        )
        raise InvalidMarketDataError(
            "dataset has missing expected sessions within its observed range: "
            f"{rendered}"
        )
    if not metadata.corporate_actions_complete:
        raise InvalidMarketDataError(
            "unadjusted market data requires complete explicit corporate actions"
        )
    if (
        metadata.ohlc_basis != "raw_provider"
        or metadata.volume_basis != "raw_provider"
        or metadata.adjusted_fields_used
    ):
        raise InvalidMarketDataError(
            "corporate-action execution requires consistent raw provider OHLCV"
        )
    if metadata.corporate_action_policy != (
        "separate_provider_reported_cash_dividends_and_splits"
    ):
        raise InvalidMarketDataError(
            "market data has unsupported corporate-action semantics"
        )
    if (
        dividend_policy is DividendPolicy.REJECT_IF_DIVIDENDS
        and metadata.dividend_count > 0
    ):
        raise InvalidMarketDataError(
            "dataset contains cash dividends; select DividendPolicy.PRICE_RETURN_ONLY "
            "or DividendPolicy.CASH_DIVIDENDS explicitly"
        )
