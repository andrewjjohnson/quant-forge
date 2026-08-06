"""Public daily-market-data ingestion orchestration."""

from datetime import UTC, date

from quantforge.data.cache import MarketDataCache, request_key
from quantforge.data.calendar import NYSE_CALENDAR, expected_sessions
from quantforge.data.corporate_actions import (
    CashDividendSeed,
    StockSplitSeed,
    corporate_action_snapshot_id,
)
from quantforge.data.exceptions import MarketDataError, ProviderError, RequestError
from quantforge.data.models import AdjustmentMode, MarketDataset
from quantforge.data.normalize import (
    normalize_response_with_corporate_actions,
    normalize_symbol,
)
from quantforge.data.providers.base import DailyBarProvider
from quantforge.data.validate import validate_bars


class MarketDataService:
    """Coordinate cache, provider, normalization, validation, and provenance."""

    def __init__(
        self,
        provider: DailyBarProvider,
        cache: MarketDataCache,
        *,
        calendar: str = NYSE_CALENDAR,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.calendar = calendar

    def get_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjustment: AdjustmentMode = AdjustmentMode.SPLIT_ADJUSTED,
        *,
        strict: bool = True,
        refresh: bool = False,
    ) -> MarketDataset:
        """Return inclusive bars, optionally refreshing the immutable cache index."""
        canonical = normalize_symbol(symbol)
        if start > end:
            raise RequestError("start must be on or before end")
        key = request_key(
            self.provider.name,
            canonical,
            start,
            end,
            adjustment,
            self.calendar,
            strict=strict,
        )
        if not refresh:
            cached = self.cache.find(key)
            if cached is not None:
                return cached
        try:
            response = self.provider.fetch_daily_bars(canonical, start, end, adjustment)
        except MarketDataError:
            raise
        except Exception as error:
            raise ProviderError(f"{self.provider.name} provider failed") from error
        if response.adjustment_mode is not adjustment:
            raise ProviderError("provider returned a different adjustment mode")
        bars, corporate_action_seeds = normalize_response_with_corporate_actions(
            response, canonical
        )
        validated = validate_bars(
            bars, canonical, start, end, self.calendar, strict=strict
        )
        expected = set(expected_sessions(start, end, self.calendar))
        missing = tuple(sorted(expected - {bar.session_date for bar in validated}))
        split_sessions = tuple(
            action.effective_session
            for action in corporate_action_seeds
            if isinstance(action, StockSplitSeed)
        )
        dividend_sessions = tuple(
            action.ex_dividend_session
            for action in corporate_action_seeds
            if isinstance(action, CashDividendSeed)
        )
        snapshot_id = corporate_action_snapshot_id(corporate_action_seeds)
        is_unadjusted = adjustment is AdjustmentMode.UNADJUSTED
        values: dict[str, object] = {
            "canonical_symbol": canonical,
            "provider_name": response.provider_name,
            "provider_symbol": response.provider_symbol,
            "retrieved_at": response.retrieved_at.astimezone(UTC),
            "requested_start": start,
            "requested_end": end,
            "actual_first_session": validated[0].session_date,
            "actual_last_session": validated[-1].session_date,
            "calendar": self.calendar,
            "provider_timezone": response.provider_timezone,
            "adjustment_mode": adjustment,
            "bar_count": len(validated),
            "missing_sessions": missing,
            "split_sessions": split_sessions,
            "dividend_sessions": dividend_sessions,
            "corporate_actions_complete": True,
            "corporate_action_count": len(corporate_action_seeds),
            "dividend_count": len(dividend_sessions),
            "split_count": len(split_sessions),
            "corporate_action_snapshot_id": snapshot_id,
            "ohlc_basis": "raw_provider" if is_unadjusted else "split_adjusted",
            "volume_basis": "raw_provider" if is_unadjusted else "split_adjusted",
            "adjusted_fields_used": False,
            "corporate_action_policy": (
                "separate_provider_reported_cash_dividends_and_splits"
            ),
            "adapter_version": response.adapter_version,
        }
        return self.cache.persist(
            response,
            validated,
            corporate_action_seeds,
            values,
            key,
            replace_request_index=refresh,
        )
