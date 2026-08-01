"""Public daily-market-data ingestion orchestration."""

from datetime import UTC, date

from quantforge.data.cache import MarketDataCache, request_key
from quantforge.data.calendar import NYSE_CALENDAR, expected_sessions
from quantforge.data.exceptions import MarketDataError, ProviderError, RequestError
from quantforge.data.models import AdjustmentMode, MarketDataset
from quantforge.data.normalize import normalize_response, normalize_symbol
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
    ) -> MarketDataset:
        """Return inclusive bars from cache, or retrieve and persist them."""
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
        bars = normalize_response(response, canonical)
        validated = validate_bars(
            bars, canonical, start, end, self.calendar, strict=strict
        )
        expected = set(expected_sessions(start, end, self.calendar))
        missing = tuple(sorted(expected - {bar.session_date for bar in validated}))
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
            "adapter_version": response.adapter_version,
        }
        return self.cache.persist(response, validated, values, key)
