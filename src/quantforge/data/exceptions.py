"""Stable public exceptions for market-data ingestion."""


class MarketDataError(Exception):
    """Base market-data failure."""


class RequestError(MarketDataError):
    """The caller supplied an invalid request."""


class UnsupportedCapabilityError(RequestError):
    """A valid request is outside a provider's declared capabilities."""


class UnsupportedIntervalError(UnsupportedCapabilityError):
    """A provider does not support the requested source interval."""


class UnsupportedFeedError(UnsupportedCapabilityError):
    """A provider does not support the requested market-data feed scope."""


class UnsupportedSessionScopeError(UnsupportedCapabilityError):
    """A provider does not support the requested market-session scope."""


class UnsupportedDateRangeError(UnsupportedCapabilityError):
    """A request falls outside a provider's declared historical range."""


class ProviderError(MarketDataError):
    """A provider could not satisfy a valid request."""


class ValidationError(MarketDataError):
    """Canonical bars violate the documented schema."""

    def __init__(self, message: str, *, missing_sessions: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.missing_sessions = missing_sessions


class CacheError(MarketDataError):
    """An immutable cache entry is incomplete or corrupt."""
