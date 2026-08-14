"""Built-in standard-indicator backend registry."""

from functools import lru_cache

from quantforge.indicators.backends.base import IndicatorBackendRegistry
from quantforge.indicators.backends.native import NativeIndicatorBackend
from quantforge.indicators.backends.talib import TalibIndicatorBackend


@lru_cache(maxsize=1)
def default_indicator_backend_registry() -> IndicatorBackendRegistry:
    """Return the process-wide immutable built-in backend resolver."""
    return IndicatorBackendRegistry((NativeIndicatorBackend(), TalibIndicatorBackend()))


__all__ = ["default_indicator_backend_registry"]
