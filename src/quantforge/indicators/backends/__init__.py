"""Pluggable standard-indicator backend contracts and built-in adapters."""

from quantforge.indicators.backends.base import (
    INDICATOR_BACKEND_CONTRACT_VERSION,
    NATIVE_INDICATOR_BACKEND,
    TALIB_INDICATOR_BACKEND,
    IndicatorBackend,
    IndicatorBackendIdentity,
    IndicatorBackendRegistry,
    IndicatorComputationRequest,
    IndicatorComputationResult,
    StandardIndicatorDefinition,
)
from quantforge.indicators.backends.native import NativeIndicatorBackend
from quantforge.indicators.backends.registry import default_indicator_backend_registry
from quantforge.indicators.backends.talib import TalibIndicatorBackend

__all__ = [
    "INDICATOR_BACKEND_CONTRACT_VERSION",
    "NATIVE_INDICATOR_BACKEND",
    "TALIB_INDICATOR_BACKEND",
    "IndicatorBackend",
    "IndicatorBackendIdentity",
    "IndicatorBackendRegistry",
    "IndicatorComputationRequest",
    "IndicatorComputationResult",
    "NativeIndicatorBackend",
    "StandardIndicatorDefinition",
    "TalibIndicatorBackend",
    "default_indicator_backend_registry",
]
