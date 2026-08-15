"""Stable public exceptions for indicator contracts and calculations."""


class IndicatorError(Exception):
    """Base indicator failure."""


class InvalidIndicatorParametersError(IndicatorError):
    """An indicator parameter set is invalid."""


class MissingMarketFieldError(IndicatorError):
    """Canonical input does not expose a declared required field."""


class UnorderedMarketDataError(IndicatorError):
    """Market sessions are duplicated or not strictly chronological."""


class MisalignedIndicatorOutputError(IndicatorError):
    """Indicator output is not aligned one-to-one with its input sessions."""


class IndicatorCalculationError(IndicatorError):
    """A canonical field cannot be represented by the indicator."""


class InvalidIndicatorBackendError(IndicatorError):
    """A backend contract, registry, or serialized identity is invalid."""


class UnsupportedIndicatorBackendError(IndicatorError):
    """A backend id or backend/indicator combination is unsupported."""


class IndicatorBackendVersionError(IndicatorError):
    """A serialized backend version differs from the installed implementation."""


class IndicatorComparisonError(IndicatorError):
    """A backend comparison or its immutable export is invalid."""


class IndicatorSourceError(IndicatorError):
    """A timeframe-bound indicator source is missing or incompatible."""


class UnsupportedDevelopingBarError(IndicatorSourceError):
    """An indicator received a developing bar it does not explicitly support."""
