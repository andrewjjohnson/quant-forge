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
