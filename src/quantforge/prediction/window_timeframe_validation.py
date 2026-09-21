"""Validate retained source timeframe definitions using the canonical domain types."""

from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.timeframes import Timeframe


def validate_source_timeframe_definition(snapshot: PrimitiveMapping) -> Timeframe:
    """Reject malformed definitions even when the provider's context was skipped."""
    try:
        canonical = Timeframe.from_primitive(
            cast(PrimitiveMapping, snapshot["configuration"])
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise InvalidPredictionOutputError(
            "source timeframe definition is invalid"
        ) from error
    expected: PrimitiveMapping = {
        "configuration_id": canonical.configuration_id,
        "configuration": canonical.to_primitive(),
    }
    if configuration_identity(snapshot) != configuration_identity(expected):
        raise InvalidPredictionOutputError(
            "source timeframe definition is noncanonical"
        )
    return canonical
