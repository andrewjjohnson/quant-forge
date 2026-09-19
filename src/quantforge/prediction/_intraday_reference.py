"""Shared completed-decision reference and canonical price-basis checks."""

from collections.abc import Iterable

from quantforge.data import (
    AdjustmentBasis,
    IntradayBar,
    MarketDataset,
    TimeframeBarSeries,
)
from quantforge.prediction.errors import InvalidPredictionDataError
from quantforge.prediction.outcome_resolution import OutcomeEvaluationRequest

REFERENCE_PRICE_CONVENTION = "completed_decision_observation_close"


def completed_decision_reference(
    request: OutcomeEvaluationRequest, source: TimeframeBarSeries
) -> IntradayBar:
    """Require QF-49's exact completed anchor, never substitute a nearby price."""
    reference = next(
        (
            bar
            for bar in source.bars
            if isinstance(bar, IntradayBar)
            and bar.end_timestamp == request.anchor.decision_timestamp
            and bar.session_date == request.anchor.signal_session
            and bar.complete
        ),
        None,
    )
    if reference is None:
        raise InvalidPredictionDataError(
            "intraday outcome requires the exact completed decision "
            "observation in the outcome source"
        )
    return reference


def validate_intraday_price_basis(
    dataset: MarketDataset, bars: Iterable[IntradayBar]
) -> None:
    """Protect direct studies and replay even without a context provider."""
    metadata = dataset.metadata
    adjustment_basis = AdjustmentBasis(
        adjustment_mode=metadata.adjustment_mode,
        ohlc_basis=metadata.ohlc_basis,
        volume_basis=metadata.volume_basis,
        corporate_action_policy=metadata.corporate_action_policy,
        adjusted_fields_used=metadata.adjusted_fields_used,
    )
    for bar in bars:
        if bar.symbol != metadata.canonical_symbol:
            raise InvalidPredictionDataError(
                "intraday outcome source symbol is incompatible with "
                "the prediction dataset"
            )
        if bar.provenance.adjustment_basis != adjustment_basis:
            raise InvalidPredictionDataError(
                "intraday outcome source adjustment basis is "
                "incompatible with the prediction dataset"
            )
