"""Compatibility and temporal boundaries for persisted developing rule contexts."""

from copy import deepcopy
from datetime import timedelta
from typing import Any

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import ContextCompletionPolicy, FeedScope
from quantforge.prediction import (
    InvalidPredictionOutputError,
    PredictionContextRequirements,
    PredictionTimeframeRequirement,
    build_prediction_rule_context,
)
from quantforge.prediction.context import available_prediction_context_manifest
from quantforge.prediction.window_context_validation import (
    validate_window_context_snapshot,
)
from tests.unit.indicators.test_timeframe_evaluation import (
    _adjustment_basis,  # pyright: ignore[reportPrivateUsage]
    _developing_context,  # pyright: ignore[reportPrivateUsage]
)


def _available_context() -> dict[str, Any]:
    source = _developing_context()
    requirements = PredictionContextRequirements(
        PredictionTimeframeRequirement(
            source.primary_timeframe, FeedScope.consolidated(), ()
        ),
        tuple(
            PredictionTimeframeRequirement(
                item.timeframe,
                FeedScope.consolidated(),
                (),
                completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
            )
            for item in source.required_timeframes
        ),
    )
    market_data: PrimitiveMapping = {
        "dataset_id": "fixture-prediction-dataset",
        "symbol": "SPY",
        **_adjustment_basis().to_primitive(),
    }
    context = build_prediction_rule_context(
        requirements,
        source,
        prediction_dataset_id="fixture-prediction-dataset",
        symbol="SPY",
        prediction_adjustment_basis=_adjustment_basis(),
    )
    return {
        "context": available_prediction_context_manifest(context),
        "source": source.to_primitive(),
        "requirements": requirements.to_primitive(),
        "market_data": market_data,
        "primary_timeframe": source.to_primitive()["primary_timeframe"],
        "timestamp": source.as_of.isoformat(),
    }


def test_valid_developing_bar_retains_its_future_completion_boundary() -> None:
    arguments = _available_context()
    original = deepcopy(arguments)
    validate_window_context_snapshot(**arguments)
    assert arguments == original
    bar = arguments["source"]["timeframes"][1]["developing_bar"]["bar"]
    assert bar["observed_end_timestamp"] == arguments["timestamp"]
    assert bar["expected_completion_boundary"] > arguments["timestamp"]


@pytest.mark.parametrize(
    "mutation",
    [
        "future_completed",
        "future_observation",
        "wrong_as_of",
        "naive_observation",
        "missing_developing",
        "completed_state",
        "stale_bar_id",
    ],
)
def test_developing_context_rejects_inconsistent_temporal_evidence(
    mutation: str,
) -> None:
    arguments = _available_context()
    contextual = arguments["source"]["timeframes"][1]
    bar = contextual["developing_bar"]["bar"]
    future = (_developing_context().as_of + timedelta(minutes=5)).isoformat()
    if mutation == "future_completed":
        contextual["latest_completed_bar_timestamp"] = future
    elif mutation == "future_observation":
        bar["observed_end_timestamp"] = future
    elif mutation == "wrong_as_of":
        bar["as_of"] = future
    elif mutation == "naive_observation":
        bar["observed_end_timestamp"] = "2024-07-09T14:00:00"
    elif mutation == "missing_developing":
        contextual.pop("developing_bar")
    elif mutation == "completed_state":
        contextual["latest_completion_state"] = "completed"
    else:
        bar["observed_end_timestamp"] = future
    if mutation not in ("missing_developing", "stale_bar_id"):
        bar_id = configuration_identity(bar)
        contextual["developing_bar"]["bar_id"] = bar_id
        contextual["visible_bar_ids"][-1] = bar_id
        arguments["context"]["timeframes"][1]["visible_bar_ids"][-1] = bar_id
    with pytest.raises(InvalidPredictionOutputError):
        validate_window_context_snapshot(**arguments)
