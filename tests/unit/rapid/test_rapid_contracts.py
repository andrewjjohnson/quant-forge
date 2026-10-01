"""QF-72 rapid contracts: declared inputs, decision windows and statistics."""

from collections.abc import Callable
from datetime import date, datetime, time
from decimal import Decimal, localcontext
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.examples.spy_ema import DAILY, TWO_MINUTES
from quantforge.prediction import PredictionDirection
from quantforge.prediction.signal_feature_models import SignalFeatureCandidate
from quantforge.rapid import (
    NON_AUTHORITATIVE_NOTICE,
    RapidAdmissionError,
    RapidBarInput,
    RapidDecisionWindow,
    RapidIndicatorInput,
    RapidRuleSpecification,
)
from quantforge.rapid.models import RapidValue, summarize_outcomes


def decide(clock: time, values: tuple[RapidValue, ...]) -> PredictionDirection | None:
    del clock, values
    return None


def candidate(
    symbol: str,
    session: date,
    timestamp: datetime,
    values: tuple[RapidValue, ...],
) -> SignalFeatureCandidate:
    raise AssertionError((symbol, session, timestamp, values))


def specification(
    *inputs: RapidIndicatorInput | RapidBarInput,
) -> RapidRuleSpecification:
    return RapidRuleSpecification(
        inputs,
        decide,
        candidate,
        "America/New_York",
        RapidDecisionWindow(time(11), time(14)),
    )


def test_decision_window_is_inclusive_local_and_frozen() -> None:
    window = RapidDecisionWindow(time(11), time(14))
    assert [
        window.contains(time(h, m)) for h, m in ((10, 58), (11, 0), (14, 0), (14, 2))
    ] == [
        False,
        True,
        True,
        False,
    ]
    assert window.to_primitive() == {
        "start": "11:00:00",
        "end": "14:00:00",
        "boundaries": "both_inclusive",
    }
    for start, end in (
        (time(14), time(11)),
        (time(11, tzinfo=ZoneInfo("UTC")), time(14)),
    ):
        with pytest.raises(RapidAdmissionError, match="decision window"):
            RapidDecisionWindow(start, end)


@pytest.mark.parametrize(
    "build",
    [
        lambda: RapidIndicatorInput("x", TWO_MINUTES, "fast", "ema", lag=-1),
        lambda: RapidIndicatorInput("x", TWO_MINUTES, "fast", "ema", lag=True),
        lambda: RapidIndicatorInput("", TWO_MINUTES, "fast", "ema"),
        lambda: RapidBarInput("x", DAILY, "vwap"),
        lambda: specification(),
        lambda: specification(
            RapidBarInput("close", DAILY, "close"),
            RapidBarInput("close", DAILY, "open"),
        ),
        lambda: RapidRuleSpecification(
            (RapidBarInput("close", DAILY, "close"),), decide, candidate, "Not/AZone"
        ),
    ],
    ids=["negative-lag", "bool-lag", "name", "bar-field", "empty", "duplicate", "zone"],
)
def test_invalid_declarations_are_refused(build: Callable[[], object]) -> None:
    with pytest.raises(RapidAdmissionError):
        build()


def test_specification_primitive_names_the_shared_kernel() -> None:
    primitive = specification(
        RapidIndicatorInput("previous", TWO_MINUTES, "fast", "ema", lag=1),
        RapidBarInput("close", DAILY, "close"),
    ).to_primitive()
    assert primitive["kernel"] == f"{__name__}.decide"
    assert primitive["inputs"] == [
        {
            "kind": "indicator",
            "name": "previous",
            "timeframe_configuration_id": TWO_MINUTES.configuration_id,
            "alias": "fast",
            "output": "ema",
            "lag": 1,
        },
        {
            "kind": "bar",
            "name": "close",
            "timeframe_configuration_id": DAILY.configuration_id,
            "field": "close",
            "lag": 0,
        },
    ]


def test_outcome_summaries_never_count_unavailable_values_as_zero() -> None:
    rows: tuple[PrimitiveMapping, ...] = (
        {"available": True, "raw_return": "0.01", "status": "available"},
        {"available": True, "raw_return": "0", "status": "available"},
        {"available": True, "raw_return": "-0.02", "status": "available"},
        {"available": False, "raw_return": None, "status": "session_overflow"},
        {"available": True, "label": "target_first", "mfe_percentage": "0.004"},
    )
    summary = summarize_outcomes("returns", rows).to_primitive()
    assert summary["requested"] == 5
    assert summary["available"] == 4
    statistics = summary["statistics"]
    assert isinstance(statistics, dict)
    with localcontext() as arithmetic:
        arithmetic.prec = 34  # the documented summary precision
        mean, fraction = Decimal("-0.01") / 3, Decimal(1) / 3
    assert statistics["raw_return"] == {
        "count": 3,
        "mean": str(mean),
        "positive_fraction": str(fraction),
    }
    assert statistics["status_counts"] == {"available": 3, "session_overflow": 1}
    assert statistics["label_counts"] == {"target_first": 1}
    assert "NON-AUTHORITATIVE" in NON_AUTHORITATIVE_NOTICE
