from dataclasses import dataclass, replace
from datetime import date
from typing import cast

import pytest

from quantforge.data.models import DailyBar
from quantforge.indicators import (
    IndicatorFieldOutput,
    IndicatorOutput,
    MisalignedIndicatorOutputError,
    MissingMarketFieldError,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
    UnorderedMarketDataError,
)
from quantforge.indicators.base import validate_indicator_alignment

from ..helpers import make_dataset


@dataclass(frozen=True, slots=True)
class SessionOnlyBar:
    session_date: date


def test_missing_declared_market_field_raises_domain_error() -> None:
    dataset = make_dataset(("1",))
    incomplete = replace(
        dataset,
        bars=cast(tuple[DailyBar, ...], (SessionOnlyBar(date(2024, 7, 1)),)),
    )

    with pytest.raises(MissingMarketFieldError, match="close"):
        SimpleMovingAverage(SimpleMovingAverageParameters(1)).calculate(incomplete)


def test_unordered_sessions_are_rejected_before_calculation() -> None:
    dataset = make_dataset(("1", "2"))
    reversed_dataset = replace(dataset, bars=tuple(reversed(dataset.bars)))

    with pytest.raises(UnorderedMarketDataError, match="chronological"):
        SimpleMovingAverage(SimpleMovingAverageParameters(1)).calculate(
            reversed_dataset
        )


def test_alignment_validator_rejects_changed_session_index() -> None:
    dataset = make_dataset(("1", "2"))
    output = IndicatorOutput(
        "test",
        "configuration-id",
        (date(2024, 7, 1), date(2024, 7, 3)),
        (IndicatorFieldOutput("result", (None, None)),),
    )

    with pytest.raises(MisalignedIndicatorOutputError, match="sessions"):
        validate_indicator_alignment(dataset, output)
