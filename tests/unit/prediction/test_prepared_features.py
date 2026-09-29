"""QF-63 admission contract: reviewed indicator formulas are prefix stable.

A prepared series is computed once from an exact input start and every decision
receives a prefix. That is equivalent to evaluating the decision's own bars only
if the value at each position depends on no later input. These tests prove that
property, including exact Decimal representation, for every admitted pair.
"""

import random
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest

from quantforge.indicators import (
    BollingerBands,
    BollingerBandsParameters,
    ExponentialMovingAverage,
    ExponentialMovingAverageParameters,
    MovingAverageConvergenceDivergence,
    MovingAverageConvergenceDivergenceParameters,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
    StochasticOscillator,
    StochasticOscillatorParameters,
    WilderAverageTrueRange,
    WilderAverageTrueRangeParameters,
    WilderDirectionalMovement,
    WilderDirectionalMovementParameters,
    WilderRelativeStrengthIndex,
    WilderRelativeStrengthIndexParameters,
)
from quantforge.indicators.backends import (
    NATIVE_INDICATOR_BACKEND,
    TALIB_INDICATOR_BACKEND,
)
from quantforge.indicators.base import IndicatorBar
from quantforge.indicators.exceptions import UnsupportedIndicatorBackendError
from quantforge.indicators.models import IndicatorFieldOutput
from quantforge.prediction.prepared_features import (
    _PREFIX_STABLE_BACKENDS,  # pyright: ignore[reportPrivateUsage]
    _PREFIX_STABLE_INDICATORS,  # pyright: ignore[reportPrivateUsage]
    PreparedContextScope,
    PreparedTimeframeRun,
)


@dataclass(frozen=True)
class Bar:
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


def walk(count: int, seed: int = 63) -> tuple[Bar, ...]:
    generator = random.Random(seed)
    level = Decimal("500")
    bars: list[Bar] = []
    for _ in range(count):
        change = Decimal(generator.randint(-150, 150)) / Decimal(100)
        opened = level
        level = max(Decimal("1"), level + change)
        high = max(opened, level) + Decimal(generator.randint(0, 40)) / Decimal(100)
        low = min(opened, level) - Decimal(generator.randint(0, 40)) / Decimal(100)
        bars.append(
            Bar(opened, high, low, level, Decimal(generator.randint(1_000, 90_000)))
        )
    return tuple(bars)


FACTORIES: tuple[tuple[str, type, Any], ...] = (
    ("sma", SimpleMovingAverage, SimpleMovingAverageParameters(9)),
    ("ema", ExponentialMovingAverage, ExponentialMovingAverageParameters(8)),
    ("rsi", WilderRelativeStrengthIndex, WilderRelativeStrengthIndexParameters(14)),
    ("atr", WilderAverageTrueRange, WilderAverageTrueRangeParameters(14)),
    ("dmi", WilderDirectionalMovement, WilderDirectionalMovementParameters(14)),
    ("bollinger", BollingerBands, BollingerBandsParameters(20)),
    (
        "macd",
        MovingAverageConvergenceDivergence,
        MovingAverageConvergenceDivergenceParameters(12, 26, 9),
    ),
    ("stochastic", StochasticOscillator, StochasticOscillatorParameters()),
)


def admitted_pairs() -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    for name, indicator_type, parameters in FACTORIES:
        for backend_id in (TALIB_INDICATOR_BACKEND, NATIVE_INDICATOR_BACKEND):
            try:
                indicator = indicator_type(parameters, backend_id=backend_id)
            except UnsupportedIndicatorBackendError:
                continue
            pairs.append((f"{name}-{backend_id}", indicator))
    return pairs


PAIRS = admitted_pairs()


def exact(fields: tuple[IndicatorFieldOutput, ...], count: int) -> list[object]:
    return [
        (
            item.name,
            tuple(
                None if value is None else value.as_tuple()
                for value in item.values[:count]
            ),
        )
        for item in fields
    ]


def test_every_reviewed_type_and_backend_is_exercised() -> None:
    covered: set[type] = {type(indicator) for _, indicator in PAIRS}
    assert covered == set(_PREFIX_STABLE_INDICATORS)
    assert {type(getattr(indicator, "_backend")) for _, indicator in PAIRS} == set(
        _PREFIX_STABLE_BACKENDS
    )
    assert "ema-talib_v1" in {name for name, _ in PAIRS}


@pytest.mark.parametrize(("name", "indicator"), PAIRS, ids=[n for n, _ in PAIRS])
def test_prefix_equals_evaluation_of_the_decisions_own_bars(
    name: str, indicator: Any
) -> None:
    bars = walk(180)
    full = indicator.calculate_bar_fields(bars)
    for count in (1, 2, 7, 8, 9, 13, 14, 15, 26, 34, 35, 61, 120, 179, 180):
        own = indicator.calculate_bar_fields(bars[:count])
        assert exact(own, count) == exact(full, count), (name, count)
        assert all(len(item.values) == count for item in own)


@pytest.mark.parametrize(("name", "indicator"), PAIRS, ids=[n for n, _ in PAIRS])
def test_sentinel_future_bars_cannot_change_earlier_values(
    name: str, indicator: Any
) -> None:
    bars = walk(120, seed=45)
    absurd = Bar(
        Decimal("999999"),
        Decimal("1000000"),
        Decimal("0.01"),
        Decimal("999999"),
        Decimal("1"),
    )
    for cutoff in (10, 50, 100):
        sentinel: tuple[IndicatorBar, ...] = (*bars[:cutoff], *(absurd,) * 30)
        assert exact(indicator.calculate_bar_fields(sentinel), cutoff) == exact(
            indicator.calculate_bar_fields(bars[:cutoff]), cutoff
        ), (name, cutoff)


def test_incremental_checks_fail_exactly_when_a_bad_bar_becomes_visible() -> None:
    """Per-bar checks run once per newly visible bar, like the reference scan."""
    scope = PreparedContextScope({})
    bars = tuple(range(10))
    run = cast(PreparedTimeframeRun, SimpleNamespace(bars=bars))
    checked: list[tuple[int, ...]] = []

    def check(pieces: tuple[Any, ...]) -> None:
        checked.append(tuple(pieces))
        if 7 in pieces:
            raise ValueError("bad bar")

    # Anchor phase (start 0), then the rolling start shifts to 1.
    assert scope.validate_span(("key",), run, 0, 3, check) == 3
    assert scope.validate_span(("key",), run, 0, 4, check) == 1
    assert scope.validate_span(("key",), run, 1, 6, check) == 2
    assert checked == [(0, 1, 2), (3,), (4, 5)]
    with pytest.raises(ValueError, match="bad bar"):
        scope.validate_span(("key",), run, 1, 8, check)
    # The failed span is not recorded as validated; the next attempt rechecks it.
    with pytest.raises(ValueError, match="bad bar"):
        scope.validate_span(("key",), run, 1, 8, check)
    # A different key (e.g. another dataset metadata object) starts afresh.
    assert scope.validate_span(("other",), run, 1, 6, check) == 5


def test_input_start_changes_ema_seed_so_series_are_keyed_by_start() -> None:
    """A later input start seeds differently: prepared series never share it."""
    bars = walk(80, seed=50)
    indicator = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(8), backend_id=TALIB_INDICATOR_BACKEND
    )
    from_zero = indicator.calculate_bar_fields(bars)[0].values
    from_one = indicator.calculate_bar_fields(bars[1:])[0].values
    assert from_zero[:7] == (None,) * 7
    assert from_one[:7] == (None,) * 7
    # Same final bar, different seed window: values differ, as QF-45 requires the
    # rolling daily start to produce a distinct EMA series.
    assert from_zero[-1] != from_one[-1]
