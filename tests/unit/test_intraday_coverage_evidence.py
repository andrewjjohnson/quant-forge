"""Retained coverage accepts diagnostic facts produced by the intraday contracts."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data import (
    IntradayBarBatch,
    IntradayValidationMode,
    validate_intraday_coverage,
)
from quantforge.data.intraday_coverage_evidence import validate_retained_coverage_report
from quantforge.timeframes import BarCompletion, DevelopingBarExposure, SessionScope
from tests.unit.data.test_intraday_validation import (
    NEW_YORK,
    _bar,  # pyright: ignore[reportPrivateUsage]
    _request,  # pyright: ignore[reportPrivateUsage]
    _session_bars,  # pyright: ignore[reportPrivateUsage]
    _timeframe,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "missing",
        "empty",
        "zero_volume",
        "early_close",
        "dst",
        "extended",
        "holiday",
        "developing",
    ],
)
def test_retained_diagnostic_coverage_roundtrip(case: str) -> None:
    session_date = date(2024, 7, 1)
    timeframe = _timeframe()
    start_hour, start_minute, end_hour = 9, 30, 16
    if case == "early_close":
        session_date = date(2024, 11, 29)
        timeframe = _timeframe(timedelta(hours=4))
        end_hour = 13
    elif case == "dst":
        session_date = date(2024, 3, 8)
    elif case == "extended":
        timeframe = _timeframe(
            timedelta(hours=1), session_scope=SessionScope.EXTENDED_HOURS
        )
        start_hour, start_minute, end_hour = 4, 0, 20
    elif case == "holiday":
        session_date = date(2024, 7, 4)
    elif case == "developing":
        timeframe = replace(
            timeframe, developing_bar_exposure=DevelopingBarExposure.INCLUDE
        )
    start = datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        start_hour,
        start_minute,
        tzinfo=NEW_YORK,
    )
    end_date = date(2024, 3, 11) if case == "dst" else session_date
    end = datetime(
        end_date.year, end_date.month, end_date.day, end_hour, tzinfo=NEW_YORK
    )
    if case == "developing":
        end = start + timedelta(minutes=3)
    request = _request(start, end, timeframe=timeframe)
    if case in {"holiday", "empty"}:
        bars = ()
    elif case == "developing":
        bars = (_bar(request, session_date, start, end, BarCompletion.DEVELOPING),)
    else:
        bars = _session_bars(request, session_date)
        if case == "dst":
            bars = (*bars, *_session_bars(request, end_date))
        elif case == "missing":
            bars = bars[1:]
        elif case == "zero_volume":
            bars = (replace(bars[0], volume=Decimal(0)), *bars[1:])
    batch = IntradayBarBatch(request, bars)
    report = validate_intraday_coverage(batch, mode=IntradayValidationMode.DIAGNOSTIC)
    manifest: PrimitiveMapping = {
        "request": {
            "request_id": request.request_id,
            "configuration": request.to_primitive(),
        },
        "batch_id": batch.batch_id,
        "bar_count": len(bars),
        "feed_scope": request.feed_scope.to_primitive(),
        "source_interval": request.source_interval.to_primitive(),
        "session_scope": request.timeframe.session_policy.scope.value,
        "quality_report": {
            "report_id": report.report_id,
            "report": report.to_primitive(),
        },
    }
    assert validate_retained_coverage_report(manifest) == report
