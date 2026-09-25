"""Deterministic HTTP fixtures shared by provider and composition tests."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta, tzinfo
from urllib.request import Request

import pytest

import quantforge.data.providers._massive_http as http_module
import quantforge.data.providers.massive as massive_module
from quantforge.data import FeedScope, IntradayBarRequest
from quantforge.data.models import JsonValue
from quantforge.data.providers import MassiveProvider
from quantforge.timeframes import IntradayInterval, Timeframe

TOKEN = "massive-test-secret-never-persist"
START = datetime(2024, 7, 1, 13, 30, tzinfo=UTC)
NEXT = "https://api.massive.com/v2/aggs/ticker/SPY/range/1/minute/1719840660000/1719840779999?cursor=page2"


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def read(self) -> bytes:
        return self.payload


def install_responses(
    monkeypatch: pytest.MonkeyPatch, responses: Sequence[JsonValue | bytes | Exception]
) -> list[Request]:
    remaining = list(responses)
    calls: list[Request] = []

    class FakeOpener:
        def open(self, request: Request, *, timeout: float) -> FakeResponse:
            calls.append(request)
            response = remaining.pop(0)
            if isinstance(response, Exception):
                raise response
            return FakeResponse(
                response
                if isinstance(response, bytes)
                else json.dumps(response).encode()
            )

    def fake_build_opener(*_args: object) -> FakeOpener:
        return FakeOpener()

    monkeypatch.setattr(http_module, "build_opener", fake_build_opener)
    monkeypatch.setattr(massive_module, "datetime", FrozenDateTime)
    return calls


def request(
    start: datetime = START,
    end: datetime = START + timedelta(minutes=3),
    *,
    minutes: int = 1,
    adjusted: bool = False,
) -> IntradayBarRequest:
    return IntradayBarRequest(
        "SPY",
        start,
        end,
        Timeframe.us_equity(IntradayInterval(timedelta(minutes=minutes))),
        FeedScope.consolidated(),
        MassiveProvider.split_adjusted_basis
        if adjusted
        else MassiveProvider.intraday_adjustment_basis,
    )


def row(start: datetime = START, **changes: JsonValue) -> dict[str, JsonValue]:
    return {
        "t": int(start.timestamp() * 1000),
        "o": 100,
        "h": 103,
        "l": 99,
        "c": 102,
        "v": 1000,
        **changes,
    }


def page(
    rows: Sequence[JsonValue], next_url: str | None = None, *, adjusted: bool = False
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {
        "status": "OK",
        "ticker": "SPY",
        "adjusted": adjusted,
        "resultsCount": len(rows),
        "results": list(rows),
    }
    if next_url is not None:
        result["next_url"] = next_url
    return result


def rows(start: datetime = START, count: int = 3, minutes: int = 1) -> list[JsonValue]:
    return [row(start + timedelta(minutes=index * minutes)) for index in range(count)]
