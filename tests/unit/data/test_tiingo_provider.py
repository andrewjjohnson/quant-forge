import json
from collections.abc import Sequence
from datetime import date
from email.message import Message
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest

import quantforge.data.providers.tiingo as tiingo_module
from quantforge.data import (
    AdjustmentMode,
    CashDividend,
    MarketDataCache,
    MarketDataService,
    ProviderError,
    RequestError,
    StockSplit,
    ValidationError,
)
from quantforge.data.providers import TiingoProvider

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "tiingo"
TOKEN = "secret-test-token"


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _fixture(name: str) -> bytes:
    return (FIXTURE_ROOT / name).read_bytes()


def _install_responses(
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[bytes | Exception],
) -> list[tuple[Request, float]]:
    remaining = list(responses)
    calls: list[tuple[Request, float]] = []

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        calls.append((request, timeout))
        next_response = remaining.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return FakeResponse(next_response)

    monkeypatch.setattr(tiingo_module, "urlopen", fake_urlopen)
    return calls


def _provider_response(
    monkeypatch: pytest.MonkeyPatch,
    price_payload: bytes | None = None,
) -> tuple[TiingoProvider, list[tuple[Request, float]]]:
    calls = _install_responses(
        monkeypatch,
        (_fixture("spy_metadata.json"), price_payload or _fixture("spy_eod.json")),
    )
    return TiingoProvider(TOKEN, timeout=7.5, retry_delays=()), calls


def test_tiingo_request_uses_header_auth_and_maps_lossless_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, calls = _provider_response(monkeypatch)

    response = provider.fetch_daily_bars(
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 8),
        AdjustmentMode.UNADJUSTED,
    )

    assert len(calls) == 2
    metadata_request, metadata_timeout = calls[0]
    price_request, price_timeout = calls[1]
    assert metadata_request.full_url.endswith("/tiingo/daily/SPY")
    parsed = urlparse(price_request.full_url)
    assert parsed.path.endswith("/tiingo/daily/SPY/prices")
    assert parse_qs(parsed.query) == {
        "startDate": ["2024-07-01"],
        "endDate": ["2024-07-08"],
        "resampleFreq": ["daily"],
        "format": ["json"],
    }
    assert metadata_timeout == price_timeout == 7.5
    for request, _ in calls:
        assert TOKEN not in request.full_url
        assert dict(request.header_items())["Authorization"] == f"Token {TOKEN}"
    assert response.provider_name == "tiingo"
    assert response.provider_symbol == "SPY"
    assert response.records[1]["session_date"] == "2024-07-02"
    assert response.records[1]["dividend_amount"] == 0.5
    assert response.records[3]["split_coefficient"] == 2
    assert response.records[0]["adjOpen"] == 50
    assert response.records[0]["adjClose"] == 50.5
    assert response.metadata["response_format"] == "json"
    asset_metadata = response.metadata["asset_metadata"]
    assert isinstance(asset_metadata, dict)
    assert asset_metadata["name"] == "Synthetic SPY fixture"
    assert asset_metadata["exchangeCode"] == "NYSE ARCA"
    assert asset_metadata["startDate"] == "1993-01-29"
    assert asset_metadata["endDate"] == "2024-07-08"


def test_tiingo_rejects_adjusted_ingestion_and_invalid_configuration() -> None:
    with pytest.raises(RequestError, match="TIINGO_API_KEY"):
        TiingoProvider("")
    with pytest.raises(RequestError, match="timeout"):
        TiingoProvider(TOKEN, timeout=0)
    provider = TiingoProvider(TOKEN, retry_delays=())
    with pytest.raises(RequestError, match="only raw unadjusted"):
        provider.fetch_daily_bars(
            "SPY",
            date(2024, 7, 1),
            date(2024, 7, 8),
            AdjustmentMode.SPLIT_ADJUSTED,
        )


@pytest.mark.parametrize(
    ("payload_name", "message"),
    [("malformed.json", "malformed historical prices JSON"), ("empty.json", "empty")],
)
def test_tiingo_rejects_malformed_or_empty_prices(
    monkeypatch: pytest.MonkeyPatch,
    payload_name: str,
    message: str,
) -> None:
    provider, _ = _provider_response(monkeypatch, _fixture(payload_name))
    with pytest.raises(ProviderError, match=message):
        provider.fetch_daily_bars(
            "SPY",
            date(2024, 7, 1),
            date(2024, 7, 8),
            AdjustmentMode.UNADJUSTED,
        )


@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "authentication"), (403, "authentication"), (429, "rate limit")],
)
def test_tiingo_translates_http_errors_without_leaking_token(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    message: str,
) -> None:
    error = HTTPError(
        "https://api.tiingo.com/tiingo/daily/SPY", status, "", Message(), None
    )
    calls = _install_responses(monkeypatch, (error,))
    provider = TiingoProvider(TOKEN, retry_delays=())

    with pytest.raises(ProviderError, match=message) as raised:
        provider.fetch_daily_bars(
            "SPY",
            date(2024, 7, 1),
            date(2024, 7, 8),
            AdjustmentMode.UNADJUSTED,
        )

    assert TOKEN not in str(raised.value)
    assert TOKEN not in calls[0][0].full_url


def test_tiingo_retries_temporary_failure_only_within_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = HTTPError(
        "https://api.tiingo.com/tiingo/daily/SPY", 503, "", Message(), None
    )
    calls = _install_responses(
        monkeypatch,
        (temporary, _fixture("spy_metadata.json"), _fixture("spy_eod.json")),
    )
    provider = TiingoProvider(TOKEN, retry_delays=(0,))

    provider.fetch_daily_bars(
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 8),
        AdjustmentMode.UNADJUSTED,
    )

    assert len(calls) == 3


def test_tiingo_service_persists_actions_raw_fields_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, calls = _provider_response(monkeypatch)
    cache = MarketDataCache(tmp_path)
    service = MarketDataService(provider, cache)

    dataset = service.get_daily_bars(
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 8),
        AdjustmentMode.UNADJUSTED,
    )
    reloaded = MarketDataService(provider, cache).get_daily_bars(
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 8),
        AdjustmentMode.UNADJUSTED,
    )

    assert len(calls) == 2
    assert reloaded == dataset
    assert dataset.metadata.corporate_actions_complete
    assert dataset.metadata.dividend_count == 1
    assert dataset.metadata.split_count == 1
    assert len(dataset.corporate_actions) == 2
    assert isinstance(dataset.corporate_actions[0], CashDividend)
    assert isinstance(dataset.corporate_actions[1], StockSplit)
    assert all(action.action_id for action in dataset.corporate_actions)
    raw_text = (tmp_path / dataset.metadata.raw_location).read_text()
    assert '"adjOpen":50' in raw_text
    assert '"divCash":0.5' in raw_text
    assert TOKEN not in raw_text


def test_refresh_advances_request_index_without_mutating_old_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    revised_prices = cast(list[dict[str, object]], json.loads(_fixture("spy_eod.json")))
    revised_prices[-1]["close"] = 54
    revised_prices[-1]["high"] = 54
    calls = _install_responses(
        monkeypatch,
        (
            _fixture("spy_metadata.json"),
            _fixture("spy_eod.json"),
            _fixture("spy_metadata.json"),
            json.dumps(revised_prices).encode(),
        ),
    )
    cache = MarketDataCache(tmp_path)
    service = MarketDataService(TiingoProvider(TOKEN, retry_delays=()), cache)

    original = service.get_daily_bars(
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 8),
        AdjustmentMode.UNADJUSTED,
    )
    refreshed = service.get_daily_bars(
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 8),
        AdjustmentMode.UNADJUSTED,
        refresh=True,
    )

    assert len(calls) == 4
    assert refreshed.metadata.dataset_id != original.metadata.dataset_id
    assert cache.load(original.metadata.dataset_id) == original
    assert (
        service.get_daily_bars(
            "SPY",
            date(2024, 7, 1),
            date(2024, 7, 8),
            AdjustmentMode.UNADJUSTED,
        )
        == refreshed
    )


@pytest.mark.parametrize("mutation", ["duplicate", "missing"])
def test_tiingo_service_rejects_duplicate_or_missing_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    prices = cast(list[dict[str, object]], json.loads(_fixture("spy_eod.json")))
    if mutation == "duplicate":
        prices.append(prices[0])
        expected_message = "duplicate"
    else:
        del prices[2]
        expected_message = "missing expected sessions"
    provider, _ = _provider_response(
        monkeypatch,
        json.dumps(prices).encode(),
    )

    with pytest.raises(ValidationError, match=expected_message):
        MarketDataService(provider, MarketDataCache(tmp_path)).get_daily_bars(
            "SPY",
            date(2024, 7, 1),
            date(2024, 7, 8),
            AdjustmentMode.UNADJUSTED,
        )
