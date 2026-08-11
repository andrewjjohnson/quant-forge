import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, time, timedelta
from email.message import Message
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest

import quantforge.data.providers.tiingo as tiingo_module
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    FeedScope,
    IntradayBarRequest,
    IntradayMarketDataCache,
    IntradayMarketDataService,
    ProviderError,
    RequestError,
    UnsupportedSessionScopeError,
)
from quantforge.data.providers import TiingoProvider
from quantforge.timeframes import (
    BarLabel,
    ExchangeSessionPolicy,
    IntradayInterval,
    SessionScope,
    Timeframe,
)

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "tiingo"
TOKEN = "secret-intraday-test-token"


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


def _adjustment_basis() -> AdjustmentBasis:
    return AdjustmentBasis(
        adjustment_mode=AdjustmentMode.UNADJUSTED,
        ohlc_basis="raw_provider",
        volume_basis="raw_provider",
        corporate_action_policy="not_provided_for_intraday_bars",
        adjusted_fields_used=False,
    )


def _request(
    *,
    feed_scope: FeedScope | None = None,
    interval_minutes: int = 5,
    end_timestamp: datetime = datetime(2024, 7, 1, 13, 45, tzinfo=UTC),
) -> IntradayBarRequest:
    return IntradayBarRequest(
        symbol="SPY",
        start_timestamp=datetime(2024, 7, 1, 13, 30, tzinfo=UTC),
        end_timestamp=end_timestamp,
        timeframe=Timeframe.us_equity(
            IntradayInterval(timedelta(minutes=interval_minutes))
        ),
        feed_scope=feed_scope or FeedScope.consolidated(),
        adjustment_basis=_adjustment_basis(),
    )


def test_tiingo_intraday_fetches_chunks_caches_and_replays_without_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_responses(
        monkeypatch,
        (
            _fixture("spy_intraday_5min_chunk_1.json"),
            _fixture("spy_intraday_5min_chunk_2.json"),
        ),
    )
    request = _request()
    cache = IntradayMarketDataCache(tmp_path)
    provider = TiingoProvider(
        TOKEN,
        timeout=7.5,
        retry_delays=(),
        intraday_chunk_duration=timedelta(minutes=10),
    )

    dataset = IntradayMarketDataService(cache, provider=provider).get_intraday_bars(
        request
    )
    replayed = IntradayMarketDataService(
        cache, provider_name="tiingo"
    ).get_intraday_bars(request)

    assert replayed == dataset
    assert len(calls) == 2
    assert [bar.start_timestamp for bar in dataset.bars] == [
        datetime(2024, 7, 1, 13, 30, tzinfo=UTC),
        datetime(2024, 7, 1, 13, 35, tzinfo=UTC),
        datetime(2024, 7, 1, 13, 40, tzinfo=UTC),
    ]
    assert len({bar.start_timestamp for bar in dataset.bars}) == 3
    assert dataset.metadata.raw_snapshot_ids == tuple(
        bar.provenance.source_snapshot_id for bar in (dataset.bars[0], dataset.bars[-1])
    )
    assert cache.load(dataset.metadata.dataset_id, request) == dataset

    first_query = parse_qs(urlparse(calls[0][0].full_url).query)
    second_query = parse_qs(urlparse(calls[1][0].full_url).query)
    assert urlparse(calls[0][0].full_url).path.endswith(
        "/tiingo/equity/intraday/SPY/prices"
    )
    assert first_query == {
        "afterHours": ["false"],
        "columns": ["open,high,low,close,volume"],
        "endDate": ["2024-07-01"],
        "forceFill": ["false"],
        "resampleFreq": ["5min"],
        "startDate": ["2024-07-01"],
    }
    assert second_query["startDate"] == first_query["endDate"]
    for call, timeout in calls:
        assert TOKEN not in call.full_url
        assert dict(call.header_items())["Authorization"] == f"Token {TOKEN}"
        assert timeout == 7.5

    manifest_path = (
        tmp_path
        / "intraday"
        / "datasets"
        / dataset.metadata.dataset_id
        / "manifest.json"
    )
    manifest = cast(dict[str, object], json.loads(manifest_path.read_text()))
    assert manifest["provider_name"] == "tiingo"
    assert manifest["adapter_version"] == provider.intraday_adapter_version
    assert isinstance(manifest["retrieved_at"], str)
    assert manifest["feed_scope"] == FeedScope.consolidated().to_primitive()
    assert manifest["source_interval"] == request.source_interval.to_primitive()
    assert manifest["session_scope"] == SessionScope.REGULAR_HOURS.value
    manifest_request = cast(dict[str, object], manifest["request"])
    assert manifest_request["configuration"] == request.to_primitive()
    chunks = cast(list[dict[str, object]], manifest["chunks"])
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1]
    assert all(
        chunk["endpoint"] == "tiingo/equity/intraday/<ticker>/prices"
        for chunk in chunks
    )
    assert all(chunk["raw_sha256"] == chunk["raw_snapshot_id"] for chunk in chunks)
    assert all(isinstance(chunk["retrieved_at"], str) for chunk in chunks)
    assert isinstance(manifest["data_sha256"], str)
    assert TOKEN not in manifest_path.read_text()
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert TOKEN not in artifact.read_text()


def test_tiingo_intraday_iex_and_one_minute_scope_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_responses(monkeypatch, (_fixture("spy_intraday_1min.json"),))
    request = _request(
        feed_scope=FeedScope.iex_only(),
        interval_minutes=1,
        end_timestamp=datetime(2024, 7, 1, 13, 31, tzinfo=UTC),
    )

    batch = TiingoProvider(TOKEN, retry_delays=()).fetch_intraday_bars(request)

    assert len(batch.bars) == 1
    assert batch.bars[0].provenance.feed_scope == FeedScope.iex_only()
    parsed = urlparse(calls[0][0].full_url)
    assert parsed.path.endswith("/iex/SPY/prices")
    assert parse_qs(parsed.query)["resampleFreq"] == ["1min"]
    capabilities = TiingoProvider.intraday_capabilities
    assert capabilities.supported_feed_scopes == (
        FeedScope.consolidated(),
        FeedScope.iex_only(),
    )
    assert capabilities.supported_session_scopes == (SessionScope.REGULAR_HOURS,)


def test_tiingo_intraday_rejects_unsupported_policies_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_responses(monkeypatch, ())
    provider = TiingoProvider(TOKEN, retry_delays=())
    request = _request()
    extended_request = replace(
        request,
        timeframe=Timeframe(
            request.source_interval,
            ExchangeSessionPolicy(
                scope=SessionScope.EXTENDED_HOURS,
                extended_hours_start=time(4),
                extended_hours_end=time(20),
            ),
        ),
    )
    noncanonical_label_request = replace(
        request,
        timeframe=Timeframe(request.source_interval, bar_label=BarLabel.END),
    )
    incompatible_basis_request = replace(
        request,
        adjustment_basis=AdjustmentBasis(
            adjustment_mode=AdjustmentMode.UNADJUSTED,
            ohlc_basis="raw_provider",
            volume_basis="raw_provider",
            corporate_action_policy=(
                "separate_provider_reported_cash_dividends_and_splits"
            ),
            adjusted_fields_used=False,
        ),
    )

    with pytest.raises(UnsupportedSessionScopeError):
        provider.fetch_intraday_bars(extended_request)
    with pytest.raises(RequestError, match="canonical XNYS"):
        provider.fetch_intraday_bars(noncanonical_label_request)
    with pytest.raises(RequestError, match="corporate actions"):
        provider.fetch_intraday_bars(incompatible_basis_request)
    assert not calls


def test_tiingo_intraday_refresh_preserves_old_immutable_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    revised = cast(
        list[dict[str, object]],
        json.loads(_fixture("spy_intraday_5min_chunk_2.json")),
    )
    revised[0]["close"] = 101.25
    revised[0]["high"] = 101.35
    calls = _install_responses(
        monkeypatch,
        (
            _fixture("spy_intraday_5min_chunk_1.json"),
            _fixture("spy_intraday_5min_chunk_2.json"),
            _fixture("spy_intraday_5min_chunk_1.json"),
            json.dumps(revised).encode(),
        ),
    )
    request = _request()
    cache = IntradayMarketDataCache(tmp_path)
    service = IntradayMarketDataService(
        cache,
        provider=TiingoProvider(
            TOKEN,
            retry_delays=(),
            intraday_chunk_duration=timedelta(minutes=10),
        ),
    )

    original = service.get_intraday_bars(request)
    original_raw = tuple(
        (tmp_path / location).read_bytes()
        for location in original.metadata.raw_locations
    )
    refreshed = service.get_intraday_bars(request, refresh=True)

    assert len(calls) == 4
    assert refreshed.metadata.dataset_id != original.metadata.dataset_id
    assert cache.load(original.metadata.dataset_id, request) == original
    assert (
        tuple(
            (tmp_path / location).read_bytes()
            for location in original.metadata.raw_locations
        )
        == original_raw
    )
    assert service.get_intraday_bars(request) == refreshed


def test_tiingo_intraday_errors_do_not_expose_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = HTTPError(
        "https://api.tiingo.com/tiingo/equity/intraday/SPY/prices",
        401,
        "",
        Message(),
        None,
    )
    calls = _install_responses(monkeypatch, (error,))

    with pytest.raises(ProviderError, match="authentication") as raised:
        TiingoProvider(TOKEN, retry_delays=()).fetch_intraday_bars(_request())

    assert TOKEN not in str(raised.value)
    assert TOKEN not in calls[0][0].full_url


@pytest.mark.parametrize(
    ("fixture_name", "message"),
    [("malformed.json", "malformed intraday prices JSON"), ("empty.json", "empty")],
)
def test_tiingo_intraday_rejects_malformed_or_empty_payloads(
    monkeypatch: pytest.MonkeyPatch,
    fixture_name: str,
    message: str,
) -> None:
    _install_responses(monkeypatch, (_fixture(fixture_name),))

    with pytest.raises(ProviderError, match=message):
        TiingoProvider(TOKEN, retry_delays=()).fetch_intraday_bars(_request())


def test_tiingo_intraday_rejects_duplicate_provider_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate_rows = cast(
        list[dict[str, object]],
        json.loads(_fixture("spy_intraday_1min.json")),
    )
    duplicate_rows.append(duplicate_rows[0].copy())
    _install_responses(monkeypatch, (json.dumps(duplicate_rows).encode(),))
    request = _request(
        feed_scope=FeedScope.iex_only(),
        interval_minutes=1,
        end_timestamp=datetime(2024, 7, 1, 13, 31, tzinfo=UTC),
    )

    with pytest.raises(ProviderError, match="duplicate bar key"):
        TiingoProvider(TOKEN, retry_delays=()).fetch_intraday_bars(request)


def test_prediction_and_strategy_layers_do_not_import_tiingo() -> None:
    package_root = Path(__file__).parents[3] / "src" / "quantforge"
    downstream_sources = tuple((package_root / "prediction").rglob("*.py")) + tuple(
        (package_root / "strategies").rglob("*.py")
    )

    assert downstream_sources
    for source_path in downstream_sources:
        source = source_path.read_text()
        assert "quantforge.data.providers.tiingo" not in source
        assert "TiingoProvider" not in source
