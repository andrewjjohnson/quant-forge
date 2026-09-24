"""QF-54 parsing, acquisition, secrets, identity, cache, and isolation."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

import pytest

import quantforge.data.providers._massive_http as http_module
from quantforge.data import (
    AdjustmentMode,
    FeedScope,
    IntradayBarBatch,
    IntradayCoverageValidationError,
    IntradayMarketDataCache,
    IntradayMarketDataService,
    IntradayValidationMode,
    ProviderError,
    RequestError,
    UnsupportedCapabilityError,
    validate_intraday_coverage,
)
from quantforge.data.models import JsonValue
from quantforge.data.providers import (
    MassiveProvider,
    TiingoProvider,
    can_reuse_intraday_cache,
    create_intraday_provider,
)
from quantforge.timeframes import IntradayInterval, Timeframe
from tests.fixtures.massive.helpers import (
    NEXT,
    START,
    TOKEN,
    install_responses,
    page,
    request,
    row,
    rows,
)


@pytest.mark.parametrize("minutes", [1, 5])
@pytest.mark.parametrize("adjusted", [False, True])
def test_mapping_boundaries_and_explicit_adjustment(
    monkeypatch: pytest.MonkeyPatch, minutes: int, adjusted: bool
) -> None:
    logical = request(
        end=START + timedelta(minutes=3 * minutes), minutes=minutes, adjusted=adjusted
    )
    records = rows(count=3, minutes=minutes)
    # Provider snapping can return both a leading bar and the exclusive end.
    records = [
        row(START - timedelta(minutes=minutes)),
        *records,
        row(logical.end_timestamp),
    ]
    calls = install_responses(monkeypatch, [page(records, adjusted=adjusted)])
    provider = MassiveProvider(TOKEN)
    result = provider.fetch_intraday(logical)
    assert len(result.batch.bars) == 3
    first = result.batch.bars[0]
    assert (first.open, first.high, first.low, first.close, first.volume) == tuple(
        Decimal(item) for item in (100, 103, 99, 102, 1000)
    )
    assert first.start_timestamp == START
    assert first.provenance.provider_name == "massive"
    assert first.provenance.adjustment_basis == logical.adjustment_basis
    assert first.end_timestamp == START + timedelta(minutes=minutes)
    assert result.batch.bars[-1].end_timestamp == logical.end_timestamp
    url = urlsplit(calls[0].full_url)
    assert url.netloc == "api.massive.com"
    assert url.path.endswith(
        f"/range/{minutes}/minute/{int(START.timestamp() * 1000)}/"
        f"{int(logical.end_timestamp.timestamp() * 1000) - 1}"
    )
    assert parse_qs(url.query) == {
        "adjusted": [str(adjusted).lower()],
        "sort": ["asc"],
        "limit": ["50000"],
    }
    assert calls[0].get_header("Authorization") == f"Bearer {TOKEN}"
    assert provider.http_request_count == 1
    assert TOKEN not in result.raw_snapshots[0].serialize().decode()


def test_fractional_range_retains_only_complete_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical = request(
        START + timedelta(microseconds=123),
        START + timedelta(minutes=2, microseconds=123),
    )
    calls = install_responses(monkeypatch, [page(rows())])
    result = MassiveProvider(TOKEN).fetch_intraday(logical)
    assert [bar.start_timestamp for bar in result.batch.bars] == [
        START + timedelta(minutes=1)
    ]
    assert urlsplit(calls[0].full_url).path.endswith(
        f"/{int(START.timestamp() * 1000)}/"
        f"{int((START + timedelta(minutes=2)).timestamp() * 1000)}"
    )


@pytest.mark.parametrize("page_count", [1, 2, 3])
def test_all_pages_deterministic_cache_and_offline_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, page_count: int
) -> None:
    records = rows()
    responses: list[JsonValue] = []
    for index in range(page_count):
        selected = (
            records[index : index + 1] if index < page_count - 1 else records[index:]
        )
        next_url = (
            NEXT.replace("page2", f"page{index + 2}")
            if index + 1 < page_count
            else None
        )
        responses.append(page(selected, next_url))
    calls = install_responses(monkeypatch, responses * 2)
    provider = MassiveProvider(TOKEN)
    first = provider.fetch_intraday(request())
    second = provider.fetch_intraday(request())
    assert first.batch == second.batch
    assert first.raw_snapshots[0].serialize() == second.raw_snapshots[0].serialize()
    assert len(first.raw_snapshots) == 1
    assert len(first.raw_snapshots[0].records) == page_count
    assert len(calls) == 2 * page_count
    assert all(call.get_header("Authorization") == f"Bearer {TOKEN}" for call in calls)
    cache = IntradayMarketDataCache(tmp_path)
    persisted = cache.persist(first)
    assert persisted.quality_report.is_complete
    # Supplying a provider on a cache hit must still avoid another HTTP call.
    assert (
        IntradayMarketDataService(cache, provider=provider).get_intraday_bars(request())
        == persisted
    )
    assert (
        IntradayMarketDataService(cache, provider_name="massive").get_intraday_bars(
            request()
        )
        == persisted
    )
    assert cache.load(persisted.metadata.dataset_id, request()) == persisted
    assert len(calls) == 2 * page_count
    assert cache.find("tiingo", request()) is None
    assert (
        cache.find(
            "massive", replace(request(), end_timestamp=START + timedelta(minutes=4))
        )
        is None
    )


def test_safe_pagination_and_raw_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = page(rows(count=1), NEXT + "&apiKey=" + TOKEN + "&token=another-secret")
    payload["diagnostic"] = TOKEN
    payload["api_key"] = "another-secret"
    calls = install_responses(
        monkeypatch, [payload, page(rows(START + timedelta(minutes=1), 2))]
    )
    result = MassiveProvider(TOKEN).fetch_intraday(request())
    raw = result.raw_snapshots[0].serialize().decode()
    assert TOKEN not in raw
    assert "another-secret" not in raw
    assert "apiKey=" not in raw
    assert all(TOKEN not in call.full_url for call in calls)
    assert parse_qs(urlsplit(calls[1].full_url).query) == {"cursor": ["page2"]}


@pytest.mark.parametrize(
    "next_url",
    [
        "http://api.massive.com/v2/aggs/ticker/SPY/range/1/minute/1/2?cursor=x",
        NEXT.replace("api.massive.com", "evil.example"),
        NEXT.replace("SPY", "QQQ"),
        NEXT.replace("/1/minute/", "/5/minute/"),
        NEXT.replace("api.massive.com", "user@api.massive.com"),
        NEXT + "#fragment",
        NEXT + "&sort=desc",
        NEXT + "&adjusted=true",
        NEXT + "&cursor=other",
        "",
        NEXT + "&cursor=" + TOKEN,
    ],
)
def test_unsafe_next_url_fails_before_second_request(
    monkeypatch: pytest.MonkeyPatch, next_url: str
) -> None:
    calls = install_responses(monkeypatch, [page(rows(count=1), next_url)])
    with pytest.raises(ProviderError, match="pagination") as caught:
        MassiveProvider(TOKEN).fetch_intraday(request())
    assert TOKEN not in str(caught.value)
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["duplicate", "loop", "malformed", "http"])
def test_partial_fetch_never_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    second: JsonValue | Exception = {
        "duplicate": page(rows()),
        "loop": page(rows(START + timedelta(minutes=1), 2), NEXT),
        "malformed": page([{"t": "bad"}]),
        "http": URLError(TOKEN),
    }[failure]
    install_responses(monkeypatch, [page(rows(count=1), NEXT), second])
    cache = IntradayMarketDataCache(tmp_path)
    with pytest.raises(ProviderError) as caught:
        IntradayMarketDataService(
            cache, provider=MassiveProvider(TOKEN, retry_delays=())
        ).get_intraday_bars(request())
    assert TOKEN not in str(caught.value)
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("o", None),
        ("o", "100"),
        ("h", False),
        ("l", "secret"),
        ("c", -1),
        ("h", 98),
        ("l", 104),
        ("v", -1),
        ("t", None),
        ("t", 1.1),
        ("t", True),
        ("t", 10**30),
        ("t", int(START.timestamp() * 1000) + 1),
    ],
)
def test_malformed_required_fields_fail_with_safe_context(
    monkeypatch: pytest.MonkeyPatch, field: str, bad: JsonValue
) -> None:
    record = row()
    record[field] = bad
    install_responses(monkeypatch, [page([record])])
    with pytest.raises(ProviderError, match="page=1 row=0 timestamp=") as caught:
        MassiveProvider(TOKEN).fetch_intraday(request(end=START + timedelta(minutes=1)))
    assert "SPY" in str(caught.value)
    assert "range [" in str(caught.value)


@pytest.mark.parametrize("field", ["o", "h", "l", "c", "v", "t"])
def test_missing_required_field(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    record = row()
    del record[field]
    install_responses(monkeypatch, [page([record])])
    with pytest.raises(ProviderError):
        MassiveProvider(TOKEN).fetch_intraday(request())


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "bad",
        {"status": "ERROR", "error": TOKEN},
        page([None]),
        {**page(rows()), "results": {}},
        {**page(rows()), "resultsCount": 2},
        {**page(rows()), "ticker": "QQQ"},
        {**page(rows()), "adjusted": True},
        {**page(rows()), "next_url": 1},
        b"not-json",
        b"\xff",
        page([row(o=float("nan"))]),
        page([row(v=float("inf"))]),
    ],
)
def test_invalid_payload(
    monkeypatch: pytest.MonkeyPatch, payload: JsonValue | bytes
) -> None:
    install_responses(monkeypatch, [payload])
    with pytest.raises(ProviderError) as caught:
        MassiveProvider(TOKEN).fetch_intraday(request())
    assert TOKEN not in str(caught.value)


@pytest.mark.parametrize("status", [301, 401, 403, 404, 429, 500, 503])
def test_http_errors_are_safe_and_bounded(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    failure = HTTPError(NEXT + "&apiKey=" + TOKEN, status, TOKEN, Message(), None)
    calls = install_responses(monkeypatch, [failure, failure])
    provider = MassiveProvider(TOKEN, retry_delays=(0,))
    with pytest.raises(ProviderError, match=f"HTTP status {status}") as caught:
        provider.fetch_intraday(request())
    assert TOKEN not in str(caught.value)
    assert len(calls) == (2 if status in (429, 500, 503) else 1)
    assert caught.value.__suppress_context__


def test_transient_retry_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_responses(monkeypatch, [URLError(TOKEN), page(rows())])
    provider = MassiveProvider(TOKEN, retry_delays=(0,))
    assert len(provider.fetch_intraday_bars(request()).bars) == 3
    assert provider.http_request_count == len(calls) == 2


def test_redirects_do_not_forward_authorization() -> None:
    from io import BytesIO

    assert (
        http_module.NoRedirect().redirect_request(
            Request(NEXT), BytesIO(), 302, "found", Message(), "https://evil.example"
        )
        is None
    )


def test_provider_selection_and_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASSIVE_API_KEY", TOKEN)
    monkeypatch.setenv("TIINGO_API_KEY", "tiingo-secret")
    assert isinstance(create_intraday_provider("massive"), MassiveProvider)
    assert isinstance(create_intraday_provider("tiingo"), TiingoProvider)
    assert TiingoProvider.intraday_adapter_version == "2"
    monkeypatch.delenv("MASSIVE_API_KEY")
    with pytest.raises(RequestError, match="MASSIVE_API_KEY"):
        create_intraday_provider("massive")
    with pytest.raises(RequestError, match="unsupported"):
        create_intraday_provider("unknown")
    calls = install_responses(monkeypatch, [page(rows())])
    provider = MassiveProvider(TOKEN)
    with pytest.raises(UnsupportedCapabilityError):
        provider.fetch_intraday(replace(request(), feed_scope=FeedScope.iex_only()))
    with pytest.raises(UnsupportedCapabilityError):
        provider.fetch_intraday(
            replace(
                request(),
                timeframe=Timeframe.us_equity(IntradayInterval(timedelta(minutes=2))),
            )
        )
    with pytest.raises(RequestError, match="explicit raw or split-adjusted"):
        provider.fetch_intraday(
            replace(
                request(),
                adjustment_basis=replace(
                    MassiveProvider.split_adjusted_basis,
                    adjustment_mode=AdjustmentMode.SPLIT_AND_DIVIDEND_ADJUSTED,
                ),
            )
        )
    assert not calls


def test_ordering_volume_and_material_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(monkeypatch, [page(list(reversed(rows())))])
    original = MassiveProvider(TOKEN).fetch_intraday(request())
    assert [bar.start_timestamp for bar in original.batch.bars] == sorted(
        bar.start_timestamp for bar in original.batch.bars
    )
    assert request().request_id != request(adjusted=True).request_id
    assert request().request_id != request(minutes=5).request_id
    assert request().request_id != request(end=START + timedelta(minutes=4)).request_id
    assert (
        request().request_id
        != replace(request(), feed_scope=FeedScope.iex_only()).request_id
    )
    snapshot = original.raw_snapshots[0]
    assert snapshot.to_primitive()["provider_name"] == "massive"
    from quantforge.data.identity import canonical_json_bytes, sha256_hex

    assert snapshot.snapshot_id != sha256_hex(
        canonical_json_bytes({**snapshot.to_primitive(), "provider_name": "tiingo"})
    )


def test_cache_policy_preserves_other_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_responses(monkeypatch, [page(rows(count=2))])
    result = MassiveProvider(TOKEN).fetch_intraday(request())
    cache = IntradayMarketDataCache(tmp_path)
    diagnostic = cache.persist(result)
    report = diagnostic.quality_report
    assert not report.is_complete
    assert can_reuse_intraday_cache("massive", report)
    assert not can_reuse_intraday_cache("tiingo", report)
    assert can_reuse_intraday_cache("fake", report)
    assert (
        IntradayMarketDataService(cache, provider_name="massive").get_intraday_bars(
            request()
        )
        == diagnostic
    )
    assert cache.load(diagnostic.metadata.dataset_id, request()) == diagnostic


def test_failed_refresh_keeps_previous_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_responses(
        monkeypatch, [page(rows()), page(rows(count=1), NEXT), b"invalid"]
    )
    cache = IntradayMarketDataCache(tmp_path)
    service = IntradayMarketDataService(cache, provider=MassiveProvider(TOKEN))
    original = service.get_intraday_bars(request())
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    with pytest.raises(ProviderError):
        service.get_intraday_bars(request(), refresh=True)
    assert cache.find("massive", request()) == original
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*.json")}


@pytest.mark.parametrize("zero_volume", [0, 0.25])
def test_nonnegative_fractional_volume_preserved(
    monkeypatch: pytest.MonkeyPatch, zero_volume: float
) -> None:
    install_responses(monkeypatch, [page([row(v=zero_volume)])])
    result = MassiveProvider(TOKEN).fetch_intraday(
        request(end=START + timedelta(minutes=1))
    )
    assert result.batch.bars[0].volume == Decimal(str(zero_volume))


def test_pages_without_retained_bars_still_follow_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_responses(
        monkeypatch,
        [
            page([row(START - timedelta(minutes=1))], NEXT),
            page([], NEXT.replace("page2", "page3")),
            page(rows()),
        ],
    )
    result = MassiveProvider(TOKEN).fetch_intraday(request())
    assert len(calls) == 3
    assert len(result.raw_snapshots[0].records) == 3
    assert len(result.batch.bars) == 3


def test_documented_date_bounds_in_next_url_keep_exact_logical_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    next_url = NEXT.replace("1719840779999", "2024-07-01")
    calls = install_responses(
        monkeypatch,
        [
            page(rows(count=1), next_url),
            page(rows(START + timedelta(minutes=1), 3)),
        ],
    )
    result = MassiveProvider(TOKEN).fetch_intraday(request())
    assert calls[1].full_url == next_url
    assert len(result.batch.bars) == 3
    assert result.batch.bars[-1].end_timestamp == request().end_timestamp


@pytest.mark.parametrize("missing_index", [0, 1, 2])
@pytest.mark.parametrize("minutes", [1, 5])
def test_missing_bar_preserved_through_pagination_and_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_index: int, minutes: int
) -> None:
    logical = request(end=START + timedelta(minutes=3 * minutes), minutes=minutes)
    records = rows(minutes=minutes)
    del records[missing_index]
    calls = install_responses(
        monkeypatch,
        [
            page(records[:1], NEXT.replace("/1/minute/", f"/{minutes}/minute/")),
            page(records[1:]),
        ],
    )
    cache = IntradayMarketDataCache(tmp_path)
    service = IntradayMarketDataService(cache, provider=MassiveProvider(TOKEN))
    dataset = service.get_intraday_bars(logical)
    assert [bar.start_timestamp for bar in dataset.bars] == [
        START + timedelta(minutes=index * minutes)
        for index in range(3)
        if index != missing_index
    ]
    report = dataset.quality_report
    assert report.validation_mode is IntradayValidationMode.DIAGNOSTIC
    assert not report.is_complete
    assert report.observed_bar_count == 2
    assert report.expected_completed_interval_count == 3
    assert [interval.start_timestamp for interval in report.missing_intervals] == [
        START + timedelta(minutes=missing_index * minutes)
    ]
    assert service.get_intraday_bars(logical) == dataset
    assert (
        IntradayMarketDataService(cache, provider_name="massive").get_intraday_bars(
            logical
        )
        == dataset
    )
    assert len(calls) == 2
    # A strict consumer still rejects the same observations using existing policy.
    with pytest.raises(IntradayCoverageValidationError) as caught:
        validate_intraday_coverage(IntradayBarBatch(logical, dataset.bars))
    assert caught.value.report.missing_intervals == report.missing_intervals


@pytest.mark.parametrize("response_kind", ["empty", "omitted", "outside_rth"])
def test_no_retained_observations_preserve_empty_batch_and_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, response_kind: str
) -> None:
    payload = page([])
    if response_kind == "omitted":
        del payload["results"]
    elif response_kind == "outside_rth":
        payload = page([row(START - timedelta(minutes=1))])
    calls = install_responses(monkeypatch, [payload])
    cache = IntradayMarketDataCache(tmp_path)
    logical = request(start=START - timedelta(minutes=1))
    dataset = IntradayMarketDataService(
        cache, provider=MassiveProvider(TOKEN)
    ).get_intraday_bars(logical)
    assert dataset.bars == ()
    assert not dataset.quality_report.is_complete
    assert dataset.quality_report.expected_completed_interval_count == 3
    assert len(dataset.quality_report.missing_intervals) == 3
    assert (
        IntradayMarketDataService(cache, provider_name="massive").get_intraday_bars(
            logical
        )
        == dataset
    )
    assert len(calls) == 1


def test_duplicate_in_partial_final_bar_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    tail = row(START + timedelta(minutes=2))
    install_responses(monkeypatch, [page([*rows(), tail])])
    with pytest.raises(ProviderError, match="duplicate canonical bar"):
        MassiveProvider(TOKEN).fetch_intraday(
            request(end=START + timedelta(minutes=2, seconds=30))
        )


@pytest.mark.parametrize("api_key", ["", " ", "bad\nkey"])
def test_invalid_key_is_not_echoed(api_key: str) -> None:
    with pytest.raises(RequestError, match="MASSIVE_API_KEY"):
        MassiveProvider(api_key)


def test_cache_artifacts_contain_no_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_responses(
        monkeypatch, [{**page(rows()), "debug": f"Bearer {TOKEN}", "apiKey": TOKEN}]
    )
    cache = IntradayMarketDataCache(tmp_path)
    IntradayMarketDataService(cache, provider=MassiveProvider(TOKEN)).get_intraday_bars(
        request()
    )
    assert all(
        TOKEN.encode() not in path.read_bytes() for path in tmp_path.rglob("*.json")
    )
