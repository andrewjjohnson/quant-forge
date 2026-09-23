"""QF-53 provider-only acquisition regressions; all HTTP responses are synthetic."""

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest

import quantforge.data.providers.tiingo as tiingo_module
from quantforge.data import (
    FeedScope,
    IntradayBarBatch,
    IntradayBarProvenance,
    IntradayBarRequest,
    IntradayFetchResult,
    IntradayMarketDataCache,
    IntradayMarketDataService,
    IntradayRawSnapshot,
    ProviderError,
    RequestError,
    validate_intraday_coverage,
)
from quantforge.data.calendar import expected_sessions
from quantforge.data.models import JsonValue
from quantforge.data.providers import TiingoProvider
from quantforge.timeframes import IntradayInterval, Timeframe, resolve_exchange_session

TOKEN = "secret-intraday-test-token"


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


FIXED_RETRIEVAL = datetime(2026, 1, 1, tzinfo=UTC)


class FixedClock(datetime):
    @classmethod
    def now(cls, tz: object = None) -> datetime:
        return FIXED_RETRIEVAL


def history_request(start: str, end: str, minutes: int = 1) -> IntradayBarRequest:
    return IntradayBarRequest(
        "SPY",
        datetime.fromisoformat(start),
        datetime.fromisoformat(end),
        Timeframe.us_equity(IntradayInterval(timedelta(minutes=minutes))),
        FeedScope.consolidated(),
        TiingoProvider.intraday_adjustment_basis,
    )


def session_rows(session_date: date, minutes: int = 1) -> list[dict[str, JsonValue]]:
    """Mimic Tiingo's nominal 390-minute day, including early-close anomalies."""
    opened = resolve_exchange_session(session_date).open_timestamp
    return [
        {
            "date": (opened + timedelta(minutes=index)).isoformat(),
            "open": "100.1",
            "high": "101.2",
            "low": "99.3",
            "close": "100.4",
            "volume": index,
        }
        for index in range(0, 390, minutes)
    ]


def install_history(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: Mapping[date, list[dict[str, JsonValue]] | Exception] | None = None,
    minutes: int = 1,
) -> list[date]:
    calls: list[date] = []

    def fetch(request: Request, *, timeout: float) -> FakeResponse:
        query = parse_qs(urlparse(request.full_url).query)
        assert query["startDate"] == query["endDate"]
        assert query["afterHours"] == query["forceFill"] == ["false"]
        session_date = date.fromisoformat(query["startDate"][0])
        calls.append(session_date)
        payload = (responses or {}).get(session_date)
        if isinstance(payload, Exception):
            raise payload
        if payload is None:
            payload = session_rows(session_date, minutes)
        return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr(tiingo_module, "urlopen", fetch)
    monkeypatch.setattr(tiingo_module, "datetime", FixedClock)
    return calls


@pytest.mark.parametrize(
    "session_date", [date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24)]
)
def test_early_closes_retain_210_minutes_and_lossless_post_close_evidence(
    monkeypatch: pytest.MonkeyPatch, session_date: date
) -> None:
    calls = install_history(monkeypatch)
    session = resolve_exchange_session(session_date)
    request = history_request(
        session.open_timestamp.isoformat(),
        (session.close_timestamp + timedelta(hours=8)).isoformat(),
    )
    result = TiingoProvider(TOKEN).fetch_intraday(request)
    assert calls == [session_date]
    assert len(result.batch.bars) == 210
    assert result.batch.bars[-1].end_timestamp == session.close_timestamp
    assert len(result.raw_snapshots[0].records) == 390
    assert (
        result.raw_snapshots[0].records[210]["date"]
        == session.close_timestamp.isoformat()
    )
    assert validate_intraday_coverage(result.batch).is_complete
    assert result.raw_snapshots[0].chunk_end_timestamp == request.end_timestamp


@pytest.mark.parametrize(
    "session_date", [date(2024, 1, 2), date(2024, 3, 11), date(2025, 6, 2)]
)
def test_normal_session_numerics_and_dst_are_unchanged(
    monkeypatch: pytest.MonkeyPatch, session_date: date
) -> None:
    install_history(monkeypatch)
    session = resolve_exchange_session(session_date)
    request = history_request(
        session.open_timestamp.isoformat(), session.close_timestamp.isoformat()
    )
    result = TiingoProvider(TOKEN).fetch_intraday(request)
    assert len(result.batch.bars) == 390
    for bar, row in zip(result.batch.bars, session_rows(session_date), strict=True):
        assert bar.start_timestamp.isoformat() == row["date"]
        assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == tuple(
            Decimal(str(row[field]))
            for field in ("open", "high", "low", "close", "volume")
        )
    assert validate_intraday_coverage(result.batch).is_complete


def test_holiday_rows_never_become_canonical_and_raw_chunks_partition_logical_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holiday_row = dict(session_rows(date(2024, 1, 12))[0], date="2024-01-15T14:30:00Z")
    responses = {date(2024, 1, 12): [*session_rows(date(2024, 1, 12)), holiday_row]}
    calls = install_history(monkeypatch, responses=responses)
    request = history_request("2024-01-12T00:00:00Z", "2024-01-17T00:00:00Z")
    result = TiingoProvider(TOKEN).fetch_intraday(request)
    assert calls == [date(2024, 1, 12), date(2024, 1, 16)]
    assert Counter(bar.session_date for bar in result.batch.bars) == dict.fromkeys(
        calls, 390
    )
    assert result.raw_snapshots[0].records[-1] == holiday_row
    assert result.raw_snapshots[0].chunk_start_timestamp == request.start_timestamp
    assert result.raw_snapshots[-1].chunk_end_timestamp == request.end_timestamp
    assert (
        result.raw_snapshots[0].chunk_end_timestamp
        == result.raw_snapshots[1].chunk_start_timestamp
    )
    assert all(
        snapshot.source_request_id == request.request_id
        for snapshot in result.raw_snapshots
    )
    assert validate_intraday_coverage(result.batch).is_complete
    # Fixed retrieval makes raw and normalized identities reproducible.
    assert TiingoProvider(TOKEN).fetch_intraday(request) == result
    assert (
        replace(request, feed_scope=FeedScope.iex_only()).request_id
        != request.request_id
    )
    assert (
        replace(
            request, end_timestamp=request.end_timestamp + timedelta(days=1)
        ).request_id
        != request.request_id
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2024-01-15T00:00:00Z", "2024-01-16T00:00:00Z"),
        ("2024-07-03T17:00:00Z", "2024-07-05T13:30:00Z"),
    ],
)
def test_closed_market_only_request_makes_no_http_calls(
    monkeypatch: pytest.MonkeyPatch, start: str, end: str
) -> None:
    calls = install_history(monkeypatch)
    with pytest.raises(ProviderError, match="empty"):
        TiingoProvider(TOKEN).fetch_intraday(history_request(start, end))
    assert not calls


@pytest.mark.parametrize(("seconds", "expected"), [(0, 390), (30, 389)])
def test_partial_first_and_last_sessions_intersect_exact_half_open_request(
    monkeypatch: pytest.MonkeyPatch, seconds: int, expected: int
) -> None:
    calls = install_history(monkeypatch)
    request = history_request("2024-07-03T15:00:00Z", "2024-07-05T18:00:00Z")
    request = replace(
        request,
        start_timestamp=request.start_timestamp + timedelta(seconds=seconds),
        end_timestamp=request.end_timestamp + timedelta(seconds=seconds),
    )
    result = TiingoProvider(TOKEN).fetch_intraday(request)
    assert calls == [date(2024, 7, 3), date(2024, 7, 5)]
    assert len(result.batch.bars) == expected
    assert all(
        request.start_timestamp
        <= bar.start_timestamp
        < bar.end_timestamp
        <= request.end_timestamp
        for bar in result.batch.bars
    )
    report = validate_intraday_coverage(result.batch)
    assert report.is_complete
    assert not any(session.request_covers_full_session for session in report.sessions)


def test_multi_month_history_has_complete_session_coverage_and_ordered_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_history(monkeypatch, minutes=5)
    request = history_request("2024-06-01T00:00:00Z", "2024-12-26T00:00:00Z", 5)
    result = TiingoProvider(TOKEN).fetch_intraday(request)
    expected = expected_sessions(date(2024, 6, 1), date(2024, 12, 25))
    assert tuple(calls) == expected
    assert len(result.raw_snapshots) == len(expected)
    report = validate_intraday_coverage(result.batch)
    assert report.is_complete
    assert tuple(session.session_date for session in report.sessions) == expected
    assert all(
        session.observed_completed_interval_count
        == (
            42
            if session.session_date
            in {date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24)}
            else 78
        )
        for session in report.sessions
    )
    assert tuple(bar.start_timestamp for bar in result.batch.bars) == tuple(
        sorted({bar.start_timestamp for bar in result.batch.bars})
    )


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "empty_session",
        "duplicate",
        "unaligned",
        "invalid_timestamp",
        "naive_timestamp",
        "ohlc",
        "volume",
        "nonfinite",
        "missing_field",
        "nonnumeric",
    ],
)
def test_bad_expected_session_fails_whole_acquisition_without_cache_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    rows = session_rows(date(2024, 7, 3))
    if defect == "missing":
        rows.pop(17)
    elif defect == "empty_session":
        rows = []
    elif defect == "duplicate":
        rows.insert(18, rows[17].copy())
    elif defect == "unaligned":
        rows[17]["date"] = "2024-07-03T13:47:30Z"
    elif defect == "invalid_timestamp":
        rows[17]["date"] = f"Authorization: Token {TOKEN}"
    elif defect == "naive_timestamp":
        rows[17]["date"] = "2024-07-03T13:47:00"
    elif defect == "ohlc":
        rows[17]["high"] = "99"
    elif defect == "volume":
        rows[17]["volume"] = -1
    elif defect == "nonfinite":
        rows[17]["close"] = "NaN"
    elif defect == "missing_field":
        del rows[17]["low"]
    else:
        rows[17]["open"] = f"Authorization: Token {TOKEN}"
    install_history(monkeypatch, responses={date(2024, 7, 3): rows})
    request = history_request("2024-07-02T00:00:00Z", "2024-07-06T00:00:00Z")
    service = IntradayMarketDataService(
        IntradayMarketDataCache(tmp_path), provider=TiingoProvider(TOKEN)
    )
    with pytest.raises(ProviderError) as raised:
        service.get_intraday_bars(request)
    message = str(raised.value)
    assert "SPY 1min" in message
    assert "tiingo/equity/intraday/<ticker>/prices" in message
    assert request.start_timestamp.isoformat() in message
    assert request.end_timestamp.isoformat() in message
    assert "2024-07-03" in message
    if defect not in {"missing", "empty_session", "duplicate"}:
        assert "row 17" in message
        assert "timestamp=" in message
        assert "reason:" in message
    assert TOKEN not in message
    assert "Authorization" not in message
    assert not tuple(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("status", [400, 401, 403, 429, 503])
def test_http_errors_keep_safe_context_and_never_echo_body_or_headers(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    error = HTTPError(
        f"https://api.tiingo.com/?token={TOKEN}",
        status,
        f"Authorization: Token {TOKEN}",
        Message(),
        None,
    )
    calls = install_history(monkeypatch, responses={date(2024, 7, 3): error})
    request = history_request("2024-07-03T00:00:00Z", "2024-07-04T00:00:00Z")
    with pytest.raises(ProviderError) as raised:
        TiingoProvider(TOKEN, retry_delays=()).fetch_intraday(request)
    assert calls == [date(2024, 7, 3)]
    assert "session=2024-07-03" in str(raised.value)
    assert f"HTTP status {status}" in str(raised.value)
    assert "17:00:00+00:00" in str(raised.value)
    assert TOKEN not in str(raised.value)
    assert "Authorization" not in str(raised.value)
    assert raised.value.__suppress_context__


def test_session_cache_reuses_before_network_and_refresh_keeps_old_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = install_history(monkeypatch)
    request = history_request("2024-07-03T00:00:00Z", "2024-07-06T00:00:00Z")
    cache = IntradayMarketDataCache(tmp_path)
    service = IntradayMarketDataService(cache, provider=TiingoProvider(TOKEN))
    original = service.get_intraday_bars(request)
    assert service.get_intraday_bars(request) == original
    assert len(calls) == 2
    original_bytes = {
        path: path.read_bytes()
        for path in tmp_path.rglob("*.json")
        if "requests" not in path.parts
    }
    changed = session_rows(date(2024, 7, 5))
    changed[0]["volume"] = 900
    install_history(monkeypatch, responses={date(2024, 7, 5): changed})
    refreshed = service.get_intraday_bars(request, refresh=True)
    assert refreshed.metadata.dataset_id != original.metadata.dataset_id
    assert refreshed.request.request_id == original.request.request_id
    assert all(path.read_bytes() == content for path, content in original_bytes.items())
    assert cache.load(original.metadata.dataset_id, request) == original
    assert (
        IntradayMarketDataService(cache, provider_name="tiingo").get_intraday_bars(
            request
        )
        == refreshed
    )
    before_failure = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    changed.pop(17)
    install_history(monkeypatch, responses={date(2024, 7, 5): changed})
    with pytest.raises(ProviderError, match="missing"):
        service.get_intraday_bars(request, refresh=True)
    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*.json")
    } == before_failure
    assert cache.find("tiingo", request) == refreshed


def single_snapshot_result(
    result: IntradayFetchResult, provider_name: str
) -> IntradayFetchResult:
    """Build a legacy single-range result through public contracts."""
    request = result.batch.request
    raw = IntradayRawSnapshot(
        provider_name,
        "SPY",
        "1",
        "historical/prices",
        request.request_id,
        request.start_timestamp,
        request.end_timestamp,
        FIXED_RETRIEVAL,
        (),
        tuple(
            record for snapshot in result.raw_snapshots for record in snapshot.records
        ),
    )
    provenance = IntradayBarProvenance(
        provider_name,
        "SPY",
        "1",
        FIXED_RETRIEVAL,
        request.request_id,
        raw.snapshot_id,
        request.feed_scope,
        request.adjustment_basis,
    )
    return IntradayFetchResult(
        IntradayBarBatch(
            request,
            tuple(replace(bar, provenance=provenance) for bar in result.batch.bars),
        ),
        (raw,),
        result.capabilities_configuration_id,
    )


def test_legacy_january_cache_still_loads_and_other_provider_keeps_one_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_history(monkeypatch)
    request = history_request("2024-01-12T00:00:00Z", "2024-01-17T00:00:00Z")
    tiingo_result = TiingoProvider(TOKEN).fetch_intraday(request)
    cache = IntradayMarketDataCache(tmp_path)
    legacy = cache.persist(single_snapshot_result(tiingo_result, "tiingo"))
    original_bytes = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    fake_result = single_snapshot_result(tiingo_result, "other-provider")
    received: list[IntradayBarRequest] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("generic service must never invoke Tiingo planning or HTTP")

    monkeypatch.setattr(tiingo_module, "plan_tiingo_intraday_chunks", forbidden)
    monkeypatch.setattr(tiingo_module, "urlopen", forbidden)
    assert (
        IntradayMarketDataService(cache, provider_name="tiingo").get_intraday_bars(
            request
        )
        == legacy
    )
    assert (
        IntradayMarketDataService(
            cache, provider=TiingoProvider(TOKEN)
        ).get_intraday_bars(request)
        == legacy
    )
    assert all(path.read_bytes() == content for path, content in original_bytes.items())

    class OtherProvider:
        name = "other-provider"
        intraday_capabilities = TiingoProvider.intraday_capabilities

        def fetch_intraday(self, request: IntradayBarRequest) -> IntradayFetchResult:
            received.append(request)
            return fake_result

    acquired = IntradayMarketDataService(
        cache, provider=OtherProvider()
    ).get_intraday_bars(request)
    assert received == [request]
    assert len(acquired.metadata.raw_snapshot_ids) == 1
    assert acquired.quality_report.is_complete
    # Generic diagnostic persistence remains available to other providers.
    incomplete = replace(
        fake_result, batch=IntradayBarBatch(request, fake_result.batch.bars[1:])
    )
    assert (
        not IntradayMarketDataCache(tmp_path / "diagnostic")
        .persist(incomplete)
        .quality_report.is_complete
    )


def test_overlapping_logical_chunks_are_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_history(monkeypatch)
    request = history_request("2024-07-02T00:00:00Z", "2024-07-04T00:00:00Z")
    result = TiingoProvider(TOKEN).fetch_intraday(request)
    first, second = result.raw_snapshots
    overlap = IntradayRawSnapshot(
        second.provider_name,
        second.provider_symbol,
        second.adapter_version,
        second.endpoint,
        second.source_request_id,
        first.chunk_end_timestamp - timedelta(minutes=1),
        second.chunk_end_timestamp,
        second.retrieved_at,
        second.request_parameters,
        second.records,
    )
    with pytest.raises(ValueError, match="ordered and contiguous"):
        replace(result, raw_snapshots=(first, overlap))


def test_calendar_validation_reason_survives_provider_error_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantforge.data.providers._tiingo_intraday import TiingoIntradayChunk

    request = history_request("2024-01-15T14:30:00Z", "2024-01-15T21:00:00Z")
    holiday_row = dict(session_rows(date(2024, 1, 12))[0], date="2024-01-15T14:30:00Z")
    install_history(monkeypatch, responses={date(2024, 1, 15): [holiday_row]})

    # Negative control reproduces the old planner's holiday acquisition. The
    # unchanged canonical validator must still reject it with its real reason.
    def old_plan(
        request: IntradayBarRequest, maximum_duration: timedelta
    ) -> tuple[TiingoIntradayChunk, ...]:
        return (
            TiingoIntradayChunk(
                date(2024, 1, 15),
                request.start_timestamp,
                request.end_timestamp,
                request.start_timestamp,
                request.end_timestamp,
            ),
        )

    monkeypatch.setattr(tiingo_module, "plan_tiingo_intraday_chunks", old_plan)
    with pytest.raises(ProviderError) as raised:
        TiingoProvider(TOKEN).fetch_intraday(request)
    message = str(raised.value)
    assert "date is not an exchange session for XNYS: 2024-01-15" in message
    assert "row 0" in message
    assert "timestamp=2024-01-15T14:30:00+00:00" in message
    assert "session=2024-01-15" in message
    assert TOKEN not in message


def test_nonfinite_json_fails_with_context_without_echoing_other_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = session_rows(date(2024, 7, 3))
    rows[17]["volume"] = float("nan")
    rows[17]["detail"] = f"Authorization: Token {TOKEN}"
    install_history(monkeypatch, responses={date(2024, 7, 3): rows})
    request = history_request("2024-07-03T00:00:00Z", "2024-07-04T00:00:00Z")
    with pytest.raises(ProviderError, match="JSON compliant") as raised:
        TiingoProvider(TOKEN).fetch_intraday(request)
    assert "session=2024-07-03" in str(raised.value)
    assert TOKEN not in str(raised.value)
    assert "Authorization" not in str(raised.value)


@pytest.mark.parametrize("end", ["2024-07-03T17:00:00Z", "2024-07-03T16:59:45Z"])
def test_malformed_in_session_extra_cannot_hide_behind_partial_end_clipping(
    monkeypatch: pytest.MonkeyPatch, end: str
) -> None:
    rows = session_rows(date(2024, 7, 3))
    rows.insert(210, dict(rows[209], date="2024-07-03T16:59:30Z"))
    install_history(monkeypatch, responses={date(2024, 7, 3): rows})
    request = history_request("2024-07-03T13:30:00Z", end)
    # All expected bars are present. The extra malformed observation must fail
    # itself; a missing-coverage check cannot detect this case.
    with pytest.raises(ProviderError, match=r"row 210.*timestamp=2024-07-03T16:59:30"):
        TiingoProvider(TOKEN).fetch_intraday(request)


@pytest.mark.parametrize("minutes", [1, 5])
@pytest.mark.parametrize("conflicting_volume", [False, True])
def test_duplicate_in_session_row_cannot_hide_behind_partial_end_clipping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minutes: int,
    conflicting_volume: bool,
) -> None:
    rows = session_rows(date(2024, 7, 3), minutes)
    last_in_session_index = 210 // minutes - 1
    duplicate = rows[last_in_session_index].copy()
    if conflicting_volume:
        duplicate["volume"] = 9999
    rows.insert(last_in_session_index + 1, duplicate)
    install_history(monkeypatch, responses={date(2024, 7, 3): rows}, minutes=minutes)
    request = history_request("2024-07-03T13:30:00Z", "2024-07-03T16:59:45Z", minutes)
    service = IntradayMarketDataService(
        IntradayMarketDataCache(tmp_path), provider=TiingoProvider(TOKEN)
    )
    with pytest.raises(ProviderError, match="duplicate bar key") as raised:
        service.get_intraday_bars(request)
    message = str(raised.value)
    assert f"row {last_in_session_index + 1};" in message
    assert f"timestamp={duplicate['date']}" in message
    assert "session=2024-07-03" in message
    assert TOKEN not in message
    assert not tuple(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("minutes", [1, 5])
def test_partial_end_clips_unique_final_bar_and_excludes_post_close_duplicates(
    monkeypatch: pytest.MonkeyPatch, minutes: int
) -> None:
    rows = session_rows(date(2024, 7, 3), minutes)
    rows.append(rows[210 // minutes].copy())  # 17:00 UTC is outside retention.
    install_history(monkeypatch, responses={date(2024, 7, 3): rows}, minutes=minutes)
    request = history_request("2024-07-03T13:30:00Z", "2024-07-03T16:59:45Z", minutes)
    result = TiingoProvider(TOKEN).fetch_intraday(request)
    assert len(result.batch.bars) == 210 // minutes - 1
    assert all(bar.end_timestamp <= request.end_timestamp for bar in result.batch.bars)
    assert validate_intraday_coverage(result.batch).is_complete
    assert result.raw_snapshots[0].records == tuple(rows)


@pytest.fixture(params=["missing_bar", "missing_session"])
def incomplete_history(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> IntradayFetchResult:
    install_history(monkeypatch)
    logical_request = history_request("2024-07-02T00:00:00Z", "2024-07-06T00:00:00Z")
    result = TiingoProvider(TOKEN).fetch_intraday(logical_request)
    bars = result.batch.bars
    retained = (
        bars[1:]
        if request.param == "missing_bar"
        else tuple(bar for bar in bars if bar.session_date != date(2024, 7, 3))
    )
    return replace(result, batch=IntradayBarBatch(logical_request, retained))


def test_incomplete_legacy_cache_is_reacquired_and_reused(
    incomplete_history: IntradayFetchResult,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = IntradayMarketDataCache(tmp_path)
    legacy = cache.persist(single_snapshot_result(incomplete_history, "tiingo"))
    assert legacy.metadata.adapter_version == "1"
    assert not legacy.quality_report.is_complete
    original_artifacts = {
        path: path.read_bytes()
        for directory in ("raw", "datasets")
        for path in (tmp_path / "intraday" / directory).rglob("*.json")
    }
    calls = install_history(monkeypatch)
    service = IntradayMarketDataService(cache, provider=TiingoProvider(TOKEN))
    acquired = service.get_intraday_bars(legacy.request)
    assert calls == [date(2024, 7, 2), date(2024, 7, 3), date(2024, 7, 5)]
    assert acquired.quality_report.is_complete
    assert acquired.metadata.adapter_version == "2"
    assert acquired.metadata.dataset_id != legacy.metadata.dataset_id
    assert cache.find("tiingo", legacy.request) == acquired
    assert cache.load(legacy.metadata.dataset_id, legacy.request) == legacy
    assert all(path.read_bytes() == body for path, body in original_artifacts.items())
    calls.clear()
    assert service.get_intraday_bars(legacy.request) == acquired
    assert (
        IntradayMarketDataService(cache, provider_name="tiingo").get_intraday_bars(
            legacy.request
        )
        == acquired
    )
    assert calls == []


def test_incomplete_legacy_cache_is_rejected_without_credentials_or_mutation(
    incomplete_history: IntradayFetchResult,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = IntradayMarketDataCache(tmp_path)
    legacy = cache.persist(single_snapshot_result(incomplete_history, "tiingo"))
    original_bytes = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    calls = install_history(monkeypatch)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("offline cache validation must not construct a provider")

    monkeypatch.setattr(TiingoProvider, "__init__", forbidden)
    with pytest.raises(RequestError, match=r"cached tiingo.*coverage"):
        IntradayMarketDataService(cache, provider_name="tiingo").get_intraday_bars(
            legacy.request
        )
    assert calls == []
    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*.json")
    } == original_bytes
    assert cache.load(legacy.metadata.dataset_id, legacy.request) == legacy


def test_failed_reacquisition_preserves_legacy_artifacts_and_request_pointer(
    incomplete_history: IntradayFetchResult,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = IntradayMarketDataCache(tmp_path)
    legacy = cache.persist(single_snapshot_result(incomplete_history, "tiingo"))
    original_bytes = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    # The last session fails after earlier sessions have already been fetched.
    calls = install_history(
        monkeypatch,
        responses={date(2024, 7, 5): session_rows(date(2024, 7, 5))[1:]},
    )
    service = IntradayMarketDataService(cache, provider=TiingoProvider(TOKEN))
    with pytest.raises(ProviderError, match="missing"):
        service.get_intraday_bars(legacy.request)
    assert calls == [date(2024, 7, 2), date(2024, 7, 3), date(2024, 7, 5)]
    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*.json")
    } == original_bytes
    assert cache.find("tiingo", legacy.request) == legacy


def test_other_provider_still_acquires_and_reuses_diagnostic_cache(
    incomplete_history: IntradayFetchResult,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = single_snapshot_result(incomplete_history, "other-provider")
    received: list[IntradayBarRequest] = []

    class OtherProvider:
        name = "other-provider"
        intraday_capabilities = TiingoProvider.intraday_capabilities

        def fetch_intraday(self, request: IntradayBarRequest) -> IntradayFetchResult:
            received.append(request)
            return result

    calls = install_history(monkeypatch)
    cache = IntradayMarketDataCache(tmp_path)
    service = IntradayMarketDataService(cache, provider=OtherProvider())
    acquired = service.get_intraday_bars(result.batch.request)
    assert not acquired.quality_report.is_complete
    assert len(acquired.metadata.raw_snapshot_ids) == 1
    assert service.get_intraday_bars(acquired.request) == acquired
    assert (
        IntradayMarketDataService(
            cache, provider_name="other-provider"
        ).get_intraday_bars(acquired.request)
        == acquired
    )
    assert received == [result.batch.request]
    assert calls == []
