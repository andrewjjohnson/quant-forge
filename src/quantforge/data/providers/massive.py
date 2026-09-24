"""Historical Massive stock aggregates behind canonical intraday contracts."""

import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import cast
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from quantforge.data.calendar import expected_sessions
from quantforge.data.exceptions import ProviderError, RequestError
from quantforge.data.intraday import (
    IntradayBar,
    IntradayBarBatch,
    IntradayBarProvenance,
    IntradayBarRequest,
    IntradayProviderCapabilities,
)
from quantforge.data.intraday_ingestion import IntradayFetchResult, IntradayRawSnapshot
from quantforge.data.intraday_validation import (
    IntradayCoverageReport,
    IntradayCoverageValidationError,
    validate_intraday_coverage,
)
from quantforge.data.lineage import AdjustmentBasis, FeedScope
from quantforge.data.models import AdjustmentMode, JsonValue, ProviderRecord
from quantforge.data.providers._massive_http import MassiveHTTPClient
from quantforge.timeframes import (
    BarCompletion,
    IntradayInterval,
    SessionScope,
    Timeframe,
    resolve_exchange_session,
)

_HOST = "https://api.massive.com"
_ENDPOINT = "/v2/aggs/ticker/{ticker}/range/{multiplier}/minute/{from}/{to}"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_SECRET_PARAMETERS = frozenset(("apikey", "api_key", "token", "authorization"))


class MassiveProvider:
    """Fetch complete XNYS/RTH 1m or 5m stock history, with no event feed."""

    name = "massive"
    intraday_adapter_version = "1"
    intraday_adjustment_basis = AdjustmentBasis(
        AdjustmentMode.UNADJUSTED,
        "raw_provider",
        "raw_provider",
        "not_provided_for_intraday_bars",
        False,
    )
    split_adjusted_basis = AdjustmentBasis(
        AdjustmentMode.SPLIT_ADJUSTED,
        "split_adjusted",
        "split_adjusted",
        "not_provided_for_intraday_bars",
        True,
    )
    intraday_capabilities = IntradayProviderCapabilities(
        provider_name=name,
        supported_intervals=tuple(
            IntradayInterval(timedelta(minutes=minutes)) for minutes in (1, 5)
        ),
        supported_feed_scopes=(FeedScope.consolidated(),),
        supported_session_scopes=(SessionScope.REGULAR_HOURS,),
    )

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 30.0,
        retry_delays: tuple[float, ...] = (0.25, 1.0),
    ) -> None:
        self._http = MassiveHTTPClient(api_key, timeout, retry_delays)

    @property
    def http_request_count(self) -> int:
        """Count actual HTTP attempts, including retries, for acquisition audits."""
        return self._http.request_count

    @staticmethod
    def can_reuse_intraday_cache(quality_report: IntradayCoverageReport) -> bool:
        return quality_report.is_complete

    def fetch_intraday_bars(self, request: IntradayBarRequest) -> IntradayBarBatch:
        return self.fetch_intraday(request).batch

    def fetch_intraday(self, request: IntradayBarRequest) -> IntradayFetchResult:
        """Acquire all pages before validating or exposing a successful result."""
        self.intraday_capabilities.validate_request(request)
        if request.timeframe != Timeframe.us_equity(request.source_interval):
            raise RequestError("Massive requires the canonical XNYS/RTH timeframe")
        if request.adjustment_basis not in (
            self.intraday_adjustment_basis,
            self.split_adjusted_basis,
        ):
            raise RequestError(
                "Massive requires explicit raw or split-adjusted OHLCV with "
                "intraday corporate-action events unavailable"
            )
        adjusted = request.adjustment_basis == self.split_adjusted_basis
        multiplier = request.source_interval.nominal_duration // timedelta(minutes=1)
        prefix = (
            f"/v2/aggs/ticker/{quote(request.symbol, safe='.-')}"
            f"/range/{multiplier}/minute/"
        )
        # Millisecond wire bounds envelope the exact microsecond logical range.
        start_ms = (request.start_timestamp - _EPOCH) // timedelta(milliseconds=1)
        end_ms = (
            request.end_timestamp - _EPOCH - timedelta(microseconds=1)
        ) // timedelta(milliseconds=1)
        parameters = (
            ("adjusted", str(adjusted).lower()),
            ("sort", "asc"),
            ("limit", "50000"),
        )
        url: str | None = f"{_HOST}{prefix}{start_ms}/{end_ms}?{urlencode(parameters)}"
        pages: list[ProviderRecord] = []
        seen_urls: set[str] = set()
        context = (
            f"Massive {request.symbol} {multiplier}m endpoint={_ENDPOINT}; "
            f"range [{request.start_timestamp.isoformat()}, "
            f"{request.end_timestamp.isoformat()}); "
        )
        try:
            while url is not None:
                url = _page_url(url, prefix, adjusted)
                if self._http.redact(url) != url:
                    raise ProviderError("pagination URL contains credentials")
                if url in seen_urls:
                    raise ProviderError("pagination loop/repeated URL")
                seen_urls.add(url)
                payload = self._http.request_json(url)
                page = _page_payload(payload, request.symbol, adjusted)
                next_url = page.get("next_url")
                if next_url is not None:
                    if not isinstance(next_url, str) or not next_url:
                        raise ProviderError("malformed pagination next_url")
                    next_url = _page_url(next_url, prefix, adjusted)
                    page["next_url"] = next_url
                pages.append(
                    {
                        "url": url,
                        "retrieved_at": datetime.now(UTC).isoformat(),
                        "response": self._redact_payload(page),
                    }
                )
                url = next_url
            snapshot = IntradayRawSnapshot(
                provider_name=self.name,
                provider_symbol=request.symbol,
                adapter_version=self.intraday_adapter_version,
                endpoint=_ENDPOINT,
                source_request_id=request.request_id,
                chunk_start_timestamp=request.start_timestamp,
                chunk_end_timestamp=request.end_timestamp,
                retrieved_at=datetime.fromisoformat(
                    cast(str, pages[-1]["retrieved_at"])
                ),
                request_parameters=(
                    *parameters,
                    ("from", str(start_ms)),
                    ("to", str(end_ms)),
                    ("multiplier", str(multiplier)),
                    ("timespan", "minute"),
                ),
                records=tuple(pages),
            )
            batch = _canonical_batch(request, snapshot)
            validate_intraday_coverage(batch)
            return IntradayFetchResult(
                batch, (snapshot,), self.intraday_capabilities.configuration_id
            )
        except IntradayCoverageValidationError as error:
            first = (
                error.report.missing_intervals or error.report.unexpected_intervals
            )[0]
            raise ProviderError(
                context + f"incomplete required coverage: {error}; "
                f"first failed interval={first.start_timestamp.isoformat()}"
            ) from None
        except (ProviderError, ValueError, TypeError, OverflowError) as error:
            # Underlying HTTP bodies and malformed field contents are never echoed.
            reason = (
                str(error)
                if isinstance(error, ProviderError)
                else "invalid canonical response or raw payload"
            )
            raise ProviderError(
                self._http.redact(
                    context + f"page={len(pages) + (url is not None)}; " + reason
                )
            ) from None

    def _redact_payload(self, payload: JsonValue) -> JsonValue:
        if isinstance(payload, str):
            return self._http.redact(payload)
        if isinstance(payload, list):
            return [self._redact_payload(item) for item in payload]
        if isinstance(payload, dict):
            return {
                self._http.redact(key): (
                    "<redacted>"
                    if key.lower() in _SECRET_PARAMETERS
                    else self._redact_payload(item)
                )
                for key, item in payload.items()
            }
        return payload


def _page_url(url: str, prefix: str, adjusted: bool) -> str:
    """Validate the destination before attaching a credential to any page."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.massive.com"
        or parsed.fragment
        or not re.fullmatch(re.escape(prefix) + r"\d+/\d+", parsed.path)
    ):
        raise ProviderError("unsafe or incompatible pagination URL")
    parameters = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if name.lower() not in _SECRET_PARAMETERS
    ]
    expected = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": "50000"}
    if len({name for name, _ in parameters}) != len(parameters) or any(
        name != "cursor" and (name not in expected or expected[name] != value)
        for name, value in parameters
    ):
        raise ProviderError("incompatible pagination query")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(sorted(parameters)), "")
    )


def _page_payload(
    payload: JsonValue, symbol: str, adjusted: bool
) -> dict[str, JsonValue]:
    if not isinstance(payload, dict):
        raise ProviderError("invalid aggregate top-level payload: expected object")
    if payload.get("status") not in ("OK", "DELAYED"):
        raise ProviderError("aggregate response did not report success")
    if payload.get("ticker") != symbol or payload.get("adjusted") is not adjusted:
        raise ProviderError("aggregate ticker/adjustment metadata mismatch")
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ProviderError("malformed aggregate results: expected array")
    count = payload.get("resultsCount")
    if count is not None and (type(count) is not int or count != len(rows)):
        raise ProviderError("aggregate resultsCount mismatch")
    return dict(payload)


def _timestamp(value: JsonValue) -> datetime:
    if type(value) is not int:
        raise ProviderError("aggregate t must be an integer Unix-millisecond timestamp")
    try:
        return _EPOCH + timedelta(milliseconds=value)
    except OverflowError:
        raise ProviderError(
            "aggregate t is outside the supported timestamp range"
        ) from None


def _decimal(value: JsonValue, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderError(f"aggregate {field} must be numeric")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ProviderError(f"aggregate {field} must be numeric") from None
    if not result.is_finite():
        raise ProviderError(f"aggregate {field} must be finite")
    return result


def _canonical_batch(
    request: IntradayBarRequest, snapshot: IntradayRawSnapshot
) -> IntradayBarBatch:
    policy = request.timeframe.session_policy
    timezone = ZoneInfo(policy.timezone_name)
    sessions = {
        session_date: resolve_exchange_session(session_date, policy)
        for session_date in expected_sessions(
            request.start_timestamp.astimezone(timezone).date(),
            (request.end_timestamp - timedelta(microseconds=1))
            .astimezone(timezone)
            .date(),
            policy.calendar_name,
        )
    }
    provenance = IntradayBarProvenance(
        snapshot.provider_name,
        snapshot.provider_symbol,
        snapshot.adapter_version,
        snapshot.retrieved_at,
        request.request_id,
        snapshot.snapshot_id,
        request.feed_scope,
        request.adjustment_basis,
    )
    bars: list[IntradayBar] = []
    seen_starts: set[datetime] = set()
    for page_index, page in enumerate(snapshot.records):
        response = cast(dict[str, JsonValue], page["response"])
        for row_index, record in enumerate(
            cast(list[JsonValue], response.get("results", []))
        ):
            timestamp_context = "<invalid or missing>"
            try:
                if not isinstance(record, dict):
                    raise ProviderError("aggregate row must be an object")
                start = _timestamp(record.get("t"))
                timestamp_context = start.isoformat()
                if not request.start_timestamp <= start < request.end_timestamp:
                    continue
                session_date = start.astimezone(timezone).date()
                session = sessions.get(session_date)
                if (
                    session is None
                    or not session.open_timestamp <= start < session.close_timestamp
                ):
                    continue
                if start in seen_starts:
                    raise ProviderError(
                        "duplicate canonical bar across aggregate rows/pages"
                    )
                seen_starts.add(start)
                end = start + request.source_interval.nominal_duration
                bar = IntradayBar(
                    symbol=request.symbol,
                    session_date=session_date,
                    start_timestamp=start,
                    end_timestamp=end,
                    timeframe=request.timeframe,
                    completion=BarCompletion.COMPLETED,
                    open=_decimal(record.get("o"), "o"),
                    high=_decimal(record.get("h"), "h"),
                    low=_decimal(record.get("l"), "l"),
                    close=_decimal(record.get("c"), "c"),
                    volume=_decimal(record.get("v"), "v"),
                    provenance=provenance,
                )
                if end <= request.end_timestamp:
                    bars.append(bar)
            except (ProviderError, ValueError) as error:
                raise ProviderError(
                    f"page={page_index + 1} row={row_index} "
                    f"timestamp={timestamp_context}: {error}"
                ) from None
    if not bars:
        raise ProviderError("empty aggregate response (no retained RTH bars)")
    bars.sort(key=lambda bar: bar.start_timestamp)
    return IntradayBarBatch(request, tuple(bars))
