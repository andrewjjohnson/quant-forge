"""Tiingo daily and session-aware intraday acquisition adapters."""

import json
import math
import time
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from quantforge.data.exceptions import ProviderError, RequestError
from quantforge.data.intraday import (
    IntradayBar,
    IntradayBarBatch,
    IntradayBarProvenance,
    IntradayBarRequest,
    IntradayContractValidationError,
    IntradayProviderCapabilities,
)
from quantforge.data.intraday_ingestion import (
    IntradayFetchResult,
    IntradayRawSnapshot,
)
from quantforge.data.intraday_validation import (
    IntradayCoverageReport,
    IntradayCoverageValidationError,
    validate_intraday_coverage,
)
from quantforge.data.lineage import AdjustmentBasis, FeedScope
from quantforge.data.models import (
    AdjustmentMode,
    JsonValue,
    ProviderRecord,
    ProviderResponse,
)
from quantforge.data.providers._tiingo_intraday import (
    TiingoIntradayChunk,
    plan_tiingo_intraday_chunks,
)
from quantforge.timeframes import (
    BarCompletion,
    IntradayInterval,
    SessionScope,
    Timeframe,
)

_BASE_URL = "https://api.tiingo.com/tiingo/daily"
_CONSOLIDATED_INTRADAY_BASE_URL = "https://api.tiingo.com/tiingo/equity/intraday"
_IEX_INTRADAY_BASE_URL = "https://api.tiingo.com/iex"
_RETRYABLE_HTTP_STATUSES = frozenset((429, 500, 502, 503, 504))
_REQUIRED_PRICE_FIELDS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjOpen",
    "adjHigh",
    "adjLow",
    "adjClose",
    "adjVolume",
    "divCash",
    "splitFactor",
)
_REQUIRED_INTRADAY_FIELDS = ("date", "open", "high", "low", "close", "volume")
_ONE_MINUTE = IntradayInterval(timedelta(minutes=1))
_FIVE_MINUTES = IntradayInterval(timedelta(minutes=5))


class TiingoProvider:
    """Retrieve Tiingo EOD and intraday JSON with header-only authentication."""

    name = "tiingo"
    adapter_version = "1"
    intraday_adapter_version = "2"
    intraday_adjustment_basis = AdjustmentBasis(
        adjustment_mode=AdjustmentMode.UNADJUSTED,
        ohlc_basis="raw_provider",
        volume_basis="raw_provider",
        corporate_action_policy="not_provided_for_intraday_bars",
        adjusted_fields_used=False,
    )
    intraday_capabilities = IntradayProviderCapabilities(
        provider_name=name,
        supported_intervals=(_ONE_MINUTE, _FIVE_MINUTES),
        supported_feed_scopes=(FeedScope.consolidated(), FeedScope.iex_only()),
        supported_session_scopes=(SessionScope.REGULAR_HOURS,),
    )

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 30.0,
        retry_delays: tuple[float, ...] = (0.25, 1.0),
        intraday_chunk_duration: timedelta = timedelta(days=30),
    ) -> None:
        if not api_key or not api_key.strip():
            raise RequestError("TIINGO_API_KEY is required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise RequestError("Tiingo timeout must be positive and finite")
        if any(not math.isfinite(delay) or delay < 0 for delay in retry_delays):
            raise RequestError("Tiingo retry delays must be finite and nonnegative")
        if intraday_chunk_duration < timedelta(
            minutes=5
        ) or intraday_chunk_duration % timedelta(minutes=1):
            raise RequestError(
                "Tiingo intraday chunk duration must be a whole number of minutes "
                "and at least five minutes"
            )
        self._api_key = api_key.strip()
        self._timeout = timeout
        self._retry_delays = retry_delays
        self._intraday_chunk_duration = intraday_chunk_duration

    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjustment: AdjustmentMode,
    ) -> ProviderResponse:
        """Return raw Tiingo OHLCV plus explicit dividend and split fields."""
        if adjustment is not AdjustmentMode.UNADJUSTED:
            raise RequestError(
                "Tiingo execution ingestion supports only raw unadjusted OHLCV"
            )
        if start > end:
            raise RequestError("start must be on or before end")
        provider_symbol = symbol.replace(".", "-").upper()
        encoded_symbol = quote(provider_symbol, safe="-")
        metadata_url = f"{_BASE_URL}/{encoded_symbol}"
        prices_query = urlencode(
            {
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "resampleFreq": "daily",
                "format": "json",
            }
        )
        prices_url = f"{metadata_url}/prices?{prices_query}"
        retrieved_at = datetime.now(UTC)
        metadata_payload = self._request_json(metadata_url, "metadata")
        prices_payload = self._request_json(prices_url, "historical prices")
        if not isinstance(metadata_payload, dict):
            raise ProviderError("Tiingo returned malformed metadata JSON")
        if not isinstance(prices_payload, list):
            raise ProviderError("Tiingo returned malformed historical-price JSON")
        if not prices_payload:
            raise ProviderError("Tiingo returned an empty historical-price response")

        records: list[ProviderRecord] = []
        for index, row_value in enumerate(prices_payload):
            if not isinstance(row_value, dict):
                raise ProviderError(f"Tiingo returned malformed price row {index}")
            row = cast(dict[str, JsonValue], row_value)
            missing = [field for field in _REQUIRED_PRICE_FIELDS if field not in row]
            if missing:
                raise ProviderError(
                    f"Tiingo price row {index} is missing required fields: "
                    f"{', '.join(missing)}"
                )
            session_date = _tiingo_session(row["date"], index)
            records.append(
                {
                    **row,
                    "session_date": session_date.isoformat(),
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "dividend_amount": row["divCash"],
                    "split_coefficient": row["splitFactor"],
                }
            )

        asset_metadata = cast(dict[str, JsonValue], metadata_payload)
        returned_ticker = asset_metadata.get("ticker")
        if isinstance(returned_ticker, str) and returned_ticker:
            provider_symbol = returned_ticker
        return ProviderResponse(
            provider_name=self.name,
            provider_symbol=provider_symbol,
            retrieved_at=retrieved_at,
            provider_timezone="America/New_York",
            adjustment_mode=adjustment,
            records=tuple(records),
            metadata={
                "endpoint": "tiingo/daily/<ticker>/prices",
                "response_format": "json",
                "resample_frequency": "daily",
                "requested_start": start.isoformat(),
                "requested_end": end.isoformat(),
                "asset_metadata": asset_metadata,
            },
            adapter_version=self.adapter_version,
        )

    def fetch_intraday_bars(self, request: IntradayBarRequest) -> IntradayBarBatch:
        """Return QF-15 canonical bars for an explicit Tiingo feed request."""
        return self.fetch_intraday(request).batch

    @staticmethod
    def can_reuse_intraday_cache(quality_report: IntradayCoverageReport) -> bool:
        """Apply current coverage requirements to any cached adapter revision."""
        return quality_report.is_complete

    def fetch_intraday(self, request: IntradayBarRequest) -> IntradayFetchResult:
        """Retrieve bounded raw chunks and normalize them into one stable batch."""
        self.intraday_capabilities.validate_request(request)
        if request.timeframe != Timeframe.us_equity(request.source_interval):
            raise RequestError(
                "Tiingo intraday ingestion supports only the canonical XNYS "
                "regular-hours timeframe policy"
            )
        if request.adjustment_basis != self.intraday_adjustment_basis:
            raise RequestError(
                "Tiingo intraday ingestion requires raw unadjusted OHLCV with "
                "intraday corporate actions explicitly unavailable"
            )
        endpoint, base_url = _intraday_endpoint(request.feed_scope)
        provider_symbol = request.symbol.replace(".", "-").upper()
        encoded_symbol = quote(provider_symbol, safe="-")
        interval_parameter = _intraday_interval_parameter(request)
        snapshots: list[IntradayRawSnapshot] = []
        bars: list[IntradayBar] = []
        chunks = plan_tiingo_intraday_chunks(request, self._intraday_chunk_duration)
        for chunk in chunks:
            try:
                snapshot = self._fetch_intraday_chunk(
                    request,
                    chunk,
                    endpoint,
                    base_url,
                    encoded_symbol,
                    provider_symbol,
                    interval_parameter,
                )
                snapshots.append(snapshot)
                bars.extend(_intraday_bars_from_snapshot(request, snapshot, chunk))
            except (ProviderError, ValueError) as error:
                raise ProviderError(
                    _intraday_context(request, endpoint, chunk)
                    + str(error).replace(self._api_key, "<redacted>")
                ) from None
        if not bars:
            raise ProviderError(
                _intraday_context(request, endpoint)
                + "Tiingo returned an empty intraday-price response (no retained bars)"
            )
        bars.sort(key=lambda bar: (bar.start_timestamp, bar.end_timestamp))
        try:
            batch = IntradayBarBatch(request, tuple(bars))
            validate_intraday_coverage(batch)
        except IntradayCoverageValidationError as error:
            first = (
                error.report.missing_intervals or error.report.unexpected_intervals
            )[0]
            chunk = next(
                item
                for item in chunks
                if item.retention_start <= first.start_timestamp < item.retention_end
            )
            raise ProviderError(
                _intraday_context(request, endpoint, chunk)
                + f"{error}; first failed interval: {first.start_timestamp.isoformat()}"
            ) from None
        except IntradayContractValidationError as error:
            raise ProviderError(
                _intraday_context(request, endpoint)
                + f"Tiingo intraday response is invalid: {error}".replace(
                    self._api_key, "<redacted>"
                )
            ) from None
        return IntradayFetchResult(
            batch,
            tuple(snapshots),
            self.intraday_capabilities.configuration_id,
        )

    def _fetch_intraday_chunk(
        self,
        request: IntradayBarRequest,
        chunk: TiingoIntradayChunk,
        endpoint: str,
        base_url: str,
        encoded_symbol: str,
        provider_symbol: str,
        interval_parameter: str,
    ) -> IntradayRawSnapshot:
        parameters = (
            ("afterHours", "false"),
            ("columns", "open,high,low,close,volume"),
            ("endDate", chunk.session_date.isoformat()),
            ("forceFill", "false"),
            ("resampleFreq", interval_parameter),
            ("startDate", chunk.session_date.isoformat()),
        )
        url = f"{base_url}/{encoded_symbol}/prices?{urlencode(parameters)}"
        payload = self._request_json(
            url, "intraday prices", user_agent="QuantForge/0.1 TiingoIntraday/2"
        )
        if not isinstance(payload, list):
            raise ProviderError("Tiingo returned malformed intraday-price JSON")
        records: list[ProviderRecord] = []
        for row_index, row_value in enumerate(payload):
            if not isinstance(row_value, dict):
                raise ProviderError(
                    f"malformed intraday price row {row_index}: expected object"
                )
            records.append(cast(ProviderRecord, row_value))
        return IntradayRawSnapshot(
            provider_name=self.name,
            provider_symbol=provider_symbol,
            adapter_version=self.intraday_adapter_version,
            endpoint=endpoint,
            source_request_id=request.request_id,
            chunk_start_timestamp=chunk.coverage_start,
            chunk_end_timestamp=chunk.coverage_end,
            retrieved_at=datetime.now(UTC),
            request_parameters=parameters,
            records=tuple(records),
        )

    def _request_json(
        self,
        url: str,
        label: str,
        *,
        user_agent: str = "QuantForge/0.1 TiingoEOD/1",
    ) -> JsonValue:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Token {self._api_key}",
                "User-Agent": user_agent,
            },
            method="GET",
        )
        for attempt in range(len(self._retry_delays) + 1):
            try:
                with urlopen(request, timeout=self._timeout) as response:
                    payload = response.read()
            except HTTPError as error:
                if error.code in (401, 403):
                    raise ProviderError(
                        f"Tiingo authentication was rejected (HTTP status {error.code})"
                    ) from None
                if error.code in _RETRYABLE_HTTP_STATUSES:
                    if attempt < len(self._retry_delays):
                        time.sleep(self._retry_delays[attempt])
                        continue
                    if error.code == 429:
                        raise ProviderError(
                            "Tiingo rate limit was exceeded (HTTP status 429)"
                        ) from None
                    raise ProviderError(
                        "Tiingo temporary server failure persisted after "
                        f"bounded retries (HTTP status {error.code})"
                    ) from None
                raise ProviderError(
                    f"Tiingo {label} request failed with HTTP status {error.code}"
                ) from None
            except (TimeoutError, URLError, OSError):
                if attempt < len(self._retry_delays):
                    time.sleep(self._retry_delays[attempt])
                    continue
                raise ProviderError(
                    f"Tiingo {label} request failed after bounded retries"
                ) from None
            try:
                loaded = cast(JsonValue, json.loads(payload))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                raise ProviderError(f"Tiingo returned malformed {label} JSON") from None
            return loaded
        raise AssertionError("bounded Tiingo retry loop did not terminate")


def _intraday_endpoint(feed_scope: FeedScope) -> tuple[str, str]:
    if feed_scope == FeedScope.consolidated():
        return (
            "tiingo/equity/intraday/<ticker>/prices",
            _CONSOLIDATED_INTRADAY_BASE_URL,
        )
    if feed_scope == FeedScope.iex_only():
        return "iex/<ticker>/prices", _IEX_INTRADAY_BASE_URL
    raise RequestError("Tiingo intraday feed scope is not supported")


def _intraday_interval_parameter(request: IntradayBarRequest) -> str:
    if request.source_interval == _ONE_MINUTE:
        return "1min"
    if request.source_interval == _FIVE_MINUTES:
        return "5min"
    raise RequestError("Tiingo intraday interval is not supported")


def _intraday_context(
    request: IntradayBarRequest,
    endpoint: str,
    chunk: TiingoIntradayChunk | None = None,
) -> str:
    context = (
        f"Tiingo {request.symbol} {_intraday_interval_parameter(request)} "
        f"endpoint={endpoint}; logical range "
        f"[{request.start_timestamp.isoformat()}, "
        f"{request.end_timestamp.isoformat()}); "
    )
    if chunk is not None:
        context += (
            f"session={chunk.session_date}; retention "
            f"[{chunk.retention_start.isoformat()}, "
            f"{chunk.retention_end.isoformat()}); "
            f"chunk [{chunk.coverage_start.isoformat()}, "
            f"{chunk.coverage_end.isoformat()}); "
        )
    return context


def _intraday_bars_from_snapshot(
    request: IntradayBarRequest,
    snapshot: IntradayRawSnapshot,
    chunk: TiingoIntradayChunk,
) -> tuple[IntradayBar, ...]:
    duration = request.source_interval.nominal_duration
    exchange_timezone = ZoneInfo(request.timeframe.session_policy.timezone_name)
    bars: list[IntradayBar] = []
    seen_starts: set[datetime] = set()
    provenance = IntradayBarProvenance(
        provider_name=snapshot.provider_name,
        provider_symbol=snapshot.provider_symbol,
        adapter_version=snapshot.adapter_version,
        retrieved_at=snapshot.retrieved_at,
        source_request_id=request.request_id,
        source_snapshot_id=snapshot.snapshot_id,
        feed_scope=request.feed_scope,
        adjustment_basis=request.adjustment_basis,
    )
    for row_index, record in enumerate(snapshot.records):
        timestamp_context = "<invalid or missing>"
        try:
            start_timestamp = _tiingo_intraday_timestamp(record["date"], row_index)
            timestamp_context = start_timestamp.isoformat()
            end_timestamp = start_timestamp + duration
            if not (chunk.retention_start <= start_timestamp < chunk.retention_end):
                continue
            missing = [
                field for field in _REQUIRED_INTRADAY_FIELDS if field not in record
            ]
            if missing:
                raise ProviderError(f"missing required fields: {', '.join(missing)}")
            bar = IntradayBar(
                symbol=request.symbol,
                session_date=start_timestamp.astimezone(exchange_timezone).date(),
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                timeframe=request.timeframe,
                completion=BarCompletion.COMPLETED,
                open=_tiingo_intraday_decimal(record["open"], "open"),
                high=_tiingo_intraday_decimal(record["high"], "high"),
                low=_tiingo_intraday_decimal(record["low"], "low"),
                close=_tiingo_intraday_decimal(record["close"], "close"),
                volume=_tiingo_intraday_decimal(record["volume"], "volume"),
                provenance=provenance,
            )
            if start_timestamp in seen_starts:
                raise ProviderError("intraday bar batch contains a duplicate bar key")
            seen_starts.add(start_timestamp)
            # Validate in-session rows before excluding a valid bar that is not
            # fully inside a partial logical range. Malformed or duplicate rows
            # must not escape validation merely because their computed end is late.
            if end_timestamp > request.end_timestamp:
                continue
            bars.append(bar)
        except (KeyError, ValueError, ProviderError) as error:
            raise ProviderError(
                f"malformed intraday price row {row_index}; "
                f"timestamp={timestamp_context}; "
                f"reason: {error}"
            ) from None
    return tuple(bars)


def _tiingo_intraday_timestamp(value: JsonValue, row_index: int) -> datetime:
    if not isinstance(value, str):
        raise ProviderError(
            f"Tiingo intraday price row {row_index} has an invalid timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ProviderError(
            f"Tiingo intraday price row {row_index} has an invalid timestamp"
        ) from None
    if parsed.utcoffset() is None:
        raise ProviderError(
            f"Tiingo intraday price row {row_index} has a timezone-naive timestamp"
        )
    return parsed.astimezone(UTC)


def _tiingo_intraday_decimal(value: JsonValue, field_name: str) -> Decimal:
    try:
        if isinstance(value, bool) or value is None or isinstance(value, (list, dict)):
            raise InvalidOperation
        decimal_value = Decimal(str(value))
        if not decimal_value.is_finite():
            raise InvalidOperation
        return decimal_value
    except InvalidOperation:
        raise ProviderError(f"{field_name} must be a finite number") from None


def _tiingo_session(value: JsonValue, row_index: int) -> date:
    if not isinstance(value, str):
        raise ProviderError(f"Tiingo price row {row_index} has an invalid date")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise ProviderError(
                f"Tiingo price row {row_index} has an invalid date"
            ) from None
    return parsed.date()
