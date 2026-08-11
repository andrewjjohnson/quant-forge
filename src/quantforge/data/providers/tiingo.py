"""Tiingo End-of-Day raw-price and corporate-action adapter."""

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
from quantforge.data.lineage import AdjustmentBasis, FeedScope
from quantforge.data.models import (
    AdjustmentMode,
    JsonValue,
    ProviderRecord,
    ProviderResponse,
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
    intraday_adapter_version = "1"
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
        for chunk_start, chunk_end in _chunk_ranges(
            request.start_timestamp,
            request.end_timestamp,
            self._intraday_chunk_duration,
        ):
            parameters = (
                ("afterHours", "false"),
                ("columns", "open,high,low,close,volume"),
                (
                    "endDate",
                    (chunk_end - timedelta(microseconds=1)).date().isoformat(),
                ),
                ("forceFill", "false"),
                ("resampleFreq", interval_parameter),
                ("startDate", chunk_start.date().isoformat()),
            )
            url = f"{base_url}/{encoded_symbol}/prices?{urlencode(parameters)}"
            payload = self._request_json(
                url,
                "intraday prices",
                user_agent="QuantForge/0.1 TiingoIntraday/1",
            )
            retrieved_at = datetime.now(UTC)
            if not isinstance(payload, list):
                raise ProviderError("Tiingo returned malformed intraday-price JSON")
            records: list[ProviderRecord] = []
            for row_index, row_value in enumerate(payload):
                if not isinstance(row_value, dict):
                    raise ProviderError(
                        f"Tiingo returned malformed intraday price row {row_index}"
                    )
                row = cast(dict[str, JsonValue], row_value)
                missing = [
                    field for field in _REQUIRED_INTRADAY_FIELDS if field not in row
                ]
                if missing:
                    raise ProviderError(
                        f"Tiingo intraday price row {row_index} is missing required "
                        f"fields: {', '.join(missing)}"
                    )
                records.append(row)
            snapshot = IntradayRawSnapshot(
                provider_name=self.name,
                provider_symbol=provider_symbol,
                adapter_version=self.intraday_adapter_version,
                endpoint=endpoint,
                source_request_id=request.request_id,
                chunk_start_timestamp=chunk_start,
                chunk_end_timestamp=chunk_end,
                retrieved_at=retrieved_at,
                request_parameters=parameters,
                records=tuple(records),
            )
            snapshots.append(snapshot)
            bars.extend(_intraday_bars_from_snapshot(request, snapshot))
        if not bars:
            raise ProviderError("Tiingo returned an empty intraday-price response")
        bars.sort(key=lambda bar: (bar.start_timestamp, bar.end_timestamp))
        try:
            batch = IntradayBarBatch(request, tuple(bars))
        except IntradayContractValidationError as error:
            raise ProviderError(
                f"Tiingo intraday response is invalid: {error}"
            ) from None
        return IntradayFetchResult(
            batch,
            tuple(snapshots),
            self.intraday_capabilities.configuration_id,
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
                    raise ProviderError("Tiingo authentication was rejected") from None
                if error.code in _RETRYABLE_HTTP_STATUSES:
                    if attempt < len(self._retry_delays):
                        time.sleep(self._retry_delays[attempt])
                        continue
                    if error.code == 429:
                        raise ProviderError("Tiingo rate limit was exceeded") from None
                    raise ProviderError(
                        "Tiingo temporary server failure persisted after "
                        "bounded retries"
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


def _chunk_ranges(
    start_timestamp: datetime,
    end_timestamp: datetime,
    maximum_duration: timedelta,
) -> tuple[tuple[datetime, datetime], ...]:
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start_timestamp
    while cursor < end_timestamp:
        chunk_end = min(cursor + maximum_duration, end_timestamp)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return tuple(chunks)


def _intraday_bars_from_snapshot(
    request: IntradayBarRequest,
    snapshot: IntradayRawSnapshot,
) -> tuple[IntradayBar, ...]:
    duration = request.source_interval.nominal_duration
    exchange_timezone = ZoneInfo(request.timeframe.session_policy.timezone_name)
    bars: list[IntradayBar] = []
    for row_index, record in enumerate(snapshot.records):
        try:
            start_timestamp = _tiingo_intraday_timestamp(record["date"], row_index)
            end_timestamp = start_timestamp + duration
            if not (
                snapshot.chunk_start_timestamp
                <= start_timestamp
                < snapshot.chunk_end_timestamp
            ):
                continue
            if end_timestamp > request.end_timestamp:
                continue
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
            bars.append(
                IntradayBar(
                    symbol=request.symbol,
                    session_date=start_timestamp.astimezone(exchange_timezone).date(),
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    timeframe=request.timeframe,
                    completion=BarCompletion.COMPLETED,
                    open=_tiingo_intraday_decimal(record["open"]),
                    high=_tiingo_intraday_decimal(record["high"]),
                    low=_tiingo_intraday_decimal(record["low"]),
                    close=_tiingo_intraday_decimal(record["close"]),
                    volume=_tiingo_intraday_decimal(record["volume"]),
                    provenance=provenance,
                )
            )
        except (KeyError, ValueError, InvalidOperation):
            raise ProviderError(
                f"Tiingo returned malformed intraday price row {row_index}"
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


def _tiingo_intraday_decimal(value: JsonValue) -> Decimal:
    if isinstance(value, bool) or value is None or isinstance(value, (list, dict)):
        raise InvalidOperation
    decimal_value = Decimal(str(value))
    if not decimal_value.is_finite():
        raise InvalidOperation
    return decimal_value


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
