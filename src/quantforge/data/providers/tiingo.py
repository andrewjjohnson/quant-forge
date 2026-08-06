"""Tiingo End-of-Day raw-price and corporate-action adapter."""

import json
import math
import time
from datetime import UTC, date, datetime
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from quantforge.data.exceptions import ProviderError, RequestError
from quantforge.data.models import (
    AdjustmentMode,
    JsonValue,
    ProviderRecord,
    ProviderResponse,
)

_BASE_URL = "https://api.tiingo.com/tiingo/daily"
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


class TiingoProvider:
    """Retrieve lossless Tiingo EOD JSON using header-only token authentication."""

    name = "tiingo"
    adapter_version = "1"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 30.0,
        retry_delays: tuple[float, ...] = (0.25, 1.0),
    ) -> None:
        if not api_key or not api_key.strip():
            raise RequestError("TIINGO_API_KEY is required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise RequestError("Tiingo timeout must be positive and finite")
        if any(not math.isfinite(delay) or delay < 0 for delay in retry_delays):
            raise RequestError("Tiingo retry delays must be finite and nonnegative")
        self._api_key = api_key.strip()
        self._timeout = timeout
        self._retry_delays = retry_delays

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

    def _request_json(self, url: str, label: str) -> JsonValue:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Token {self._api_key}",
                "User-Agent": "QuantForge/0.1 TiingoEOD/1",
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
