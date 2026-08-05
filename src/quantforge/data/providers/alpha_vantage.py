"""Alpha Vantage TIME_SERIES_DAILY_ADJUSTED adapter."""

import json
from datetime import UTC, date, datetime
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from quantforge.data.exceptions import ProviderError, RequestError
from quantforge.data.models import (
    AdjustmentMode,
    JsonValue,
    ProviderRecord,
    ProviderResponse,
)


class AlphaVantageProvider:
    """Retrieve raw daily bars and corporate actions from Alpha Vantage."""

    name = "alpha_vantage"
    adapter_version = "2"

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        if not api_key:
            raise RequestError("ALPHA_VANTAGE_API_KEY is required")
        self._api_key = api_key
        self._timeout = timeout

    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjustment: AdjustmentMode,
    ) -> ProviderResponse:
        if adjustment is AdjustmentMode.SPLIT_AND_DIVIDEND_ADJUSTED:
            raise RequestError(
                "Alpha Vantage adapter does not expose coherent dividend-adjusted OHLCV"
            )
        query = urlencode(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self._api_key,
            }
        )
        try:
            with urlopen(
                f"https://www.alphavantage.co/query?{query}", timeout=self._timeout
            ) as response:
                payload = cast(JsonValue, json.load(response))
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            raise ProviderError("Alpha Vantage request failed") from error
        if not isinstance(payload, dict):
            raise ProviderError("Alpha Vantage returned a malformed response")
        series_value = payload.get("Time Series (Daily)")
        if not isinstance(series_value, dict):
            message = (
                payload.get("Error Message")
                or payload.get("Note")
                or payload.get("Information")
            )
            detail = f": {message}" if isinstance(message, str) else ""
            raise ProviderError(f"Alpha Vantage returned no daily series{detail}")
        records: list[ProviderRecord] = []
        for session_text, row_value in series_value.items():
            if not isinstance(row_value, dict):
                raise ProviderError("Alpha Vantage returned a malformed daily row")
            try:
                session = date.fromisoformat(session_text)
                if start <= session <= end:
                    records.append(
                        {
                            "session_date": session_text,
                            "open": row_value["1. open"],
                            "high": row_value["2. high"],
                            "low": row_value["3. low"],
                            "close": row_value["4. close"],
                            "volume": row_value["6. volume"],
                            "dividend_amount": row_value["7. dividend amount"],
                            "split_coefficient": row_value["8. split coefficient"],
                        }
                    )
            except (KeyError, ValueError) as error:
                raise ProviderError(
                    "Alpha Vantage returned a malformed daily row"
                ) from error
        metadata_value = payload.get("Meta Data")
        provider_symbol = symbol
        if isinstance(metadata_value, dict):
            symbol_value = metadata_value.get("2. Symbol")
            if isinstance(symbol_value, str):
                provider_symbol = symbol_value
        return ProviderResponse(
            self.name,
            provider_symbol,
            datetime.now(UTC),
            "US/Eastern",
            adjustment,
            tuple(records),
            {"endpoint": "TIME_SERIES_DAILY_ADJUSTED"},
            self.adapter_version,
        )
