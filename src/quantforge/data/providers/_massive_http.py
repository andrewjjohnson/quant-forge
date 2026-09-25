"""Massive-only HTTP transport with header credentials and bounded retries."""

import json
import math
import time
from email.message import Message
from typing import IO, cast
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from quantforge.data.exceptions import ProviderError, RequestError
from quantforge.data.models import JsonValue

_RETRYABLE = frozenset((429, 500, 502, 503, 504))


class NoRedirect(HTTPRedirectHandler):
    """Never forward the Authorization header to a redirect destination."""

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> Request | None:
        return None


class MassiveHTTPClient:
    """Keep credentials and untrusted HTTP diagnostics out of public errors."""

    def __init__(
        self, api_key: str, timeout: float, retry_delays: tuple[float, ...]
    ) -> None:
        if not api_key or not api_key.strip():
            raise RequestError("MASSIVE_API_KEY is required")
        if any(character.isspace() for character in api_key.strip()):
            raise RequestError("MASSIVE_API_KEY contains invalid whitespace")
        if not math.isfinite(timeout) or timeout <= 0:
            raise RequestError("Massive timeout must be positive and finite")
        if any(not math.isfinite(delay) or delay < 0 for delay in retry_delays):
            raise RequestError("Massive retry delays must be finite and nonnegative")
        self._api_key = api_key.strip()
        self._timeout = timeout
        self._retry_delays = retry_delays
        self._opener = build_opener(NoRedirect())
        self.request_count = 0

    def redact(self, text: str) -> str:
        return text.replace(self._api_key, "<redacted>")

    def request_json(self, url: str) -> JsonValue:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "QuantForge/0.1 MassiveIntraday/1",
            },
            method="GET",
        )
        for attempt in range(len(self._retry_delays) + 1):
            self.request_count += 1
            try:
                with self._opener.open(request, timeout=self._timeout) as response:
                    payload = response.read()
            except HTTPError as error:
                status = error.code
                error.close()
                if status in _RETRYABLE and attempt < len(self._retry_delays):
                    time.sleep(self._retry_delays[attempt])
                    continue
                label = {
                    401: "authentication rejected",
                    403: "authorization/subscription rejected",
                    429: "rate limit exceeded after bounded retries",
                }.get(status, "aggregate HTTP request failed")
                raise ProviderError(f"{label} (HTTP status {status})") from None
            except (TimeoutError, URLError, OSError):
                if attempt < len(self._retry_delays):
                    time.sleep(self._retry_delays[attempt])
                    continue
                raise ProviderError(
                    "network request failed after bounded retries"
                ) from None
            try:
                return cast(JsonValue, json.loads(payload))
            except (UnicodeDecodeError, ValueError, TypeError):
                raise ProviderError("malformed aggregate JSON") from None
        raise AssertionError("bounded Massive retry loop did not terminate")
