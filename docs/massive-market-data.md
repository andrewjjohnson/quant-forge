# Massive historical stock aggregates (QF-54)

`massive` is a canonical provider identity behind the existing intraday
contracts. It supplies consolidated U.S. stock 1-minute and 5-minute OHLCV
for the canonical XNYS regular-hours policy. This adapter implements historical
REST aggregates only. QF-45 can select it at provider setup without changing
indicators, predictions, outcome labels, validation, or reporting.

## Configuration

Set `MASSIVE_API_KEY` in the process environment using your shell or approved
secret manager. QuantForge does not load `.env` files automatically. Never
commit the value. The setup factory also accepts an explicit `api_key` argument.

```python
from datetime import UTC, datetime, timedelta
from pathlib import Path

from quantforge.data import (
    FeedScope,
    IntradayBarRequest,
    IntradayMarketDataCache,
    IntradayMarketDataService,
)
from quantforge.data.lineage import AdjustmentBasis
from quantforge.data.models import AdjustmentMode
from quantforge.data.providers import create_intraday_provider
from quantforge.timeframes import IntradayInterval, Timeframe

provider_name = "massive"  # "tiingo" remains available through the same factory.
request = IntradayBarRequest(
    symbol="SPY",
    start_timestamp=datetime(2025, 7, 1, tzinfo=UTC),
    end_timestamp=datetime(2025, 7, 2, tzinfo=UTC),
    timeframe=Timeframe.us_equity(IntradayInterval(timedelta(minutes=1))),
    feed_scope=FeedScope.consolidated(),
    adjustment_basis=AdjustmentBasis(
        AdjustmentMode.UNADJUSTED,
        "raw_provider",
        "raw_provider",
        "not_provided_for_intraday_bars",
        False,
    ),
)
cache = IntradayMarketDataCache(Path("data/market-data"))
service = IntradayMarketDataService(
    cache,
    provider=create_intraday_provider(provider_name),
)
dataset = service.get_intraday_bars(request)

# Offline: no provider construction, API key, or network is needed.
replayed = IntradayMarketDataService(
    cache,
    provider_name=provider_name,
).get_intraday_bars(request)
```

The repository previously used direct adapter construction. The additive
`create_intraday_provider` setup helper supports both intraday providers;
existing constructors and injected fake providers are unchanged. The service
continues to check the cache before calling any provider. A caller wanting
credential-free replay uses the existing `provider_name`-only service.

## Acquisition and time semantics

The adapter uses Massive's current
[Custom Bars endpoint](https://massive.com/docs/rest/stocks/aggregates/custom-bars):
`https://api.massive.com/v2/aggs/ticker/{ticker}/range/{multiplier}/minute/{from}/{to}`.
It maps canonical 1m/5m to multipliers 1/5, requests `sort=asc`, `limit=50000`,
and explicit `adjusted=true` or `false`. The limit counts base aggregates,
not necessarily returned bars. One logical range starts one request followed
by every `next_url`; there is no per-session acquisition loop.

[Authentication](https://massive.com/docs/rest/quickstart) uses only
`Authorization: Bearer ...` on every page. Pagination URLs must stay on the
HTTPS Massive host and the same ticker/interval endpoint. Credential query
parameters are removed; changed query policies, repeated URLs, and redirects
are rejected before credentials can be forwarded. HTTP attempts, including
bounded retries, are observable through `http_request_count`.

The wire bounds are integer Unix milliseconds: floor the start, and floor
`end - 1 microsecond` for the inclusive provider end. Provider snapping or
pagination can return extra observations. Canonical retention enforces the
original exact UTC `[start_timestamp, end_timestamp)` range and requires the
entire bar to fit before the end. `t` is the bar start; the canonical end is
start plus the requested interval. No float conversion is used for timestamps.

Exchange-local session dates and actual opens/closes come from the existing
XNYS calendar. Pre-market, post-market, holiday, and out-of-range rows with
valid timestamps are excluded. Early closes retain only the actual RTH window
(210 expected one-minute intervals rather than 390 on a normal session).
Observed aggregate counts may be lower. Invalid timestamps fail closed. Retained
rows use the existing canonical Decimal, OHLC, volume, alignment, session, and
batch validators. Rows are sorted explicitly; duplicate in-session starts fail,
including at a partial final request bar. VWAP and trade count are optional and
remain in raw responses only.

Massive documents that an interval can have no aggregate when no qualifying trade
occurs. The adapter preserves missing intervals, including successful responses
with no retained bars; it never synthesizes or forward-fills an observation.
An omission alone does not establish its cause. The existing cache's diagnostic
coverage report records missing intervals and incomplete sessions truthfully.
Massive adds no universal requirement for every expected RTH interval to contain
an aggregate.

Acceptance remains with existing QuantForge completeness policies: strict
coverage validation and default aggregation reject incomplete inputs, while
diagnostic acquisition and replay preserve them with their quality evidence.
QF-45/SPY contracts and the optional SPY verifier can require complete RTH coverage.
Zero volume remains permitted by the canonical model and produces its existing
quality warning.

## Adjustment and events

`MassiveProvider.intraday_adjustment_basis` is raw OHLCV (`adjusted=false`).
`MassiveProvider.split_adjusted_basis` maps to `adjusted=true`, with both OHLC
and volume marked `split_adjusted` and `adjusted_fields_used=true`.
Massive documents [split adjustments to OHLCV, including fractional volume](https://massive.com/knowledge-base/article/why-does-volume-return-as-a-decimal-value-from-the-aggregates-endpoint).
The adapter checks each page's ticker and adjustment metadata against the
request. Dividend-adjusted requests and contradictory basis declarations fail.
There is no local adjustment calculation.

Both modes use `not_provided_for_intraday_bars`. No corporate-action event
endpoint is called, and unavailable events do not mean no events occurred.
QF-51 preserves `CorporateActionAvailability.UNAVAILABLE`, incomplete event
coverage, and the empty supplied-event snapshot truthfully. Provider-adjusted
history reflects the provider's retrieved adjustment basis; it is not a claim
of point-in-time corporate-action knowledge or suitability for QF-5 accounting.

## Provenance, cache, and failures

One existing `IntradayRawSnapshot` covers the exact logical range. Its records
are the ordered pages, each with a safe request URL, retrieval time, and full
JSON response except credential redaction. The snapshot hash binds all pages,
including optional fields and excluded observations. Every canonical bar points
to that snapshot. The source retrieval timestamp is the final page's retrieval
instant. Endpoint family, wire parameters, adapter version, logical request,
feed/session/adjustment basis, and canonical digest use existing manifest fields.
No cache or provenance schema changes are needed; see ADR 0026.

The existing request ID describes provider-neutral requested semantics. The
cache key is `(provider_name, request_id)`, and source/dataset/family identities
also bind `massive`. Numerically identical Tiingo bars cannot alias the source.
Page layout and retrieval timestamps affect immutable acquisition identity,
as existing contracts require, but do not change the logical request ID.

All pages and canonical structural validation must succeed before persistence.
A failed or interrupted page acquisition leaves no successful partial dataset or
new request pointer; restarting it refetches the range. Successfully acquired
sparse or empty datasets can be cached and replayed with their diagnostic coverage
reports. Finishing pagination does not assert complete market-session coverage.
There is no persisted page-level resume checkpoint. Refresh preserves old
snapshots and advances the pointer only after success. Changed range, interval,
adjustment, feed, or session policies cannot reuse an incompatible request.
Massive uses the existing default diagnostic cache reuse policy, as do fake
providers; Tiingo retains its own policy.

Errors distinguish configuration, unsupported capabilities, authentication (401),
subscription/authorization (403), other HTTP failures, malformed JSON/payloads,
row errors, and pagination. Strict consumers report incomplete required coverage
through the existing validation/aggregation errors. Safe diagnostics include ticker,
interval, endpoint family, range, and available page/row/timestamp context.
HTTP bodies, exception causes, and malformed field contents are not echoed.
Configured secrets are redacted from translated errors and retained responses.

## Verification

Use uv 0.12.1 and the frozen dependency workflow:

```bash
uv run --frozen pytest tests/unit/data/test_massive_provider.py \
  tests/integration/test_massive_prediction_inputs.py
```

Synthetic tests exercise both adjustment modes, pagination/security, exact range
boundaries, sparse/empty responses, strict consumer rejection of incomplete inputs,
immutable cache/replay, and unchanged QF-51/QF-52
composition. A three-session fixture spanning July 4, 2024 contains 990 source
bars, 495 derived two-minute bars, and three completed daily bars, with the July 3
early close respected. No study or EMA logic is added.

With the key set, the opt-in verifier acquires normal/early-close sessions,
a month, a pagination range, and a full 2025 SPY year through the normal service.
It requires complete coverage using the existing quality report and checks 390/210
bars for its specific normal/early-close SPY cases. These are verification
requirements for SPY, not universal Massive-provider invariants:

```bash
uv run --frozen python scripts/verify_massive_history.py
uv run --frozen python scripts/verify_massive_history.py --offline
# Explicitly request more continuous history when account access permits it:
uv run --frozen python scripts/verify_massive_history.py \
  --history-start 2023-01-01 --history-end 2026-01-01
```

Each JSON result reports exact request/observed bounds, HTTP attempts, retained
page count, raw/canonical row counts, source identities, complete coverage, and
credential-free cache replay. Licensed responses remain in ignored `data/`.
Account entitlements can prevent acquisition. Historical gaps can fail the strict
SPY verification even when acquisition and diagnostic caching succeed. Failed
acquisition is never reported as successfully cached, and incomplete coverage is
never reported as complete. QF-45 strategy/study execution, flat files, streaming,
options, ML, and trading remain outside QF-54.
