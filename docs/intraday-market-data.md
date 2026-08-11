# Provider-neutral intraday market-data contracts

QF-15 defines the typed boundary for requesting and normalizing historical
intraday OHLCV. QF-16 implements the first HTTP and immutable-cache path behind
that boundary using Tiingo. Both build on QF-13 timeframe/session semantics and
QF-14 dataset-family provenance; neither implements aggregation or session-gap
validation.

The public records are exported from `quantforge.data`. Provider adapters use
`IntradayBarProvider` from `quantforge.data.providers`.

## Requests

`IntradayBarRequest` is an immutable, serializable request for a half-open
timestamp range: `start_timestamp` is inclusive and `end_timestamp` is
exclusive. Both values must be timezone-aware, are normalized to UTC, and must
be strictly ordered. A request requires an intraday QF-13 `Timeframe`; daily and
weekly intervals cannot pass as elapsed intraday durations.

```python
from datetime import UTC, datetime, timedelta

from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    FeedScope,
    IntradayBarRequest,
)
from quantforge.timeframes import IntradayInterval, Timeframe

request = IntradayBarRequest(
    symbol="SPY",
    start_timestamp=datetime(2024, 7, 1, tzinfo=UTC),
    end_timestamp=datetime(2024, 7, 2, tzinfo=UTC),
    timeframe=Timeframe.us_equity(IntradayInterval(timedelta(minutes=5))),
    feed_scope=FeedScope.consolidated(),
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
```

The embedded timeframe supplies the source interval, exchange calendar,
exchange timezone, regular- or extended-hours scope, intraday anchor,
cross-session policy, label policy, and developing-bar exposure. The request
also binds feed coverage and the complete adjustment/corporate-action basis.
Copying only a display interval such as `5m` is not sufficient provenance.

`to_primitive()` returns canonical JSON-compatible values, `serialize()` emits
canonical sorted JSON bytes, and `request_id` is the SHA-256 identity of that
complete primitive record. Equivalent instants expressed with different UTC
offsets receive the same identity. Changing the range, interval, session
policy, feed, or adjustment basis changes it.

## Feed scope

QF-15 reuses the QF-14 `FeedScope` contract instead of introducing a second
feed vocabulary:

- `FeedScope.iex_only()` is single-venue coverage for market center `IEX`;
- `FeedScope.consolidated()` is consolidated coverage;
- `FeedScope.unknown()` records that coverage is genuinely unknown;
- `FeedScope.provider_defined(name)` retains an explicit provider scope that
  has no canonical equivalent.

Unknown never means consolidated. A family or request that uses unknown feed
coverage receives a different deterministic identity from IEX-only or
consolidated data.

## Canonical bars

`IntradayBar` is one provider-neutral normalized observation. It records:

- canonical symbol and exchange-session identifier;
- explicit bar start and end timestamps normalized to UTC;
- the complete QF-13 timeframe and nominal source interval;
- actual elapsed duration derived from the explicit boundaries;
- completion state;
- exact Decimal OHLCV;
- provider-neutral `IntradayBarProvenance`.

The existing QF-13 `IntradayBarWindow` validates exchange-session membership,
anchor alignment, ordering, nominal versus actual duration, completion state,
partial terminal bars, cross-session policy, and developing-bar opt-in. A
completed partial-duration terminal bar therefore remains structurally
different from a developing bar. Bar labels do not control availability;
consumers use the explicit end and completion state.

Bars additionally reject non-finite or nonpositive prices, negative volume,
and impossible OHLC relationships. Zero volume is permitted because a valid
intraday interval can contain no executions. A provenance retrieval timestamp
cannot precede the observed bar end.

`IntradayBarProvenance` retains only provider-neutral primitives and domain
records:

- provider and provider symbol;
- adapter version and UTC retrieval timestamp;
- deterministic source request and immutable source snapshot identifiers;
- feed scope;
- adjustment/corporate-action basis.

Provider SDK or HTTP response objects never cross the adapter boundary.
`IntradayBarProvider.fetch_intraday_bars()` returns an `IntradayBarBatch`, never
a provider response or a bare tuple. The batch binds bars to the exact request,
requires chronological ordering, rejects duplicate keys, and verifies every
bar's symbol, timeframe, request identity, feed scope, adjustment basis, and
requested timestamp range. It deliberately does not infer missing bars or
validate session gaps. An ingestion adapter is responsible for preserving its
raw response separately before mapping it to these contracts. QF-16's Tiingo
adapter satisfies that boundary with `IntradayFetchResult` and immutable raw
chunk snapshots.

Like requests, bars and batches expose `to_primitive()`, canonical `serialize()`
bytes, and deterministic identities. The bar identity binds timestamps,
interval and session policy, completion, OHLCV, request/snapshot provenance,
feed, and adjustment basis. The batch identity additionally binds the exact
ordered collection and its complete request.

## Provider capability metadata

`IntradayProviderCapabilities` declares one adapter's:

- supported typed source intervals;
- supported feed scopes;
- supported regular- or extended-hours session scopes;
- optional earliest and latest supported timestamps.

Capability tuples are sorted for stable serialization, and duplicates or naive
range boundaries are rejected. `configuration_id` fingerprints the complete
declaration. Open-ended history is represented with a `None` start or end,
never by inventing a timestamp.

Call `validate_request()` before retrieval. It raises a precise public domain
exception:

- `UnsupportedIntervalError`;
- `UnsupportedFeedError`;
- `UnsupportedSessionScopeError`;
- `UnsupportedDateRangeError`.

All inherit from `UnsupportedCapabilityError` and the existing `RequestError`,
so callers may handle one capability class or an exact unsupported dimension.
These errors describe a valid provider-neutral request that a particular
adapter cannot satisfy; malformed contracts fail construction instead.

## Tiingo intraday ingestion

`TiingoProvider` implements both the QF-15 canonical adapter and the QF-16 raw
acquisition boundary. It currently declares:

- 1-minute and 5-minute source intervals;
- the complete canonical XNYS regular-hours timeframe policy only;
- raw, unadjusted OHLCV only;
- consolidated equity and explicitly IEX-only feed scopes.

Five-minute consolidated data is the preferred canonical source where the
account and endpoint provide it. Tiingo's
[consolidated equity documentation](https://www.tiingo.com/documentation/equity-realtime-stock-data)
maps a consolidated request to
`/tiingo/equity/intraday/<ticker>/prices`; an IEX-only request uses
the endpoint documented by the
[IEX API](https://www.tiingo.com/documentation/iex),
`/iex/<ticker>/prices`. The adapter never silently falls back from one to the
other because doing so would change the QF-15 request identity and source-family
meaning. Callers that cannot access consolidated history must make a new
request with `FeedScope.iex_only()`. Tiingo documents the consolidated endpoint
as beta and the IEX historical volume field as IEX-only coverage.

Authentication uses only `Authorization: Token ...`. Tokens are absent from
query strings, endpoint metadata, raw artifacts, normalized artifacts,
manifests, and translated exception text. Requests explicitly select OHLCV,
disable provider force-fill, and exclude after-hours data. Force-filled chart
bars would invent observations and are therefore not accepted as canonical
source data.

Intraday corporate actions are not supplied by these endpoints. Requests must
therefore use `TiingoProvider.intraday_adjustment_basis`, whose policy records
`not_provided_for_intraday_bars`; the adapter rejects a request that claims the
daily provider-reported dividend/split policy.

The adapter divides a half-open request into contiguous chunks of at most 30
days by default. The bound is configurable for testing and provider-plan
constraints. Chunk boundaries are deterministic from the exact request start,
end, and bound. Tiingo's documented date parameters can cause adjacent raw
responses to include the same boundary session; normalization assigns a row to
exactly one half-open chunk by bar start, sorts the merged bars, and then lets
`IntradayBarBatch` reject any remaining duplicate key or ordering problem.
Detailed expected-session completeness checks remain deferred.

## Intraday immutable cache

`IntradayMarketDataService` checks an `IntradayMarketDataCache` before provider
access. A cached request can therefore be replayed with only
`provider_name="tiingo"`; constructing a provider or supplying
`TIINGO_API_KEY` is unnecessary. A cache miss without a provider fails
explicitly.

The intraday namespace is separate from QF-3 daily schema version 4:

```text
intraday/raw/<raw-snapshot-sha256>.json
intraday/datasets/<dataset-sha256>/bars.json
intraday/datasets/<dataset-sha256>/manifest.json
intraday/requests/<provider>/<request-sha256>.json
```

Each raw chunk is canonical JSON containing the lossless response records,
provider and provider symbol, non-secret request parameters, endpoint, exact
chunk range, retrieval timestamp, parent request ID, and adapter version. Its
SHA-256 is both its immutable filename and the `source_snapshot_id` retained by
every canonical bar derived from it.

The normalized artifact is the QF-15 canonical `IntradayBarBatch`. Dataset
schema version 1 manifests bind:

- the full request and request identity;
- provider, endpoint/feed scope, source interval, and session scope;
- adapter and provider-capability versions;
- every ordered chunk range, retrieval timestamp, raw location, and raw hash;
- normalized batch identity, bar count, content hash, and dataset identity.

Writes use fsynced temporary files and hard-link creation. Existing immutable
content is accepted only when its bytes match; a collision raises `CacheError`.
Refresh creates new raw and dataset identities and atomically advances only the
small request pointer, leaving every prior snapshot reloadable.

```python
cache = IntradayMarketDataCache(Path("data/market-data"))
dataset = IntradayMarketDataService(
    cache,
    provider=TiingoProvider(os.environ["TIINGO_API_KEY"]),
).get_intraday_bars(request)

# Later, including in an offline process with no API key:
replayed = IntradayMarketDataService(
    cache,
    provider_name="tiingo",
).get_intraday_bars(request)
```

Ordinary tests use small synthetic JSON fixtures. Live verification is opt-in:

```bash
TIINGO_API_KEY=... QUANTFORGE_RUN_LIVE_TIINGO_INTRADAY=1 \
  uv run pytest -m integration \
  tests/integration/test_tiingo_intraday_market_data.py
```

The live test prefers consolidated data. Set
`QUANTFORGE_TIINGO_INTRADAY_FEED=iex` to explicitly validate the IEX-only path.

## Daily compatibility

QF-15 does not change:

- `DailyBar` or its date-based completed-session meaning;
- QF-3 schema version 4;
- `DailyBarProvider.fetch_daily_bars()` or `ProviderResponse`;
- `MarketDataService.get_daily_bars()`;
- daily request/cache keys, manifests, dataset IDs, artifacts, or provider
  behavior.

Intraday contracts and QF-16 datasets use independently versioned schema-1
artifacts in a separate namespace, leaving legacy QF-3 artifacts unchanged.

## Deliberate limitations

QF-15/QF-16 do not implement:

- session-gap or missing-source-bar validation;
- bar aggregation;
- multi-timeframe as-of alignment;
- provider capability discovery over the network.

Those are sibling QF-12 stories. They must consume these contracts without
weakening QF-13 completion semantics or QF-14 source-family consistency.
