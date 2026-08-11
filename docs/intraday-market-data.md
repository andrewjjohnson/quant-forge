# Provider-neutral intraday market-data contracts

QF-15 defines the typed boundary for requesting and normalizing historical
intraday OHLCV. It builds on QF-13 timeframe/session semantics and QF-14
dataset-family provenance without implementing a provider HTTP client, cache,
aggregation, or session-gap validation.

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

`IntradayBar` is the provider-neutral adapter output. It records:

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
`IntradayBarProvider.fetch_intraday_bars()` returns only a tuple of canonical
`IntradayBar` records. A future adapter is responsible for preserving its raw
response separately before mapping it to these contracts.

Like requests, bars expose `to_primitive()`, canonical `serialize()` bytes, and
a deterministic `bar_id`. The bar identity binds timestamps, interval and
session policy, completion, OHLCV, request/snapshot provenance, feed, and
adjustment basis.

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

## Daily compatibility

QF-15 does not change:

- `DailyBar` or its date-based completed-session meaning;
- QF-3 schema version 4;
- `DailyBarProvider.fetch_daily_bars()` or `ProviderResponse`;
- `MarketDataService.get_daily_bars()`;
- daily request/cache keys, manifests, dataset IDs, artifacts, or provider
  implementations.

Intraday contracts use schema version 1 in a separate namespace. Future cache
and ingestion stories must create new intraday artifacts and bind the complete
request/timeframe identities rather than modifying legacy QF-3 artifacts.

## Deliberate limitations

QF-15 does not implement:

- Tiingo or another provider's intraday HTTP mapping;
- raw or normalized intraday cache population;
- chunked retrieval;
- session-gap or missing-source-bar validation;
- bar aggregation;
- multi-timeframe as-of alignment;
- provider capability discovery over the network.

Those are sibling QF-12 stories. They must consume these contracts without
weakening QF-13 completion semantics or QF-14 source-family consistency.
