# Provider-neutral intraday market-data contracts

QF-15 defines the typed boundary for requesting and normalizing historical
intraday OHLCV. QF-16 implements the first HTTP and immutable-cache path behind
that boundary using Tiingo. QF-17 validates completed source-interval coverage
against the configured exchange calendar and persists typed quality evidence.
QF-18 aggregates those immutable source datasets into deterministic larger
intraday bars. QF-19 derives exchange-session daily and exchange-weekly bars
from the same source. All five build on QF-13 timeframe/session semantics and
QF-14 dataset-family provenance.

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

Adjacent canonical bars may not overlap. Duplicate keys, out-of-order bars,
overlapping timestamps, or bars outside their request range fail at the typed
batch boundary before coverage validation.

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

## Intraday coverage and quality validation

`validate_intraday_coverage()` consumes a provider-neutral
`IntradayBarBatch`; it performs no retrieval and never changes, fills,
interpolates, or invents a bar. Expected completed source intervals are derived
from the request's complete QF-13 timeframe and the installed exchange calendar.
The calculation therefore uses actual XNYS sessions, session opens and closes,
early closes, exchange-local daylight-saving behavior, and the request's
regular- or explicit extended-hours scope. Weekends and holidays do not enter
the expected set.

The typed `IntradayCoverageReport` records:

- request, batch, timeframe, source-interval, feed, and session-scope identity;
- expected and observed completed interval counts;
- exact missing and unexpected interval boundaries;
- incomplete exchange sessions;
- developing bars and zero-volume bars as warnings;
- per-session open, close, requested coverage, and quality details.

Completed terminal partial-duration bars are expected when a session ends
before the nominal interval boundary, including an early close. They are not
reported as gaps. Completed bars that do not match the canonical expected
boundaries are material unexpected intervals. Developing bars do not satisfy a
completed interval and remain separately visible.

The modes change control flow, not calendar or quality semantics:

- `IntradayValidationMode.STRICT` raises
  `IntradayCoverageValidationError` when completed intervals are missing or
  unexpected. The exception retains the complete report.
- `IntradayValidationMode.DIAGNOSTIC` always returns the report so a caller can
  inspect or apply its own tolerance policy.

Zero volume remains permitted by the QF-15 canonical contract because a valid
interval can contain no executions. QF-17 records it as a warning. Negative or
non-finite volume, non-finite/nonpositive prices, null/non-Decimal values, and
invalid OHLC relationships remain hard `IntradayBar` construction failures in
both modes. Session-scope, range, ordering, duplicate, and overlap violations
likewise fail at the canonical bar or batch boundary rather than being softened
by diagnostic mode.

Intraday cache schema version 2 embeds the diagnostic report and its
deterministic identity in every dataset manifest. Cache loads recompute the
report from canonical bars and the exchange calendar and reject a mismatch.
`IntradayDataset.quality_report` exposes the verified report directly to later
derived-dataset and study code. Raw snapshot schema version 1 remains unchanged,
so the same raw response bytes keep their content identity. Existing intraday
dataset-schema-1 manifests are not rewritten; refresh or re-ingest them to
create a quality-bearing schema-2 dataset.

## Session-aware derived intraday bars

`aggregate_intraday_dataset()` accepts one immutable `IntradayDataset`, a QF-13
target `Timeframe`, and an optional `IntradayAggregationPolicy`. The target must
be intraday, strictly larger than the source, and an exact duration multiple.
Source and target use the same exchange-session and anchor policies, both
prohibit cross-session continuation, and the target excludes developing bars.
This supports 5-minute sources resampled to 15-minute, 30-minute, 1-hour,
2-hour, and 4-hour bars as well as other valid exact multiples.

Target windows begin at the actual session open under the default policy and
end at the earlier of their nominal boundary or actual session close. A normal
XNYS 4-hour session therefore emits 09:30-13:30 and a completed partial-duration
13:30-16:00 bar. The 2024-11-29 early close emits one completed partial-duration
09:30-13:00 bar. Boundaries are resolved per session, so DST changes affect UTC
offsets without changing the exchange-local 09:30 open, and no aggregate can
contain constituents from two sessions.

OHLCV uses first open, maximum high, minimum low, final close, and summed
volume. Every output window records ordered source bar IDs, expected and
observed constituent counts, exact missing intervals, and its output bar ID.
The default `MissingConstituentPolicy.STRICT` raises
`IntradayAggregationQualityError` before returning a dataset if the verified
source report is incomplete. `DIAGNOSTIC` excludes unexpected source intervals,
never fills gaps, and may emit an explicitly incomplete aggregate from the
available expected constituents; a window with no observations is recorded but
not emitted. Consumers must not treat a diagnostic incomplete bar as equivalent
to a quality-complete bar.

Derived bars identify `quantforge` as their producer and point their source
snapshot at the immutable normalized source dataset ID. The derived manifest
embeds the complete QF-17 source report and binds:

- source dataset, request, batch, normalized content, and raw snapshot IDs;
- complete target request/timeframe semantics;
- complete aggregation policy and per-window quality report;
- normalized derived bar content and identities;
- a QF-14 family manifest whose only members are the canonical source dataset
  and the QuantForge-derived child.

Provider-native higher-timeframe bars therefore cannot enter this derived
family by matching a provider name or symbol. Repeating aggregation with the
same source snapshot, target timeframe, and policy produces identical bars,
bar IDs, dataset ID, family manifest, and bytes. Changing the source snapshot
or aggregation policy changes the derived dataset identity.

`IntradayAggregationCache` writes immutable artifacts separately from source
ingestion:

```text
intraday/derived/<dataset-sha256>/bars.json
intraday/derived/<dataset-sha256>/manifest.json
```

Loading rederives the result from the verified source dataset and compares the
complete canonical bytes, identities, and checksums. Existing content is never
overwritten; differing bytes at a content-addressed path raise `CacheError`.
Before creating any artifact, persistence also recomputes the supplied derived
dataset's batch ID, bar count, content checksum, source/report bindings, dataset
identity, canonical paths, producer provenance, and exact family lineage. A
reconstructed or replaced dataclass therefore cannot poison a content-addressed
entry with bytes that disagree with its metadata.

## Exchange-session daily and weekly derived bars

`aggregate_session_dataset()` accepts one immutable `IntradayDataset`, a
`Timeframe` containing exactly `SessionInterval(1)` or
`TradingWeekInterval(1)`, and an optional `SessionAggregationPolicy`. Source and
target session policies must match exactly. The default strict policy rejects a
QF-17 report with missing or unexpected completed intervals. Diagnostic mode
excludes unexpected observations, never fills missing constituents, and
preserves every gap in the source report and target-period evidence.

Daily bars map to one fully covered exchange session. Their timestamps run from
that session's configured actual open to actual close, including early closes.
Weekly bars map to the QF-13 Monday-Sunday exchange trading week and list every
constituent session explicitly. Holidays are absent because the exchange
calendar publishes no session; a holiday-shortened week is complete when all
of its actual sessions are covered. Source ranges beginning or ending inside a
session or exchange week are recorded as excluded partial periods and are not
emitted as completed bars.

Both targets use first open, maximum high, minimum low, final close, and summed
volume. Each bar carries the ordered immutable intraday source-bar IDs used in
the calculation. Validation rejects duplicate source IDs within a bar and any
reuse across bars in one derived dataset. Equivalent runs produce identical
bars, bar IDs, dataset IDs, family manifests, and bytes.

The derived manifest embeds and verifies:

- the normalized source dataset, raw snapshots, request, batch, and complete
  QF-17 quality report;
- the complete QF-13 target timeframe and configured regular/extended-hours
  session scope;
- the source adjustment and corporate-action basis through the QF-14 family;
- the aggregation policy and per-period constituent evidence;
- a two-member QF-14 lineage from the source snapshot to the QuantForge-derived
  target dataset.

`SessionAggregationCache` persists immutable artifacts at:

```text
session/derived/<dataset-sha256>/bars.json
session/derived/<dataset-sha256>/manifest.json
```

Cache loads rederive the requested dataset from the verified source and compare
canonical bytes, identities, and checksums. Existing artifacts are never
overwritten with different bytes.

### Differences from provider-native EOD data

Provider-native daily or weekly bars are never substituted. Even for the same
symbol and apparent date, provider EOD values can differ because of feed venue
coverage, regular-versus-extended hours, provider correction timing, corporate-
action adjustment, weekly labeling, or proprietary aggregation rules. QF-19
bars identify `quantforge` as producer and trace to the exact canonical
intraday snapshot. Provider EOD remains a separate dataset family unless a
future explicit external-bar validation policy proves compatibility.

## Daily compatibility

QF-15 does not change:

- `DailyBar` or its date-based completed-session meaning;
- QF-3 schema version 4;
- `DailyBarProvider.fetch_daily_bars()` or `ProviderResponse`;
- `MarketDataService.get_daily_bars()`;
- daily request/cache keys, manifests, dataset IDs, artifacts, or provider
  behavior.

Intraday contracts and datasets use independently versioned artifacts in a
separate namespace, leaving legacy QF-3 artifacts unchanged.

## Deliberate limitations

QF-15/QF-16/QF-17/QF-18/QF-19 do not implement:

- multi-timeframe as-of alignment;
- downstream study-specific tolerance decisions beyond exposing the report;
- developing target bars;
- provider capability discovery over the network.

Those are sibling QF-12 stories. They must consume these contracts without
weakening QF-13 completion semantics or QF-14 source-family consistency.
