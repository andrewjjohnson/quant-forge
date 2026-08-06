# Daily market data

QF-3 provides provider-independent completed daily stock and ETF bars plus
typed, immutable corporate actions. It does not calculate indicators, signals,
returns, or portfolio state.

## Public API

```python
import os
from datetime import date
from pathlib import Path

from quantforge.data import AdjustmentMode, MarketDataCache, MarketDataService
from quantforge.data.providers import TiingoProvider

service = MarketDataService(
    TiingoProvider(os.environ["TIINGO_API_KEY"]),
    MarketDataCache(Path("data/market-data")),
)
dataset = service.get_daily_bars(
    "SPY",
    date(2020, 1, 1),
    date(2025, 12, 31),
    AdjustmentMode.UNADJUSTED,
)
```

Dates are inclusive XNYS trading-session labels. Symbols are trimmed and
uppercased. `dataset.bars` is ascending canonical Decimal OHLCV,
`dataset.metadata` is complete provenance, and `dataset.corporate_actions`
exposes typed `CashDividend` and `StockSplit` records. `MarketDataCache.load()`
loads a dataset directly by immutable dataset ID.

`get_daily_bars(..., refresh=True)` makes a new provider request even when the
request key is cached. Raw responses and datasets remain write-once; only the
small request-key pointer atomically advances to the newly retrieved immutable
dataset. The previous dataset remains loadable by ID. Without `refresh`, an
identical request reuses the cache and makes no provider call.

## Tiingo EOD adapter

`TiingoProvider` uses [Tiingo End-of-Day](https://www.tiingo.com/documentation/end-of-day)
because it returns raw daily OHLCV,
adjusted audit fields, and per-session dividend and split facts together. Its
public constructor is:

```python
TiingoProvider(
    api_key: str,
    *,
    timeout: float = 30.0,
    retry_delays: tuple[float, ...] = (0.25, 1.0),
)
```

It requests metadata from `/tiingo/daily/<ticker>` and JSON prices from
`/tiingo/daily/<ticker>/prices` with `startDate`, `endDate`, daily resampling,
and JSON format. [Authentication](https://www.tiingo.com/documentation/general/connecting)
is only the `Authorization: Token ...` header;
the token is never placed in the URL, raw artifact, or provider error text.
Set `TIINGO_API_KEY` in the process environment. The repository does not parse
or mutate `.env` files; a shell or approved environment loader may load the
ignored local file.

The adapter preserves Tiingo's original fields losslessly in the raw artifact:

```text
date, open, high, low, close, volume,
adjOpen, adjHigh, adjLow, adjClose, adjVolume,
divCash, splitFactor
```

Available ticker, asset name, exchange code, provider start/end dates,
requested range, endpoint, response format, retrieval timestamp, provider, and
adapter version are preserved in raw metadata. Canonical execution bars use
only raw `open`, `high`, `low`, `close`, and `volume`. Adjusted Tiingo values are
audit fields and are never used by this execution path.

The provider supports only `AdjustmentMode.UNADJUSTED`; asking it for adjusted
ingestion fails explicitly. It reports authentication failures (401/403), rate
limits (429), other HTTP failures, malformed JSON, malformed/missing fields,
and empty responses through QF-3 exceptions. Network and retryable 5xx failures
receive only the configured bounded retries. Ordinary tests replace the HTTP
boundary and never call Tiingo.

Tiingo's terms and attribution requirements apply to retrieved data. Keep
downloaded data for authorized internal use and do not redistribute provider
responses or commit them to this repository.

## Alpha Vantage compatibility

`AlphaVantageProvider` remains supported with `ALPHA_VANTAGE_API_KEY` and its
existing `TIME_SERIES_DAILY_ADJUSTED` mapping. It preserves raw OHLCV, split
coefficients, and dividends. `split_adjusted` remains available for causal
indicator research, but QF-5 execution accepts only raw unadjusted OHLCV with
complete explicit actions. No existing Alpha Vantage cache is rewritten.

## Corporate actions and adjustment policy

Schema version 4 requires every provider record to carry a positive finite
`split_coefficient` and a finite nonnegative `dividend_amount`:

- zero dividends and unit split factors create no action;
- a nonzero dividend becomes `CashDividend(symbol, ex_dividend_session,
  amount_per_share, provider_name, source_dataset_id, action_id)`;
- a non-unit factor becomes `StockSplit(symbol, effective_session,
  split_factor, provider_name, source_dataset_id, action_id)`;
- negative dividends, nonpositive splits, duplicate symbol/type/session
  actions, malformed values, and actions without an observed bar are rejected.

Tiingo documents [`divCash`](https://www.tiingo.com/documentation/corporate-actions/dividends)
on the ex-dividend date. It is not a payment-date field. Tiingo defines
[`splitFactor`](https://www.tiingo.com/documentation/corporate-actions/splits)
as shares after divided by shares before, so
QF-5 multiplies existing shares by the factor and divides average cost basis by
it. See ADR 0003 and `docs/backtesting.md`.

`DatasetMetadata` records:

- raw, normalized, and corporate-action artifact locations;
- provider, provider symbol, adapter version, and UTC retrieval time;
- requested/actual range, calendar, timezone, and missing sessions;
- adjustment mode, `ohlc_basis`, `volume_basis`, and adjusted-field usage;
- corporate-action completeness, policy, counts, and action sessions;
- raw/data SHA-256 values, corporate-action snapshot ID, dataset ID, and schema.

The corporate-action snapshot ID fingerprints ordered action economics. Stable
action IDs additionally bind every action to that snapshot and source dataset.
The dataset ID includes metadata, raw and canonical-bar hashes, and action
snapshot identity. A provider correction therefore produces a different raw
snapshot/dataset; an action correction also changes the action snapshot. QF-5
includes both dataset and action snapshot identity in its run identity.

`unadjusted` means raw provider OHLCV with dividends and splits exposed as
separate events. QF-5 always applies splits and explicitly selects price-only,
cash-dividend, or strict-rejection treatment for dividends. `split_adjusted`
divides earlier prices by all later split
factors and multiplies volume by those factors, retaining a coherent basis for
indicator research. `split_and_dividend_adjusted` is rejected. QF-5 also
rejects any adjusted dataset because adjusted executions plus explicit cash
dividends could double count distributions and adjusted historical share units
are incompatible with point-in-time whole-share fills.

## Calendar and validation

The maintained `exchange-calendars` XNYS schedule determines expected sessions.
Weekends and holidays are not gaps. Stable ascending sort is the only automatic
correction. Duplicate sessions, out-of-range rows, empty input, inconsistent
symbols or adjustment modes, invalid/nonpositive OHLCV, impossible high/low
relationships, malformed dates, incomplete action fields, duplicate actions,
invalid action economics, and non-canonical manifest JSON field types are
rejected. Cached retrieval timestamps must be ISO 8601 values with a defined UTC
offset; timezone-naive values are rejected before UTC normalization. Strict
ingestion (the default) rejects missing exchange sessions. Non-strict ingestion
records them; QF-5 still rejects gaps inside the observed range. Values are
never filled, interpolated, rounded, or invented.

## Immutable storage

The caller-selected cache root contains:

```text
raw/<raw-sha256>.json
datasets/<dataset-sha256>/bars.csv
datasets/<dataset-sha256>/corporate_actions.json
datasets/<dataset-sha256>/manifest.json
requests/<request-sha256>.json
```

Raw JSON is a lossless representation of the complete provider response and
request metadata. Bars use exact decimal-text CSV. Corporate actions use stable
canonical JSON, including string-encoded decimal economics. Raw, bar, action,
and manifest artifacts are immutable and written with fsynced temporary files.
Loads verify raw/bar hashes, canonical
paths, metadata, the exact supported corporate-action artifact schema, action
snapshot/action IDs, dataset identity, calendar facts, and all complete-dataset
invariants. Dataset hashes prove consistency, not provider authenticity.

Schema version 4 changes request and dataset identities. Older artifacts remain
immutable and are not upgraded silently; re-ingest them for action-aware QF-5.

## Live verification and SPY example

Ordinary tests are offline. The opt-in integration range includes a known SPY
dividend and performs one ingestion, cache reload, price-only and cash-dividend
backtests, and both exports:

```bash
TIINGO_API_KEY=... QUANTFORGE_RUN_LIVE_TIINGO=1 \
  uv run pytest -m integration tests/integration/test_tiingo_market_data.py
```

Run the fixed real-data example with:

```bash
TIINGO_API_KEY=... uv run python scripts/run_spy_backtest.py
```

Use `--refresh` to retrieve a new immutable provider snapshot,
`--dataset-id <id>` to replay a cached dataset, or `--fixture` for a short
synthetic offline validation whose output is explicitly not market performance.
The real example is fixed at 2020-01-01 through 2025-12-31, explicitly selects
`DividendPolicy.PRICE_RETURN_ONLY`, prints ignored-dividend disclosures and a
price-return warning, and writes only under Git-ignored `data/` and `reports/`
roots.
