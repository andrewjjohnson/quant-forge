# Daily market data

QF-3 provides provider-independent, completed daily stock and ETF bars. It does
not calculate indicators, signals, or returns.

## Usage

```python
import os
from datetime import date
from pathlib import Path

from quantforge.data import AdjustmentMode, MarketDataCache, MarketDataService
from quantforge.data.providers import AlphaVantageProvider

service = MarketDataService(
    AlphaVantageProvider(os.environ["ALPHA_VANTAGE_API_KEY"]),
    MarketDataCache(Path("data/market-data")),
)
dataset = service.get_daily_bars(
    "SPY",
    date(2020, 1, 1),
    date(2020, 12, 31),
    AdjustmentMode.SPLIT_ADJUSTED,
)
```

Dates are inclusive XNYS trading-session labels, not timestamps. Symbols are
trimmed and uppercased. Canonical rows are ascending and contain `symbol`,
`session_date`, and exact decimal `open`, `high`, `low`, `close`, and `volume`.
The metadata remains available as `dataset.metadata`.

## Provider and adjustment policy

The first adapter uses Alpha Vantage `TIME_SERIES_DAILY_ADJUSTED`, selected
because it supplies daily SPY data, original OHLC and volume, and explicit split
coefficients in a lossless JSON response. It requires
`ALPHA_VANTAGE_API_KEY`, is rate limited, and supplies neither an authoritative
point-in-time universe nor complete corporate-action records.

`unadjusted` preserves provider OHLCV. `split_adjusted` applies every later
split coefficient to all earlier OHLC prices (prices are divided) and volume
(volume is multiplied). The effective split session itself remains on its
post-split basis. This creates one coherent OHLCV basis; adjusted close is never
combined with raw OHLC. Although Alpha Vantage also returns adjusted close and
dividends, this adapter deliberately does not consume them:
`split_and_dividend_adjusted` is rejected because applying its adjusted-close
factor to volume would not define coherent OHLCV semantics. No dividend cash
flows or corporate-action ledger are exposed in QF-3.

Schema version 2 requires every provider record to include a finite positive
`split_coefficient`. QF-3 records the effective session for every non-unit
coefficient in immutable `DatasetMetadata.split_sessions`. An empty tuple is
therefore verified provider provenance for the requested records, not a silent
assumption that splits did not occur.

`split_adjusted` is suitable for causal indicator and strategy research on a
coherent price basis, but QF-5 does not execute shares against it. QF-5 accepts
only schema-version-2 `unadjusted` datasets whose `split_sessions` has no event
inside the observed interval. Split-bearing execution still requires preserving
the coefficient itself plus a quantity and cost-basis transformation policy.

## Calendar and validation

The maintained `exchange-calendars` XNYS schedule determines expected sessions.
Weekends and exchange holidays are therefore not gaps. Stable ascending sort is
the only automatic correction. Duplicate sessions, rows outside the request,
empty input, inconsistent symbols or adjustment modes, null/non-numeric/nonfinite
values, nonpositive OHLCV, impossible high/low relationships, and malformed
dates are rejected. Missing, nonfinite, or nonpositive split coefficients are
also rejected because their absence would make split-free provenance
unverifiable. Strict ingestion (the default) rejects missing exchange sessions
and reports their dates. Non-strict mode records missing sessions in the
manifest. Values are never filled, interpolated, or otherwise invented.

## Immutable storage and metadata

Runtime artifacts live below the caller-provided ignored `data/` directory:

```text
raw/<raw-sha256>.json
datasets/<dataset-sha256>/bars.csv
datasets/<dataset-sha256>/manifest.json
requests/<request-sha256>.json
```

Canonical JSON and decimal-text CSV preserve raw fields, dates, and numeric
values. Writes use a temporary fsynced file and an atomic hard link. Existing
artifacts are accepted only when byte-identical and are never overwritten.
Loads verify raw and normalized SHA-256 checksums, the dataset identifier, and
bar count; partial or corrupt entries fail. The request key covers provider,
symbol, inclusive dates, adjustment, calendar, and schema version.

The stable manifest records canonical and provider symbols, provider and adapter
versions, UTC retrieval time, requested and actual session bounds, XNYS calendar,
provider timezone, adjustment mode, raw and normalized locations, immutable
dataset ID, schema version, bar count, missing expected sessions, and verified
provider-reported split sessions.

Schema version 2 changes request and dataset identities. Existing version-1
artifacts remain immutable and are not silently upgraded; because their
manifests do not prove split completeness, they must be re-ingested before use
with QF-5.

## Optional live verification

Ordinary tests never use the network. To explicitly retrieve, validate, cache,
and reload a short SPY range:

```bash
ALPHA_VANTAGE_API_KEY=... QUANTFORGE_RUN_LIVE_MARKET_DATA=1 \
  uv run pytest -m integration tests/integration/test_alpha_vantage_market_data.py
```
