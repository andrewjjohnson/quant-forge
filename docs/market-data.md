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

`split_adjusted` is suitable for causal indicator and strategy research on a
coherent price basis, but QF-5 does not execute shares against it. Because the
current `MarketDataset` omits point-in-time split factors, QF-5 accepts only
`unadjusted` requests whose range contains no split. Split-aware backtesting
requires a future corporate-action schema and quantity transformation policy.

## Calendar and validation

The maintained `exchange-calendars` XNYS schedule determines expected sessions.
Weekends and exchange holidays are therefore not gaps. Stable ascending sort is
the only automatic correction. Duplicate sessions, rows outside the request,
empty input, inconsistent symbols or adjustment modes, null/non-numeric/nonfinite
values, nonpositive OHLCV, impossible high/low relationships, and malformed
dates are rejected. Strict ingestion (the default) also rejects missing exchange
sessions and reports their dates. Non-strict mode records missing sessions in the
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
dataset ID, schema version, bar count, and missing expected sessions.

## Optional live verification

Ordinary tests never use the network. To explicitly retrieve, validate, cache,
and reload a short SPY range:

```bash
ALPHA_VANTAGE_API_KEY=... QUANTFORGE_RUN_LIVE_MARKET_DATA=1 \
  uv run pytest -m integration tests/integration/test_alpha_vantage_market_data.py
```
