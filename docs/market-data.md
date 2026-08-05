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
because it supplies daily SPY data, original OHLC and volume, explicit split
coefficients, and cash dividend amounts in a lossless JSON response. It requires
`ALPHA_VANTAGE_API_KEY`, is rate limited, and supplies neither an authoritative
point-in-time universe nor complete corporate-action records.

`unadjusted` preserves provider OHLCV. `split_adjusted` applies every later
split coefficient to all earlier OHLC prices (prices are divided) and volume
(volume is multiplied). The effective split session itself remains on its
post-split basis. This creates one coherent OHLCV basis; adjusted close is never
combined with raw OHLC. The adapter preserves Alpha Vantage's per-session
dividend amount as corporate-action provenance but does not apply it to prices
or cash. It deliberately does not consume adjusted close:
`split_and_dividend_adjusted` is rejected because applying its adjusted-close
factor to volume would not define coherent OHLCV semantics. No dividend cash
flows or corporate-action ledger are exposed in QF-3.

Schema version 3 requires every provider record to include both a finite
positive `split_coefficient` and a finite nonnegative `dividend_amount`. QF-3
records every non-unit split session in immutable
`DatasetMetadata.split_sessions` and every nonzero cash-dividend session in
`DatasetMetadata.dividend_sessions`. Empty tuples are therefore verified
provider provenance for the requested records, not silent assumptions that no
corporate action occurred.

`split_adjusted` is suitable for causal indicator and strategy research on a
coherent price basis, but QF-5 does not execute shares against it. QF-5 accepts
only schema-version-3 `unadjusted` datasets whose `split_sessions` and
`dividend_sessions` have no event inside the observed interval. Split-bearing
execution still requires preserving the coefficient itself plus a quantity and
cost-basis transformation policy. Dividend-bearing execution requires an
explicit entitlement, payment-date, withholding, and cash-credit policy.

## Calendar and validation

The maintained `exchange-calendars` XNYS schedule determines expected sessions.
Weekends and exchange holidays are therefore not gaps. Stable ascending sort is
the only automatic correction. Duplicate sessions, rows outside the request,
empty input, inconsistent symbols or adjustment modes, null/non-numeric/nonfinite
values, nonpositive OHLCV, impossible high/low relationships, and malformed
dates are rejected. Missing, nonfinite, or nonpositive split coefficients and
missing, nonfinite, or negative dividend amounts are also rejected because
their absence would make corporate-action provenance unverifiable. Strict
ingestion (the default) rejects missing exchange sessions and reports their
dates. Non-strict mode records missing sessions in the manifest. Values are
never filled, interpolated, or otherwise invented.

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
`DatasetMetadata` exposes the immutable `raw_sha256` and `data_sha256` digests.
Loads verify the raw and normalized files against those digests, recompute the
dataset identifier from the complete metadata and digests, verify canonical
artifact paths, and run the complete-dataset validator; partial, corrupt, or
identity-inconsistent entries fail. `validate_market_dataset` reserializes bars
and verifies types, OHLCV relationships, symbols, ordering, bounds, counts,
calendar membership, the exact recomputed missing-session tuple,
corporate-action-session structure, digests, canonical paths, and dataset
identity. This prevents both retained-ID mutations and self-consistent metadata
that contradicts derivable calendar facts. The request key covers provider,
symbol, inclusive dates, adjustment, calendar, and schema version.

Dataset identity is an integrity and consistency mechanism, not a signature of
provider authenticity. Only cache loading has the raw provider bytes needed to
verify `raw_sha256`; split and dividend completeness also depends on schema-v3
provider ingestion because those events cannot be reconstructed from OHLCV.

The stable manifest records canonical and provider symbols, provider and adapter
versions, UTC retrieval time, requested and actual session bounds, XNYS calendar,
provider timezone, adjustment mode, raw and normalized locations and SHA-256
digests, immutable dataset ID, schema version, bar count, missing expected
sessions, and verified provider-reported split and cash-dividend sessions.

Schema version 3 changes request and dataset identities. Existing version-1 and
version-2 artifacts remain immutable and are not silently upgraded; because
their manifests do not prove complete split and dividend provenance, they must
be re-ingested before use with QF-5.

## Optional live verification

Ordinary tests never use the network. To explicitly retrieve, validate, cache,
and reload a short SPY range:

```bash
ALPHA_VANTAGE_API_KEY=... QUANTFORGE_RUN_LIVE_MARKET_DATA=1 \
  uv run pytest -m integration tests/integration/test_alpha_vantage_market_data.py
```
