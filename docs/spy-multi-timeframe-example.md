# Reproducible SPY multi-timeframe context example

QF-30 provides a small, indicator-free example for inspecting QuantForge's
timeframe aggregation, common-source lineage, immutable cache replay, and
as-of alignment. It does not evaluate a prediction or trading strategy.

## Run the example

From the repository root:

```bash
uv run python scripts/export_spy_multi_timeframe_context.py
```

The script needs no credentials and constructs no provider client. On the
first run it materializes the committed fixture into the ignored local cache
at `data/qf30-spy-context-cache/`. Later runs reload and validate that exact
immutable QF-16 cache artifact. Both paths derive the higher timeframes again
from the cache-validated source.

The checked-in golden export is under
`examples/spy_multi_timeframe/exports/<example-id>/`. A rerun computes the
same content-addressed directory and verifies every existing byte. It never
overwrites a differing artifact. Use `--cache-root` and `--output-root` to run
an isolated copy.

## Data scope and limitations

The input is a deterministic, redistributable synthetic fixture with canonical
symbol `SPY`. It covers XNYS regular-hours sessions from 2024-06-24 through
2024-07-12, including the 2024-07-03 early close and the 2024-07-04 holiday.
It is deliberately not represented as observed provider market data:

- provider: `quantforge_example_fixture`;
- feed scope: provider-defined `synthetic_regular_hours_fixture`;
- prices: raw, unadjusted synthetic decimals;
- corporate actions: unavailable for the synthetic intraday fixture;
- source interval: canonical session-open-anchored 5-minute bars;
- calendar/timezone: XNYS and `America/New_York` regular hours.

The values are suitable for auditing arithmetic and temporal boundaries only.
They must not be used for market-performance claims. The script imports the
data and timeframe layers only; it does not load `quantforge.indicators`,
`native_v1`, `talib_v1`, or TA-Lib. It also creates no predictions, signals,
orders, fills, portfolio records, or P&L.

## Deterministic source recipe

The first 5-minute source bar has sequence number 0. For sequence number
`i`, the fixture uses:

```text
open   = 500.00 + 0.01 * i
high   = open + 0.05
low    = open - 0.03
close  = open + 0.02
volume = 1000 + i
```

QuantForge derives every 4-hour, daily, and weekly value from that one cached
5-minute snapshot. Each aggregate uses the first open, maximum high, minimum
low, final close, and summed volume. The composed dataset-family manifest binds
the exact QF-18 and QF-19 artifact-family manifests, and all four context series
carry the same family ID and canonical source dataset ID.

## Fixed decision timestamps

The example exports completed-only and developing-as-of contexts separately
for four decisions:

| Scenario | Exchange-local decision | Reason |
| --- | --- | --- |
| `normal_session` | 2024-07-01 11:00 ET | Normal session with a completed prior week |
| `early_close_near_close` | 2024-07-03 12:55 ET | Five minutes before the scheduled 13:00 close |
| `midweek` | 2024-07-10 12:00 ET | Current daily and weekly periods are forming |
| `normal_session_near_close` | 2024-07-12 15:55 ET | Five minutes before a normal 16:00 close |

Completed-only contexts retain only terminal bars whose explicit end is at or
before the decision time. Developing contexts retain the same completed
history and append a structurally distinct forming 4-hour, daily, or weekly bar
built only from completed 5-minute constituents available at that decision.

## Hand audit: 2024-07-03 early close

The early-close session contains 42 canonical 5-minute bars from 09:30 through
13:00 ET. Their global fixture sequence numbers are 546 through 587. Because a
four-hour nominal window cannot cross the session close, QF-18 emits one
completed 210-minute terminal partial bar:

```text
boundary: 2024-07-03 09:30-13:00 ET
open:     500.00 + 0.01 * 546                         = 505.46
high:     (500.00 + 0.01 * 587) + 0.05                = 505.92
low:      (500.00 + 0.01 * 546) - 0.03                = 505.43
close:    (500.00 + 0.01 * 587) + 0.02                = 505.89
volume:   sum(1000 + i for i in 546..587)             = 65,793
```

At the fixed 12:55 ET decision, the 12:55-13:00 source bar is not yet
available. The developing daily and 4-hour bars therefore use only sequence
numbers 546 through 586:

```text
observed boundary:       2024-07-03 09:30-12:55 ET
expected completion:     2024-07-03 13:00 ET
source count:            41
open/high/low/close:     505.46 / 505.91 / 505.43 / 505.88
volume:                  sum(1546..1586) = 64,206
completion:              developing
```

The completed-only context at 12:55 does not expose either eventual 2024-07-03
aggregate. Its latest daily bar remains 2024-07-02
(`504.68 / 505.50 / 504.65 / 505.47`, volume `117,507`), and its latest weekly
bar remains the completed 2024-06-24 exchange week
(`500.00 / 503.94 / 499.97 / 503.91`, volume `465,855`). This is the expected
causal difference between the two policies.

## Export contents

The immutable directory contains:

- `bars_5m.csv`, `bars_4h.csv`, `bars_daily.csv`, and `bars_weekly.csv` with
  explicit UTC boundaries, completion states, OHLCV, identities, and source
  counts;
- `contexts_completed.json` and `contexts_developing.json`, kept separate to
  prevent completion-policy ambiguity;
- `context_table.md`, a compact human-readable view of every latest value;
- the source, QF-18, QF-19, and composed dataset-family manifests;
- `manifest.json`, which binds the fixture, source snapshot, derived datasets,
  family, contexts, policy limitations, and SHA-256 digest of every other
  exported file.

Normal XNYS sessions visibly contain a full 09:30-13:30 ET 4-hour bar and a
completed 13:30-16:00 ET terminal partial bar. The early close visibly contains
only the completed 09:30-13:00 ET terminal partial bar.
