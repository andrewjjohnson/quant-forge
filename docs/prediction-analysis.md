# Overnight gap prediction analysis

QF-11 adds `quantforge.prediction`, a provider-neutral research boundary for
testing whether information available after one completed daily session predicts
the next exchange session's opening gap. It does not place orders, simulate
fills, or change QF-5's next-open execution safeguards.

```text
completed QF-3 bars
        |
        v
causal QF-4 indicators
        |
        v
direction prediction after session t closes
        |
        +-------------------- feature boundary --------------------+
                                                                  |
                                                                  v
                                      next-session open outcome label
                                                                  |
                                                                  v
                                      accuracy and gap-size metrics
```

The signal close is a prediction anchor, not a claimed executable fill. A
strategy that requires the completed daily close cannot also claim it purchased
at that exact close.

## Baseline rules

The baseline owns Wilder RSI with period 2 and Wilder +DI, -DI, and ADX with
period 5. Indicator values use only the current and earlier completed bars.
Wilder smoothing starts from an initial full-period mean or sum and then uses
the recursive Wilder update. The first ADX is the mean of the first five DX
values.

Friday signal sessions are excluded by default. Python weekday numbers are used
in configuration (`Monday=0` through `Sunday=6`), so the default excluded value
is `4`.

Rules are evaluated in this order:

1. If current ADX is greater than 60, emit no prediction.
2. If ADX is inclusively between current +DI and -DI and was not inclusively
   between the prior session's DI values:
   - +DI above -DI predicts an upward gap;
   - -DI above +DI predicts a downward gap.
3. An RSI stab from above occurs when prior RSI was above both prior DI values
   and current RSI is at or below the larger current DI value. It predicts up.
4. An RSI stab from below occurs when prior RSI was below both prior DI values
   and current RSI is at or above the smaller current DI value. It predicts
   down.
5. RSI below 15 predicts up; RSI above 85 predicts down.
6. RSI equal to 15 or 85 is in the middle range. Within the inclusive middle
   range, a bullish candle predicts up and a bearish candle predicts down. A
   doji emits no prediction.

The ADX zone-entry rule has priority over the RSI-stab rule, which has priority
over the base RSI/candle rule. An ADX value greater than 60 is an absolute veto.

## Outcome and metrics

For signal session `t` and its immediate exchange-calendar successor `t+1`:

```text
overnight gap = open[t+1] / close[t] - 1
gap size      = abs(overnight gap)
signed return = overnight gap       for an up prediction
                -overnight gap      for a down prediction
correct       = signed return > 0
```

All percentage-named fields are decimal ratios: `0.01` means 1%. Any positive
or negative gap is evaluated. An exactly flat gap is counted as incorrect. The
summary records prediction, correct, and incorrect counts; accuracy; average
absolute gap size for correct and incorrect predictions; and average signed
prediction return for each group. A final-session signal is preserved in the
generated-signal count but cannot receive a label and is counted separately.

## Public API

```python
from pathlib import Path

from quantforge.data import MarketDataCache
from quantforge.prediction import (
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    export_prediction_analysis,
    run_prediction_analysis,
)

dataset = MarketDataCache(Path("data/market-data")).load("<dataset-id>")
strategy = OvernightGapPredictionStrategy(OvernightGapPredictionParameters())
result = run_prediction_analysis(dataset, strategy)
artifact_path = export_prediction_analysis(result, Path("reports/predictions"))
```

The maintained SPY command wraps the same API:

```bash
TIINGO_API_KEY=... uv run python scripts/run_spy_gap_prediction.py
```

By default it requests raw, unadjusted Tiingo EOD bars for SPY from 2020-01-01
through 2025-12-31, stores the immutable QF-3 dataset under
`data/market-data`, and exports the QF-11 analysis under
`reports/predictions`. The JSON summary prints direction counts, accuracy, and
average absolute gap size for correct and incorrect predictions. Repeating the
same run verifies and reuses the exact immutable export.

Reuse a previously cached QF-3 dataset without making a provider request:

```bash
uv run python scripts/run_spy_gap_prediction.py --dataset-id <dataset-id>
```

Use `--refresh` only when intentionally retrieving a new immutable Tiingo
snapshot. Use `--cache-root` and `--output-root` to override the default
locations. The analysis manifest records schema-v4 corporate-action provenance,
including dividend and split counts and the corporate-action snapshot ID. The
baseline period contains no special options handling; all labels use underlying
SPY close-to-next-open prices.

Tiingo ingestion uses raw, unadjusted OHLCV. Ex-dividend price effects therefore
remain part of the observed underlying-price gap; this analysis does not convert
them into total returns. A raw dataset containing a stock split fails closed
because the mechanical price change must not be counted as a predicted gap.
Split-adjusted QF-3 datasets are not subject to that raw-price rejection. These
policies and the action counts are recorded in the manifest.

## Export schema

Each analysis exports atomically to `<output-root>/<analysis-id>/`:

- `manifest.json` records engine and schema versions, complete strategy and
  indicator configuration, QF-3 dataset identity and bar fingerprint, counts,
  metrics, and limitations.
- `predictions.csv` contains one labeled row per evaluable signal.

Prediction rows record:

- stable prediction, dataset, strategy, and strategy-configuration identities;
- QF-3 provider, raw-price basis, and corporate-action snapshot provenance;
- symbol, signal session, and outcome session;
- direction, originating rule, full parameter set, and contemporaneous feature
  values;
- signal close and next open;
- overnight gap, absolute gap size, prediction-signed return, and correctness.

Exports never contain orders, fills, quantities, option prices, or portfolio
profit and loss. The results measure direction prediction only and remain
in-sample descriptive evidence until evaluated on untouched data.
