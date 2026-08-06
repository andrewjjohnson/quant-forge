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

## Original baseline rules

The original QF-11 strategy is retained unchanged as the
`combined_original` baseline. New hypotheses are separate prediction strategy
classes and must not silently replace this baseline. The baseline owns Wilder
RSI with period 2 and Wilder +DI, -DI, and ADX with period 5. Indicator values
use only the current and earlier completed bars.
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

## Comparison configurations

The maintained comparison evaluates four stable configurations independently:

- `combined_original` runs the complete original QF-11 implementation above.
- `focused_rules` retains only RSI below the lower threshold (UP), ADX entering
  the positive-DI-controlled zone (UP), RSI stabbing the DI zone from above
  (UP), and RSI stabbing it from below (DOWN). It excludes the negative-DI ADX
  entry, RSI above the upper threshold, and both middle-range candle rules.
- `rsi_oversold_up` predicts UP only when completed-session RSI is strictly
  below its configured lower threshold. Equality is not a prediction. Its
  defaults are RSI(2), threshold 15, and Friday exclusion.
- `always_up` predicts UP on every eligible session. It is a structural-bias
  control, not a proposed trading strategy.

Every configuration uses the same QF-3 dataset, immediate-successor outcome
label, raw-price basis, weekday eligibility, and flat-gap-as-incorrect policy.
An optional weekday-inclusion tuple supports explicit future subset studies;
the default report still shows Monday through Thursday and does not remove a
weekday because it looked weak.

SPY has historically exhibited an upward overnight tendency, so directional
accuracy alone can make an UP-heavy rule look more informative than it is. The
report therefore includes two always-UP comparisons. The all-eligible-session
row describes each configuration alongside the full eligible baseline. The
matched-session row recomputes always-UP statistics on the exact dates when the
other configuration predicted. For an UP-only rule, matched accuracy is simply
the upward-gap frequency on those dates. A DOWN prediction records the
always-UP signed return and the strategy-minus-baseline incremental result for
that same session.

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

The comparison study adds raw and signed return means, medians, population
standard deviation, absolute gap magnitude, positive and adverse tail
frequencies at 0.10%, 0.25%, 0.50%, and 1.00%, and correct/incorrect magnitude
splits. It reports 95% Wilson score intervals for accuracy with their sample
counts. These intervals communicate binomial uncertainty; they are not p-values
or evidence that a rule is tradable.

Prediction-level streak statistics include longest correct and incorrect
streaks, counts of incorrect streaks at least three and five predictions long,
and maximum decline in the ordered cumulative signed-gap series. The last is
explicitly labeled a prediction-study statistic, not portfolio drawdown. The 20
best and 20 worst available outcomes per configuration remain visible so an
average cannot conceal dependence on a few large gaps.

## Causal derived features and range summaries

The comparison export exposes typed completed-session columns for RSI, prior
RSI, RSI change, ADX, prior ADX, ADX change, +DI, -DI, current and prior DI
spread, Wilder ATR, ATR divided by close, OHLC, candle return, volume, trailing
average volume, volume ratio, and weekday. Definitions are:

```text
RSI change       = RSI - previous RSI
ADX change       = ADX - previous ADX
DI spread        = positive DI - negative DI
candle return    = close / open - 1
ATR percentage   = Wilder ATR(14) / close
volume ratio     = volume / trailing 20-session average volume
```

All rolling values are trailing and use the current and earlier completed
sessions only. Outcome labels are attached after every strategy has generated
its predictions and cannot influence these features or a rule decision.

Default bins are deterministic and configurable. RSI uses 0–5, 5–10, 10–15,
15–20, then 10-point ranges through 90–100. ADX uses 10-point ranges through
50–60 and 60+. DI spread uses `<-20`, `-20–-10`, `-10–10`, `10–20`, and `20+`.
ATR/close uses `<0.50%`, `0.50–1.00%`, `1.00–1.50%`, `1.50–2.00%`,
`2.00–3.00%`, and `3.00%+`. Volume ratio uses `<0.75`, `0.75–1.00`,
`1.00–1.25`, `1.25–1.50`, `1.50–2.00`, and `2.00+`.

Intervals are left-inclusive and right-exclusive (`[lower, upper)`) except the
final bounded RSI bin, which includes 100. Unbounded labels state their exact
interval convention. Each row reports both eligible observation count and
prediction count; `adequate_sample` is false until the prediction count reaches
the configurable minimum, 30 by default. Empty and inadequate bins are retained
rather than presented as useful ranges.

## Exploratory thresholds and chronological periods

The simple RSI strategy is rerun at the fixed ordered thresholds 5, 10, 15, 20,
25, and 30 while RSI period remains 2. Results include full-period, annual, and
configured chronological-period rows. This study does not mutate the default
threshold or automatically select a winner.

The default periods are development (2020–2022), exploratory validation
(2023–2024), and observed 2025. The periods are nonoverlapping and configurable.
Because all of them, including 2025, have already been inspected, none is a
pristine holdout and the full report remains exploratory in-sample analysis.
Apparent RSI strength must be evaluated for sample adequacy and consistency
across these periods, then tested on untouched data.

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

Run the multi-configuration comparison against the same fixed Tiingo request or
an existing immutable dataset:

```bash
TIINGO_API_KEY=... uv run python scripts/analyze_spy_gap_predictions.py
uv run python scripts/analyze_spy_gap_predictions.py --dataset-id <dataset-id>
```

The concise JSON output includes each configuration, matched always-UP results,
strongest and weakest years and weekdays, the threshold table, and the export
location. It never prints the API key.

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

Each comparison study similarly exports atomically and immutably to
`<output-root>/<comparison-study-id>/` with `manifest.json`, `metrics.json`, and
CSV files for configuration, prediction, rule, weekday, annual, period,
threshold, feature-bin, baseline, best-outcome, and worst-outcome records.
Configuration and dataset fingerprints participate in the study identity.
Repeated runs validate exact bytes before reusing an existing directory.

Underlying close-to-next-open gaps do not model option spreads, implied
volatility changes, strike selection, contract multipliers, theta, liquidity,
or fills. Neither a favorable average gap nor high directional accuracy proves
option profitability. In particular, RSI below 15 is an exploratory candidate,
not a validated or profitable rule.
