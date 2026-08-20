# Static multi-timeframe study inspection

QF-34 adds a reporting-layer artifact for visually auditing one or more selected
prediction timestamps. The output is a content-addressed directory containing
`report.html` and `manifest.json`. The HTML is self-contained: it has inline CSS,
inline SVG, and an embedded copy of the exact report metadata, with no external
assets, running server, Python process, or network access.

## Fixed SPY example

From the repository root:

```bash
uv run python scripts/render_spy_study_inspection.py
```

The command reuses the committed QF-30 synthetic SPY fixture, QF-33 cache-only
source, exact QF-33 historical-study record, and configured `talib_v1` SMA(2)
outputs. It does not retrieve market data or read credentials. By default it
renders weekly, daily, 4-hour, and optional 5-minute panels around the fixture's
fixed midweek timestamp. Use `--completion-policy completed_bars_only` for the
separate completed-bars QF-33 study, or `--exclude-primary-timeframe` to omit the
optional 5-minute panel.

When several rule condition aliases identify the same timeframe, its panel title
lists the aliases in deterministic sorted order instead of rejecting the valid
rule configuration.

The checked-in intentional example is under
`examples/spy_multi_timeframe/study_inspection_reports/<report-id>/`. Generated
reports under `reports/` remain ignored.

The fixture is synthetic and validates rendering, timestamps, provenance, and
temporal separation only. It is not evidence of predictive accuracy,
profitability, or a trade recommendation.

## Exact inputs and causality

The generator accepts only a validated `PredictionRuleContext`, the matching
QF-31 `TechnicalConfluenceEvaluation` and rule, an exact
`HistoricalPredictionStudyReference`, and the QF-14 `DatasetFamily`. It renders:

- canonical OHLCV bars already selected by QF-20/QF-21;
- normalized `TimeframeIndicatorOutput` rows already produced through QF-28;
- completed, completed-terminal-partial-duration, and developing completion
  states without collapsing them;
- exact bar start/end or observed-through boundaries;
- the primary decision boundary, prediction direction, and every rule condition;
- study, rule, indicator, timeframe-bound indicator, backend, feed, session,
  aggregation, dataset-family, and source-dataset provenance.

The reporting module has no indicator computation path. It does not import
TA-Lib or another backend and does not translate backend-specific tuples or
parameters. SMA, EMA, Bollinger Bands, MACD, stochastic, Wilder, and volume
series are selected from normalized output field names only. The manifest and
HTML therefore contain the exact values used by the study rather than a visual
recalculation.

Every panel must resolve to the same common dataset family. The generator
revalidates symbol and adjustment basis, context references against the supplied
family manifest, every timeframe's exact bar IDs against the captured source
context, each timeframe's complete rule-declared requirement,
one-to-one declared indicator outputs, indicator-to-bar
IDs/timestamps/completion states, each indicator's exact panel dataset reference,
normalized indicator configuration, timeframe binding, developing-bar
capability, backend identity, and the historical study/rule identity. A changed
backend or configuration needs a matching new study reference and produces a
different content-addressed report ID.

## Developing bars and future outcomes

Developing candles use an unfilled amber dashed style. Their embedded records
retain both observed-through and expected-completion timestamps. Completed bars
are never restyled as developing, and developing bars are never implied to have
their eventual completed OHLCV values.

An optional `FutureOutcomeRegion` is stored with
`availability=post_decision_only`, cannot begin before the primary decision
timestamp, and is rendered in a separately shaded region and card labeled
`NOT AVAILABLE AT DECISION TIME`. It is never added to an indicator series or
causal panel input.

## Reproducibility and immutable export

The report ID hashes the schema and engine versions, visual selection policy,
exact selected bar payloads, exact normalized indicator rows, full backend and
configuration provenance, rule evaluation, context manifest, historical study,
dataset-family manifest, and any post-decision annotation. There is no wall-clock
generation timestamp. Repeating an identical export verifies both files
byte-for-byte and reuses the directory; differing existing bytes fail closed.
Validated report inputs are captured as an immutable primitive snapshot during
construction, so later mutation of a caller-owned rule cannot alter the report
ID or either serialized artifact.

No generated report belongs in Git except a deliberately reviewed small fixture
or example. The normal `reports/` location is ignored.
