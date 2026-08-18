# Multi-timeframe technical-confluence prediction rule

QF-31 adds one reusable prediction rule for testing explicit weekly, daily, and
4-hour technical-confluence hypotheses. It composes the existing QF-28
restricted context and QF-29 feature-capture contracts; it does not create a
second alignment engine, call an indicator backend directly, optimize
parameters, or model trades.

The public entry points are `TechnicalConfluencePredictionRule` for custom
typed rules and `create_reference_technical_confluence_rule()` for the fixed
SPY research example below.

## Rule semantics

Each `TechnicalCondition` has a globally unique name, display-safe timeframe
name, complete canonical `Timeframe`, typed left operand, exact operator, typed
right operand or finite `Decimal` threshold, and explicit enabled state.
Indicator operands name only a QF-28 alias and normalized QuantForge output;
they cannot reference TA-Lib parameter names, tuple positions, arrays, or
library objects. Canonical `open`, `high`, `low`, `close`, and `volume` bar
fields may be compared with normalized outputs on the same timeframe.

Supported operators have these exact boundaries:

| Operator | Pass rule |
| --- | --- |
| `greater_than` | current left `>` current right |
| `greater_than_or_equal` | current left `>=` current right |
| `less_than` | current left `<` current right |
| `less_than_or_equal` | current left `<=` current right |
| `equal` | current left `==` current right |
| `crosses_above` | previous left `<=` previous right and current left `>` current right |
| `crosses_below` | previous left `>=` previous right and current left `<` current right |

UP and DOWN each use all-of semantics over their enabled conditions. Exactly
one passing side emits that direction. If neither side passes, both sides pass,
or a required value is unavailable, the outcome is `NO_PREDICTION`. A
directional result becomes an accepted QF-7 candidate; `NO_PREDICTION` becomes
an auditable rejected candidate rather than disappearing. Disabled conditions
are recorded as disabled and are excluded from their side's all-of decision.
Each side must retain at least one enabled condition.

Every candidate records:

- each condition's enabled state and `passed`, `failed`, `unavailable`, or
  `disabled` status;
- current left and right values;
- prior values for crossovers;
- the latest source timestamp used by each condition;
- the latest rule-visible source-bar timestamp for weekly, daily, and 4-hour
  context;
- UP-side and DOWN-side all-of results; and
- the final `up`, `down`, or `no_prediction` outcome.

Warm-up values remain unavailable and are never backfilled. The rule evaluates
only the latest values in `PredictionRuleContext`; QF-28 has already restricted
that view to declared timeframes, indicators, completion policies, compatible
source provenance, and the primary decision boundary.

## Fixed SPY reference rule

The reference is a deliberately fixed, unoptimized research hypothesis. It is
not a claim that every favorable-looking chart should predict correctly, and
it is not a trading strategy.

All standard indicators explicitly select `talib_v1`. Relative volume uses the
normalized QF-27 implementation with the same declared feed scope and therefore
has no invented standard-backend identity. Periods are counts of bars in the
condition's assigned timeframe.

### UP conditions

Every enabled row must pass:

| Timeframe | Condition | Exact comparison |
| --- | --- | --- |
| Weekly | Price trend | close `>` EMA(10) |
| Weekly | MACD state | MACD(12,26,9) histogram `>` 0 |
| Daily | Moving-average trend | SMA(20) `>` EMA(50) |
| Daily | Bollinger location | close `>=` Bollinger(20, 2 population deviations) middle band |
| Daily | Bollinger bandwidth | normalized bandwidth `>= 0.02` |
| Daily | Relative volume | relative volume(20, including current bar) `>= 1` |
| Daily | Volume average | current volume `>=` volume moving average(20) |
| Daily | RSI ceiling | RSI(14) `<= 70` |
| Daily | ADX floor | ADX(14) `>= 20` |
| Daily | ATR ceiling | ATR(14) `<= 15` price units |
| 4-hour | EMA trend | EMA(9) `>` EMA(21) |
| 4-hour | MACD crossover | MACD(12,26,9) histogram crosses above 0 |
| 4-hour | Stochastic crossover | stochastic(5,3,3) %K crosses above %D |
| 4-hour | Stochastic ceiling | stochastic %K `<= 30` |

### DOWN conditions

Every enabled row must pass:

| Timeframe | Condition | Exact comparison |
| --- | --- | --- |
| Weekly | Price trend | close `<` EMA(10) |
| Weekly | MACD state | MACD(12,26,9) histogram `<` 0 |
| Daily | Moving-average trend | SMA(20) `<` EMA(50) |
| Daily | Bollinger location | close `<=` Bollinger(20, 2 population deviations) middle band |
| Daily | Bollinger bandwidth | normalized bandwidth `>= 0.02` |
| Daily | Relative volume | relative volume(20, including current bar) `>= 1` |
| Daily | Volume average | current volume `>=` volume moving average(20) |
| Daily | RSI floor | RSI(14) `>= 30` |
| Daily | ADX floor | ADX(14) `>= 20` |
| Daily | ATR ceiling | ATR(14) `<= 15` price units |
| 4-hour | EMA trend | EMA(9) `<` EMA(21) |
| 4-hour | MACD crossover | MACD(12,26,9) histogram crosses below 0 |
| 4-hour | Stochastic crossover | stochastic(5,3,3) %K crosses below %D |
| 4-hour | Stochastic floor | stochastic %K `>= 70` |

The fixed ATR threshold is SPY-specific and belongs to the reference
configuration identity. Researchers should define a separate named
configuration rather than silently changing this example.

## Completion and causality policy

Completed bars only are the default. The supplied primary decision timeframe
always remains completed-only. A caller may explicitly request
`DEVELOPING_BAR_AS_OF` for the weekly, daily, and 4-hour requirements; QF-28
accepts that configuration only because every reference indicator declares
causal developing-bar support. QF-21 reconstructs forming bars solely from
completed lower-timeframe constituents available at the decision timestamp.

The rule itself cannot obtain an undeclared timeframe or indicator. QF-28 also
rejects a contextual bar ending after the latest primary decision bar. Adding
future source data therefore cannot revise a historical result when the same
as-of context is rebuilt.

## Identity and provenance

The rule configuration and resulting study identity bind:

- every condition name, timeframe, operand, normalized output, operator,
  threshold, unit, and enabled state;
- UP/DOWN combination and conflict semantics;
- the primary, weekly, daily, and 4-hour timeframe configurations;
- completion, freshness, failure, session, and feed-scope policies;
- every normalized indicator configuration and configuration ID;
- standard-backend ID, contract version, mapped function, wrapper version, and
  native runtime version when applicable; and
- the resolved source context, visible bars, dataset family, and source
  datasets through QF-28 study provenance.

Historical single-timeframe/native rules are untouched. Creating the reference
rule explicitly opts into `talib_v1`; omitted-backend legacy constructors keep
their existing native identities and values.

## QF-7 and QF-29 capture

`strategy_feature_definitions` supplies QF-7 with typed flat columns for every
condition value, status, and source timestamp. The rule's
`multi_timeframe_feature_requests` property deterministically derives one
QF-29 request for every normalized indicator field consumed by a condition.
Pass that tuple directly to `build_signal_feature_dataset()` so each flat
indicator value receives its sibling `__metadata` column with source-bar
timing, completion state, staleness, backend identity, dataset-family identity,
source dataset, and feed scope.

```python
rule = create_reference_technical_confluence_rule(
    primary_timeframe=five_minute,
    four_hour_timeframe=four_hour,
    daily_timeframe=daily,
    weekly_timeframe=weekly,
    feed_scope=FeedScope.consolidated(),
)

result = build_signal_feature_dataset(
    dataset=prediction_dataset,
    prediction_study=study,
    contextual_features=(),
    multi_timeframe_features=rule.multi_timeframe_feature_requests,
    context_provider=context_provider,
    outcomes=outcomes,
    output_root=output_root,
)
```

The deterministic test fixture includes accepted UP, accepted DOWN, and
rejected `NO_PREDICTION` cases. These synthetic values validate rule semantics
and auditability only; they are not market evidence or profitability results.

## Deliberate limits

QF-31 does not select a best rule, compare parameters, alert, place orders,
simulate fills, calculate portfolio returns, model options, or claim
profitability. Deterministic parameter comparison remains QF-32 scope, and
scanner/alert behavior remains QF-33 scope.
