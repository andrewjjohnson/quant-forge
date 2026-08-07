# Research Integrity

QuantForge results may influence real financial decisions. The platform must make invalid or overstated conclusions difficult to produce accidentally.

## Core principle

A result is useful only when another person can understand:

- what information was available at each decision time;
- what assumptions produced each order and fill;
- which data and code produced the result;
- which observations influenced strategy selection;
- whether performance survived untouched out-of-sample evaluation.

## Look-ahead bias

Never allow a strategy or filter to use information that was not available at the decision timestamp.

Common failure modes:

- shifting indicators in the wrong direction;
- using the current bar close and filling at that same close;
- centered rolling windows;
- backfilled features;
- revised fundamentals applied to earlier dates;
- final earnings values used before their publication timestamp;
- full-period normalization;
- selecting parameters using test-period results.

Required safeguards:

- explicit signal and execution timestamps;
- tests with sentinel future values;
- chronological transformations;
- next-permitted-execution rules;
- separate modules for contemporaneous features and forward labels.

QF-4 enforces these safeguards through aligned trailing indicator windows,
explicit unavailable warm-up values, distinct signal and earliest-executable
sessions, exchange-calendar resolution, and append-future causality tests. A
calendar-resolved execution session is eligibility metadata only; it does not
claim an order or fill occurred. See `docs/strategy-contracts.md`.

QF-5 turns that eligibility metadata into auditable orders but fills only at the
eligible session's open. Its chronological state machine applies explicit
adverse slippage, commission, and additional transaction fees before
cash/position updates, preserves final unexecuted signals, and verifies through
golden and append-future tests that later bars cannot revise earlier execution.
Run and trade provenance includes an explicit strategy implementation version.
Commission, fee, and slippage model versions also participate in run identity,
and configuration provenance is deeply snapshotted before execution. A
separately persisted fingerprint of the actual validated OHLCV bars also
participates in run identity, preventing reused or stale QF-3 metadata from
aliasing different market inputs. QF-4's adjustment-mode and trading-calendar
reference participates as well because it controls signal eligibility and
execution timing. Commission and fee models are accepted only when they
explicitly guarantee nondecreasing buy-side costs by quantity, which makes
whole-share affordability search sound.

Built-in and custom commission, fee, slippage, and cost-configuration callbacks
execute under the same serialized 34-digit Decimal policy for both the strategy
engine and benchmark. A caller's ambient Decimal precision therefore cannot
change cost results while preserving the same run ID.

Before execution, QF-5 additionally reserializes the current bars and verifies
that their QF-3 normalized digest, the complete provenance metadata, raw digest,
schema, and canonical artifact paths reproduce the declared QF-3 dataset ID.
Removing a corporate-action session or replacing bars while retaining an old ID
therefore fails closed before the strategy or benchmark runs.

The QF-3 validator independently recomputes expected sessions from the declared
calendar across the requested range and requires exact agreement with
`missing_sessions`. QF-5 uses those recomputed facts—not the tuple alone—to
reject gaps inside the observed range, so a multi-session equity change cannot
be annualized as one daily return.
QF-3 schema version 4 requires a provider split coefficient and cash-dividend
amount for every bar and binds every non-unit split and nonzero dividend to an
immutable typed action snapshot. QF-5 rejects legacy/incomplete schemas and
adjusted datasets, then causally applies verified raw-price actions. Its run
identity changes with the dataset, action snapshot, or selected dividend
policy. See `docs/backtesting.md` and ADRs 0001 and 0003.

Dataset hashes prove consistency, not provider authenticity. Raw-byte digest
verification occurs when QF-3 loads the immutable cache, and corporate-action
completeness remains dependent on schema-v4 provider ingestion because split and
dividend events are not derivable from OHLCV alone.

## Survivorship bias

Testing only securities that exist today can overstate historical performance.

When testing changing universes:

- use point-in-time membership data where possible;
- include delisted securities when supported;
- record universe-construction methodology;
- clearly label tests that use a current-universe approximation;
- do not claim index-wide historical validity from today’s constituent list.

## Corporate actions and adjusted data

Record whether each field is:

- raw;
- split adjusted;
- dividend adjusted;
- total-return adjusted.

Do not mix conventions silently.

Execution prices, cash dividends, position quantities, and benchmark returns
must use a coherent disclosed policy. QF-5 always uses raw OHLCV and point-in-
time split factors for fills, marks, and both accounts. To prevent a split from
appearing to price indicators as a crash or jump, strategy features use a
causal forward-normalized view: effective and prior split factors scale the
current OHLCV row, while later split factors never alter earlier rows.
`CASH_DIVIDENDS` credits explicit ex-date
cash, `PRICE_RETURN_ONLY` preserves events but excludes cash from all economics,
and `REJECT_IF_DIVIDENDS` fails closed. It never uses Tiingo adjusted fields for
fills or marks and rejects adjusted prices with any raw-price policy.

Price return and total economic return answer different research questions.
Price-only mode can isolate signal behavior during early research, but it
understates long-period strategy and benchmark wealth and must carry the
exported warning and return-basis label. Its estimated ignored cash is disclosure,
not a correction to equity. Cash mode uses prior-close entitlement; splits apply
before opening orders and dividends credit afterward. This prevents an ex-date
opening purchase from earning or being funded by that dividend while preserving
entitlement for an ex-date opening sale.

Tiingo's `divCash` marks the ex-dividend date. Crediting it that session is a
documented daily-model approximation, not the real payment date. Splits multiply
shares by Tiingo's shares-after/shares-before factor, reconstructed as a bounded
rational ratio only when it exactly round-trips through the provider's canonical
float representation. Splits conserve aggregate basis and cash and create no
realized P&L. A fractional result after reconstruction is rejected because
cash-in-lieu is not modeled. Dividend selection never disables split handling.
Tests cover all policy/data combinations, entitlement boundaries, exactly-once
cash, ignored-cash non-effects, split continuity under every policy, benchmark
parity, identity changes, and fail-closed adjustment/provenance validation.

Because the buy-and-hold benchmark enters at the first open, its risk-return
series includes the initial-capital-to-first-close return, including entry costs
and that session's price move. Treating that invested observation as a zero
would understate benchmark volatility and distort Sharpe and Sortino.

## Timestamp and calendar integrity

Record:

- source timezone;
- normalized timezone;
- exchange calendar;
- session boundaries;
- bar-label convention;
- daylight-saving handling.

Do not treat every calendar date as a trading session. Distinguish an exchange holiday from missing provider data.

## Costs and market impact

Include realistic assumptions for reportable results:

- commissions;
- regulatory and exchange fees where material;
- bid/ask spread;
- slippage;
- liquidity constraints;
- order size;
- partial or rejected fills when relevant.

A perfect-fill run may be retained as a debugging baseline, but it must be labeled clearly.

For options, later models must account for wide and unstable spreads, contract multipliers, exercise/assignment, expiration, and chain availability.

## Overfitting

Overfitting includes more than excessive model complexity. It also occurs through repeated human inspection.

Safeguards:

- separate development, validation, walk-forward, and final holdout periods;
- consume a final holdout sparingly;
- record every optimization trial, including failures;
- report parameter surfaces and stability;
- prefer simple rules when performance is similar;
- test sensitivity to nearby parameters;
- test across market regimes and suitable symbols;
- rerun under worse costs;
- penalize low trade counts and concentrated outcomes.

An isolated optimum surrounded by poor neighbors is a warning, not a discovery.

QF-6 makes that warning operational for deterministic Cartesian studies. It
records the full grid size, exclusions, failures, ineligible successes, hard
constraints, objective/tie-breaker policy, and immediate candidate-index
neighborhood statistics. Objective rank and stability rank remain separate;
failed, missing, and ineligible neighbors are never converted to zero. Its
`recommended_robust` result is restricted to stable, non-isolated candidates in
a configured top objective band, but remains an in-sample descriptive result.
It is not a substitute for untouched validation, walk-forward evaluation, or
multiple-comparison controls. See `docs/optimization.md`.

## Multiple comparisons

Testing many indicators, ranges, time horizons, symbols, and parameter sets creates false discoveries.

Feature-analysis reports should include:

- total hypotheses considered;
- sample sizes;
- effect sizes;
- uncertainty intervals where appropriate;
- multiple-comparison controls when formal significance is claimed;
- untouched validation performance.

Do not add a discovered feature directly to production logic. Register it as a candidate hypothesis and test it on data that did not generate the hypothesis.

## Signal-feature analysis

For every qualifying setup, capture features consistently, including signals that were:

- accepted;
- blocked by a filter;
- rejected for risk or portfolio constraints;
- overlapping while already in a position.

Avoid comparing only losing trades. Compare feature distributions and conditional outcomes for the full eligible signal population.

Keep the two concepts separate:

- **features:** information available at signal time;
- **labels:** outcomes calculated using later observations.

QF-11 enforces this split through generic typed study contracts. The orchestrator
first fixes every causal prediction, then invokes the configured outcome
labeler, then invokes the evaluator with only the fixed prediction and its typed
outcome. Study identity includes the rule, outcome, evaluator, feature, horizon,
market-field, schema, and dataset configurations. The generic core does not
assume an outcome is a gap or an evaluation is directional correctness.
Strategy warm-up declarations must be positive integers, and every emitted
signal must follow at least that many completed dataset observations.
Returned outcome sessions must match the labeler's exact declared session
horizon, and a labeler may report an unavailable outcome only when that declared
future session lies beyond the dataset boundary. The runner snapshots prediction
records and their complete generated collection once before validating parameter
payloads or exposing the dataset to labeler validation. The same fixed collection
drives validation, evaluation, and reported record counts. The runner reads
configured parameters from the canonical snapshots and checks them after
validation and around every label call. It also snapshots
prediction and outcome primitives around evaluation, revalidates component-owned
values after all evaluations to catch delayed mutation, and includes the complete
contemporaneous feature payload in row identity. Outcome and evaluation
identities hash their already-captured canonical value snapshots rather than
re-invoking component serializers. Returned rows use detached typed payloads and
immutable primitive snapshots so later component reuse cannot change an earlier
result's serialization or invalidate its identities. Strategy, labeler, and
evaluator execution uses a detached copy of the validated market dataset and
checks it against a pristine snapshot after component calls, so custom component
mutation cannot alter caller-owned data or desynchronize rows from dataset
provenance. Prediction manifests preserve the QF-3 OHLC and volume bases,
whether adjusted provider fields were used, and the corporate-action policy so
volume-derived features and gap labels remain interpretable without an external
cache lookup. The backward-compatible
overnight-gap adapter likewise includes the complete fixed causal signal
snapshot in each legacy prediction ID, preventing changed direction, reason, or
features from aliasing an existing scientific record.

Comparison weekday summaries cover every day observed among eligible sessions
after applying weekday filters. This leaves exchange-traded XNYS studies on
their actual weekdays, preserves header-only summaries when no session matches,
and prevents weekend observations in `24/7` datasets from disappearing from
segmented reports.

For the original overnight-gap study, the prediction strategy sees
completed-session OHLC and causal Wilder indicator outputs but no next-open
price. Its concrete labeler pairs an already-generated signal with the immediate
next exchange session. The signal close is explicitly a label anchor, not a
fabricated same-close fill, and exported accuracy is not labeled as trade
performance. See `docs/prediction-analysis.md`.

QF-11 comparison experiments preserve that original strategy as an immutable
baseline and implement narrower hypotheses as separate strategy classes. Sparse
rules are shown against always-UP both over the full eligible population and on
their exact matched prediction sessions, making SPY's structural upward-gap
bias visible. Threshold, range, weekday, year, and already-observed period rows
retain small samples, weak periods, neutral outcomes, and adverse outliers. The
2025 segment is labeled observed rather than pristine holdout, and no comparison
output claims an executable close fill or options profitability.

The maintained Tiingo SPY commands additionally require exact XNYS boundary
sessions and an empty missing-session set for the declared 2020–2025 request.
A non-strict cache with truncated edges or other missing sessions is a different
experiment and cannot be presented as the maintained baseline.

Use versioned schemas for both.

## Walk-forward testing

A valid walk-forward process:

1. defines a training window;
2. selects parameters using only that window;
3. freezes those parameters;
4. evaluates the immediately following test window;
5. advances the windows;
6. combines only out-of-sample test segments.

Report:

- results by window;
- profitable-window percentage;
- parameter turnover;
- trade counts;
- drawdowns;
- cost sensitivity;
- aggregate out-of-sample performance.

Never splice in-sample portions into the final walk-forward equity curve.

## Reproducibility

Every material experiment should record:

- run ID;
- code commit;
- repository dirty state;
- dependency lock fingerprint;
- dataset fingerprint;
- provider and retrieval metadata;
- strategy and parameter configuration;
- universe definition;
- cost and fill assumptions;
- random seed;
- train/test/holdout boundaries;
- environment information.

If a result cannot be reproduced, label it exploratory.

## Numerical correctness

- Use explicit units.
- Define annualization assumptions.
- Define risk-free-rate assumptions when used.
- Handle zero denominators and empty samples explicitly.
- Avoid silently replacing invalid values with zero.
- Test metrics against hand-calculated fixtures.
- Document whether returns are arithmetic or logarithmic.
- Document how overlapping trades and exposure are treated.

## Result communication

Reports must distinguish:

- gross and net returns;
- in-sample and out-of-sample results;
- strategy and benchmark;
- hypothetical and executed performance;
- exploratory and validated findings.

Do not describe backtests as predictions or guarantees. Do not imply that high historical returns establish future profitability.

## Paper and live promotion

A strategy should not progress merely because it tops an optimization table.

Promotion criteria should eventually include:

- minimum out-of-sample trade count;
- performance across multiple windows;
- acceptable drawdown;
- stability to nearby parameters;
- acceptable results under worse costs;
- deterministic reproduction;
- paper-trading agreement with modeled behavior;
- operational safeguards and monitoring.

Human approval is required before live trading.
