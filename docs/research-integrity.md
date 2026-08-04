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
QF-3 schema version 3 requires a provider split coefficient and cash-dividend
amount for every bar and retains every non-unit split and nonzero dividend
session as immutable provenance. QF-5 rejects legacy schemas, observed split-
or dividend-bearing ranges, and adjusted datasets because it does not yet
receive or apply the event details needed for causal share, cost, and cash
accounting. See `docs/backtesting.md` and ADR 0001.

Dataset hashes prove consistency, not provider authenticity. Raw-byte digest
verification occurs when QF-3 loads the immutable cache, and corporate-action
completeness remains dependent on schema-v3 provider ingestion because split and
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
must use a coherent policy. QF-5 verifies split- and dividend-free unadjusted
ranges from QF-3 schema-version-3 provenance and rejects every range with an
observed corporate action until point-in-time factors and dividend cash-flow
semantics are available. Tests cover both rejection boundaries; applying those
events still requires dedicated accounting tests before broad equity-universe
work is considered reliable.

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
