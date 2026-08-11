# QuantForge Architecture

## Goals

QuantForge must make quantitative experiments:

- reproducible;
- auditable;
- resistant to time-series leakage;
- easy to extend with new strategies;
- consistent between research, paper trading, and eventual live trading.

The architecture should favor explicit domain boundaries over convenience shortcuts.

## High-level flow

```text
Market-data provider
        |
        v
Raw immutable extract
        |
        v
Normalization and validation
        |
        v
Canonical timeframe + session policy
        |
        v
Canonical bars + typed corporate actions
        |
        +----------------------------+
        |                            |
        v                            v
Aligned indicators             Dataset fingerprint
        |
        v
Strategy target-state decisions
        |
        v
Position-sizing intent
        |
        v
Execution simulation
        |
        v
Portfolio accounting
        |
        v
Trades, equity, and artifacts
        |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
Metrics/validation       Feature analysis       Reports/manifests
```

QF-6 optimization orchestrates repeated strategy runs but does not alter the
semantics of the underlying QF-4 strategy or QF-5 backtest. See
`docs/optimization.md` and ADR 0002.

QF-11 prediction analysis is a separate branch from execution. A generic study
orchestrator fixes causal predictions before invoking a separately configured
outcome labeler, then passes each typed prediction/outcome pair to a separately
configured evaluator. The original QF-11 study supplies the concrete
next-session-open gap label and directional evaluator. Prediction studies never
create orders, fills, or portfolio results. See `docs/prediction-analysis.md`.

QF-7 extends that branch after causal candidate classification. It enriches
already-fixed QF-11 prediction records with configurable QF-4 context, reuses
QF-11 labeler/evaluator compositions for multi-session outcomes, and exports a
versioned deterministic signal-feature dataset. It does not create a parallel
prediction framework or feed future outcomes back into rule execution. See
`docs/signal-feature-datasets.md`.

## Module boundaries

### Data acquisition

QF-3 implements this boundary in `quantforge.data`: an injected provider feeds
pure normalization and XNYS-aware validation, followed by content-addressed raw
JSON, canonical decimal CSV, and a stable manifest. See `docs/market-data.md`.

Responsibilities:

- communicate with a provider;
- retrieve raw data;
- capture provider metadata;
- handle provider-specific errors and rate limits.

Must not:

- calculate indicators;
- generate signals;
- simulate trades;
- silently normalize provider-specific behavior.

### Data normalization and validation

Responsibilities:

- transform provider data into canonical schemas;
- normalize timezone and identifiers;
- validate ordering, uniqueness, nulls, and numerical relationships;
- record adjustment policies;
- validate and retain typed provider-reported split and cash-dividend records;
- generate deterministic fingerprints;
- authoritatively validate complete datasets by recomputing all derivable bar,
  calendar, gap, digest, path, and identity invariants.

Must not:

- infer missing prices from future observations;
- mix adjusted and unadjusted series silently;
- generate strategy decisions.

### Timeframe and exchange-session semantics

QF-13 implements the shared temporal domain in `quantforge.timeframes`. It sits
below provider ingestion, aggregation, indicators, prediction, and backtesting.
Distinct typed intervals represent sub-day durations, exchange-session counts,
and exchange-trading-week counts. Immutable policy records define calendar,
timezone, regular or explicit extended hours, anchoring, cross-session behavior,
labels, and developing-bar exposure. See `docs/timeframe-semantics.md` and ADR
0004.

Responsibilities:

- define provider-neutral interval and completion-state vocabulary;
- resolve exchange sessions and trading weeks from the configured calendar;
- validate intraday boundaries, anchors, clock-leading and session-terminal
  partial durations, and developing-bar exposure;
- serialize every material semantic policy and derive deterministic identity;
- preserve explicit start/end timestamps independently of label convention.

Must not:

- retrieve provider data;
- aggregate OHLCV;
- align multiple timeframes as of a decision timestamp;
- calculate indicators, predictions, signals, orders, or fills;
- silently emulate a provider or chart platform's proprietary candle policy.

### Indicators and features

QF-4 implements the reusable indicator boundary in `quantforge.indicators`.
Indicators consume QF-3 `MarketDataset` values and return immutable, session-
aligned fields with explicit unavailable values. See
`docs/strategy-contracts.md`.

Responsibilities:

- compute reusable contemporaneous values;
- document lookback and warm-up requirements;
- return aligned, typed outputs;
- avoid mutation of inputs.

Must not:

- use future rows;
- create orders;
- own portfolio state;
- calculate forward outcome labels in the same namespace as contemporaneous features.

### Strategy

QF-4 implements this boundary in `quantforge.strategies`: immutable parameters,
owned indicator declarations, target-position decisions, next-session timing,
normalized sizing intent, and a minimal generic runner. Strategy output retains
QF-3 dataset provenance but never claims execution. See
`docs/strategy-contracts.md`.

Responsibilities:

- declare parameters and required fields;
- transform available historical information into entry/exit signals;
- expose signal metadata and rationale when useful.

Must not:

- download data;
- assume fills occurred;
- modify cash or positions;
- render reports;
- inspect validation or holdout results during signal generation.

### Position-sizing intent

Responsibilities:

- convert a desired position state into a normalized target request;
- declare any context a future policy needs;
- validate the requested target range.

Must not:

- calculate final share quantity;
- reserve cash, apply leverage, or round lots;
- decide prices, orders, or fills;
- own portfolio accounting.

### Execution simulation

QF-5 implements deterministic daily-bar execution in
`quantforge.backtesting`. It consumes the generic QF-4 target-state/weight
contract, creates an auditable order for every decision, enforces next-exchange-
session eligibility, sizes whole shares, and applies separately configured
commission, additional fees, and adverse slippage. See `docs/backtesting.md` and
ADR 0001.

The QF-5 execution boundary accepts only schema-version-4 unadjusted QF-3
datasets with complete typed corporate actions and consistent raw OHLCV. It
rejects adjusted executions, missing/incomplete actions, and unknown action
semantics. Dividend policy is independently configurable as price-only,
cash-credit, or strict rejection; raw-data split accounting remains mandatory.
Before trusting provenance, QF-5 reserializes bars and verifies their
digest, complete metadata, raw digest, action snapshot and records, and
canonical paths against the QF-3 dataset ID. The shared QF-3 validator also
recomputes exchange sessions and exact gap provenance. Raw-byte and provider
authenticity remain QF-3 ingestion/cache responsibilities.

The boundary also rejects QF-3 datasets with expected market sessions missing
between their first and last observed bars. This preserves daily metric
semantics; leading and trailing requested-range gaps remain explicit provenance.
Strategy and benchmark execution share one cost-evaluation boundary that invokes
custom slippage, commission, fee, and configuration callbacks under the
serialized Decimal policy.

Responsibilities:

- convert eligible signals into orders;
- enforce timing rules;
- apply commission, fee, slippage, and fill models;
- record rejected and partial orders when supported.

Must not:

- change the strategy signal;
- calculate optimization scores;
- hide unfilled orders.

### Portfolio accounting

QF-5 owns a chronological single-symbol, long-only state machine. Each session
captures prior-close dividend entitlement, applies splits, executes eligible
orders at the open, applies or discloses dividends according to policy, and
marks shares at the close.
It emits immutable action, position, trade, cash, equity, return, peak, and
drawdown records, and rejects negative cash/shares or fractional split outcomes.

Responsibilities:

- own cash, positions, cost basis, realized/unrealized P&L, and equity;
- enforce accounting invariants;
- emit auditable ledgers.

Must not:

- infer strategy intent;
- alter historical fills;
- calculate indicators.

### Metrics and validation

QF-5 calculates typed, nullable performance summaries and a cost-matched
full-period buy-and-hold benchmark without altering the underlying ledgers.
Undefined ratios serialize as `null`; metrics never insert `NaN` or infinity.

Responsibilities:

- calculate performance and risk metrics;
- compare against benchmarks;
- implement chronological splits;
- assemble out-of-sample results;
- flag low sample sizes and fragile outcomes.

Must not:

- tune strategy parameters using test or holdout periods;
- rewrite trades;
- conceal invalid runs.

### Optimization

QF-6 implements this boundary in `quantforge.optimization`. Typed finite search
spaces use QF-4 strategy parameter-contract order, declarative constraints and
the real strategy factory exclude invalid assignments before QF-5, and stable
study/trial identities reuse QF-3 dataset identity plus QF-4/QF-5
configuration/version provenance. Sequential execution is the reference;
bounded local process execution initializes the immutable dataset once per
worker. The parent process alone owns atomic persistence, QF-5 artifact export,
resume state, ranking, stability, and final ordering.

Ranking reads only typed QF-5 performance fields and applies explicit hard
constraints before a deterministic objective/tie-breaker policy. Stability is
computed in finite candidate-index coordinates, reports failed/ineligible
neighbors without replacing them with zero, and keeps objective rank separate
from stability rank. All outputs are explicitly in-sample.

Responsibilities:

- define search spaces and constraints;
- schedule trials;
- persist status and metrics;
- resume interrupted studies;
- assess parameter stability.
- record excluded, failed, pending, ineligible, and successful trials without
  hiding poor outcomes;
- export reproducible study manifests, trial summaries, rankings, neighborhood
  statistics, and parameter-value summaries.

Must not:

- introduce different execution assumptions between trials;
- select using holdout outcomes;
- discard failed trials without a record.
- retrieve provider data during trials;
- recalculate QF-5 metrics or duplicate strategy/backtest representations;
- treat stability as holdout or out-of-sample validation;
- deploy or promote a selected strategy automatically.

### Prediction analysis

QF-11 implements this boundary in `quantforge.prediction`. A prediction rule
consumes a QF-3 dataset and causal features, an outcome labeler attaches typed
later observations only after predictions are fixed, and an evaluator compares
each fixed prediction/outcome pair. The generic runner imposes no direction,
correctness, next-open, or gap schema. The original QF-11 study composes this
boundary with a directional rule, an immediate-next-session-open gap labeler,
and a directional evaluator. Its signal close is retained only as the outcome
reference price and is never represented as a fill.

Responsibilities:

- preserve a strict boundary between contemporaneous features and future labels;
- declare and validate each outcome's future-session horizon and market fields;
- keep label construction separate from study-specific evaluation;
- retain rule, outcome, evaluator, feature, dataset, and schema provenance;
- preserve deterministic typed study rows and identities;
- export study-specific deterministic immutable analysis artifacts.

Must not:

- create orders, fills, positions, or trade P&L;
- alter QF-5 timing or execution assumptions;
- expose future outcome values to prediction generation;
- assume every prediction is directional or every evaluation is classification;
- describe direction accuracy as executable performance.

### Signal-feature datasets

QF-7 implements this boundary in `quantforge.prediction.feature_dataset`. A
candidate rule emits one stable `SignalFeatureCandidate` for every identifiable
opportunity and fixes its accepted, rejected, blocked, or genuinely supported
overlapping disposition. Contextual features receive only history through the
completed signal session. The builder then composes and runs configured QF-11
outcome labelers/evaluators, flattens their typed values, and checkpoints rows
atomically for resume.

Responsibilities:

- capture every strategy input and configured unused contextual feature;
- retain rejected and blocked candidates without duplicate opportunities;
- calculate explicit session-based returns, MFE/MAE, and target/stop labels
  only after dispositions are fixed;
- preserve daily-bar target/stop ambiguity;
- document every flattened field's type, unit, source, and timing;
- bind QF-3, rule, feature, labeler, evaluator, and schema configuration into a
  deterministic identity;
- produce resumable CSV and machine-readable schema/manifest artifacts.

Must not:

- expose any forward label to prediction generation or contextual features;
- invent overlap semantics for stateless rules;
- represent excursions as executable prices;
- place orders or alter QF-5 execution/accounting;
- turn exploratory feature differences into production filters.

### Reporting and experiment manifests

QF-5 exports a stable manifest and ordered CSV ledgers into an immutable
run-identity directory. The deterministic identity covers QF-3 dataset
provenance, QF-4's execution-relevant adjustment and calendar reference, a
canonical fingerprint of the actual validated OHLCV bars, QF-4 strategy
configuration and implementation version, all execution/cost/sizing assumptions
and their implementation versions, and engine/result schema versions. The
validated-bar fingerprint, QF-3 raw and normalized content digests, selected
dividend/split policies, return basis, and the corporate-action snapshot are
persisted in the manifest. QF-5 first rejects a
dataset identifier that is
inconsistent with the current bars or execution metadata, then independently
binds the validated inputs into the run identity. Canonical immutable snapshots
prevent later caller-owned configuration mutation from changing the manifest
under a fixed run identity.

Responsibilities:

- record code, dependency, configuration, and data identities;
- render immutable human-readable reports;
- link results to underlying artifacts;
- clearly distinguish in-sample and out-of-sample results.

Must not:

- recompute results with undocumented assumptions;
- imply future profitability.

### Broker adapters

Reserved for paper and live trading.

They should adapt broker APIs to domain-level orders and events. Core strategy and accounting code should not import broker SDKs directly.

## Dependency direction

Prefer dependencies toward stable domain abstractions.

```text
CLI / notebooks / reports
            |
            v
Application orchestration
            |
            v
Domain interfaces and models
            |
            v
Data, execution, storage, and broker adapters
```

Infrastructure code may implement domain interfaces. Domain code must not depend on infrastructure-specific packages.

## Data and result objects

Prefer explicit, versioned records for:

- dataset metadata;
- strategy configuration;
- signal records;
- orders and fills;
- portfolio snapshots;
- trades;
- optimization trials;
- validation windows;
- experiment manifests.

Every persisted schema should have a version and migration policy before it becomes externally depended upon.

## Reproducibility

A reportable run should be identifiable by:

- unique run ID;
- code commit SHA;
- dirty-working-tree indicator;
- dependency lock fingerprint;
- strategy and parameter configuration;
- data fingerprint;
- provider and adjustment metadata;
- cost and fill assumptions;
- random seeds;
- execution environment;
- start and completion timestamps.

## Extensibility

New strategies should usually require:

1. a strategy implementation;
2. strategy-specific configuration;
3. unit tests;
4. optional feature declarations;
5. no change to execution, portfolio, optimization, or reporting internals.

New providers and brokers should be introduced through adapters rather than conditionals scattered across the codebase.

## Architectural decisions

Use ADRs under `docs/decisions/` for choices that are expensive to reverse, including:

- vectorized versus event-driven source of truth;
- canonical time and price conventions;
- storage engines and schemas;
- corporate-action handling;
- options-data model;
- broker abstraction;
- deterministic parallelism;
- experiment-tracking backend.
