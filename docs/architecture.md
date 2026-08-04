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
Canonical market dataset
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

Optimization orchestrates repeated strategy runs but must not alter the semantics of the underlying backtest.

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
- generate deterministic fingerprints.

Must not:

- infer missing prices from future observations;
- mix adjusted and unadjusted series silently;
- generate strategy decisions.

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
executes eligible orders at its open before marking shares at its close. It
emits immutable position, trade, cash, equity, return, peak, and drawdown
records, and rejects negative cash or shares.

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

Responsibilities:

- define search spaces and constraints;
- schedule trials;
- persist status and metrics;
- resume interrupted studies;
- assess parameter stability.

Must not:

- introduce different execution assumptions between trials;
- select using holdout outcomes;
- discard failed trials without a record.

### Reporting and experiment manifests

QF-5 exports a stable manifest and ordered CSV ledgers into an immutable
run-identity directory. The deterministic identity covers QF-3 dataset
provenance, QF-4 strategy configuration and implementation version, all
execution/cost/sizing assumptions, and engine/result schema versions.

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
