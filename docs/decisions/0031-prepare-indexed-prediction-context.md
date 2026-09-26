# ADR 0031: Prepare exact prediction context once per execution scope

- Status: Accepted
- Date: 2026-09-25
- Jira: [QF-59](https://frostfiredigital-37308542.atlassian.net/browse/QF-59)

## Context

After QF-58, each permitted historical decision still serialized/hashed the same
QF-8 plan once per context timeframe. Each call rebuilt source timestamp keys,
validated the full chronology, scanned history and visible membership, then
scanned the full source again to retrieve matching bars. QF-45 uses two context
timeframes, hence two full plan hashes per decision.

Plans are frozen, but their graph can include runtime subclasses. Globally
memoizing an arbitrary live plan by object identity would create stale-identity
risk. QF-48 already indexes exact membership timestamps to session labels;
QF-20's bounded alignment remains authoritative for completion and freshness.

## Decision

Add an explicit, detached `PreparedPredictionContext` scoped to a plan, window,
source family and already validated prediction-input view. Capture the existing
`ValidationPlan.plan_id` once. Copy only immutable selection metadata: window
identity, boundaries, warm-up counts, observed-session bounds/missing sessions,
calendar/timeframe policy, and exact permitted plan timestamps. Do not retain a
live plan or duplicate its full source-provenance evidence.

Reuse QF-58's recursive exact-type immutability check for source backing. Unknown
or mutable source graphs retain the reference selector. Index each supported
source's completed UTC bar ends once and retain integer window-start positions.
Per-decision binary search resolves the inclusive completion cutoff; warm-up and
visible membership are exact slices. A source without an in-window completed
bar retains the preceding anchor plus declared warm-up, as before.

The QF-39 permitted-context provider prepares before QF-42 execution and retains
its exact restricted schedule in a set. QF-11/QF-20/QF-28 receive ordinary causal
series through the existing provider interface. Indicators, outcome sources,
completion/freshness policies and failed/skipped context behavior remain owned
by their existing contracts. Context preparation does not own outcome membership.

Preparation never enters scientific identity or checkpoint serialization. Resume
reconstructs indexes, then QF-56 executes only its verified incomplete suffix.
Explicit reuse across scopes must validate current plan identity, exact window,
input-view identity and source artifacts. Indicator parameters need no additional
index key: source selection depends on declared plan warm-up and timeframe;
QF-28 continues checking each actual indicator/configuration declaration.

## Consequences

Full-plan hashing and full-source chronology scans become one-time scope costs.
Preparation costs and memory remain proportional to indexed observations. Indexes
contain references, timestamp keys and integer positions, not another price store.
Known calendar timestamps may need the existing QF-58 immutable backing shells.
No global cache, cursor or ordering dependency is introduced.

Selection still validates/materializes the selected subset, and QF-20 still sorts
and validates bounded series. These costs can grow with visible window history;
this decision does not change indicator history or optimize outcome validation.
QF-52 projection and independent lineage reconstruction remain unchanged and are
separate startup work. No QF-40/QF-9/QF-41 changes are required.
