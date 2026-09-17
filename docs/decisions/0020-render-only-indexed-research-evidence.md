# ADR 0020: Render only indexed research evidence

- Status: Accepted
- Date: 2026-09-16
- Jira: [QF-41](https://frostfiredigital-37308542.atlassian.net/browse/QF-41)
- Pull request: [#43](https://github.com/andrewjjohnson/quant-forge/pull/43)

## Context

QF-9 already binds source provenance and typed immutable artifacts. QF-40 owns
OOS summaries and permanent holdout state. A report that reconstructs missing
metrics, reranks trials or trusts an old reservation would create a second
research path or misstate validation evidence.

## Decision

Use fixed study-family section mappings over QF-9 references with a small
standard-library HTML renderer. Show saved values, bounded table previews and
exact artifact links. Missing fields remain unavailable. Do not execute any
producer, metric or chart code. Reuse QF-9 verification, safe metadata and
content identity helpers; do not modify producer contracts.

Query the original QF-40 ledger for current holdout state, tied to the manifest's
exact validation lineage. Never infer unseen status from saved reservations.
Retain consumption evidence permanently in presentation, even when result
artifacts are missing. Recheck authority before publication and clearly disclose
that exported HTML is a static snapshot.

## Consequences

Prediction, feature and trading studies keep distinct concepts. Reports work
offline without dependencies beyond the existing Python package. QF-34 charts
remain linked existing artifacts. Producer-specific research summaries must be
produced and indexed separately; reporting cannot fill gaps. Relative artifact
links require preserving the common source layout. Reports do not update after
publication, so users must rebuild to inspect later consumption or file changes.
