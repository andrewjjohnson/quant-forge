# ADR 0024: Separate causal prediction views from canonical source evidence

- Status: Accepted
- Jira: QF-52

## Context

Rebinding a canonical QF-51 input's range/content while retaining complete source
evidence creates an invalid hybrid at QF-39/QF-40 evaluator boundaries. Preserving
only retrieval time cannot resolve the other evidence contradictions.

## Decision

Add a distinct bounded provenance variant to the existing prediction carrier.
Validate the canonical input before projection. Retain opaque canonical ancestry,
static family semantics, and only permitted original session observations. Use a
separate deterministic namespace with cutoff-bearing identity.

Keep canonical validation strict. Local bounded validation checks evidence,
content, calendar, cutoff, and semantics. A shared ancestry check compares a view
with the independently retained canonical plan source at offline QF-39/QF-40
boundaries. A flat source hash alone cannot prove subset membership; do not claim
otherwise or attach future source evidence to the evaluator to obtain that proof.

Reuse the existing partition helper, artifact relationships, and resume comparisons.
Membership, outcomes, daily projections, and holdout state semantics stay unchanged.

## Consequences

The view contains no parent object or future observation evidence. Retrieval and
cutoff remain distinct. Local evidence grows with visible session history. Offline
lineage checks use the canonical plan evidence without provider I/O or research
execution. No cache, manifest type, or ingestion format is added.

See [the bounded-input contract](../bounded-prediction-inputs.md).
