# ADR 0036: Separate decision coverage receipts from rich observations

- Status: Accepted
- Date: 2026-09-29
- Jira: [QF-64](https://frostfiredigital-37308542.atlassian.net/browse/QF-64)

## Context

Schemas 2 and 3 persist one complete normalized decision for every QF-42
scheduled timestamp. Each record carries the full QF-28 context, empty signal and
row lists, and physical identities, whether or not the rule produced anything.
In the QF-45 fixed window, 4,070 of 4,095 decisions produced no candidate.
After QF-62 those records still took 88.3 of the 93.5 MB.

After QF-63, a late no-candidate decision cost about 45 ms, and most of that was
persistence work. Each decision built the full `PredictionWindowDecision`
snapshot, which also encodes the shared market data and configuration. It then
converted and hashed that snapshot, normalized its catalogue membership, and
re-validated the whole persisted context before commit.

Downstream consumers (QF-32, QF-39, QF-40, QF-45 export) read only signals and
rows. They still decoded every no-candidate record to find the few that had any.

Current QF-42 dispositions are `evaluated` (at least one generated signal, of any
disposition, labeled or not), `no_prediction` (an available context and no
signal) and `skipped` (a context rejected under the SKIP policy, retained for
audit). A failed decision is never persisted: the window stops at the committed
prefix. QF-42 validation allows at most one generated signal per decision
session. QF-11 results and the persisted structure allow any number.

## Decision

Add explicit, opt-in compact schema `"4"`. Schemas 2 and 3 are unchanged, and
QF-45's frozen configuration keeps schema 2.

- **One receipt per scheduled decision.** Every scheduled decision gets exactly
  one `decision_receipt` line, in schedule order. The line holds `sequence`, the
  unchanged `status`, the original QF-20 `context_id` and QF-11
  `prediction_study_id`, `observation_count`, and `receipt_id`. The timestamp
  comes from the authenticated schedule.
- **Rich evidence only where it carries meaning.** An `evaluated` or `skipped`
  receipt nests its complete schema 3 decision, with QF-62 catalogue segments
  and ranges. An ordinary `no_prediction` receipt carries no rich payload.
- **Zero to many.** `observation_count` equals the nested decision's number of
  generated signals. Readers number observations across the window in schedule
  order, then signal order.
- **Durability.** QF-56 is unchanged. Each receipt, together with any nested
  evidence and catalogue growth, is one journal line committed by one fsync and
  one atomic checkpoint. A committed receipt proves the decision executed; a
  decision with no committed receipt did not.
- **Validation.** Rich evidence is fully validated on append, on resume and
  offline, exactly as in schema 3. A no-prediction decision decodes its
  in-memory context once before commit. It is checked against the scheduled
  timestamp and session, the window's family, dataset and shared-input
  provenance, the configured requirements and primary timeframe, and zero counts.
  Both persisted identities must be the hashes of that exact context under the
  window's scope. The full QF-20/QF-28 structural re-validation runs only for
  contexts that are persisted.
- **Readers.** `iterate_decision_receipts()` and `iterate_observations()` serve
  versions 1 to 4. `iterate_decisions()` stays for versions 1 to 3 and raises for
  version 4, because version 4 does not keep a rich record for every timestamp.
  Internal consumers use the new views.

We rejected three alternatives:

- **Delta-encoding every no-candidate context.** This would keep offline
  re-validation of those contexts. It would also keep most of the per-decision
  encode, hash and validation cost, and normalizing repeated context metadata
  is separate, deferred work.
- **A second journal for observations.** This adds a second commit offset and a
  new crash window.
- **A bitmap of dispositions.** It cannot carry the three existing statuses and
  their identities.

## Consequences

Storage grows with decisions times a receipt of about 350 bytes, plus
observations times their rich payload, plus shared evidence. It no longer grows
with decisions times a rich context. Observation consumers never build rich
objects for no-prediction decisions.

Scientific identities are unchanged: schedule and QF-42 window identity,
context, study, row, outcome and evaluation IDs, and signals. The physical
shared-evidence, window, receipt, decision, result and catalogue-content
identities are new for schema 4.

A persisted no-prediction receipt cannot be re-validated offline against a
context body, because the body is not stored. Offline checks confirm its
coverage, order, identity format and binding into the window result hash.
Re-executing the decision reproduces both identities deterministically. A
rehashed forgery of a whole window, including its header, is therefore detected
for rich records but, for bare receipts, only by re-execution.

No-prediction receipts are coverage evidence, not negative labels. Dense
(all-timestamp) research should use a columnar representation, not this event
format.

See [sparse event windows](../sparse-event-windows.md).
