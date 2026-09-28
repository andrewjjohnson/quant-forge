# ADR 0034: Normalize window membership into append-only catalogues

- Status: Accepted
- Jira: [QF-62](https://frostfiredigital-37308542.atlassian.net/browse/QF-62)

## Context

In QF-45's fixed schema 2 window, 96.36% of the bytes are ordered
`visible_bar_ids` lists. Each list is repeated in the QF-20 source context, the
QF-28 rule timeframe and every normalized indicator: four copies of the 2m list
and three of the daily list per decision. Consecutive decisions repeat almost the
whole history.

The lists are evidence. Offline validation checks that the copies agree. They
also feed the QF-20 `context_id` and the QF-11 `study_id`. Daily membership also
has a legitimate rolling start bound from QF-48's warm-up lookback.

The generic QF-42 context provider cannot enumerate membership before execution.
QF-56 also writes the shared evidence before any decision runs.

## Decision

Add explicit compact representation schema `"3"`.

- **Catalogues.** Store each list's bar IDs once per window, in append-only
  catalogues keyed by the hash of an exact source dataset reference and timeframe
  (including session) binding.
- **Growth.** Carry catalogue growth as ordered segments inside the decision
  that first needs the entries. The decision and its catalogue growth then commit
  atomically under the unchanged QF-56 journal and checkpoint. A reference can
  never reach an entry introduced later.
- **Ranges.** Replace a list with `{catalogue_id, start_index, stop_index}` only
  when it is one contiguous catalogue run. Starts may move.
- **Exceptions.** Keep empty, developing-bar, unbound, non-contiguous, reordered
  and duplicated lists explicit.
- **Canonical form.** Readers replay the encoding from the preceding catalogue
  state and reject any other representation. The header names each complete
  catalogue's content hash.

Do not redefine scientific identities. Validation reconstructs the exact expanded
lists and verifies the unchanged QF-20/QF-11/row IDs. The physical shared-evidence,
window, decision, result and catalogue IDs are new for schema 3. The reader
negotiates versions 1, 2 and 3 explicitly, and header, evidence and decision
forms must agree. Existing files are neither rewritten nor migrated. QF-45 keeps
schema 2. Consumers gain version plumbing, not catalogue knowledge.

We rejected putting the catalogue in the up-front shared evidence. That would
have required a new provider protocol, stored bars that no decision observed
(including post-window bars), and made range causality depend on the entries
alone.

## Consequences

The QF-45 fixed window is exactly re-encoded from 2,453,216,703 to 93,450,669
bytes without recomputation. All 4,095 expanded decisions are byte-identical to
schema 2. Membership storage and per-decision records no longer grow with visible
history.

Readers retain catalogues whose size is bounded by the source bars observed.
Structural reading is faster because decisions are small. Scientific validation
still hashes the expanded context to verify the original IDs, so it improves
less.

The remaining bytes are repeated context metadata, feature manifests and study
envelopes. Those, sparse decisions and feature-series preparation are deferred
to later tickets.

See [the membership contract](../normalized-window-membership.md).
