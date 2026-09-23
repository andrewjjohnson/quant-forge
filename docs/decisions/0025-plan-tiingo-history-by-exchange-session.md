# ADR 0025: Plan Tiingo history by exchange session

- Status: Accepted
- Date: 2026-09-23
- Jira: [QF-53](https://frostfiredigital-37308542.atlassian.net/browse/QF-53)
- PR: [#51](https://github.com/andrewjjohnson/quant-forge/pull/51)

## Context

QF-45 diagnostics found HTTP-200 Tiingo intraday range responses containing
XNYS holiday observations and post-early-close observations despite RTH and
force-fill exclusions. Exact session requests work. Canonical validation
correctly rejects invalid session timestamps and must remain strict.

`IntradayFetchResult` and QF-51 require raw chunks to partition the complete
logical request. Literal disjoint session bounds would violate this established
contract and force unrelated source/prediction changes.

## Decision

Keep calendar decomposition inside Tiingo. Use the existing calendar utilities
to resolve actual XNYS sessions, request each date separately, and clip by the
intersection of the requested range and exact session window. Honor the existing
maximum chunk duration within sessions; default requests use one response per
session. No weekend/holiday acquisition unit is needed.

Distinguish physical acquisition/retention from logical coverage. Existing raw
chunk bounds are contiguous logical coverage envelopes. The first begins at the
logical start; subsequent envelopes begin at their retained window start. Each
ends at the next retained window start or the logical end. Each envelope has
one session/subwindow; intervening closed-market time has no expected bars.
Wire date parameters remain the actual single session date. Raw bodies remain
lossless and hashed, including excluded observations. The existing canonical
coverage report supplies actual session boundaries and observed coverage.
Together these records reconstruct retention without a new provenance schema.

The Tiingo adapter runs unchanged strict canonical coverage validation before
returning one fetch result. Failed acquisition cannot reach service persistence.
Other providers retain their own decomposition and completeness policies.

Bump only the Tiingo intraday adapter version to 2. Logical request identity and
raw/dataset schemas remain unchanged. Raw layout/version are already identity
material, and old valid cache artifacts remain reusable before provider access.

## Consequences

Request counts rise to roughly one per session; correctness takes precedence
over batching. Memory and persistence still scale with the entire logical
request. There is no new resumable partial-acquisition cache. A failure requires
retrying the uncached logical request, using the existing bounded HTTP retries.

Half-open clipping excludes provably out-of-window timestamps only. In-session
malformation, duplicate observations, missing intervals, and incomplete expected
sessions fail closed; there is no data repair or change to price/adjustment,
feature, outcome, validation-partition, or execution semantics. Session dates and
UTC ranges in errors make these failures diagnosable without echoing credentials
or untrusted HTTP bodies.

## Alternatives considered

- Multi-session wire requests: retain the demonstrated provider ambiguity.
- Disjoint raw chunk bounds: require a generic provenance and QF-51 contract
  change with no need for other providers.
- Synthetic empty holiday snapshots: misrepresent responses that were never
  acquired and obscure the distinction between no session and missing data.
- Weaken validation or rewrite bad timestamps: destroys research integrity.

## Validation

Deterministic tests exercise MLK Day, all three known early closes, normal and
DST sessions, seven months of history, partial boundaries, missing/malformed
rows, duplicate/overlapping chunks, safe diagnostics, immutable old/new caches,
a non-Tiingo single-range provider, and QF-51/QF-52 composition. Controlled live
verification uses the normal market-data service/cache; results are reported in
the PR without committing licensed observations or credentials.
