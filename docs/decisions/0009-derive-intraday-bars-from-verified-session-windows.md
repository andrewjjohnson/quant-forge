# ADR 0009: Derive intraday bars from verified session windows

- Status: Accepted
- Date: 2026-08-12
- Jira: [QF-18](https://frostfiredigital-37308542.atlassian.net/browse/QF-18)

## Context

QF-13 defines exchange-session boundaries and completion states, QF-14 defines
common-source dataset-family lineage, and QF-17 binds exact coverage evidence to
an immutable intraday source dataset. Larger intraday bars must consume those
contracts without relying on fixed UTC hours, mixing provider-native intervals,
or hiding missing constituents. Aggregation also needs stable identities and
immutable persistence so an apparently identical chart can be reproduced from
the exact same source revision and policy.

## Decision

QuantForge derives larger intraday bars from one immutable normalized source
dataset. The target is a strictly larger exact multiple of the source duration,
uses the same exchange-session and anchor policies, prohibits cross-session
continuation, and exposes completed target bars only. Windows are resolved from
each actual exchange open and close. A shorter terminal window at a normal or
early close is emitted as a completed partial-duration bar.

OHLCV aggregation is first open, maximum high, minimum low, final close, and
summed volume. The default strict policy rejects an incomplete QF-17 source
report before returning a dataset. An explicit diagnostic policy excludes
unexpected intervals, never fills missing data, and retains exact per-window
expected, observed, missing, source-bar, and output-bar evidence. A diagnostic
window with observations may be emitted but remains structurally marked
incomplete in the aggregation report; a window with no observations is not
invented.

Each derived identity binds the source dataset, raw snapshot IDs, source batch
and quality report, target timeframe, complete aggregation policy, output batch,
and per-window report. The manifest embeds the full source quality report and a
QF-14 family whose root is the normalized source dataset ID and whose child is
the QuantForge-derived dataset. Derived bars identify QuantForge as producer,
so provider-native higher-timeframe observations cannot alias a family member.

Derived artifacts use a separate content-addressed namespace. Existing bytes
are accepted only when identical, and loads rederive from the verified source
before comparing canonical bars, manifest, IDs, and checksums.

## Consequences

Normal sessions, early closes, and DST dates use the same calendar-derived
semantics, and an aggregate cannot blend two sessions. Reaggregation is byte-
deterministic. Source revisions and strict-versus-diagnostic choices produce
different dataset identities even when resulting OHLCV happens to match.

Diagnostic output is intentionally not a substitute for complete data. A
consumer must inspect the aggregation report before using an incomplete bar.
Rederiving on cache load costs CPU but provides strong verification and avoids a
second, potentially divergent deserialization path.

## Alternatives considered

- **Use provider-native 15-minute, hourly, or 4-hour bars.** Rejected because
  provider products can differ in feed, session, anchor, adjustment, and source
  revision and cannot prove common-source lineage.
- **Fill missing constituents.** Rejected because invented OHLCV would hide
  data quality and can change indicators or research outcomes.
- **Drop every terminal partial bar.** Rejected because it discards completed
  market observations and mishandles early closes.
- **Use fixed UTC buckets.** Rejected because XNYS local sessions change UTC
  offset across DST and have calendar-defined early closes.
- **Include daily, weekly, alignment, or developing-bar behavior.** Rejected as
  sibling-ticket scope outside QF-18.

## Validation

Offline fixture and invariant-style tests cover hand-auditable OHLCV, every
required 5-minute exact multiple, normal 4-hour terminal bars, early closes,
multiple sessions, missing strict/diagnostic behavior, deterministic bar and
manifest identities, source and policy identity sensitivity, provider-native
separation, immutable cache verification, invalid targets, and pre/post-DST
session offsets. The full repository checks protect earlier behavior.
