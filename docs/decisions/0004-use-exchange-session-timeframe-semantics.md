# ADR 0004: Use exchange-session timeframe semantics

- Status: Accepted
- Date: 2026-08-10
- Jira: QF-13

## Context

QuantForge needs one provider-neutral meaning for intraday, daily, and weekly
bars before later QF-12 stories add ingestion, aggregation, multi-timeframe
alignment, or generalized indicators. A generic timedelta is insufficient:
XNYS daily bars are exchange sessions, weekly bars are trading weeks, regular
sessions can close early, and local session times move relative to UTC across
daylight-saving transitions.

The model must also prevent completed short terminal bars from being mistaken
for developing bars. That distinction controls whether a consumer could observe
future high, low, close, or volume. Existing QF-3 daily caches and dataset IDs
are already public scientific artifacts and must not be rewritten merely to add
the new vocabulary.

## Decision

QuantForge uses separate immutable interval types for positive sub-day elapsed
durations, positive exchange-session counts, and positive exchange-trading-week
counts. Daily means one exchange session; weekly means one Monday-Sunday set of
actual exchange sessions. Neither is encoded as a timedelta.

The default U.S. equity policy is XNYS, `America/New_York`, regular trading
hours, actual-session-open intraday anchoring, prohibited cross-session bars,
start labels, and completed-bars-only exposure. Bar labels never determine
availability; explicit end timestamps and completion state do.

Intraday bar state distinguishes full completed, developing, completed leading
partial-duration, and completed terminal partial-duration bars. The leading
partial state applies only when a clock bucket begins before the session and
ends after the open. A completed terminal partial-duration bar must be shorter
than its nominal interval and end at the schedule's actual session close. A
developing bar requires explicit consumer opt-in and must be strictly before its
applicable full, clock, or session-close terminal boundary.

Regular boundaries are resolved from `exchange-calendars` and stored in UTC,
with the configured exchange timezone retained in policy. Extended-hours scope
requires explicit same-day exchange-local bounds. Session-open and clock
anchoring are different identity-bearing policies. Clock anchoring requires an
explicit local origin. Cross-session permission is also explicit and is not the
default.

Every material setting has a stable primitive representation in timeframe
schema version 1. The canonical sorted-JSON SHA-256 is the timeframe
configuration ID. Later datasets, aggregations, contexts, indicators, and
studies must bind that ID when the policy becomes part of their scope.

QF-3 `DailyBar`, schema version 4, provider contracts, cache artifacts, and
dataset identities remain unchanged. This ADR supplies the canonical model but
does not implement QF-14 or later child-story behavior.

## Consequences

The type system prevents daily and weekly bars from silently collapsing into
minute arithmetic. XNYS holidays, weekends, early closes, and DST behavior are
calendar-derived and deterministic. Four-hour bars retain both nominal and
actual duration, and terminal completion cannot alias a developing state.
Clock-aligned leading intersections likewise remain explicit completed states
instead of being discarded or mislabeled as developing.

The configuration is more verbose than a string such as `4h`, but it preserves
the policies required to reproduce a chart. A platform/provider correction or
semantic policy change produces a different identity instead of silently
reusing results.

Existing daily research remains byte-for-byte compatible. Until a later ticket
adds timeframe identity to a new dataset schema, legacy QF-3 manifests retain
their existing calendar/date-label semantics and IDs. Consumers must not infer
that old manifests acquired new persisted fields.

## Alternatives considered

- **Use timedeltas for intraday, daily, and weekly bars.** Rejected because 24
  hours is not an exchange session and seven days is not an exchange trading
  week. It also obscures holidays and early closes.
- **Use clock-aligned bars by default.** Retained as an explicit supported
  policy, but rejected as the U.S. equity default because it can create a
  leading partial window at the 09:30 open and does not express the requested
  session-open 4-hour example.
- **Copy a provider or charting platform's candle identifiers.** Rejected as
  canonical semantics because identifiers such as `4h` omit session scope,
  anchor, label, partial-bar, and developing-bar behavior. Future
  platform-compatible policies should be explicit adapters or new versioned
  anchor policies, not silent changes to the default.
- **Discard partial terminal bars.** Rejected because it loses real completed
  session data and makes early closes/non-even divisions inconsistent.
- **Expose developing bars by default.** Rejected because completed-only is the
  safer causal research default. Developing bars remain available only through
  explicit policy and later as-of reconstruction work.
- **Permit cross-session intraday bars by default.** Rejected because it can
  blend disjoint market sessions and overnight gaps without disclosure.
- **Retrofit timeframe fields into QF-3 schema version 4.** Rejected because it
  would break existing daily dataset identities and prematurely enter later
  ingestion scope.

## Validation

Deterministic unit and invariant-style parameterized tests cover arbitrary
positive sub-day durations, distinct session/week counts, stable serialization,
identity sensitivity, session-open and clock anchoring, explicit cross-session
permission, clock-leading partials, developing-bar opt-in, the 09:30-13:30 and
13:30-16:00 XNYS four-hour windows, the 2024-11-29 early close, the 2024
Independence Day holiday week, a weekend, and the March 2024 DST boundary.
Existing full-suite tests protect QF-3/QF-4/QF-5/QF-6/QF-7/QF-11 compatibility.
