# Current-data multi-timeframe prediction scanner

QF-33 adds a research-only application boundary for evaluating a validated
QF-31 technical-confluence rule against a current QF-20/QF-21 context. The
scanner does not define a live-only rule, calculate indicators itself, label
future outcomes, or submit orders. It reuses the exact QF-28 restricted
context, QF-35 backend-neutral indicator path, and QF-31 condition evaluator
used by historical studies.

## Scanner boundary

`PredictionScanner` receives one or more `PredictionScannerRuleBinding`
objects. Every binding pairs a QF-31 rule with a
`HistoricalPredictionStudyReference` captured from the validated historical
study manifest. Before any data access, before handling a preparation failure,
immediately before evaluation, and again after rule-controlled evaluation and
output generation, the scanner verifies all of the following against that
reference:

- rule name and implementation version;
- complete rule configuration and configuration ID;
- every condition and normalized operand;
- timeframe, session, feed, completion, freshness, and failure policies;
- the exact validated symbol universe;
- the immutable historical input-bars SHA-256 fingerprint;
- the exact price, volume, and corporate-action adjustment basis;
- every indicator configuration ID; and
- resolved standard-backend ID, contract version, mapped function, Python
  wrapper version, and native runtime version.

Any difference raises `HistoricalStudyMismatchError`. There is no fallback to
another backend or indicator implementation.

The injected `PredictionScannerDataSource` owns acquisition orchestration. Its
`prepare_context(requirements, as_of=..., refresh=...)` implementation may
refresh canonical source data or replay an immutable cache, rebuild the
required derived timeframes, and return a `PredictionScannerSnapshot`.
Provider clients, endpoints, and credentials remain behind that data/ingestion
adapter. The scanner receives only the canonical context, dataset identity,
symbol, and adjustment basis. The current symbol must belong to the historical
study's immutable validated universe; cross-universe inference fails closed.
Transient acquisition details such as whether an immutable cache was seeded or
replayed are run-level diagnostics, not causal alert provenance.
`PredictionScannerSnapshot` rejects a reported prediction dataset ID unless it
matches the canonical source snapshot ID carried by every context lineage
reference, preventing provenance from naming unrelated or stale data.

`dry_run=True` always passes `refresh=False` to the source. A cache-only source
should reject `refresh=True` rather than silently accessing a provider.

## Temporal and evaluation semantics

The requested `as_of` must be timezone-aware and must exactly match the
returned context. `build_prediction_rule_context()` then enforces the existing
QF-28 contract before the rule runs:

- only bars visible at `as_of` are available;
- the primary decision bar is completed;
- contextual bars cannot end after the primary decision boundary;
- missing and stale declared timeframes fail clearly;
- symbol, adjustment basis, feed scope, session policy, dataset family, and
  completion policy must match; and
- indicators are evaluated only through their captured normalized
  configuration and backend identity.

The scanner calls the normal QF-31 evaluator and QF-11-compatible
`generate_with_context()` path, then proves their accepted/rejected direction
agrees. It accepts only output contract version 1 and requires both the outer
output and its candidate to carry the bound rule's exact strategy and source-
rule identity. It does not construct or call an outcome labeler.
Historical/current parity therefore means that the same fixed context, rule
configuration, and backend environment produce the same normalized indicator
values and rule decision in both paths.

Context failure policy is preserved per bound rule. `fail` propagates missing,
stale, or incompatible context errors and stops the scan. `skip` records a
stable skipped-context manifest in `PredictionRuleScanResult.context_failure`
and continues to later rule bindings without evaluating or alerting the skipped
rule. Rule, provenance, and scanner-contract mismatches still fail closed.

Completed-bars-only remains the conservative policy. Developing mode must be
declared by the historical rule. QF-21 then reconstructs each developing
higher-timeframe bar only from completed canonical source intervals whose ends
are at or before `as_of`.

## Alert schema

Only an accepted UP or DOWN candidate creates a `PredictionAlert`. A
no-prediction evaluation remains available in `PredictionRuleScanResult` and
is not sent to alert sinks.

Alert schema version 1 includes:

- symbol, UTC `as_of`, primary decision timestamp, and predicted direction;
- rule name, implementation version, and configuration ID;
- condition definitions, passed/failed/unavailable status, normalized values,
  prior crossover values, and source timestamps;
- latest normalized indicator values grouped by timeframe and alias;
- both unbound indicator and timeframe-bound configuration IDs;
- exact backend identity/version metadata when a standard backend applies;
- source bar IDs, start/end timestamps, completion states, and session dates;
- timeframe, session, feed, dataset-family, source-dataset, and adjustment
  provenance;
- the referenced historical study, its input-dataset fingerprint, optional
  validated symbol universe, optional summary, and optional sample count; and
- an explicit research-only/no-order disclaimer.

The alert ID binds symbol, the complete serialized historical-study reference,
rule/version/configuration, normalized indicator configuration/backend
identities, primary decision timestamp, completion policy, exact context
identity. Binding the full study reference ensures that metadata changes cannot
produce different serialized bytes under an existing alert ID. Transient cache
state is excluded so an interrupted delivery keeps one stable ID and byte-for-
byte payload when retried from the same canonical dataset.
Alert construction has no broker, order, fill, portfolio, or outcome-label
dependency.

## Deduplication

Deduplication uses an exclusive pending claim so two scanner processes cannot
emit the same key concurrently. The file-backed store holds an operating-system
lock while sinks run, marks the claim `published` only after every sink succeeds,
and releases a pending claim when a sink fails. A competing scan waits for the
active claim: after the lock becomes available, it either retries a released or
abandoned claim or returns the ID stored by the successful publisher. Duplicate
scan results therefore identify the actual emitted alert artifact. If a scanner
process exits while a claim is pending, the operating system releases its lock
and the same alert identity can recover the abandoned claim instead of being
suppressed forever. A different alert identity sharing the decision-bar key
cannot overwrite pending evidence; recovery fails closed until the original
identity is retried or the state is explicitly reconciled.

Lock files use stable inodes and remain in the state directory. Pending and
published JSON states are written and synced to private temporary files, then
atomically replaced while the stable lock remains held. An interrupted state
transition therefore leaves either the prior complete state or the next
complete state, never a truncated live marker.

Delivery is therefore at least once across an abrupt exit: a process that exits
after a sink succeeds but before the claim becomes `published` may cause that
sink to receive the alert again. Sinks should use `alert_id` as their idempotency
key. The bundled JSON file sink already enforces idempotent create-only writes.

Two explicit policies are available:

- `decision_bar` is the default. It emits at most once for the same symbol,
  historical study, rule/configuration/backend semantics, completion policy,
  and primary decision timestamp. A refreshed dataset or a later `as_of`
  cannot repeatedly alert on the same decision bar.
- `exact_context` keys by the complete alert identity. It permits another
  alert when a developing or completed context identity changes, even if the
  latest primary decision bar has not changed. Use this only when intermediate
  developing-context updates are intentionally actionable.

`InMemoryAlertDeduplicationStore` supports one process.
`JsonFileAlertDeduplicationStore` persists private, content-addressed lifecycle
and lock files for cross-run deduplication and recovers abandoned pending
claims.

## Sinks

`ConsolePredictionAlertSink` writes the complete alert as formatted JSON.
`JsonFilePredictionAlertSink` writes and syncs a private temporary file, then
atomically installs `<alert-id>.json` with create-only hard-link semantics. A
process exit can therefore leave either no final path or a complete final
artifact, never a partially written final artifact. If the final path already
exists, identical bytes are accepted and different content fails instead of
overwriting prior evidence. Additional sinks implement the small
`PredictionAlertSink` protocol; push, email, and SMS providers are not included
in QF-33.

## Offline SPY dry run

From the repository root:

```bash
uv run python scripts/scan_spy_predictions.py \
  --cache-root data/qf30-spy-context-cache \
  --alert-root reports/qf33-spy-alerts \
  --state-root data/qf33-spy-alert-state
```

The script uses the committed QF-30 synthetic SPY fixture. It creates no
provider client and reads no credentials. On the first run, it materializes
the fixture through the immutable QF-16 cache; subsequent runs validate and
replay that cache. It rebuilds 4-hour, daily, and weekly artifacts, constructs
the fixed midweek developing context, and evaluates the exact `talib_v1`
SMA(2) configuration captured under
`qf33_spy_fixture_parity_study_v1`. The scanner loads that identity and its
unadjusted-price policy, plus the fixture input-bars fingerprint, from a
committed immutable study record; it does not derive the study reference from
the current rule. The default developing-bar
mode uses `examples/spy_multi_timeframe/qf33_historical_study_reference.json`;
the completed-bars mode uses the independently validated
`examples/spy_multi_timeframe/qf33_completed_bars_historical_study_reference.json`.
The deterministic ascending fixture emits one UP alert. Repeating the command
emits no second alert under the default `decision_bar` policy.

Use completed bars only with:

```bash
uv run python scripts/scan_spy_predictions.py \
  --completion-policy completed_bars_only
```

Use `--deduplication-policy exact_context` only when a new context identity is
intended to create another alert. The fixture validates temporal and
historical/current parity mechanics only; it is synthetic, is not evidence of
predictive accuracy or profitability, and must not be used for a trading
decision.

## Deliberate limits

QF-33 adds no brokerage integration, automated execution, options selection,
provider-specific notification integration, scheduler, daemon, dashboard, or
separate live indicator calculation. Production acquisition adapters may be
added behind `PredictionScannerDataSource` without changing scanner, rule,
indicator, alert, or deduplication semantics.
