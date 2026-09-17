# Static research reports (QF-41)

`quantforge.reporting.build_research_report` reads a saved QF-9 experiment
manifest and its artifact index. `export_research_report` writes one standalone
HTML file with inline CSS and no JavaScript, network assets, server or running
Python requirement. This is a presentation layer over persisted research.

## API and local layout

```python
from pathlib import Path
from decimal import Decimal
from quantforge.reporting import (
    ResearchReportConfig,
    build_research_report,
    export_research_report,
)

root = Path(".").resolve()
report = build_research_report(
    root / "reports/experiments/<manifest-id>.json",
    artifact_root=root,
    config=ResearchReportConfig(
        maximum_preview_rows=20,
        minimum_sample_size=30,  # An explicit researcher-selected warning threshold.
        high_trial_count=100,  # Optional; neither value is a statistical guarantee.
        maximum_configuration_change_frequency=Decimal("0.5"),
    ),
)
html = export_research_report(report, root / "reports/research")
```

The existing manifest must retain its QF-9 content-addressed filename. Both
manifest and output must be inside `artifact_root`. Publication creates
`<output-root>/<report-id>.html` atomically without overwriting different bytes.
Exact retries reuse the file. Source artifacts, hashes and manifests are never
modified. Generated reports normally remain under ignored `reports/`.

The HTML is independently viewable; links to original JSON, CSV, Parquet, HTML,
SVG and other indexed files require their existing relative layout. Move the
common root together to preserve links. Large artifacts are not copied into the
HTML. Tables show at most `maximum_preview_rows` records in published order and
disclose truncation. Provenance is available through native HTML disclosures.
There is no report DSL, plugin registry, frontend application or chart engine.

Public immutable presentation records are `ResearchReport`,
`ResearchReportConfig`, `ReportSection`, `ReportPhase`, `ReportArtifact` and
`ResearchWarning`; invalid output/configuration raises `ResearchReportError`.
QF-9 and QF-40 errors propagate when the authoritative manifest/ledger cannot
be trusted. QF-34's public inspection API remains compatible and loads lazily.

## Study-type sections

| QF-9 study type | Rendered evidence |
| --- | --- |
| Prediction / prediction window | Recorded counts; rule/outcome/indicator context; indexed summary, metrics or analysis; direction/frequency, accuracy/intervals, baseline comparisons, signed outcomes, MFE/MAE, events and segmented summaries when present |
| Feature dataset | QF-7/QF-29 schema, candidate/disposition counts, namespaces, contextual/backend/timeframe provenance, outcome definitions, related prediction IDs; published distributions or QF-7 group/bin summaries when explicitly indexed |
| Parameter study | Search space, constraints, minimum samples, trial status/analysis, published rankings and parameter neighborhoods/stability |
| Backtest | QF-5 performance/drawdown, benchmark configuration/performance, capital, strategy, costs, slippage, execution/corporate-action policy, trade counts and original equity/benchmark/trade/fill/position table previews |
| Optimization | QF-6 search metadata, native eligible rankings, trial metrics/costs, stability, robust/best trial references and links to original per-trial QF-5 exports |
| Walk-forward / OOS / holdout validation | QF-8 plan, rolling/expanding policy and purging/embargo; QF-39 frozen selections, partition membership and fold states; QF-40 OOS summary, normalized equity when supplied, turnover, completeness and consumed holdout result |

Every family also exposes manifest/study/run/index identities, original code,
dependency and source lineage, indexed artifact relationships, validation and
final-holdout sections. Source/backend versions are retained exactly; absent
historical provenance is never replaced with the renderer's runtime.

Phases are visibly labeled **IN-SAMPLE**, **VALIDATION / SELECTION**,
**WALK-FORWARD OOS** and **FINAL HOLDOUT**. A primary result attached through
QF-9's validated fold/holdout relationship receives that phase. Standalone
results and search rankings remain in-sample; a top-ranked configuration is not
presented as a validated winner. The renderer never reranks trials.

Generic QF-11 results intentionally have no universal accuracy summary. Their
rows are linked, and unavailable summary metrics stay unavailable. Existing
producer summaries can be explicitly indexed with the appropriate artifact type
and JSON pointer using QF-9's `additional_artifacts`/relationships API. Reporting
does not discover unindexed sibling files or infer their relationship from a
shared directory. Evidence used in sections, research warnings and holdout
history must belong to the primary producer or connect through explicit QF-9
artifact relationships. A QF-9 validation attachment also retains its owning
validation study's evidence, including fold states without individual edges.
Unrelated additional artifacts remain in the full index with their integrity
status and links; their values cannot affect the reported study. QF-7 feature
analysis likewise requires an indexed published analysis artifact; the feature
dataset alone does not imply distributions.

Prediction reports have no equity, trading-performance or transaction-cost
sections/warnings. Execution metrics are explicitly not applicable. Missing
applicable metrics use **Unavailable**, never zero. An explicitly recorded zero
stays zero; an empty collection is labeled empty. Unknown producer fields are
preserved in displayed structured metadata when selected, or remain in the
linked original artifact; no producer schema is migrated or reconstructed.
Monthly returns and other absent QF-5 metrics remain unavailable.

## QF-40 is the holdout authority

For current holdout state, provide the existing QF-40 `OOSSource` and permanent
`HoldoutLedger` alongside the manifest:

```python
report = build_research_report(
    manifest_path,
    artifact_root=root,
    holdout_source=source,  # Original QF-40 source, or load_oos_source(original_plan, path).
    holdout_ledger=ledger,  # Open the existing permanent ledger; never create a replacement.
)
html = export_research_report(report, root / "reports/research")
```

The source must match the manifest's exact plan, study, lineage and study
definition. Reporting calls only `ledger.state(source)`, which owns exposure
auditing and lineage checks (including its existing ledger lock). It never
reserves, consumes, prepares or evaluates a holdout. It queries state before and
after building the report and rechecks the retained authority before export;
changes require rebuilding. A contradictory consumed-to-reserved state fails.

Without the authority, historical consumed state or an indexed consumption/result
reference always remains **CONSUMED**, including when that artifact is missing or
invalid. A historical reservation alone yields **CURRENT STATE UNAVAILABLE**,
never an assertion that the holdout is still unseen. With verified current state,
the header explicitly says **RESERVED / UNSEEN at this snapshot** or **CONSUMED —
never unseen**. Plan reservation metadata is labeled historical, separately from
the authoritative current section.

If the ledger has a new result absent from the old QF-9 index, its current state
and reference are shown but its metrics are not rendered. Create a new manifest
through QF-9 to include the result. Missing results never reset consumption.
Ledger errors fail closed rather than inventing pristine status. Like every
static artifact, an already-exported report does not update after later ledger
changes; its header discloses snapshot semantics. Rebuild to inspect current state.

## Warning rules

Warnings have stable codes, deterministic order and an evidence source. Rules
do not parse producer prose, infer thresholds or compute research statistics.
Original producer disclosures are also shown without interpreting their text.

| Code | Explicit condition |
| --- | --- |
| `IN_SAMPLE_ONLY` | No verified indexed OOS aggregate, fold result or consumed holdout result is available |
| `LOW_SAMPLE_SIZE` | Recorded prediction/labeled-row/trade/candidate count is below the configured minimum; when omitted, QF-32's recorded `ranking.minimum_prediction_count` is used if available |
| `HIGH_TRIAL_COUNT` | Recorded trials/Cartesian combination count reaches the explicitly configured threshold |
| `PARAMETER_INSTABILITY` | A published parameter/configuration-stability assessment classifies a result as `fragile` or an isolated peak, or its configuration-change frequency exceeds the configured maximum |
| `INCOMPLETE_DATA_COVERAGE` | Recorded missing sessions/intervals or incomplete sessions are nonempty, or explicit `coverage_complete` is false |
| `FAILED_OR_MISSING_OOS_WINDOWS` | A recorded fold is not completed, aggregate completeness is false, or indexed fold/OOS evidence is missing or invalid |
| `HOLDOUT_ALREADY_CONSUMED` | Current authority or historical consumption evidence says consumed |
| `HOLDOUT_STATE_UNAVAILABLE` | Current authority is absent and there is no consumption evidence |
| `ARTIFACT_INTEGRITY` | An indexed artifact is missing, invalid, mismatched or deliberately absent optional evidence |
| `MISSING_TRANSACTION_COST_ASSUMPTIONS` | Backtest/optimization configuration lacks commission, fees or slippage metadata; explicitly configured zero costs count as supplied assumptions |

Numerical warnings are disabled when neither a renderer threshold nor the
documented source constraint is available. Preview limits are presentation only;
they do not change warning denominators or underlying results. Warnings do not
certify statistical significance, economic validity or future performance.
Instability assessments come from indexed parameter summaries (their direct
records, `stability`, `summaries` and `top_stability_trials`), indexed configuration
stability, or an OOS aggregate's `stability` record. Strategy/factory configuration,
ranking parameters and nested per-window configurations do not supply assessments.
Count, coverage and completeness warnings follow only known statistic containers
such as `analysis`, `counts`, `record_counts`, `performance`, `coverage`, `summary`
and published ranking records. They do not traverse strategy/factory configuration
or trial/ranking parameter mappings. Native configuration artifacts supply only
their producer-owned counts, performance and market-data records.

## Integrity, security and determinism

QF-9 remains authoritative: its strict manifest reader validates identity and
canonical bytes; `verify_artifacts` verifies exact hashes, root containment,
JSON pointers and bindings. Captured JSON/CSV buffers are bound to the index
using QF-9's existing `ProducerReadSet`, including replacement-race detection.
Only verified evidence can supply displayed values or active file links.
Missing, tampered, unsafe and absent optional artifacts remain visible in the
index with their issue status and original hash. Verified inputs are checked
again before export. Nothing repairs hashes or suppresses integrity failures.
CSV previews require a header with nonempty, unique column names; missing,
empty or duplicate names mark the artifact invalid so cells cannot be silently
discarded or displayed under ambiguous labels.
JSON pointers into raw `rows`, `decisions`, `observations` or `preview` collections,
and QF-9 feature checkpoint artifacts (`rows/<id>`), remain verified links only.
The report retains no payload for these artifacts and never interprets their
fields as warning metadata. QF-9 still reads and validates the original JSON;
this avoids retained row copies rather than introducing streaming verification.
Prediction-result metadata also omits embedded `prediction_study.rows` and
`prediction_window.decisions`, including QF-32 trial exports indexed at the
document root. Published analysis and source manifests stay unchanged in the
retained metadata; complete observations remain available through the original
artifact link.
The same projection applies to walk-forward and consumed-holdout result wrappers.
Prediction observations and backtest signal, order, fill, position, trade, equity
and corporate-action histories remain in the source artifact rather than in each
retained fold snapshot. Manifests, counts, performance, summary fields and arbitrary
strategy configuration remain intact; no research values are recomputed.
OOS aggregates also omit their raw prediction `observations` arrays from retained
metadata. Their summaries, stability and provenance remain intact, as do backtest
`normalized_equity` series used by report sections. The original aggregate remains
available through its verified artifact link and hash.

All text, labels, configuration values, filenames and attributes are escaped.
Local file URLs are percent-encoded and explicitly relative; source strings
never become executable URL schemes. No artifact HTML/SVG is inlined. QF-34
inspection/chart artifacts are linked with their exact hash and provenance;
chart generation and normalized indicator calculation are never invoked.
The HTML has a restrictive Content Security Policy and no executable scripts.
QF-9's credential-field/value guard rejects unsafe JSON and manifest metadata;
the same guard applies to CSV previews. Errors never echo unsafe content. This
does not discover arbitrarily disguised secrets in unrelated free text; inputs
must contain research evidence only. Environment variables are never read.

The report ID binds renderer version, exact manifest ID/relative path,
configuration, observed artifact statuses and holdout snapshot. It uses the
repository's existing `configuration_identity` helper, not a separate artifact
hashing contract. Identical inputs and relative output layout produce identical
HTML bytes. There is no generated-now timestamp. Relocating the complete root
preserves content; changing relative output depth changes link spelling, so use
the same layout for byte comparisons. Original producer timestamps remain data.

The renderer never calls TA-Lib, indicator/context builders, prediction/outcome
engines, feature generators, parameter search/ranking, walk-forward execution,
OOS aggregation, holdout evaluation, backtesting, metric calculation or QF-34
chart generation. Missing research stays missing. Tests block these execution
paths and also exercise the ordinary API in a fresh interpreter where research
and numerical-package imports are prohibited.

## Offline verification

```bash
uv run pytest tests/unit/reporting
```

Tests use native synthetic prediction, feature, parameter, optimization,
backtest, walk-forward, OOS and holdout exports. They verify section/phase
selection, exact stored values, ranking order, no recomputation, consumption
transitions, unavailable evidence, warning boundaries, corruption/races,
credential rejection, HTML injection, local links and deterministic publication.
Strategy-specific calculations, extra outcome statistics, monthly-return
calculation, plot generation and dashboards remain outside QF-41.
