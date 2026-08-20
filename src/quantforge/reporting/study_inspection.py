"""Standalone synchronized charts for immutable multi-timeframe study inputs.

This module renders only canonical bars and normalized indicator outputs already
present in a ``PredictionRuleContext``. It deliberately contains no indicator
calculation path and no standard-indicator backend dependency.
"""

# HTML/SVG templates are kept as readable literal fragments.
# ruff: noqa: E501

from __future__ import annotations

import html
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.lineage import (
    DatasetFamily,
    SourceConsistencyMode,
)
from quantforge.indicators.timeframe import TIMEFRAME_INDICATOR_CONTRACT_VERSION
from quantforge.prediction.context import PredictionRuleContext
from quantforge.prediction.scanner import (
    HistoricalPredictionStudyReference,
    PredictionScannerRule,
)
from quantforge.prediction.technical_confluence import (
    TechnicalConfluenceEvaluation,
)
from quantforge.timeframes import (
    BarCompletion,
    SessionInterval,
    Timeframe,
    TradingWeekInterval,
)

STUDY_INSPECTION_REPORT_SCHEMA_VERSION = "1"
STUDY_INSPECTION_REPORT_ENGINE_VERSION = "1"
STUDY_INSPECTION_ARTIFACT_FILENAMES = ("manifest.json", "report.html")

_PRICE_OVERLAY_OUTPUTS = frozenset(
    (
        "simple_moving_average",
        "exponential_moving_average",
        "bollinger_middle_band",
        "bollinger_upper_band",
        "bollinger_lower_band",
    )
)
_VOLUME_OVERLAY_OUTPUTS = frozenset(("volume_moving_average",))
_PALETTE = (
    "#4cc9f0",
    "#f72585",
    "#f9c74f",
    "#90be6d",
    "#c77dff",
    "#f9844a",
    "#43aa8b",
    "#577590",
)


class StudyInspectionReportError(ValueError):
    """A report input or immutable export is inconsistent."""


class StudyInspectionExportStatus(StrEnum):
    """Whether an immutable report was created or byte-verified and reused."""

    CREATED = "created_immutable_report"
    REUSED = "reused_immutable_report"


def _aware_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise StudyInspectionReportError(
            f"{field_name} must be a timezone-aware timestamp"
        )
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class FutureOutcomeRegion:
    """Explicitly post-decision annotation, kept separate from causal features."""

    start_timestamp: datetime
    end_timestamp: datetime
    label: str
    values: PrimitiveMappingSnapshot | None = None

    def __post_init__(self) -> None:
        start = _aware_utc(self.start_timestamp, "future outcome start")
        end = _aware_utc(self.end_timestamp, "future outcome end")
        if start >= end:
            raise StudyInspectionReportError(
                "future outcome start must precede its end"
            )
        if not isinstance(cast(object, self.label), str) or not self.label.strip():
            raise StudyInspectionReportError("future outcome label is required")
        object.__setattr__(self, "start_timestamp", start)
        object.__setattr__(self, "end_timestamp", end)
        object.__setattr__(self, "label", self.label.strip())

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "availability": "post_decision_only",
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "label": self.label,
            "values": None if self.values is None else self.values.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class StudyInspectionReportConfig:
    """Deterministic visual selection policy for one static artifact."""

    max_bars_per_panel: int = 80
    include_primary_timeframe: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(cast(object, self.max_bars_per_panel), bool)
            or not isinstance(cast(object, self.max_bars_per_panel), int)
            or self.max_bars_per_panel < 2
        ):
            raise StudyInspectionReportError(
                "max bars per panel must be an integer of at least two"
            )
        if not isinstance(cast(object, self.include_primary_timeframe), bool):
            raise StudyInspectionReportError(
                "include primary timeframe must be a boolean"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "max_bars_per_panel": self.max_bars_per_panel,
            "include_primary_timeframe": self.include_primary_timeframe,
        }


@dataclass(frozen=True, slots=True)
class StudyInspectionSelection:
    """One selected prediction timestamp and its exact immutable provenance."""

    name: str
    context: PredictionRuleContext
    rule: PredictionScannerRule
    evaluation: TechnicalConfluenceEvaluation
    historical_study: HistoricalPredictionStudyReference
    dataset_family: DatasetFamily
    future_outcome: FutureOutcomeRegion | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.name), str) or not self.name.strip():
            raise StudyInspectionReportError("selection name is required")
        if not isinstance(cast(object, self.context), PredictionRuleContext):
            raise StudyInspectionReportError("selection context is invalid")
        if not isinstance(cast(object, self.evaluation), TechnicalConfluenceEvaluation):
            raise StudyInspectionReportError("selection evaluation is invalid")
        if not isinstance(
            cast(object, self.historical_study), HistoricalPredictionStudyReference
        ):
            raise StudyInspectionReportError(
                "selection historical study reference is invalid"
            )
        if not isinstance(cast(object, self.dataset_family), DatasetFamily):
            raise StudyInspectionReportError("selection dataset family is invalid")
        if self.future_outcome is not None and not isinstance(
            cast(object, self.future_outcome), FutureOutcomeRegion
        ):
            raise StudyInspectionReportError("selection future outcome is invalid")
        object.__setattr__(self, "name", self.name.strip())
        _validate_selection(self)


@dataclass(frozen=True, slots=True)
class StudyInspectionReport:
    """Complete deterministic report model with standalone HTML serialization."""

    selections: tuple[StudyInspectionSelection, ...]
    config: StudyInspectionReportConfig
    _identity_snapshot: PrimitiveMappingSnapshot = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.selections), tuple) or not self.selections:
            raise StudyInspectionReportError("report requires at least one selection")
        if any(
            not isinstance(item, StudyInspectionSelection)
            for item in cast(tuple[object, ...], self.selections)
        ):
            raise StudyInspectionReportError("report selections are invalid")
        if len({item.name for item in self.selections}) != len(self.selections):
            raise StudyInspectionReportError("report selection names must be unique")
        if not isinstance(cast(object, self.config), StudyInspectionReportConfig):
            raise StudyInspectionReportError("report configuration is invalid")
        for selection in self.selections:
            _validate_selection(selection)
        object.__setattr__(
            self,
            "_identity_snapshot",
            PrimitiveMappingSnapshot.capture(
                {
                    "schema_version": STUDY_INSPECTION_REPORT_SCHEMA_VERSION,
                    "artifact_type": "multi_timeframe_study_inspection_report",
                    "engine_version": STUDY_INSPECTION_REPORT_ENGINE_VERSION,
                    "configuration": self.config.to_primitive(),
                    "selections": [
                        _selection_primitive(selection, self.config)
                        for selection in self.selections
                    ],
                }
            ),
        )

    @property
    def report_id(self) -> str:
        return configuration_identity(self._identity_primitive())

    def _identity_primitive(self) -> PrimitiveMapping:
        return self._identity_snapshot.to_primitive()

    def manifest_primitive(self) -> PrimitiveMapping:
        return {"report_id": self.report_id, **self._identity_primitive()}

    def manifest_bytes(self) -> bytes:
        return (
            json.dumps(
                self.manifest_primitive(),
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

    def html_bytes(self) -> bytes:
        return _render_html(self.manifest_primitive()).encode("utf-8")


def build_study_inspection_report(
    selections: tuple[StudyInspectionSelection, ...],
    *,
    config: StudyInspectionReportConfig | None = None,
) -> StudyInspectionReport:
    """Build a report from exact contexts without recalculating any indicator."""
    return StudyInspectionReport(selections, config or StudyInspectionReportConfig())


def export_study_inspection_report(
    report: StudyInspectionReport, output_root: Path
) -> tuple[Path, StudyInspectionExportStatus]:
    """Create or byte-verify one content-addressed standalone report directory."""
    if not isinstance(cast(object, report), StudyInspectionReport):
        raise StudyInspectionReportError("report is invalid")
    expected = {
        "manifest.json": report.manifest_bytes(),
        "report.html": report.html_bytes(),
    }
    destination = output_root / report.report_id
    if destination.exists():
        _validate_existing_export(destination, expected)
        return destination, StudyInspectionExportStatus.REUSED
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{report.report_id}.", dir=str(output_root))
        )
        try:
            for filename, content in expected.items():
                path = temporary / filename
                path.write_bytes(content)
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            os.rename(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    except (OSError, TypeError, ValueError) as error:
        raise StudyInspectionReportError(
            "failed to export immutable study inspection report"
        ) from error
    return destination, StudyInspectionExportStatus.CREATED


def _validate_existing_export(destination: Path, expected: dict[str, bytes]) -> None:
    try:
        if destination.is_symlink() or not destination.is_dir():
            raise StudyInspectionReportError(
                "existing immutable report path is invalid"
            )
        entries = {entry.name: entry for entry in destination.iterdir()}
        if set(entries) != set(STUDY_INSPECTION_ARTIFACT_FILENAMES) or any(
            entry.is_symlink() or not entry.is_file() for entry in entries.values()
        ):
            raise StudyInspectionReportError(
                "existing immutable report file set is invalid"
            )
        if any(
            entries[name].read_bytes() != content for name, content in expected.items()
        ):
            raise StudyInspectionReportError(
                "existing immutable report content differs"
            )
    except StudyInspectionReportError:
        raise
    except OSError as error:
        raise StudyInspectionReportError(
            "failed to validate existing immutable report"
        ) from error


def _validate_selection(selection: StudyInspectionSelection) -> None:
    context = selection.context
    family = selection.dataset_family
    rule = selection.rule
    try:
        selection.historical_study.validate_rule(rule)
        selection.historical_study.validate_adjustment_basis(context.adjustment_basis)
        selection.historical_study.validate_symbol(context.symbol)
        live_rule_id = configuration_identity(rule.configuration())
        expected_evaluation = rule.evaluate(context)
    except (AttributeError, TypeError, ValueError) as error:
        raise StudyInspectionReportError(
            "selection rule or historical-study provenance is incompatible"
        ) from error
    if (
        context.requirements != rule.context_requirements
        or rule.configuration_id != live_rule_id
        or selection.evaluation != expected_evaluation
    ):
        raise StudyInspectionReportError(
            "selection evaluation does not match the exact current rule context"
        )
    if (
        context.symbol != family.canonical_symbol
        or context.adjustment_basis != family.adjustment_basis
    ):
        raise StudyInspectionReportError(
            "selection symbol or adjustment basis differs from its dataset family"
        )
    source_context = context.source_context_snapshot.to_primitive()
    consistency = _mapping(
        source_context.get("source_consistency"), "source consistency"
    )
    if (
        consistency.get("mode") != SourceConsistencyMode.COMMON_DATASET_FAMILY.value
        or consistency.get("family_id") != family.family_id
    ):
        raise StudyInspectionReportError(
            "all report panels must use one common dataset family"
        )
    timeframes = source_context.get("timeframes")
    if not isinstance(timeframes, list) or not timeframes:
        raise StudyInspectionReportError(
            "selection context has no dataset-family timeframe metadata"
        )
    for raw_timeframe in cast(list[object], timeframes):
        timeframe = _mapping(raw_timeframe, "context timeframe")
        raw_reference = timeframe.get("dataset_reference")
        if raw_reference is None:
            raise StudyInspectionReportError(
                "every rendered timeframe must have dataset-family lineage"
            )
        reference = _mapping(raw_reference, "dataset reference")
        dataset_id = reference.get("dataset_id")
        if not isinstance(dataset_id, str):
            raise StudyInspectionReportError("context dataset ID is invalid")
        expected = family.reference(dataset_id).to_primitive()
        if reference != expected:
            raise StudyInspectionReportError(
                "context dataset reference does not match the supplied family"
            )
    for timeframe_input in context.timeframes:
        if not timeframe_input.bars:
            raise StudyInspectionReportError("rendered timeframe has no bars")
        requirement_by_alias = {
            item.alias: item for item in timeframe_input.requirement.indicators
        }
        for named_output in timeframe_input.indicators:
            output = named_output.output
            requirement = requirement_by_alias.get(named_output.alias)
            if requirement is None:
                raise StudyInspectionReportError(
                    "indicator output is not declared by the timeframe"
                )
            requirement.validate_unchanged()
            expected_bound_configuration: PrimitiveMapping = {
                "component_type": "timeframe_indicator",
                "contract_version": TIMEFRAME_INDICATOR_CONTRACT_VERSION,
                "indicator": {
                    "configuration_id": requirement.configuration_id,
                    "configuration": requirement.indicator.configuration(),
                },
                "source": {
                    "timeframe": {
                        "configuration_id": (
                            timeframe_input.requirement.timeframe.configuration_id
                        ),
                        "configuration": (
                            timeframe_input.requirement.timeframe.to_primitive()
                        ),
                    },
                    "fields": [item.value for item in output.source_fields],
                    "completion_policy": output.completion_policy.value,
                    "developing_bar_support": (
                        requirement.developing_bar_support.value
                    ),
                    "observation_unit": "bar",
                    "warm_up_bars": output.warm_up_bars,
                    "aggregation_provenance": (
                        output.dataset_reference.to_primitive(include_feed_scope=True)
                    ),
                    "feed_scope": output.feed_scope.to_primitive(),
                },
            }
            if (
                output.configuration_id
                != configuration_identity(expected_bound_configuration)
                or output.indicator_name != requirement.indicator.name
                or output.source_timeframe != timeframe_input.requirement.timeframe
                or output.completion_policy
                is not timeframe_input.requirement.completion_policy
                or output.developing_bar_support
                is not requirement.developing_bar_support
                or output.source_fields
                != tuple(
                    sorted(
                        requirement.indicator.required_fields,
                        key=lambda item: item.value,
                    )
                )
                or output.warm_up_bars != requirement.indicator.warm_up_observations
                or output.bar_ids != tuple(bar.bar_id for bar in timeframe_input.bars)
                or output.bar_end_timestamps
                != tuple(bar.end_timestamp for bar in timeframe_input.bars)
                or output.completion_states
                != tuple(bar.completion for bar in timeframe_input.bars)
                or output.backend_identity != requirement.backend_identity
                or output.dataset_reference.family_id != family.family_id
                or output.dataset_reference
                != family.reference(output.dataset_reference.dataset_id)
                or tuple(field.name for field in output.fields)
                != requirement.indicator.output_fields
            ):
                raise StudyInspectionReportError(
                    "indicator values or provenance do not match their causal bars"
                )
    decision_timestamp = context.latest_bar_for(
        context.requirements.primary.timeframe
    ).end_timestamp
    if (
        selection.future_outcome is not None
        and selection.future_outcome.start_timestamp < decision_timestamp
    ):
        raise StudyInspectionReportError(
            "future outcome region cannot begin before the decision timestamp"
        )


def _mapping(value: object, field_name: str) -> PrimitiveMapping:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) for key in cast(dict[object, object], value)
    ):
        raise StudyInspectionReportError(f"{field_name} must be a JSON object")
    return cast(PrimitiveMapping, value)


def _selection_primitive(
    selection: StudyInspectionSelection, config: StudyInspectionReportConfig
) -> PrimitiveMapping:
    context = selection.context
    primary_id = context.requirements.primary.timeframe.configuration_id
    aliases = _timeframe_aliases(selection)
    references = _context_references(context, selection.dataset_family)
    inputs = tuple(
        item
        for item in sorted(
            context.timeframes,
            key=lambda item: _timeframe_sort_key(item.requirement.timeframe),
        )
        if config.include_primary_timeframe
        or item.requirement.timeframe.configuration_id != primary_id
    )
    panels: list[Primitive] = []
    for timeframe_input in inputs:
        timeframe = timeframe_input.requirement.timeframe
        bars = timeframe_input.bars[-config.max_bars_per_panel :]
        start_index = len(timeframe_input.bars) - len(bars)
        bar_rows: list[Primitive] = [
            {"bar_id": bar.bar_id, **bar.to_primitive()} for bar in bars
        ]
        indicators: list[Primitive] = []
        requirement_by_alias = {
            item.alias: item for item in timeframe_input.requirement.indicators
        }
        for named_output in timeframe_input.indicators:
            output = named_output.output
            requirement = requirement_by_alias[named_output.alias]
            rows = cast(list[Primitive], output.to_rows()[start_index:])
            indicators.append(
                {
                    "alias": named_output.alias,
                    "indicator_name": output.indicator_name,
                    "indicator_configuration_id": requirement.configuration_id,
                    "timeframe_indicator_configuration_id": output.configuration_id,
                    "backend": (
                        None
                        if output.backend_identity is None
                        else output.backend_identity.to_primitive()
                    ),
                    "source_fields": [item.value for item in output.source_fields],
                    "warm_up_bars": output.warm_up_bars,
                    "output_fields": [field.name for field in output.fields],
                    "rows": rows,
                }
            )
        panels.append(
            {
                "name": aliases.get(
                    timeframe.configuration_id,
                    f"primary_{_timeframe_label(timeframe)}",
                ),
                "timeframe": {
                    "configuration_id": timeframe.configuration_id,
                    "configuration": timeframe.to_primitive(),
                },
                "completion_policy": (
                    timeframe_input.requirement.completion_policy.value
                ),
                "dataset_reference": references[timeframe.configuration_id],
                "bars": bar_rows,
                "indicators": indicators,
            }
        )
    selection_value: PrimitiveMapping = {
        "name": selection.name,
        "symbol": context.symbol,
        "as_of": context.as_of.isoformat(),
        "decision_timestamp": context.latest_bar_for(
            context.requirements.primary.timeframe
        ).end_timestamp.isoformat(),
        "prediction_direction": selection.evaluation.outcome.value,
        "historical_study": selection.historical_study.to_primitive(),
        "rule": {
            "rule_id": selection.rule.name,
            "implementation_version": selection.rule.implementation_version,
            "configuration_id": selection.rule.configuration_id,
            "configuration": selection.rule.configuration(),
        },
        "evaluation": selection.evaluation.to_primitive(),
        "context": context.manifest_primitive(),
        "dataset_family": selection.dataset_family.to_manifest(),
        "panels": panels,
        "future_outcome": (
            None
            if selection.future_outcome is None
            else selection.future_outcome.to_primitive()
        ),
    }
    return {
        "selection_id": configuration_identity(selection_value),
        **selection_value,
    }


def _context_references(
    context: PredictionRuleContext, family: DatasetFamily
) -> dict[str, PrimitiveMapping]:
    source_context = context.source_context_snapshot.to_primitive()
    raw_timeframes = cast(list[Primitive], source_context["timeframes"])
    references: dict[str, PrimitiveMapping] = {}
    for raw_timeframe in raw_timeframes:
        timeframe = cast(PrimitiveMapping, raw_timeframe)
        requirement = cast(PrimitiveMapping, timeframe["requirement"])
        timeframe_value = cast(PrimitiveMapping, requirement["timeframe"])
        configuration_id = cast(str, timeframe_value["configuration_id"])
        reference = cast(PrimitiveMapping, timeframe["dataset_reference"])
        dataset_id = cast(str, reference["dataset_id"])
        references[configuration_id] = family.reference(dataset_id).to_primitive(
            include_feed_scope=True
        )
    return references


def _timeframe_aliases(selection: StudyInspectionSelection) -> dict[str, str]:
    aliases: dict[str, set[str]] = {}
    for result in selection.evaluation.condition_results:
        condition = result.condition
        aliases.setdefault(condition.timeframe.configuration_id, set()).add(
            condition.timeframe_name
        )
    return {
        configuration_id: " / ".join(sorted(timeframe_aliases))
        for configuration_id, timeframe_aliases in aliases.items()
    }


def _timeframe_sort_key(timeframe: Timeframe) -> tuple[int, int, str]:
    interval = timeframe.interval
    if isinstance(interval, TradingWeekInterval):
        return (0, -interval.week_count, timeframe.configuration_id)
    if isinstance(interval, SessionInterval):
        return (1, -interval.session_count, timeframe.configuration_id)
    duration = interval.nominal_duration
    return (2, -int(duration.total_seconds()), timeframe.configuration_id)


def _timeframe_label(timeframe: Timeframe) -> str:
    interval = timeframe.interval
    if isinstance(interval, TradingWeekInterval):
        return f"{interval.week_count}w"
    if isinstance(interval, SessionInterval):
        return f"{interval.session_count}d"
    seconds = int(interval.nominal_duration.total_seconds())
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"


def _render_html(manifest: PrimitiveMapping) -> str:
    selections = cast(list[PrimitiveMapping], manifest["selections"])
    selection_sections = "".join(_render_selection(item) for item in selections)
    embedded = (
        json.dumps(
            manifest,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    report_id = html.escape(cast(str, manifest["report_id"]))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantForge study inspection {report_id}</title>
<style>
:root{{--bg:#07111f;--panel:#101d2d;--panel2:#15263a;--text:#edf6ff;--muted:#9db2c7;--grid:#31465e;--up:#2dd4bf;--down:#fb7185;--developing:#fbbf24;--accent:#60a5fa;--future:#7f1d1d}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
main{{max-width:1280px;margin:auto;padding:24px}} h1,h2,h3{{font-family:ui-sans-serif,system-ui,sans-serif}} h1{{margin:0 0 6px}} h2{{margin-top:30px}} h3{{margin:18px 0 8px}}
.subtitle,.muted{{color:var(--muted)}} .selection,.panel,.meta,.future-card{{background:var(--panel);border:1px solid #263b52;border-radius:10px;padding:16px;margin:16px 0}}
.summary-grid,.meta-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px}} .pill{{background:var(--panel2);padding:8px 10px;border-radius:7px;overflow-wrap:anywhere}}
.direction-up{{color:var(--up)}} .direction-down{{color:var(--down)}} .direction-no_prediction{{color:var(--developing)}}
.chart{{width:100%;height:auto;background:#091522;border-radius:8px}} .grid{{stroke:var(--grid);stroke-width:1}} .wick{{stroke-width:1.4}} .completed-up{{fill:var(--up);stroke:var(--up)}} .completed-down{{fill:var(--down);stroke:var(--down)}} .developing{{fill:transparent;stroke:var(--developing);stroke-width:2;stroke-dasharray:4 3}} .decision{{stroke:var(--accent);stroke-width:2;stroke-dasharray:6 4}} .future{{fill:var(--future);opacity:.45}} .series{{fill:none;stroke-width:1.8}} .hist-positive{{fill:var(--up);opacity:.65}} .hist-negative{{fill:var(--down);opacity:.65}} .volume{{fill:#54708d;opacity:.7}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0;color:var(--muted)}} .legend span::before{{content:"";display:inline-block;width:12px;height:8px;margin-right:5px;border-radius:2px;background:currentColor}}
table{{width:100%;border-collapse:collapse;margin-top:8px}} th,td{{padding:7px;border-bottom:1px solid #263b52;text-align:left;vertical-align:top}} th{{color:#bcd0e4}} code{{color:#c4b5fd;overflow-wrap:anywhere}} details{{margin-top:10px}} .future-card{{border-color:#7f1d1d;background:#26141a}} .future-card strong{{color:#fca5a5}}
@media print{{body{{background:white;color:black}}.selection,.panel,.meta,.future-card{{break-inside:avoid;background:white;border-color:#aaa}}.chart{{background:#f7f7f7}}}}
</style>
</head>
<body><main>
<h1>QuantForge multi-timeframe study inspection</h1>
<div class="subtitle">Static, deterministic, research-only artifact. No web server or Python runtime is required to view it.</div>
<div class="pill"><strong>Report ID</strong><br><code>{report_id}</code></div>
{selection_sections}
<details class="meta"><summary>Complete embedded report metadata</summary><pre id="metadata-preview">Use the embedded JSON payload for exact machine-readable provenance.</pre></details>
<script type="application/json" id="quantforge-report-metadata">{embedded}</script>
</main></body></html>
"""


def _render_selection(selection: PrimitiveMapping) -> str:
    name = html.escape(cast(str, selection["name"]))
    symbol = html.escape(cast(str, selection["symbol"]))
    direction = cast(str, selection["prediction_direction"])
    study = cast(PrimitiveMapping, selection["historical_study"])
    rule = cast(PrimitiveMapping, selection["rule"])
    family = cast(PrimitiveMapping, selection["dataset_family"])
    canonical_source = cast(PrimitiveMapping, family["canonical_source"])
    feed = cast(PrimitiveMapping, canonical_source["feed_scope"])
    aggregation = cast(PrimitiveMapping, canonical_source["aggregation_policy"])
    panels = "".join(
        _render_panel(cast(PrimitiveMapping, panel), selection)
        for panel in cast(list[Primitive], selection["panels"])
    )
    conditions = _render_conditions(cast(PrimitiveMapping, selection["evaluation"]))
    future = selection["future_outcome"]
    future_html = (
        "" if future is None else _render_future(cast(PrimitiveMapping, future))
    )
    return f"""<section class="selection">
<h2>{name} — {symbol}</h2>
<div class="summary-grid">
<div class="pill"><strong>Decision</strong><br><code>{html.escape(cast(str, selection["decision_timestamp"]))}</code></div>
<div class="pill"><strong>As of</strong><br><code>{html.escape(cast(str, selection["as_of"]))}</code></div>
<div class="pill"><strong>Prediction</strong><br><span class="direction-{html.escape(direction)}">{html.escape(direction.upper())}</span></div>
<div class="pill"><strong>Study ID</strong><br><code>{html.escape(cast(str, study["study_id"]))}</code></div>
</div>
<div class="meta-grid">
<div class="pill"><strong>Rule</strong><br>{html.escape(cast(str, rule["rule_id"]))}<br><code>{html.escape(cast(str, rule["configuration_id"]))}</code></div>
<div class="pill"><strong>Dataset family</strong><br><code>{html.escape(cast(str, family["family_id"]))}</code></div>
<div class="pill"><strong>Feed scope</strong><br>{html.escape(_compact_json(feed))}</div>
<div class="pill"><strong>Aggregation policy</strong><br><code>{html.escape(cast(str, aggregation["configuration_id"]))}</code></div>
</div>
{panels}
{conditions}
{future_html}
</section>"""


def _render_panel(panel: PrimitiveMapping, selection: PrimitiveMapping) -> str:
    name = html.escape(cast(str, panel["name"]))
    bars = cast(list[PrimitiveMapping], panel["bars"])
    indicators = cast(list[PrimitiveMapping], panel["indicators"])
    price = _price_chart_svg(bars, indicators, selection["future_outcome"] is not None)
    volume = _volume_chart_svg(bars, indicators)
    oscillator_charts = "".join(_indicator_chart_svg(item) for item in indicators)
    metadata_rows = "".join(_indicator_metadata_row(item) for item in indicators)
    dataset_reference = panel["dataset_reference"]
    dataset_text = (
        "available in source context metadata"
        if dataset_reference is None
        else _compact_json(cast(PrimitiveMapping, dataset_reference))
    )
    return f"""<article class="panel">
<h3>{name}</h3>
<div class="muted">{len(bars)} exact causal bars · completion policy: {html.escape(cast(str, panel["completion_policy"]))}</div>
<div class="muted">dataset: <code>{html.escape(dataset_text)}</code></div>
<div class="legend"><span style="color:var(--up)">completed up</span><span style="color:var(--down)">completed down</span><span style="color:var(--developing)">developing as-of</span><span style="color:var(--accent)">decision boundary</span></div>
{price}
{volume}
{oscillator_charts}
<details><summary>Indicator and backend provenance</summary><table><thead><tr><th>Alias</th><th>Indicator config</th><th>Timeframe-bound config</th><th>Backend</th><th>Latest normalized values</th></tr></thead><tbody>{metadata_rows or '<tr><td colspan="5">No configured indicators on this optional primary panel.</td></tr>'}</tbody></table></details>
</article>"""


def _price_chart_svg(
    bars: list[PrimitiveMapping],
    indicators: list[PrimitiveMapping],
    has_future: bool,
) -> str:
    width, height = 1160, 300
    left, top, bottom = 64.0, 24.0, 36.0
    right = 160.0 if has_future else 40.0
    plot_right = width - right
    plot_bottom = height - bottom
    values = [_decimal(bar[field]) for bar in bars for field in ("low", "high")]
    overlay_series: list[tuple[str, list[Decimal | None]]] = []
    for indicator in indicators:
        rows = cast(list[PrimitiveMapping], indicator["rows"])
        for field_name in cast(list[str], indicator["output_fields"]):
            if field_name in _PRICE_OVERLAY_OUTPUTS:
                series = [_optional_decimal(row.get(field_name)) for row in rows]
                overlay_series.append((f"{indicator['alias']}.{field_name}", series))
                values.extend(value for value in series if value is not None)
    minimum, maximum = _padded_range(values)
    slot = (plot_right - left) / len(bars)

    def x(index: int) -> float:
        return left + slot * (index + 0.5)

    def y(value: Decimal) -> float:
        return top + float((maximum - value) / (maximum - minimum)) * (
            plot_bottom - top
        )

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Candlestick chart with exact source boundaries">',
        f'<line class="grid" x1="{left}" y1="{top}" x2="{left}" y2="{plot_bottom}"/>',
        f'<line class="grid" x1="{left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>',
        f'<text x="4" y="{top + 5}" fill="#9db2c7">{html.escape(str(maximum))}</text>',
        f'<text x="4" y="{plot_bottom}" fill="#9db2c7">{html.escape(str(minimum))}</text>',
    ]
    for index, bar in enumerate(bars):
        boundary_x = left + slot * index
        parts.append(
            f'<line class="grid" x1="{boundary_x:.2f}" y1="{top}" x2="{boundary_x:.2f}" y2="{plot_bottom}" opacity=".28"/>'
        )
        open_price = _decimal(bar["open"])
        high = _decimal(bar["high"])
        low = _decimal(bar["low"])
        close = _decimal(bar["close"])
        completion = cast(str, bar["completion"])
        candle_class = (
            "developing"
            if completion == BarCompletion.DEVELOPING.value
            else "completed-up"
            if close >= open_price
            else "completed-down"
        )
        center = x(index)
        body_top = min(y(open_price), y(close))
        body_height = max(1.5, abs(y(open_price) - y(close)))
        candle_width = max(2.0, slot * 0.55)
        start_timestamp = bar.get(
            "start_timestamp", bar.get("observed_start_timestamp")
        )
        end_timestamp = bar.get("end_timestamp", bar.get("observed_end_timestamp"))
        title = html.escape(
            f"{start_timestamp} → {end_timestamp} | "
            f"{completion} | O {open_price} H {high} L {low} C {close}"
        )
        parts.extend(
            (
                f'<g><title>{title}</title><line class="wick {candle_class}" x1="{center:.2f}" y1="{y(high):.2f}" x2="{center:.2f}" y2="{y(low):.2f}"/>',
                f'<rect class="{candle_class}" x="{center - candle_width / 2:.2f}" y="{body_top:.2f}" width="{candle_width:.2f}" height="{body_height:.2f}"/></g>',
            )
        )
    for series_index, (series_name, series) in enumerate(overlay_series):
        parts.extend(
            _polyline_parts(
                series,
                x,
                y,
                _PALETTE[series_index % len(_PALETTE)],
                series_name,
            )
        )
    parts.append(
        f'<line class="decision" x1="{plot_right:.2f}" y1="{top}" x2="{plot_right:.2f}" y2="{plot_bottom}"/><text x="{plot_right - 6:.2f}" y="16" fill="#60a5fa" text-anchor="end">DECISION</text>'
    )
    if has_future:
        parts.append(
            f'<rect class="future" x="{plot_right + 4:.2f}" y="{top}" width="{width - plot_right - 20:.2f}" height="{plot_bottom - top:.2f}"/><text x="{plot_right + 12:.2f}" y="{top + 18}" fill="#fca5a5">POST-DECISION</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _volume_chart_svg(
    bars: list[PrimitiveMapping], indicators: list[PrimitiveMapping]
) -> str:
    width, height = 1160, 150
    left, top, right, bottom = 64.0, 20.0, 40.0, 28.0
    plot_right, plot_bottom = width - right, height - bottom
    volumes = [_decimal(bar["volume"]) for bar in bars]
    overlays: list[tuple[str, list[Decimal | None]]] = []
    maximum_values = list(volumes)
    for indicator in indicators:
        rows = cast(list[PrimitiveMapping], indicator["rows"])
        for field_name in cast(list[str], indicator["output_fields"]):
            if field_name in _VOLUME_OVERLAY_OUTPUTS:
                series = [_optional_decimal(row.get(field_name)) for row in rows]
                overlays.append((f"{indicator['alias']}.{field_name}", series))
                maximum_values.extend(value for value in series if value is not None)
    maximum = max(maximum_values, default=Decimal(1))
    if maximum <= 0:
        maximum = Decimal(1)
    slot = (plot_right - left) / len(bars)

    def x(index: int) -> float:
        return left + slot * (index + 0.5)

    def y(value: Decimal) -> float:
        return top + float((maximum - value) / maximum) * (plot_bottom - top)

    parts = [
        f'<h3>Volume</h3><svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Canonical volume chart">',
        f'<line class="grid" x1="{left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>',
    ]
    for index, value in enumerate(volumes):
        bar_width = max(2.0, slot * 0.65)
        parts.append(
            f'<rect class="volume" x="{x(index) - bar_width / 2:.2f}" y="{y(value):.2f}" width="{bar_width:.2f}" height="{plot_bottom - y(value):.2f}"><title>{html.escape(str(value))} shares</title></rect>'
        )
    for series_index, (series_name, series) in enumerate(overlays):
        parts.extend(
            _polyline_parts(
                series,
                x,
                y,
                _PALETTE[(series_index + 3) % len(_PALETTE)],
                series_name,
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def _indicator_chart_svg(indicator: PrimitiveMapping) -> str:
    fields = [
        name
        for name in cast(list[str], indicator["output_fields"])
        if name not in _PRICE_OVERLAY_OUTPUTS and name not in _VOLUME_OVERLAY_OUTPUTS
    ]
    if not fields:
        return ""
    rows = cast(list[PrimitiveMapping], indicator["rows"])
    series = [
        (name, [_optional_decimal(row.get(name)) for row in rows]) for name in fields
    ]
    available = [value for _, values in series for value in values if value is not None]
    title = html.escape(f"{indicator['alias']} · {indicator['indicator_name']}")
    if not available:
        return f'<h3>{title}</h3><div class="muted">All displayed normalized values are unavailable during indicator warm-up.</div>'
    width, height = 1160, 180
    left, top, right, bottom = 64.0, 20.0, 40.0, 28.0
    plot_right, plot_bottom = width - right, height - bottom
    minimum, maximum = _padded_range(available)
    slot = (plot_right - left) / len(rows)

    def x(index: int) -> float:
        return left + slot * (index + 0.5)

    def y(value: Decimal) -> float:
        return top + float((maximum - value) / (maximum - minimum)) * (
            plot_bottom - top
        )

    parts = [
        f'<h3>{title}</h3><svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Normalized indicator chart">',
        f'<line class="grid" x1="{left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>',
    ]
    zero_y = y(Decimal(0)) if minimum <= 0 <= maximum else None
    if zero_y is not None:
        parts.append(
            f'<line class="grid" x1="{left}" y1="{zero_y:.2f}" x2="{plot_right}" y2="{zero_y:.2f}"/>'
        )
    for series_index, (field_name, values) in enumerate(series):
        if field_name == "histogram" and zero_y is not None:
            bar_width = max(2.0, slot * 0.55)
            for index, value in enumerate(values):
                if value is None:
                    continue
                value_y = y(value)
                parts.append(
                    f'<rect class="{"hist-positive" if value >= 0 else "hist-negative"}" x="{x(index) - bar_width / 2:.2f}" y="{min(value_y, zero_y):.2f}" width="{bar_width:.2f}" height="{max(1.0, abs(value_y - zero_y)):.2f}"><title>{html.escape(field_name)} {html.escape(str(value))}</title></rect>'
                )
        else:
            parts.extend(
                _polyline_parts(
                    values,
                    x,
                    y,
                    _PALETTE[series_index % len(_PALETTE)],
                    field_name,
                )
            )
    parts.append("</svg>")
    return "".join(parts)


def _polyline_parts(
    values: list[Decimal | None],
    x: Callable[[int], float],
    y: Callable[[Decimal], float],
    color: str,
    label: str,
) -> list[str]:
    parts: list[str] = []
    segment: list[str] = []
    for index, value in enumerate(values):
        if value is None:
            if len(segment) >= 2:
                parts.append(
                    f'<polyline class="series" stroke="{color}" points="{" ".join(segment)}"><title>{html.escape(label)}</title></polyline>'
                )
            segment = []
            continue
        segment.append(f"{x(index):.2f},{y(value):.2f}")
    if len(segment) >= 2:
        parts.append(
            f'<polyline class="series" stroke="{color}" points="{" ".join(segment)}"><title>{html.escape(label)}</title></polyline>'
        )
    return parts


def _indicator_metadata_row(indicator: PrimitiveMapping) -> str:
    rows = cast(list[PrimitiveMapping], indicator["rows"])
    latest = rows[-1]
    values = {
        field: latest.get(field)
        for field in cast(list[str], indicator["output_fields"])
    }
    backend = indicator["backend"]
    backend_text = (
        "not applicable"
        if backend is None
        else _compact_json(cast(PrimitiveMapping, backend))
    )
    return f"""<tr><td>{html.escape(cast(str, indicator["alias"]))}</td>
<td><code>{html.escape(cast(str, indicator["indicator_configuration_id"]))}</code></td>
<td><code>{html.escape(cast(str, indicator["timeframe_indicator_configuration_id"]))}</code></td>
<td>{html.escape(backend_text)}</td><td><code>{html.escape(_compact_json(values))}</code></td></tr>"""


def _render_conditions(evaluation: PrimitiveMapping) -> str:
    rows: list[str] = []
    for result_value in cast(list[Primitive], evaluation["condition_results"]):
        result = cast(PrimitiveMapping, result_value)
        condition = cast(PrimitiveMapping, result["condition"])
        rows.append(
            f"<tr><td>{html.escape(cast(str, condition['name']))}</td>"
            f"<td>{html.escape(cast(str, condition['timeframe_name']))}</td>"
            f"<td>{html.escape(cast(str, result['status']))}</td>"
            f"<td>{html.escape(str(result['left_value']))}</td>"
            f"<td>{html.escape(str(result['right_value']))}</td>"
            f"<td><code>{html.escape(str(result['source_timestamp']))}</code></td></tr>"
        )
    return (
        """<h3>Prediction rule conditions</h3>
<table><thead><tr><th>Condition</th><th>Timeframe</th><th>Status</th><th>Left</th><th>Right</th><th>Exact source timestamp</th></tr></thead><tbody>"""
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_future(future: PrimitiveMapping) -> str:
    return f"""<aside class="future-card"><strong>POST-DECISION OUTCOME — NOT AVAILABLE AT DECISION TIME</strong><br>
{html.escape(cast(str, future["label"]))}<br>
<code>{html.escape(cast(str, future["start_timestamp"]))} → {html.escape(cast(str, future["end_timestamp"]))}</code>
<details><summary>Outcome-only values</summary><code>{html.escape(_compact_json(cast(PrimitiveMapping, future["values"]) if future["values"] is not None else {}))}</code></details></aside>"""


def _decimal(value: Primitive) -> Decimal:
    try:
        result = Decimal(cast(str | int | float, value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise StudyInspectionReportError("chart value is not a decimal") from error
    if not result.is_finite():
        raise StudyInspectionReportError("chart value must be finite")
    return result


def _optional_decimal(value: Primitive | None) -> Decimal | None:
    return None if value is None else _decimal(value)


def _padded_range(values: list[Decimal]) -> tuple[Decimal, Decimal]:
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        padding = max(abs(minimum) * Decimal("0.01"), Decimal("1"))
    else:
        padding = (maximum - minimum) * Decimal("0.05")
    return minimum - padding, maximum + padding


def _compact_json(value: PrimitiveMapping) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "STUDY_INSPECTION_ARTIFACT_FILENAMES",
    "STUDY_INSPECTION_REPORT_ENGINE_VERSION",
    "STUDY_INSPECTION_REPORT_SCHEMA_VERSION",
    "FutureOutcomeRegion",
    "StudyInspectionExportStatus",
    "StudyInspectionReport",
    "StudyInspectionReportConfig",
    "StudyInspectionReportError",
    "StudyInspectionSelection",
    "build_study_inspection_report",
    "export_study_inspection_report",
]
