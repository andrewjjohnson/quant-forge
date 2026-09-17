"""Escaped, script-free HTML fragments; research values are never recalculated."""

# Readable static CSS/HTML literals.
# ruff: noqa: E501

import os
from html import escape
from pathlib import Path
from urllib.parse import quote

from quantforge.configuration import Primitive
from quantforge.reporting.research_models import ReportPhase, ResearchReport

STYLE = """
:root { color-scheme: light; font-family: system-ui, sans-serif; color: #172c40; background: #f3f5f7; }
* { box-sizing: border-box; } body { margin: 0; } main { max-width: 1280px; padding: 32px; margin: auto; }
header { border-top: 5px solid #247b76; padding: 28px; background: #102b3f; color: white; border-radius: 8px; }
h1 { margin: 6px 0 16px; font-size: 30px; } h2 { font-size: 20px; margin: 12px 0; } h3 { font-size: 15px; }
p { line-height: 1.6; } a { color: #166866; overflow-wrap: anywhere; } header a { color: #98e6de; }
.eyebrow { text-transform: uppercase; letter-spacing: .15em; font-size: 12px; } .muted { color: #526578; }
.phase { display: inline-block; font-weight: 700; font-size: 12px; padding: 5px 9px; border-radius: 4px; background: #e3e9ef; }
.oos .phase { background: #d7eee8; color: #155d4e; } .holdout .phase { background: #f3e5cb; color: #684918; }
.in_sample .phase { background: #e5e0f0; color: #514270; } .selection .phase { background: #dce8f8; color: #264f81; }
section { padding: 22px; margin-top: 18px; background: white; border: 1px solid #d9e1e7; border-radius: 8px; min-width: 0; }
.warnings { border-left: 5px solid #b57524; } .warnings li { padding: 5px 0; }
.unavailable { font-style: italic; color: #6b5860; } .status { font-weight: 700; } .failed { color: #a13232; }
.scroll { max-width: 100%; overflow-x: auto; } table { border-collapse: collapse; width: 100%; font-size: 13px; }
td, th { border-bottom: 1px solid #e1e7ec; padding: 9px; text-align: left; vertical-align: top; min-width: 110px; max-width: 420px; overflow-wrap: anywhere; }
th { color: #526578; background: #f4f7f9; font-weight: 600; } dl { display: grid; grid-template-columns: minmax(130px, 1fr) minmax(0, 3fr); gap: 0; margin: 8px 0; }
dt, dd { margin: 0; padding: 8px; border-bottom: 1px solid #edf0f3; overflow-wrap: anywhere; } dt { color: #526578; }
header dt { color: #b3c9d8; } code { font-size: 12px; overflow-wrap: anywhere; } summary { cursor: pointer; padding: 8px 0; font-weight: 600; }
nav { display: flex; flex-wrap: wrap; gap: 14px; margin: 20px 0; } .sources { font-size: 12px; margin-top: 14px; }
footer { color: #526578; font-size: 12px; margin: 24px 0; }
@media(max-width: 720px) { main { padding: 12px; } section, header { padding: 16px; } dl { grid-template-columns: 1fr; } dt { font-weight: 600; } }
@media print { body { background: white; } main { padding: 0; } .scroll { overflow: visible; } section { break-inside: avoid; } }
"""


def label(text: str) -> str:
    return escape(text.replace("_", " "))


def render_value(value: Primitive, limit: int, depth: int = 0) -> str:
    if value is None:
        return '<span class="unavailable">Unavailable — not supplied by the indexed artifact</span>'
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, dict):
        if not value:
            return '<span class="muted">Empty object recorded</span>'
        body = (
            "<dl>"
            + "".join(
                f"<dt>{label(key)}</dt><dd>{render_value(child, limit, depth + 1)}</dd>"
                for key, child in value.items()
            )
            + "</dl>"
        )
        return (
            f"<details><summary>Recorded fields</summary>{body}</details>"
            if depth > 1
            else body
        )
    if isinstance(value, list):
        if not value:
            return '<span class="muted">None recorded (empty list)</span>'
        preview = value[:limit]
        if all(isinstance(row, dict) for row in preview):
            rows = [row for row in preview if isinstance(row, dict)]
            keys = list(dict.fromkeys(key for row in rows for key in row))
            body = (
                '<div class="scroll"><table><thead><tr>'
                + "".join(f"<th>{label(key)}</th>" for key in keys)
                + "</tr></thead><tbody>"
                + "".join(
                    "<tr>"
                    + "".join(
                        f"<td>{render_value(row.get(key), limit, depth + 1)}</td>"
                        for key in keys
                    )
                    + "</tr>"
                    for row in rows
                )
                + "</tbody></table></div>"
            )
        else:
            body = (
                "<ol>"
                + "".join(
                    f"<li>{render_value(child, limit, depth + 1)}</li>"
                    for child in preview
                )
                + "</ol>"
            )
        if len(value) > limit:
            body += f'<p class="muted">Preview limited to the first {limit} records in published order. Open the source artifact for all records.</p>'
        return body
    return escape(str(value))


def local_link(root: Path, location: str, output: Path) -> str:
    relative = os.path.relpath(root / location, output.parent)
    # Prefix prevents a filename such as javascript:... from becoming a URI scheme.
    return escape("./" + quote(relative, safe="/"), quote=True)


def render_html(report: ResearchReport, output: Path) -> str:
    root = Path(report.artifact_root)
    header = report.header.to_primitive()
    limit = report.config.maximum_preview_rows
    title = str(header["study_type"]).replace("_", " ").title() + " research report"
    holdout_label = {
        "consumed": "FINAL HOLDOUT: CONSUMED — never unseen",
        "reserved_unconsumed": "FINAL HOLDOUT: RESERVED / UNSEEN at this snapshot",
        "unavailable": "FINAL HOLDOUT: CURRENT STATE UNAVAILABLE",
    }[str(header["holdout_state"])]
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">",
        f"<title>{escape(title)}</title><style>{STYLE}</style></head><body><main>",
        f'<header><div class="eyebrow">QuantForge · Static research evidence</div><h1>{escape(title)}</h1>',
        f"<p><strong>{escape(holdout_label)}</strong></p>",
        render_value(header, limit),
        f'<p><a href="{local_link(root, report.manifest_path, output)}">Exact QF-9 manifest</a></p>',
        "<p>Values and phase labels describe the supplied evidence. This static snapshot does not update when artifacts or holdout state change.</p></header>",
        '<nav><a href="#warnings">Research warnings</a><a href="#results">Study results</a><a href="#final-holdout">Final holdout</a><a href="#artifacts">Artifacts and integrity</a></nav>',
        '<section id="warnings" class="warnings"><h2>Research-integrity warnings</h2>',
    ]
    if report.warnings:
        parts.append(
            "<ul>"
            + "".join(
                f"<li><strong>{escape(warning.code)}</strong> — {escape(warning.message)}<br><code>{escape(warning.source)}</code></li>"
                for warning in report.warnings
            )
            + "</ul>"
        )
    else:
        parts.append(
            "<p>No configured warning condition was triggered. This is not validation of a trading edge.</p>"
        )
    parts.append(
        "<details><summary>Warning thresholds and preview configuration</summary>"
        + render_value(report.config.to_primitive(), limit)
        + "</details></section>"
    )
    for index, section in enumerate(report.sections):
        anchor = (
            ' id="results"'
            if index == 0
            else ' id="final-holdout"'
            if section.title == "Final holdout state"
            else ""
        )
        parts.append(
            f'<section class="{section.phase.name.lower()}"{anchor}><span class="phase">{escape(section.phase.value)}</span><h2>{escape(section.title)}</h2>'
        )
        content = render_value(section.content.to_primitive()["value"], limit)
        if section.phase is ReportPhase.PROVENANCE:
            content = (
                "<details><summary>Inspect exact provenance</summary>"
                + content
                + "</details>"
            )
        parts.append(content)
        if section.artifact_ids:
            parts.append(
                '<div class="sources">Indexed sources: '
                + ", ".join(
                    f'<a href="#artifact-{escape(identifier)}"><code>{escape(identifier)}</code></a>'
                    for identifier in section.artifact_ids
                )
                + "</div>"
            )
        parts.append("</section>")
    parts.append(
        '<section id="artifacts"><h2>Artifact index and integrity</h2><p>Only verified artifacts supply displayed values. Unavailable or invalid references remain visible. Source hashes and records are never repaired.</p><div class="scroll"><table><thead><tr><th>Type / schema</th><th>File / JSON pointer</th><th>Integrity</th><th>SHA-256 / identity</th></tr></thead><tbody>'
    )
    for item in report.artifacts:
        entry = item.entry
        filename = escape(entry.path)
        if item.status == "verified":
            filename = (
                f'<a href="{local_link(root, entry.path, output)}">{filename}</a>'
            )
        status_class = "status" if item.status == "verified" else "status failed"
        parts.append(
            f'<tr id="artifact-{entry.artifact_id}"><td>{escape(entry.artifact_type.value)}<br>Schema {escape(entry.schema_version)} · {entry.file_format.value}</td><td>{filename}<br><code>{escape(entry.json_pointer)}</code></td><td class="{status_class}">{escape(item.status)}</td><td><code>{escape(entry.sha256 or "Unavailable")}</code><details><summary>Artifact / producer IDs</summary><code>{entry.artifact_id}<br>{escape(entry.producer_study_id)}<br>{escape(entry.producer_artifact_id)}<br>{escape(entry.producer_run_id or "Unavailable")}</code></details></td></tr>'
        )
    parts.append(
        "</tbody></table></div></section><footer>Report "
        + escape(report.report_id)
        + " · Presentation only. No indicators, labels, research metrics, rankings, execution, or charts were computed. Keep the source artifact layout with this report to preserve local links.</footer></main></body></html>\n"
    )
    return "".join(parts)
