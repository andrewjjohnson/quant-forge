import json
import subprocess
import sys
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_spy_inspection_script_creates_and_reuses_static_report(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    output_root = tmp_path / "reports"
    command = [
        sys.executable,
        "scripts/render_spy_study_inspection.py",
        "--cache-root",
        str(cache_root),
        "--output-root",
        str(output_root),
        "--max-bars-per-panel",
        "12",
    ]

    first = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    first_summary = cast(PrimitiveMapping, json.loads(first.stderr))
    second_summary = cast(PrimitiveMapping, json.loads(second.stderr))

    assert first_summary["status"] == "created_immutable_report"
    assert second_summary["status"] == "reused_immutable_report"
    assert first_summary["report_id"] == second_summary["report_id"]
    assert first_summary["prediction_direction"] == "up"
    report_path = Path(cast(str, first_summary["report_path"]))
    assert report_path.is_file()
    assert report_path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_spy_inspection_script_uses_matching_completed_bars_study(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/render_spy_study_inspection.py",
            "--cache-root",
            str(tmp_path / "cache"),
            "--output-root",
            str(tmp_path / "reports"),
            "--completion-policy",
            "completed_bars_only",
            "--exclude-primary-timeframe",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = cast(PrimitiveMapping, json.loads(completed.stderr))

    assert summary["completion_policy"] == "completed_bars_only"
    assert summary["historical_study_id"] == (
        "qf33_spy_fixture_completed_bars_parity_study_v1"
    )
