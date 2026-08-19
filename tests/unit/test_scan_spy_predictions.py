import json
import subprocess
import sys
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPOSITORY_ROOT / "examples" / "spy_multi_timeframe" / "fixture.json"


def test_custom_fixture_supplies_default_as_of(tmp_path: Path) -> None:
    fixture = cast(PrimitiveMapping, json.loads(FIXTURE_PATH.read_text()))
    scenarios = cast(list[PrimitiveMapping], fixture["decision_scenarios"])
    midweek = next(item for item in scenarios if item["name"] == "midweek")
    midweek["as_of"] = "2024-07-10T11:55:00-04:00"
    custom_fixture = tmp_path / "custom-fixture.json"
    custom_fixture.write_text(json.dumps(fixture), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/scan_spy_predictions.py",
            "--fixture",
            str(custom_fixture),
            "--cache-root",
            str(tmp_path / "cache"),
            "--alert-root",
            str(tmp_path / "alerts"),
            "--state-root",
            str(tmp_path / "state"),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = cast(PrimitiveMapping, json.loads(completed.stderr))

    assert summary["as_of"] == "2024-07-10T15:55:00+00:00"
