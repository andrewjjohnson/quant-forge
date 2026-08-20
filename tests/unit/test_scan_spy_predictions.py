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


def test_completed_bars_policy_uses_matching_committed_study_record(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/scan_spy_predictions.py",
            "--completion-policy",
            "completed_bars_only",
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

    assert summary["completion_policy"] == "completed_bars_only"
    assert summary["historical_study_id"] == (
        "qf33_spy_fixture_completed_bars_parity_study_v1"
    )


def test_cache_seed_and_replay_produce_identical_alert_artifacts(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"

    def run(scan_name: str) -> tuple[PrimitiveMapping, bytes]:
        alert_root = tmp_path / f"{scan_name}-alerts"
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/scan_spy_predictions.py",
                "--cache-root",
                str(cache_root),
                "--alert-root",
                str(alert_root),
                "--state-root",
                str(tmp_path / f"{scan_name}-state"),
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        artifacts = tuple(alert_root.glob("*.json"))
        assert len(artifacts) == 1
        return (
            cast(PrimitiveMapping, json.loads(completed.stderr)),
            artifacts[0].read_bytes(),
        )

    seeded_summary, seeded_artifact = run("seeded")
    replayed_summary, replayed_artifact = run("replayed")

    assert seeded_summary["cache_status"] == (
        "seeded_local_cache_from_committed_fixture"
    )
    assert replayed_summary["cache_status"] == "replayed_local_cache"
    assert seeded_artifact == replayed_artifact
    alert = cast(PrimitiveMapping, json.loads(seeded_artifact))
    provenance = cast(PrimitiveMapping, alert["provenance"])
    assert "source_mode" not in provenance
