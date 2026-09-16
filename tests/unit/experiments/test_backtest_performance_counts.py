"""Headline backtest sample counts must agree with saved record counts."""

from pathlib import Path
from typing import cast

import pytest

from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._json import mapping
from tests.unit.experiments.test_backtest_manifest_schema import inspect_manifest
from tests.unit.experiments.test_backtest_table_counts import export as export
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record


@pytest.mark.parametrize(
    "field",
    [
        "trade_count",
        "open_trade_count",
        "winning_trades",
        "losing_trades",
        "joint_bounds",
    ],
)
def test_performance_counts_cannot_contradict_captured_records(
    export: Path, field: str
) -> None:
    manifest = read_record(export / "manifest.json")
    performance = mapping(manifest["performance"])
    completed = cast(int, mapping(manifest["record_counts"])["completed_trades"])
    if field == "joint_bounds":
        performance["winning_trades"] = completed
        performance["losing_trades"] = completed
    else:
        performance[field] = (
            completed + cast(int, mapping(manifest["record_counts"])["open_trades"]) + 1
        )
    inspect_manifest(export, manifest)
    detached = export.parent / "detached.json"
    write_json(detached, manifest)
    with pytest.raises(ManifestError, match="performance trade counts"):
        inspect_study(StudyType.BACKTEST, detached, artifact_root=export.parent)
