"""Rehashed execution identities cannot certify an unusable position-sizing rule."""

from pathlib import Path

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments._json import mapping
from quantforge.experiments._producer_integrity import validate_backtest_identity
from tests.unit.experiments.test_backtest_manifest_schema import (
    export as export,
)
from tests.unit.experiments.test_backtest_manifest_schema import inspect_manifest
from tests.unit.experiments.test_grid_integrity import read_record


def reidentify(manifest: PrimitiveMapping) -> str:
    market = mapping(manifest["market_data"])
    strategy = mapping(manifest["strategy"])
    run_id = configuration_identity(
        {
            "component": "quantforge_backtest",
            "engine_version": manifest["engine_version"],
            "result_schema_version": manifest["result_schema_version"],
            "market_data": {
                key: market[key]
                for key in (
                    "dataset_id",
                    "schema_version",
                    "adjustment_mode",
                    "calendar",
                    "corporate_action_snapshot_id",
                    "bars_fingerprint",
                )
            },
            "strategy": {
                key: strategy[key]
                for key in (
                    "strategy_id",
                    "strategy_implementation_version",
                    "strategy_configuration_id",
                    "configuration",
                )
            },
            "backtest_configuration": manifest["backtest_configuration"],
        }
    )
    manifest["run_id"] = run_id
    benchmark = mapping(manifest["benchmark"])
    benchmark["benchmark_id"] = configuration_identity(
        {
            "run_id": run_id,
            "record_type": "benchmark",
            "configuration": benchmark["configuration"],
        }
    )
    return run_id


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        (None, {}),
        (None, None),
        (None, []),
        (None, "discrete_target_weight"),
        ("model", "unsupported"),
        ("model", None),
        ("whole_shares_only", False),
        ("rebalance_existing_position", True),
        *(
            (field, value)
            for field in ("whole_shares_only", "rebalance_existing_position")
            for value in (None, 0, 1, "true", list[Primitive](), dict[str, Primitive]())
        ),
        ("undeclared", None),
    ],
)
def test_rehashed_export_requires_complete_supported_sizing(
    export: Path, field: str | None, invalid: Primitive
) -> None:
    manifest = read_record(export / "manifest.json")
    configuration = mapping(manifest["backtest_configuration"])
    if field is None:
        configuration["sizing"] = invalid
    else:
        mapping(configuration["sizing"])[field] = invalid
    renamed = export.rename(export.with_name(reidentify(manifest)))
    validate_backtest_identity(manifest)
    inspect_manifest(renamed, manifest)


@pytest.mark.parametrize(
    "field", ["model", "whole_shares_only", "rebalance_existing_position"]
)
def test_sizing_fields_cannot_be_omitted(export: Path, field: str) -> None:
    manifest = read_record(export / "manifest.json")
    del mapping(mapping(manifest["backtest_configuration"])["sizing"])[field]
    renamed = export.rename(export.with_name(reidentify(manifest)))
    validate_backtest_identity(manifest)
    inspect_manifest(renamed, manifest)


def test_native_sizing_remains_unchanged(export: Path) -> None:
    manifest = read_record(export / "manifest.json")
    assert reidentify(manifest) == export.name
    inspect_manifest(export, manifest, reject=False)
