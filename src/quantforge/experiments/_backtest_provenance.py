"""Reconcile recorded QF-5 execution assumptions without simulating a benchmark."""

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping


def benchmark_configuration(manifest: PrimitiveMapping) -> PrimitiveMapping:
    """Check the producer's complete execution and derived benchmark metadata."""
    configuration = mapping(manifest.get("backtest_configuration"))
    for key in (
        "initial_capital",
        "commission",
        "fees",
        "slippage",
        "execution",
        "dividend_policy",
        "split_policy",
        "sizing",
    ):
        if key not in configuration or configuration[key] is None:
            raise ManifestError("backtest execution provenance is incomplete")
    sizing = mapping(configuration["sizing"])
    if configuration_identity(sizing) != configuration_identity(
        {
            "model": "discrete_target_weight",
            "whole_shares_only": True,
            "rebalance_existing_position": False,
        }
    ):
        raise ManifestError(
            "backtest position-sizing configuration is unsupported or incomplete"
        )
    # QF-5 benchmark version 4's fixed metadata and original recorded inputs.
    # Do not construct BacktestConfig: its defaults could rewrite historical inputs.
    expected: PrimitiveMapping = {
        "model": "buy_and_hold",
        "implementation_version": "4",
        "start": "first_dataset_session_open",
        "return_series_start": "initial_capital_to_first_session_close",
        "forced_liquidation": False,
        **{
            key: configuration[key]
            for key in (
                "initial_capital",
                "commission",
                "fees",
                "slippage",
                "dividend_policy",
                "split_policy",
            )
        },
        "corporate_action_snapshot_id": mapping(manifest.get("market_data"))[
            "corporate_action_snapshot_id"
        ],
    }
    if configuration.get("evaluation_interval") is not None:
        expected["start"] = "first_evaluation_session_open"
        expected["evaluation_interval"] = configuration["evaluation_interval"]
    benchmark = mapping(manifest.get("benchmark"))
    expected_id = configuration_identity(
        {
            "run_id": manifest.get("run_id"),
            "record_type": "benchmark",
            "configuration": expected,
        }
    )
    recorded = mapping(benchmark.get("configuration"))
    if (
        configuration_identity(recorded) != configuration_identity(expected)
        or benchmark.get("benchmark_id") != expected_id
    ):
        raise ManifestError(
            "backtest benchmark configuration or identity is inconsistent"
        )
    return recorded
