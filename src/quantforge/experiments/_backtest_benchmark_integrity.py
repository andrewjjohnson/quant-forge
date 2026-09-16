"""Bind saved benchmark execution records to their fixed QF-5 provenance."""

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping


def validate_benchmark_execution(manifest: PrimitiveMapping) -> None:
    """Check identities and links without calculating quantities, costs or fills."""
    benchmark = mapping(manifest.get("benchmark"))
    benchmark_id = benchmark.get("benchmark_id")
    identifiers = {
        kind: configuration_identity(
            {"benchmark_id": benchmark_id, "record_type": kind}
        )
        for kind in ("signal", "order", "fill")
    }
    common = {
        "order_id": identifiers["order"],
        "originating_signal_id": identifiers["signal"],
        "symbol": mapping(manifest.get("market_data"))["canonical_symbol"],
        "side": "buy",
        "strategy_id": "buy_and_hold_benchmark",
        "strategy_configuration_id": benchmark_id,
    }
    order = mapping(benchmark.get("order"))
    quantity = order.get("requested_quantity")
    expected_order = {
        **common,
        "run_id": manifest.get("run_id"),
        "order_type": "market",
        "target_position": "long",
        "target_weight": "1",
        "signal_session": order.get("decision_session"),
        "earliest_permitted_execution_session": order.get("decision_session"),
    }
    if any(order.get(key) != value for key, value in expected_order.items()):
        raise ManifestError("backtest benchmark order provenance is inconsistent")
    fill = benchmark.get("fill")
    if fill is None:
        if (
            type(quantity) is not int
            or quantity != 0
            or order.get("status") != "rejected"
            or order.get("reason") != "insufficient_cash_for_one_share"
        ):
            raise ManifestError("backtest benchmark rejection is inconsistent")
        return
    expected_fill = {
        **common,
        "fill_id": identifiers["fill"],
        "quantity": quantity,
        "execution_session": order.get("decision_session"),
    }
    if (
        type(quantity) is not int
        or quantity < 1
        or order.get("status") != "filled"
        or order.get("reason") is not None
        or any(mapping(fill).get(key) != value for key, value in expected_fill.items())
    ):
        raise ManifestError("backtest benchmark fill linkage is inconsistent")
