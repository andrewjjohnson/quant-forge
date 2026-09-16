"""Validate saved signal/order/fill/trade identities and direct record links."""

from decimal import Decimal

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text


def _same(row: PrimitiveMapping, expected: PrimitiveMapping, label: str) -> None:
    if any(row.get(name) != value for name, value in expected.items()):
        raise ManifestError(f"backtest {label} provenance or linkage is inconsistent")


def _indexed(rows: list[PrimitiveMapping], key: str) -> dict[str, PrimitiveMapping]:
    result = {text(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise ManifestError(f"backtest duplicate {key}")
    return result


def validate_backtest_records(
    manifest: PrimitiveMapping, tables: dict[str, list[PrimitiveMapping]]
) -> None:
    """Compare persisted evidence; do not recreate signals, fills or trade P&L."""
    run_id = manifest["run_id"]
    strategy = mapping(manifest["strategy"])
    symbol = mapping(manifest["market_data"])["canonical_symbol"]
    common: PrimitiveMapping = {
        "symbol": symbol,
        "strategy_id": strategy["strategy_id"],
        "strategy_configuration_id": strategy["strategy_configuration_id"],
    }
    signals = tables["signals"]
    for ordinal, signal in enumerate(signals):
        if "parameters" in mapping(strategy["configuration"]):
            _same(
                signal,
                {
                    "strategy_parameters": mapping(strategy["configuration"])[
                        "parameters"
                    ]
                },
                "signal parameters",
            )
        _same(
            signal,
            {
                "canonical_symbol": symbol,
                **{key: value for key, value in common.items() if key != "symbol"},
            },
            "signal",
        )
        decision = {key: value for key, value in signal.items() if key != "signal_id"}
        identity = {
            "run_id": run_id,
            "record_type": "signal",
            "ordinal": ordinal,
            "decision": decision,
        }
        if configuration_identity(identity) != signal["signal_id"]:
            if decision["reason"] is None:
                decision["reason"] = ""
            if configuration_identity(identity) != signal["signal_id"]:
                raise ManifestError("backtest signal identity is inconsistent")
            signal["reason"] = ""
        weight = Decimal(text(signal["target_weight"]))
        target = signal["target_position"]
        eligible = signal["earliest_executable_session"]
        if (
            signal["execution_timing"] != "next_session_after_close"
            or target not in {"long", "flat"}
            or not 0 <= weight <= 1
            or (target == "flat" and weight != 0)
            or (target == "long" and weight <= 0)
            or (eligible is None and signal["execution_session_status"] != "unresolved")
            or (
                eligible is not None
                and (
                    text(eligible) <= text(signal["signal_session"])
                    or signal["execution_session_status"] != "pending"
                )
            )
        ):
            raise ManifestError("backtest signal execution contract is inconsistent")
    signal_index = _indexed(signals, "signal_id")
    orders = tables["orders"]
    if [order["originating_signal_id"] for order in orders] != [
        signal["signal_id"] for signal in signals
    ]:
        raise ManifestError("backtest orders differ from captured signals")
    for order in orders:
        signal = signal_index[text(order["originating_signal_id"])]
        _same(
            order,
            {
                **common,
                "run_id": run_id,
                "order_id": configuration_identity(
                    {
                        "run_id": run_id,
                        "record_type": "order",
                        "signal_id": signal["signal_id"],
                    }
                ),
                "side": "buy" if signal["target_position"] == "long" else "sell",
                "signal_session": signal["signal_session"],
                "decision_session": signal["signal_session"],
                "earliest_permitted_execution_session": signal[
                    "earliest_executable_session"
                ],
                "target_position": signal["target_position"],
                "target_weight": signal["target_weight"],
                "order_type": "market",
            },
            "order",
        )
        if order["status"] == "filled":
            if (
                type(order["requested_quantity"]) is not int
                or order["requested_quantity"] < 1
                or order["reason"] is not None
            ):
                raise ManifestError("backtest filled order is inconsistent")
        elif not isinstance(order["reason"], str) or not order["reason"]:
            raise ManifestError("backtest unfilled order requires a reason")
    order_index = _indexed(orders, "order_id")
    fills = tables["fills"]
    filled_orders = [text(fill["order_id"]) for fill in fills]
    if len(set(filled_orders)) != len(filled_orders) or set(filled_orders) != {
        text(order["order_id"]) for order in orders if order["status"] == "filled"
    }:
        raise ManifestError("backtest fills differ from filled orders")
    for fill in fills:
        order = order_index[text(fill["order_id"])]
        _same(
            fill,
            {
                **common,
                "fill_id": configuration_identity(
                    {
                        "run_id": run_id,
                        "record_type": "fill",
                        "order_id": order["order_id"],
                    }
                ),
                "originating_signal_id": order["originating_signal_id"],
                "side": order["side"],
                "quantity": order["requested_quantity"],
                "execution_session": order["earliest_permitted_execution_session"],
            },
            "fill",
        )
        if any(
            Decimal(text(fill[name])) <= 0
            for name in ("reference_price", "fill_price", "gross_notional")
        ) or any(
            Decimal(text(fill[name])) < 0
            for name in ("commission", "fees", "slippage_basis_points")
        ):
            raise ManifestError("backtest fill amount is outside its domain")
    fill_index = _indexed(fills, "fill_id")
    trades = tables["trades"]
    _indexed(trades, "trade_id")
    used_entries: list[str] = []
    used_exits: list[str] = []
    for trade in trades:
        _same(
            trade,
            {
                **common,
                "strategy_implementation_version": strategy[
                    "strategy_implementation_version"
                ],
            },
            "trade",
        )
        identity: PrimitiveMapping = {
            "run_id": run_id,
            "record_type": "trade",
            "entry_fill_id": trade["entry_fill_id"],
        }
        for prefix, side, used in (
            ("entry", "buy", used_entries),
            ("exit", "sell", used_exits),
        ):
            if prefix == "exit" and trade["is_open"] is True:
                absent = (
                    "exit_signal_id",
                    "exit_order_id",
                    "exit_fill_id",
                    "exit_session",
                    "exit_price",
                    "exit_quantity",
                    "exit_commission",
                    "exit_fees",
                    "gross_profit_loss",
                    "net_profit_loss",
                    "return_percentage",
                    "holding_period_sessions",
                    "total_economic_profit_loss",
                    "total_economic_return",
                )
                if any(trade[name] is not None for name in absent):
                    raise ManifestError("backtest open trade has completed fields")
                continue
            fill_id = text(trade[prefix + "_fill_id"])
            if fill_id not in fill_index or fill_index[fill_id]["side"] != side:
                raise ManifestError("backtest trade references an incompatible fill")
            fill = fill_index[fill_id]
            used.append(fill_id)
            _same(
                trade,
                {
                    prefix + "_" + key: fill[source]
                    for key, source in (
                        ("signal_id", "originating_signal_id"),
                        ("order_id", "order_id"),
                        ("session", "execution_session"),
                        ("price", "fill_price"),
                        ("quantity", "quantity"),
                        ("commission", "commission"),
                        ("fees", "fees"),
                    )
                },
                "trade " + prefix,
            )
            if prefix == "exit":
                identity["exit_fill_id"] = fill_id
                if text(trade["exit_session"]) < text(trade["entry_session"]) or any(
                    trade[name] is None
                    for name in (
                        "gross_profit_loss",
                        "net_profit_loss",
                        "return_percentage",
                        "holding_period_sessions",
                        "total_economic_profit_loss",
                        "total_economic_return",
                    )
                ):
                    raise ManifestError("backtest completed trade is incomplete")
        if trade["trade_id"] != configuration_identity(identity):
            raise ManifestError("backtest trade identity is inconsistent")
    if (
        len(used_entries) != len(set(used_entries))
        or len(used_exits) != len(set(used_exits))
        or set(used_entries)
        != {text(fill["fill_id"]) for fill in fills if fill["side"] == "buy"}
        or set(used_exits)
        != {text(fill["fill_id"]) for fill in fills if fill["side"] == "sell"}
    ):
        raise ManifestError("backtest trades do not cover captured fills")
