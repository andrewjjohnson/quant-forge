"""Bind saved ledger references and corporate-action provenance without accounting."""

from collections import defaultdict

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text


def _groups(
    rows: list[PrimitiveMapping], session: str, identifier: str
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[text(row[session])].append(text(row[identifier]))
    return grouped


def validate_backtest_ledgers(
    manifest: PrimitiveMapping, tables: dict[str, list[PrimitiveMapping]]
) -> None:
    """Check copied links and identifiers without calculating cash, equity or P&L."""
    market = mapping(manifest["market_data"])
    benchmark = mapping(manifest["benchmark"])
    sessions = [text(row["session"]) for row in tables["daily_equity"]]
    if not sessions or sessions != sorted(set(sessions)):
        raise ManifestError("backtest equity sessions must be ordered and unique")
    for name in ("positions", "benchmark_daily_equity"):
        if [row["session"] for row in tables[name]] != sessions:
            raise ManifestError("backtest ledger sessions differ")
    for position, equity in zip(
        tables["positions"], tables["daily_equity"], strict=True
    ):
        if position["symbol"] != market["canonical_symbol"] or any(
            position[key] != equity[key] for key in ("shares", "market_value")
        ):
            raise ManifestError("backtest position provenance differs from equity")
    for prefix, account in (("", "strategy"), ("benchmark_", "benchmark")):
        for suffix, kind, session_key in (
            ("dividend_cashflows", "dividend_cashflow", "ex_dividend_session"),
            ("split_adjustments", "split_adjustment", "effective_session"),
        ):
            records = tables[prefix + suffix]
            identifiers: set[str] = set()
            for row in records:
                identifier = text(row[kind + "_id"])
                expected = configuration_identity(
                    {
                        "run_id": manifest["run_id"],
                        "account_id": account,
                        "record_type": kind,
                        "corporate_action_id": row["corporate_action_id"],
                    }
                )
                if (
                    identifier != expected
                    or identifier in identifiers
                    or row["run_id"] != manifest["run_id"]
                    or row["account_id"] != account
                    or row["symbol"] != market["canonical_symbol"]
                    or row["source_dataset_id"] != market["dataset_id"]
                    or row[session_key] not in sessions
                ):
                    raise ManifestError(
                        "backtest corporate-action provenance is inconsistent"
                    )
                identifiers.add(identifier)
            if prefix and records != benchmark[suffix]:
                raise ManifestError(
                    "backtest benchmark corporate records differ from manifest"
                )
        equity = tables[prefix + "daily_equity"]
        orders = [mapping(benchmark["order"])] if prefix else tables["orders"]
        fills = (
            ([] if benchmark["fill"] is None else [mapping(benchmark["fill"])])
            if prefix
            else tables["fills"]
        )
        groups = {
            "order_ids": _groups(orders, "signal_session", "order_id"),
            "fill_ids": _groups(fills, "execution_session", "fill_id"),
            "dividend_cashflow_ids": _groups(
                tables[prefix + "dividend_cashflows"],
                "ex_dividend_session",
                "dividend_cashflow_id",
            ),
            "split_adjustment_ids": _groups(
                tables[prefix + "split_adjustments"],
                "effective_session",
                "split_adjustment_id",
            ),
        }
        if any(set(group) - set(sessions) for group in groups.values()):
            raise ManifestError(
                "backtest ledger references lie outside recorded sessions"
            )
        for row in equity:
            if any(
                row[key] != group.get(text(row["session"]), [])
                for key, group in groups.items()
            ):
                raise ManifestError(
                    "backtest equity references differ from captured records"
                )
