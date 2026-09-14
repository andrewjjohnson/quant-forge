"""Native QF-5 statistics plus a dimensionless chain of independent OOS windows."""

from decimal import Decimal

from quantforge.backtesting._arithmetic import arithmetic
from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    decimal_to_primitive,
)
from quantforge.oos._records import OOSIntegrityError, mapping, number, records
from quantforge.oos.common import completeness, configuration_stability, provenance
from quantforge.oos.models import BacktestOOSAggregate, OOSSource, optional_decimal
from quantforge.validation import ResearchStudyType
from quantforge.walk_forward.models import BacktestOOSArtifact


def aggregate_backtest(source: OOSSource) -> BacktestOOSAggregate:
    """Chain E[k,t]/C[k] onto the previous segment's index, starting at one.

    This is a reporting index of reset-account returns, never a capital-carryover
    simulation. Open positions remain open in native artifacts; no liquidation
    or cross-window transaction is manufactured. Non-OOS dates have no rows.
    """
    if source.plan.environment.study_type is not ResearchStudyType.TRADING_BACKTEST:
        raise OOSIntegrityError("backtest aggregation requires a trading study")
    source_provenance = provenance(source)
    native: list[PrimitiveMappingSnapshot] = []
    normalized: list[PrimitiveMappingSnapshot] = []
    windows: list[PrimitiveMapping] = []
    metrics: list[PrimitiveMapping] = []
    ending = benchmark_ending = peak = benchmark_peak = Decimal(1)
    drawdown = benchmark_drawdown = Decimal(0)
    exposed_sessions = session_count = profitable_windows = 0
    commission = fees = slippage = Decimal(0)
    with arithmetic():
        for fold in source.folds:
            artifact = fold.artifact
            if artifact is None:
                windows.append(
                    {
                        "fold_id": fold.fold_id,
                        "status": fold.status.value,
                        "performance": None,
                    }
                )
                continue
            if not isinstance(artifact, BacktestOOSArtifact):
                raise OOSIntegrityError("wrong OOS artifact family")
            payload = artifact.snapshot.to_primitive()
            manifest = mapping(payload["manifest"])
            performance = mapping(manifest["performance"])
            metrics.append(performance)
            capital = number(
                mapping(manifest["backtest_configuration"])["initial_capital"]
            )
            if capital <= 0:
                raise OOSIntegrityError("fold capital must be positive")
            base, benchmark_base = ending, benchmark_ending
            equity = records(payload["daily_equity"])
            benchmark = records(payload["benchmark_daily_equity"])
            if not equity or len(equity) != len(benchmark):
                raise OOSIntegrityError("native equity/benchmark membership differs")
            for position, (row, reference) in enumerate(
                zip(equity, benchmark, strict=True)
            ):
                if row["session"] != reference["session"]:
                    raise OOSIntegrityError("benchmark is not aligned with OOS equity")
                ending = base * number(row["total_equity"]) / capital
                benchmark_ending = (
                    benchmark_base * number(reference["total_equity"]) / capital
                )
                peak = max(peak, ending)
                benchmark_peak = max(benchmark_peak, benchmark_ending)
                current_drawdown = ending / peak - 1
                current_benchmark_drawdown = benchmark_ending / benchmark_peak - 1
                drawdown = min(drawdown, current_drawdown)
                benchmark_drawdown = min(benchmark_drawdown, current_benchmark_drawdown)
                exposed_sessions += row["exposed"] is True
                session_count += 1
                normalized.append(
                    PrimitiveMappingSnapshot.capture(
                        {
                            "fold_id": fold.fold_id,
                            "run_id": artifact.run_id,
                            "session": row["session"],
                            "timestamp_semantics": "exchange_session_close",
                            "window_start": position == 0,
                            "window_start_index": decimal_to_primitive(base),
                            "equity_index": decimal_to_primitive(ending),
                            "benchmark_index": decimal_to_primitive(benchmark_ending),
                            "drawdown": decimal_to_primitive(current_drawdown),
                            "benchmark_drawdown": decimal_to_primitive(
                                current_benchmark_drawdown
                            ),
                        }
                    )
                )
            profitable_windows += number(performance["total_return"]) > 0
            for fill in records(payload["fills"]):
                commission += number(fill["commission"])
                fees += number(fill["fees"])
                slippage += abs(number(fill["slippage_per_share"])) * number(
                    fill["quantity"]
                )
            native.append(
                PrimitiveMappingSnapshot.capture(
                    {
                        "fold_id": fold.fold_id,
                        "selection_id": artifact.selection_id,
                        "result": payload,
                        "export_location": artifact.export_location,
                        "export_fingerprint": artifact.export_fingerprint,
                    }
                )
            )
            windows.append(
                {
                    "fold_id": fold.fold_id,
                    "status": fold.status.value,
                    "performance": performance,
                    "benchmark_performance": mapping(manifest["benchmark"])[
                        "performance"
                    ],
                }
            )

        def total(key: str) -> Decimal:
            return sum((number(metric[key]) for metric in metrics), Decimal(0))

        trades, winners = int(total("trade_count")), int(total("winning_trades"))
        gross_profit, gross_loss = total("gross_profit"), total("gross_loss")
        summary: PrimitiveMapping = {
            "completeness": completeness(source),
            "windows": [item for item in windows],
            "stitching": "reset_account_relative_equity_chain_v1",
            "starting_index": "1",
            "ending_index": optional_decimal(ending if metrics else None),
            "total_return": optional_decimal(ending - 1 if metrics else None),
            "maximum_drawdown": optional_decimal(drawdown if metrics else None),
            "benchmark_total_return": optional_decimal(
                benchmark_ending - 1 if metrics else None
            ),
            "benchmark_maximum_drawdown": optional_decimal(
                benchmark_drawdown if metrics else None
            ),
            "trade_count": trades,
            "open_trade_count": int(total("open_trade_count")),
            "winning_trades": winners,
            "losing_trades": int(total("losing_trades")),
            "win_rate": optional_decimal(Decimal(winners) / trades if trades else None),
            "profit_factor": optional_decimal(
                gross_profit / abs(gross_loss) if gross_loss else None
            ),
            "gross_profit": decimal_to_primitive(gross_profit),
            "gross_loss": decimal_to_primitive(gross_loss),
            "profitable_window_fraction": optional_decimal(
                Decimal(profitable_windows) / len(metrics) if metrics else None
            ),
            "profitable_window_denominator": "completed_windows_only; see completeness",
            "exposure": optional_decimal(
                Decimal(exposed_sessions) / session_count if session_count else None
            ),
            "oos_session_count": session_count,
            "native_commissions": decimal_to_primitive(commission),
            "native_fees": decimal_to_primitive(fees),
            "native_slippage_cost": decimal_to_primitive(slippage),
            "native_dividend_income": decimal_to_primitive(
                total("total_dividend_income")
            ),
            "warnings": [
                "normalized reporting index; independent fold accounts and "
                "costs; no live capital carryover",
                "open trades retain terminal marks without synthetic closing fills",
            ],
        }
    return BacktestOOSAggregate(
        PrimitiveMappingSnapshot.capture(summary),
        configuration_stability(source),
        tuple(normalized),
        tuple(native),
        source_provenance,
    )
