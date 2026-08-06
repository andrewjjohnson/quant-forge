#!/usr/bin/env python3
"""Run the baseline moving-average strategy on Tiingo EOD SPY data.

The default source is the fixed 2020-01-01 through 2025-12-31 Tiingo request.
Set ``TIINGO_API_KEY`` or reuse a cached dataset with ``--dataset-id``. The
explicit ``--fixture`` mode exists only for deterministic offline validation;
its output is not real market performance.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

from quantforge.backtesting import (
    BacktestConfig,
    BacktestResult,
    BasisPointSlippage,
    DividendPolicy,
    ExplicitZeroFees,
    NextSessionOpenExecution,
    PerShareCommission,
    ResultExportError,
    SplitAccountingPolicy,
    export_backtest_result,
    load_backtest_manifest,
    run_backtest,
)
from quantforge.data import (
    AdjustmentMode,
    MarketDataCache,
    MarketDataError,
    MarketDataService,
    MarketDataset,
    RequestError,
)
from quantforge.data.models import ProviderRecord, ProviderResponse
from quantforge.data.providers import TiingoProvider
from quantforge.strategies import (
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "SPY"
REQUESTED_START = date(2020, 1, 1)
REQUESTED_END = date(2025, 12, 31)
FAST_WINDOW = 20
SLOW_WINDOW = 50
FIXTURE_FAST_WINDOW = 2
FIXTURE_SLOW_WINDOW = 3
TARGET_LONG_WEIGHT = Decimal("0.75")
INITIAL_CAPITAL = Decimal("100000")
COMMISSION_PER_SHARE = Decimal("0.005")
MINIMUM_COMMISSION = Decimal("1")
SLIPPAGE_BASIS_POINTS = Decimal("5")
ANNUAL_RISK_FREE_RATE = Decimal("0.03")
FIXTURE_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "backtesting"
    / "deterministic_moving_average.json"
)


class DeterministicFixtureProvider:
    """Adapt the existing golden fixture to the public QF-3 provider contract."""

    name = "deterministic_backtest_fixture"
    adapter_version = "2"

    def __init__(self, fixture_path: Path) -> None:
        self._fixture_path = fixture_path

    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjustment: AdjustmentMode,
    ) -> ProviderResponse:
        if adjustment is not AdjustmentMode.UNADJUSTED:
            raise RequestError("the deterministic fixture is unadjusted")
        fixture = cast(dict[str, object], json.loads(self._fixture_path.read_text()))
        input_values = cast(dict[str, object], fixture["input"])
        equity_rows = cast(list[dict[str, object]], fixture["equity"])
        closes = cast(list[str], input_values["closes"])
        session_dates = [
            date.fromisoformat(cast(str, row["session"])) for row in equity_rows
        ]
        if len(closes) != len(session_dates):
            raise RequestError("fixture closes and sessions are misaligned")
        records: list[ProviderRecord] = []
        for session_date, close_text in zip(session_dates, closes, strict=True):
            if not start <= session_date <= end:
                continue
            close = Decimal(close_text)
            records.append(
                {
                    "session_date": session_date.isoformat(),
                    "open": str(close),
                    "high": str(close + Decimal(1)),
                    "low": str(close),
                    "close": str(close),
                    "volume": "1000",
                    "dividend_amount": "0",
                    "split_coefficient": "1",
                }
            )
        return ProviderResponse(
            provider_name=self.name,
            provider_symbol=symbol,
            retrieved_at=datetime(2024, 7, 15, tzinfo=UTC),
            provider_timezone="America/New_York",
            adjustment_mode=adjustment,
            records=tuple(records),
            metadata={"fixture": self._fixture_path.name, "synthetic": True},
            adapter_version=self.adapter_version,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset-id", help="load this immutable dataset ID from --cache-root"
    )
    source.add_argument(
        "--fixture",
        action="store_true",
        help="run the short deterministic synthetic fixture without credentials",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="retrieve a new immutable Tiingo snapshot and advance the cache index",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "market-data",
        help="QF-3 cache root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "backtests",
        help="root for immutable QF-5 structured exports",
    )
    arguments = parser.parse_args()
    if arguments.refresh and (arguments.dataset_id or arguments.fixture):
        parser.error("--refresh applies only to a Tiingo retrieval")
    return arguments


def load_dataset(arguments: argparse.Namespace) -> tuple[MarketDataset, str, int, int]:
    cache = MarketDataCache(arguments.cache_root)
    if arguments.dataset_id is not None:
        return (
            cache.load(arguments.dataset_id),
            "cached QF-3 dataset",
            FAST_WINDOW,
            SLOW_WINDOW,
        )
    if arguments.fixture:
        service = MarketDataService(DeterministicFixtureProvider(FIXTURE_PATH), cache)
        dataset = service.get_daily_bars(
            SYMBOL,
            date(2024, 7, 1),
            date(2024, 7, 12),
            AdjustmentMode.UNADJUSTED,
        )
        return (
            dataset,
            "deterministic synthetic fixture (not real market performance)",
            FIXTURE_FAST_WINDOW,
            FIXTURE_SLOW_WINDOW,
        )
    api_key = os.environ.get("TIINGO_API_KEY")
    if not api_key:
        raise RequestError(
            "TIINGO_API_KEY is required; set it in the environment, pass "
            "--dataset-id for an existing cache entry, or use --fixture offline"
        )
    service = MarketDataService(TiingoProvider(api_key), cache)
    dataset = service.get_daily_bars(
        SYMBOL,
        REQUESTED_START,
        REQUESTED_END,
        AdjustmentMode.UNADJUSTED,
        refresh=arguments.refresh,
    )
    return dataset, "Tiingo End-of-Day", FAST_WINDOW, SLOW_WINDOW


def export_result(result: BacktestResult, output_root: Path) -> tuple[Path, str]:
    expected_path = output_root / result.run_id
    try:
        return export_backtest_result(result, output_root), "created"
    except ResultExportError:
        if not expected_path.is_dir():
            raise
        manifest = load_backtest_manifest(expected_path / "manifest.json")
        if manifest.get("run_id") != result.run_id:
            raise
        return expected_path, "reused existing immutable export"


def format_percentage(value: Decimal | None) -> str:
    return "N/A" if value is None else f"{value * Decimal(100):.6f}%"


def format_decimal(value: Decimal | None) -> str:
    return "N/A" if value is None else str(value)


def main() -> None:
    arguments = parse_arguments()
    dataset, source_label, fast_window, slow_window = load_dataset(arguments)
    strategy_parameters = MovingAverageCrossoverParameters(
        fast_window=fast_window,
        slow_window=slow_window,
        target_long_weight=TARGET_LONG_WEIGHT,
    )
    strategy = MovingAverageCrossoverStrategy(strategy_parameters)
    config = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        commission=PerShareCommission(
            amount_per_share=COMMISSION_PER_SHARE,
            minimum=MINIMUM_COMMISSION,
        ),
        fees=ExplicitZeroFees(),
        slippage=BasisPointSlippage(SLIPPAGE_BASIS_POINTS),
        dividend_policy=DividendPolicy.PRICE_RETURN_ONLY,
        split_policy=SplitAccountingPolicy(),
        execution=NextSessionOpenExecution(),
        annual_risk_free_rate=ANNUAL_RISK_FREE_RATE,
    )
    result = run_backtest(dataset, strategy, config)
    artifact_path, export_status = export_result(result, arguments.output_root)
    market_data = result.market_data
    performance = result.performance
    dividend_accounting = result.dividend_accounting

    print(f"Data source: {source_label}")
    print(f"Symbol: {market_data.canonical_symbol}")
    print(
        "Requested date range: "
        f"{market_data.requested_start.isoformat()} to "
        f"{market_data.requested_end.isoformat()}"
    )
    print(
        "Actual date range: "
        f"{market_data.actual_first_session.isoformat()} to "
        f"{market_data.actual_last_session.isoformat()}"
    )
    print(f"Provider: {market_data.provider_name}")
    print(f"Raw snapshot identifier: {market_data.raw_sha256}")
    print(f"Normalized dataset identifier: {market_data.dataset_id}")
    print(
        "Corporate-action snapshot identifier: "
        f"{market_data.corporate_action_snapshot_id}"
    )
    print(f"Adjustment mode: {market_data.adjustment_mode}")
    print(f"Dividend count: {market_data.dividend_count}")
    print(f"Dividend policy: {dividend_accounting.dividend_policy.value}")
    print(f"Return basis: {dividend_accounting.return_basis.value}")
    print(
        f"Ignored dividend-event count: {dividend_accounting.dividend_events_ignored}"
    )
    print(
        "Estimated ignored dividend cash (informational, excluded from equity): "
        f"${dividend_accounting.estimated_ignored_dividend_cash}"
    )
    print(f"Total dividend income: ${performance.total_dividend_income}")
    print(f"Split count: {market_data.split_count}")
    print(
        "Strategy parameters: "
        f"fast_window={fast_window}, slow_window={slow_window}, "
        f"source_field=close, target_long_weight={TARGET_LONG_WEIGHT}"
    )
    print(f"Initial equity: ${performance.starting_equity}")
    print(f"Ending equity: ${performance.ending_equity}")
    print(f"Total return: {format_percentage(performance.total_return)}")
    print(f"CAGR: {format_percentage(performance.cagr)}")
    print(
        f"Annualized volatility: {format_percentage(performance.annualized_volatility)}"
    )
    print(f"Sharpe: {format_decimal(performance.sharpe_ratio)}")
    print(f"Sortino: {format_decimal(performance.sortino_ratio)}")
    print(f"Maximum drawdown: {format_percentage(performance.maximum_drawdown)}")
    print(f"Trade count: {performance.trade_count}")
    print(f"Win rate: {format_percentage(performance.win_rate)}")
    print(
        "Benchmark total return: "
        f"{format_percentage(result.benchmark.performance.total_return)}"
    )
    print(f"Deterministic run identifier: {result.run_id}")
    print(f"Output/export location: {artifact_path.resolve()}")
    print(f"Export status: {export_status}")
    print(
        "WARNING: "
        + (
            dividend_accounting.warning
            or "PRICE-RETURN-ONLY policy selected; reported performance does not "
            "include cash dividends."
        )
    )


if __name__ == "__main__":
    try:
        main()
    except MarketDataError as error:
        raise SystemExit(f"market-data error: {error}") from None
