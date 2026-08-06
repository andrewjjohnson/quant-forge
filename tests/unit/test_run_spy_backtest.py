from decimal import Decimal
from pathlib import Path

import pytest
from scripts.run_spy_backtest import export_result

from quantforge.backtesting import (
    BacktestConfig,
    BasisPointSlippage,
    ExplicitZeroFees,
    FixedCommission,
    ResultExportError,
    run_backtest,
)
from quantforge.strategies import (
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
)

from .helpers import make_dataset


def test_spy_export_reuse_requires_the_complete_exact_result(tmp_path: Path) -> None:
    result = run_backtest(
        make_dataset(("3", "2", "1", "2", "3", "4", "3", "2", "1")),
        MovingAverageCrossoverStrategy(MovingAverageCrossoverParameters(2, 3)),
        BacktestConfig(
            Decimal(100),
            FixedCommission(Decimal(0)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(0)),
        ),
    )
    output_root = tmp_path / "reports"

    artifact_path, status = export_result(result, output_root)
    assert status == "created"
    assert export_result(result, output_root) == (
        artifact_path,
        "reused existing immutable export",
    )

    (artifact_path / "equity.csv").write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(ResultExportError, match="expected immutable result"):
        export_result(result, output_root)
