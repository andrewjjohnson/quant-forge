"""The QF-69 CLI uses the existing permanent ledger and refuses unsafe roots."""

import runpy
import sys
from pathlib import Path
from typing import NoReturn

import pytest

from quantforge.examples import spy_ema_inputs
from quantforge.oos import HoldoutLedger

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_conditional_ml_study.py"


def forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("inputs must not load before the CLI guards pass")


@pytest.mark.parametrize(
    ("output_root", "ledger", "message"),
    [
        ("reports/qf69-conditional-ml", False, "never creates one"),
        ("elsewhere/qf69", True, "directly inside reports/"),
        ("reports/holdout-ledger", True, "separate from the permanent holdout"),
        ("reports/HOLDOUT-LEDGER", True, "separate from the permanent holdout"),
    ],
)
def test_cli_guards_run_before_any_load_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    output_root: str,
    ledger: bool,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    if ledger:
        HoldoutLedger.create(tmp_path / "reports" / "holdout-ledger")
    before = sorted(tmp_path.rglob("*"))
    monkeypatch.setattr(spy_ema_inputs, "load_inputs", forbidden)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--output-root", output_root, "--preflight"]
    )
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert raised.value.code == 2
    assert message in capsys.readouterr().err
    assert sorted(tmp_path.rglob("*")) == before
