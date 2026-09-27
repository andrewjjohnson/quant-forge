"""The smoke CLI keeps disposable study outputs separate from holdout authority."""

import runpy
import sys
from pathlib import Path
from typing import NoReturn

import pytest

from quantforge.examples import spy_ema_inputs, spy_ema_runner
from quantforge.experiments import ExecutionProvenance
from quantforge.oos import HoldoutLedger
from tests.unit.experiments.test_contracts import execution

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_spy_ema_smoke.py"


@pytest.mark.parametrize("path_kind", ["relative", "absolute", "normalized", "symlink"])
@pytest.mark.parametrize("existing", [False, True], ids=["fresh", "existing"])
@pytest.mark.parametrize(
    "arguments",
    [(), ("--resume",), ("--preflight",)],
    ids=["normal", "resume", "preflight"],
)
def test_cli_rejects_ledger_as_output_before_any_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    path_kind: str,
    existing: bool,
    arguments: tuple[str, ...],
) -> None:
    monkeypatch.chdir(tmp_path)
    ledger_path = tmp_path / "reports" / "holdout-ledger"
    if existing:
        HoldoutLedger.create(ledger_path)
    output_root = Path("reports/holdout-ledger")
    if path_kind == "absolute":
        output_root = ledger_path
    elif path_kind == "normalized":
        output_root = Path("reports/another-run/../holdout-ledger")
    elif path_kind == "symlink":
        output_root = Path("reports/ledger-alias")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        output_root.symlink_to(ledger_path, target_is_directory=True)

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("invalid output root must be rejected before execution or inputs")

    monkeypatch.setattr(spy_ema_runner, "execution_for_run", forbidden)
    monkeypatch.setattr(spy_ema_inputs, "load_inputs", forbidden)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--output-root", str(output_root), *arguments]
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None
        for path in tmp_path.rglob("*")
    }
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert caught.value.code == 2
    assert (
        "output-root must be separate from the permanent holdout ledger"
        in capsys.readouterr().err
    )
    assert {
        path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None
        for path in tmp_path.rglob("*")
    } == before
    assert ledger_path.exists() is existing


class InputLoadingReachedError(Exception):
    """Stop after CLI path validation, before any actual market-data access."""


@pytest.mark.parametrize("custom_output", [False, True], ids=["default", "sibling"])
@pytest.mark.parametrize(
    "arguments",
    [(), ("--resume",), ("--preflight",)],
    ids=["normal", "resume", "preflight"],
)
def test_cli_accepts_separate_study_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    custom_output: bool,
    arguments: tuple[str, ...],
) -> None:
    monkeypatch.chdir(tmp_path)

    def captured(output_root: Path, repository: Path) -> ExecutionProvenance:
        assert output_root == tmp_path / "reports" / (
            "another-run" if custom_output else "qf45-smoke-2025"
        )
        assert repository == tmp_path
        return execution()

    def inputs(cache_root: Path) -> NoReturn:
        raise InputLoadingReachedError

    monkeypatch.setattr(spy_ema_runner, "execution_for_run", captured)
    monkeypatch.setattr(spy_ema_inputs, "load_inputs", inputs)
    output_arguments = ["--output-root", "reports/another-run"] if custom_output else []
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *output_arguments, *arguments])
    with pytest.raises(InputLoadingReachedError):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert (tmp_path / "reports" / "holdout-ledger").exists() is (
        "--preflight" not in arguments
    )
