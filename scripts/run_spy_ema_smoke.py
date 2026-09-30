"""Run the offline QF-45 smoke workflow; always stop before holdout consumption."""

import argparse
import json
from pathlib import Path

from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples.spy_ema_inputs import load_inputs
from quantforge.examples.spy_ema_plan import verify_configuration
from quantforge.examples.spy_ema_runner import execution_for_run, run_pre_holdout
from quantforge.oos import HoldoutLedger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("data/market-data"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("reports/qf45-smoke-2025")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight", action="store_true", help="validate configuration only"
    )
    arguments = parser.parse_args()
    root = arguments.output_root.resolve()
    ledger_path = Path("reports/holdout-ledger").resolve()
    # Reserve the ledger name case-insensitively, even before it exists, so
    # case-insensitive filesystems cannot alias a fresh study to the ledger.
    if (
        root.parent == ledger_path.parent
        and root.name.casefold() == ledger_path.name.casefold()
    ):
        parser.error("output-root must be separate from the permanent holdout ledger")
    if (
        not arguments.preflight
        and (root / "execution.json").exists()
        and not arguments.resume
    ):
        parser.error("existing execution requires --resume; never delete checkpoints")
    if root.parent != ledger_path.parent:
        parser.error(
            "output-root must be directly inside reports/ for the permanent ledger"
        )
    ledger = None
    if not arguments.preflight:
        # A missing previously-used ledger is not a new unconsumed workspace.
        if not ledger_path.exists() and any(root.parent.glob("*/execution.json")):
            parser.error(
                "workspace holdout ledger is missing; restore it before resuming"
            )
        ledger = (
            HoldoutLedger(ledger_path)
            if ledger_path.exists()
            else HoldoutLedger.create(ledger_path)
        )
    execution = None if arguments.preflight else execution_for_run(root, Path.cwd())
    # QF-65: one execution-local session authenticates the canonical inputs once
    # for loading, configuration checks and research; it ends with this process.
    with canonical_preparation():
        inputs = load_inputs(arguments.cache_root)
        print(
            json.dumps(verify_configuration(inputs, root), indent=2, sort_keys=True),
            flush=True,
        )
        if arguments.preflight:
            return
        assert ledger is not None
        assert execution is not None
        print(
            json.dumps(run_pre_holdout(inputs, root, ledger, execution), indent=2),
            flush=True,
        )


if __name__ == "__main__":
    main()
