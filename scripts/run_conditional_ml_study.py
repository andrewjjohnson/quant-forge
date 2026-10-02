"""Run the offline QF-69 conditional ML smoke study; always stop before holdout.

The study specification is frozen and persisted before any research stage. A
repeated invocation verifies and reuses every completed stage; it never
re-selects, refits differently or changes the frozen design.
"""

import argparse
import json
from pathlib import Path

from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples.spy_ema_inputs import load_inputs
from quantforge.examples.spy_ema_ml_runner import run_pre_holdout
from quantforge.examples.spy_ema_ml_study import (
    prepare_ml_walk_forward,
    specification,
)
from quantforge.examples.spy_ema_runner import execution_for_run
from quantforge.ml.sources import WORKSPACE_HOLDOUT_LEDGER
from quantforge.oos import HoldoutLedger
from quantforge.walk_forward import WalkForwardStudy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("data/market-data"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("reports/qf69-conditional-ml")
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="print the frozen specification identities; write nothing",
    )
    arguments = parser.parse_args()
    workspace = Path.cwd()
    root = arguments.output_root.resolve()
    ledger_path = (workspace / WORKSPACE_HOLDOUT_LEDGER).resolve()
    if root.parent != ledger_path.parent:
        parser.error("output-root must be directly inside reports/ (QF-9 artifacts)")
    if root.name.casefold() == ledger_path.name.casefold():
        parser.error("output-root must be separate from the permanent holdout ledger")
    if not ledger_path.exists():
        parser.error(
            "the workspace's permanent holdout ledger is missing; QF-69 uses the "
            "existing ledger and never creates one"
        )
    with canonical_preparation():
        inputs = load_inputs(arguments.cache_root)
        if arguments.preflight:
            config, adapter = prepare_ml_walk_forward(inputs, root / "walk-forward")
            adapter.validate(config.plan)
            study = WalkForwardStudy(config, adapter, root / "walk-forward")
            frozen = specification(inputs, config, adapter, study.study_id)
            print(
                json.dumps(
                    {
                        "specification_id": frozen.specification_id,
                        "plan_id": config.plan.plan_id,
                        "final_holdout_id": config.plan.final_holdout.holdout_id,
                        "qf39_study_id": study.study_id,
                        "candidate": frozen.candidate.combination_id,
                        "model_configuration_id": frozen.model.configuration_id,
                    },
                    indent=2,
                ),
                flush=True,
            )
            return
        execution = execution_for_run(root, workspace, prefix="qf69")
        gate = run_pre_holdout(
            inputs,
            root,
            workspace=workspace,
            ledger=HoldoutLedger(ledger_path),
            execution=execution,
            log=lambda line: print(line, flush=True),
        )
    print(json.dumps({"outcome": gate["outcome"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
