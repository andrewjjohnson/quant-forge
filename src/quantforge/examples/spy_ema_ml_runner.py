"""QF-69 pre-holdout orchestration; always stops with the holdout reserved.

Order (each step reuses an existing contract; nothing here fits or scores):

1. Freeze and persist the study specification before any research.
2. Persist the QF-39 study definition and reserve the plan's final holdout in
   the workspace's permanent ledger (normal QF-40 reservation).
3. Run QF-39 (QF-32 trial on selection, freeze, walk-forward test).
4. Execute the frozen candidate on the fold's development membership.
5. QF-67 dataset, QF-68 training/selection/freeze, frozen OOS inference,
   deterministic rebuild/refit and offline reproduction.
6. Only after the freeze: QF-40 OOS aggregate, QF-9 manifest and the QF-41
   reserved report of the base strategy.
7. Feature-leakage audit, artifact validation, report and pre-holdout gate.

Holdout consumption is never imported or performed here.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import cast

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples.spy_ema import EmaSmokeRule
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.examples.spy_ema_ml_study import (
    event_row_audit,
    prepare_ml_walk_forward,
    specification,
)
from quantforge.examples.spy_ema_runner import publish_pre_holdout
from quantforge.experiments import ExecutionProvenance
from quantforge.ml import read_event_dataset
from quantforge.ml.modeling import read_event_model, read_prediction_set
from quantforge.ml.study import (
    STAGES,
    ConditionalStudy,
    schema_leakage_audit,
)
from quantforge.ml.study_report import render_markdown
from quantforge.oos import HoldoutLedger, load_oos_source
from quantforge.prediction import SchemaFieldCategory
from quantforge.walk_forward import FoldStatus, WalkForwardStudy
from quantforge.walk_forward.persistence import write_record

LIMITATIONS = (
    "Smoke study of platform behavior; a null, negative or unstable result is "
    "acceptable and nothing here is evidence of an edge or of profitability.",
    "One frozen population (SPY EMA 8/48, UP only, 11:00-14:00 New York); the "
    "model is conditional on that rule having triggered.",
    "Target is the 30-minute raw return > 0 without costs, fills or direction "
    "adjustment; classification metrics are not trading performance.",
    "Overlapping intraday labels are not independent samples; no standard "
    "errors, confidence intervals or significance tests are reported.",
    "Small partitions: tens of events per role, so metrics and bucket "
    "summaries are unstable and descriptive only.",
    "The six features are highly collinear EMA and price levels; coefficients "
    "are not interpretable as effects.",
    "Development reuses 2025 periods inspected during QF-45 (training data "
    "only); 2025 market history is public, so later windows are not blind in "
    "an absolute sense.",
    "QF-9 indexes the base-strategy study, OOS aggregate and reserved report; "
    "ML artifacts are validated by their own offline readers and bound through "
    "the specification and stage records.",
)


def _holdout_state(ledger: HoldoutLedger, source_path: Path, plan: object) -> str:
    from quantforge.validation import ValidationPlan

    return ledger.state(
        load_oos_source(cast(ValidationPlan, plan), source_path)
    ).state.value


def run_pre_holdout(
    inputs: SmokeInputs,
    output_root: Path,
    *,
    workspace: Path,
    ledger: HoldoutLedger,
    execution: ExecutionProvenance,
    log: Callable[[str], None] = print,
) -> PrimitiveMapping:
    """Run every pre-holdout stage and stop at the gate."""
    with canonical_preparation():
        config, adapter = prepare_ml_walk_forward(inputs, output_root / "walk-forward")
        study = WalkForwardStudy(config, adapter, output_root / "walk-forward")
        ml = ConditionalStudy(
            specification(inputs, config, adapter, study.study_id),
            root=output_root,
            study_path=study.study_path,
            workspace=workspace,
        )
        log(f"STAGE specification {ml.freeze_specification()}")
        write_record(
            study.study_path / "manifest.json",
            study._identity.to_primitive(),  # pyright: ignore[reportPrivateUsage]
            immutable=True,
        )
        state = ledger.reserve(load_oos_source(config.plan, study.study_path))
        if state.state.value != "reserved_unconsumed":
            raise ValueError("holdout already consumed; pre-holdout research is closed")
        log(f"STAGE reserved plan {config.plan.plan_id}; study {study.study_id}")
        result = study.resume()
        if [fold.status for fold in result.folds] != [FoldStatus.COMPLETED]:
            raise ValueError("QF-39 fold did not complete; preserve failure evidence")
        fold_id = config.plan.folds[0].fold_id
        log("STAGE development evidence of the frozen candidate")
        study.evaluate_development(fold_id)
        log("STAGE dataset")
        ml.build_dataset()
        log("STAGE training, selection predictions and freeze")
        outcome = ml.train_and_freeze()
        log(f"STAGE training status {outcome.status}")
        publication: PrimitiveMapping | None = None
        if outcome.frozen:
            log("STAGE out-of-sample predictions from the frozen model")
            ml.evaluate_out_of_sample()
            log("STAGE deterministic rebuild/refit and offline reproduction")
            ml.verify_reproduction()
            log("STAGE base-strategy OOS aggregate, QF-9 manifest, reserved report")
            publication = {
                key: value
                for key, value in publish_pre_holdout(
                    config, study.study_path, output_root, ledger, execution
                ).items()
                if key != "manual_audit"
            }
        log("STAGE feature leakage audit")
        dataset = read_event_dataset(ml.dataset_path())
        declared = frozenset(
            field.name
            for field in EmaSmokeRule().strategy_feature_definitions
            if field.category is SchemaFieldCategory.CONTEMPORANEOUS_FEATURE
        )
        audit = schema_leakage_audit(
            dataset,
            declared_contemporaneous=declared,
            row_audit=event_row_audit(inputs, config.plan),
        )
        validation = validate_artifacts(ml, publication)
        holdout = _holdout_state(ledger, study.study_path, config.plan)
        report: PrimitiveMapping = {
            **ml.report(),
            "leakage_audit": audit,
            "artifact_validation": validation,
            "publication": publication,
            "holdout_state": holdout,
            "limitations": cast(list[Primitive], list(LIMITATIONS)),
        }
        write_record(output_root / "report.json", report)
        gate = ml.pre_holdout_gate(
            leakage_audit=audit,
            artifact_validation=validation,
            holdout_state=holdout,
        )
        (output_root / "report.md").write_text(
            render_markdown(report, gate), encoding="utf-8"
        )
        log(f"{gate['outcome']}: holdout {holdout.upper()}; manual review required")
        return gate


def validate_artifacts(
    ml: ConditionalStudy, publication: PrimitiveMapping | None
) -> PrimitiveMapping:
    """Re-read every persisted ML artifact offline; QF-9 checks the strategy side."""
    checked: PrimitiveMapping = {
        "specification": ml.specification.specification_id,
        "dataset": read_event_dataset(ml.dataset_path()).dataset_id,
    }
    for name in STAGES:
        if ml.stage(name) is not None:
            checked[f"stage:{name}"] = "valid"
    freeze = ml.stage("freeze")
    if freeze is not None:
        checked["model"] = read_event_model(
            ml.root / cast(str, freeze["model_path"])
        ).model_id
        checked["selection_predictions"] = read_prediction_set(
            ml.root / cast(str, freeze["selection_path"])
        ).prediction_set_id
        oos = ml.stage("out_of_sample")
        if oos is not None:
            checked["out_of_sample_predictions"] = read_prediction_set(
                ml.root / cast(str, oos["path"])
            ).prediction_set_id
    qf9: PrimitiveMapping = {
        "status": "withheld",
        "reason": "no model was frozen, so no OOS publication was made",
    }
    if publication is not None:
        # publish_pre_holdout already verified every indexed artifact
        # (verify_artifacts before and inside write_manifest) and rebuilt the
        # QF-41 report from the manifest with the authoritative holdout state.
        path = Path(cast(str, publication["manifest_path"]))
        qf9 = {
            "status": "verified",
            "manifest": path.name,
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checks": "verify_artifacts, write_manifest, build_research_report",
        }
    return {"status": "passed", "ml_artifacts": checked, "qf9": qf9}


__all__ = ["LIMITATIONS", "run_pre_holdout", "validate_artifacts"]
