"""Candidate-only QF-7/29 exports for the QF-45 inspection gate."""

from datetime import datetime
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping
from quantforge.examples.spy_ema import (
    EmaSmokeRule,
    EmaStudyFactory,
    configured_outcomes,
)
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.prediction import build_signal_feature_dataset
from quantforge.prediction.window import (
    _DecisionContextProvider,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.window_encoding import mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.walk_forward import WalkForwardConfig
from quantforge.walk_forward.partitions import PermittedPartition
from quantforge.walk_forward.prediction import (
    _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
)


def export_candidate_features(
    inputs: SmokeInputs,
    config: WalkForwardConfig,
    permitted: PermittedPartition,
    reader: PredictionWindowReader,
    parameters: PrimitiveMapping,
    output_root: Path,
) -> tuple[Path, ...]:
    """Replay only recorded candidates using the exact QF-39 causal context.

    Reuse the internal adapter bridges deliberately: no second context selector,
    scheduler, indicator implementation, or outcome implementation belongs here.
    Every resulting feature dataset retains the native QF-7/29 format.
    """
    if reader.evidence.schedule.decision_timestamps != permitted.decision_timestamps:
        raise ValueError("candidate export requires exact retained QF-8 membership")
    provider = _PermittedContextProvider(
        config.plan, permitted, (inputs.primary, inputs.daily), reader.evidence.schedule
    )
    factory = EmaStudyFactory(inputs.primary)
    study = factory.build(parameters)
    rule = cast(EmaSmokeRule, study.strategy)
    exports: list[Path] = []
    for receipt in reader.iterate_decision_receipts():
        if receipt.decision is None:
            continue  # Sparse no-prediction receipt: no candidate.
        record = receipt.decision.to_primitive()
        signals = cast(list[PrimitiveMapping], record["generated_signals"])
        if not signals:
            continue  # Event study: no candidate => no QF-7 outcome request.
        timestamp = datetime.fromisoformat(cast(str, record["decision_timestamp"]))
        context = _DecisionContextProvider(provider, timestamp, inputs.family.family_id)
        result = build_signal_feature_dataset(
            dataset=permitted.dataset,
            prediction_study=study,
            contextual_features=(),
            multi_timeframe_features=rule.multi_timeframe_feature_requests,
            context_provider=context,
            outcomes=configured_outcomes(inputs.primary),
            output_root=output_root,
        )
        if len(signals) != 1 or len(result.rows) != 1:
            raise ValueError("candidate replay changed the recorded event population")
        row = result.rows[0].to_primitive()
        if row["decision_timestamp"] != timestamp.isoformat():
            raise ValueError("candidate replay changed its timestamp")
        stored_features = mapping(signals[0]["features"])
        for name in (field.name for field in rule.strategy_feature_definitions):
            if row[f"feature_{name}"] != stored_features[name]:
                raise ValueError("candidate replay changed a recorded EMA feature")
        exports.append(output_root / result.dataset_id)
    return tuple(exports)
