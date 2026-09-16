from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import MultiTimeframeContext
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import (
    PredictionContextFailurePolicy,
    PredictionContextRequirements,
    PredictionRuleContext,
    PredictionStrategyOutput,
)
from quantforge.prediction.window import (
    _window_record_counts,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record, trial_path
from tests.unit.prediction.test_multi_timeframe_study import (
    _requirements,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_window import (
    END,
    WindowProvider,
    WindowRule,
    grid,
    run_window,
    schedule,
)


def rehash_window(window: PrimitiveMapping) -> None:
    decisions = cast(list[PrimitiveMapping], window["decisions"])
    for decision in decisions:
        study = cast(PrimitiveMapping, decision["prediction_study"])
        manifest = cast(PrimitiveMapping, study["manifest"])
        manifest["study_id"] = decision["prediction_study_id"] = configuration_identity(
            {
                "component": "quantforge_prediction_study",
                "engine_version": manifest["engine_version"],
                "market_data": manifest["market_data"],
                "study_configuration": manifest["configuration"],
                "prediction_context": manifest["prediction_context"],
            }
        )
        for row in cast(list[PrimitiveMapping], study["rows"]):
            row["study_id"] = manifest["study_id"]
            row["row_id"] = configuration_identity(
                {
                    "record_type": "prediction_study_row",
                    "study_id": manifest["study_id"],
                    "outcome_id": cast(PrimitiveMapping, row["outcome"])["outcome_id"],
                    "evaluation_id": cast(PrimitiveMapping, row["evaluation"])[
                        "evaluation_id"
                    ],
                    "signal": {
                        "features": row["features"],
                        "prediction": row["prediction"],
                    },
                }
            )
    manifest = cast(PrimitiveMapping, window["manifest"])
    manifest["window_result_id"] = configuration_identity(
        {"window_id": manifest["window_id"], "decisions": window["decisions"]}
    )
    manifest["record_counts"] = _window_record_counts(decisions)


def alter_decision(window: PrimitiveMapping, change: str) -> None:
    decisions = cast(list[PrimitiveMapping], window["decisions"])
    decision = decisions[0]
    manifest = cast(
        PrimitiveMapping,
        cast(PrimitiveMapping, decision["prediction_study"])["manifest"],
    )
    context = cast(PrimitiveMapping, manifest["prediction_context"])
    source = cast(PrimitiveMapping, context["source_context"])
    if change == "future_context":
        future_manifest = cast(
            PrimitiveMapping,
            cast(PrimitiveMapping, decisions[1]["prediction_study"])["manifest"],
        )
        manifest["prediction_context"] = deepcopy(future_manifest["prediction_context"])
        decision["context_id"] = decisions[1]["context_id"]
    elif change == "family":
        cast(PrimitiveMapping, source["source_consistency"])["family_id"] = (
            "foreign-family"
        )
    elif change == "requirements":
        cast(PrimitiveMapping, context["requirements"])["failure_policy"] = "skip"
    elif change == "source_id":
        source["context_id"] = "0" * 64
    elif change == "decision_id":
        decision["context_id"] = "0" * 64
    elif change == "source_lineage":
        aligned = cast(list[PrimitiveMapping], source["timeframes"])[0]
        cast(PrimitiveMapping, aligned["dataset_reference"])["family_id"] = (
            "foreign-family"
        )
    elif change == "selected_bars":
        cast(list[PrimitiveMapping], context["timeframes"])[0]["visible_bar_ids"] = [
            "foreign-bar"
        ]
    elif change == "future_bar":
        cast(list[PrimitiveMapping], source["timeframes"])[0][
            "latest_completed_bar_timestamp"
        ] = (END + timedelta(minutes=5)).isoformat()
    elif change == "indicator_backend":
        selected = cast(list[PrimitiveMapping], context["timeframes"])[0]
        indicator = cast(list[PrimitiveMapping], selected["indicators"])[0]
        indicator["backend"] = {"backend_id": "foreign-backend"}
    elif change == "missing_source":
        context["source_context"] = None
    elif change == "signal_substitution":
        assert decision["generated_signals"] != decisions[1]["generated_signals"]
        decision["generated_signals"] = deepcopy(decisions[1]["generated_signals"])
    elif change in {"skipped", "no_prediction"}:
        decision["status"] = change
    elif change == "duplicate_signals":
        signals = cast(list[PrimitiveMapping], decision["generated_signals"])
        signals.append(deepcopy(signals[0]))
        counts = cast(PrimitiveMapping, manifest["record_counts"])
        counts["generated_predictions"] = len(signals)
        counts["unavailable_outcomes"] = 1
    elif change == "signal_counts":
        counts = cast(PrimitiveMapping, manifest["record_counts"])
        counts["generated_predictions"] = 2
        counts["unavailable_outcomes"] = 1
    else:
        raise AssertionError(change)
    if change not in {"source_id", "decision_id", "future_context"}:
        source["context_id"] = decision["context_id"] = configuration_identity(
            {key: value for key, value in source.items() if key != "context_id"}
        )
    rehash_window(window)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("future_context", "context timing"),
        ("family", "context timing or lineage"),
        ("requirements", "context requirements"),
        ("source_id", "context identity"),
        ("decision_id", "context identity"),
        ("source_lineage", "context metadata"),
        ("selected_bars", "context metadata"),
        ("future_bar", "context metadata"),
        ("indicator_backend", "context metadata"),
        ("missing_source", "context evidence"),
        ("signal_substitution", "distinct generated signal"),
        ("duplicate_signals", "signals contain duplicates"),
        ("signal_counts", "record counts differ from generated signals"),
        ("skipped", "decision status"),
        ("no_prediction", "decision status"),
    ],
)
def test_rehashed_decision_cannot_contradict_its_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str, message: str
) -> None:
    window = run_window().to_primitive()
    block_research(monkeypatch)
    alter_decision(window, change)
    path = tmp_path / "window.json"
    write_json(path, window)
    with pytest.raises(ManifestError, match=message):
        inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)


@pytest.mark.parametrize("change", ["future_context", "signal_substitution", "skipped"])
def test_nested_grid_rejects_rehashed_decision_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    study = grid(tmp_path / "grid", WindowProvider())
    result = study.run()
    root = tmp_path / "grid" / result.study_id
    record_path = trial_path(root)
    trial = read_record(record_path)
    artifact_path = root / cast(str, trial["artifact_location"])
    artifact = read_record(artifact_path)
    window = cast(PrimitiveMapping, artifact["prediction_window"])
    alter_decision(window, change)
    artifact["prediction_window_id"] = cast(PrimitiveMapping, window["manifest"])[
        "window_result_id"
    ]
    artifact["artifact_fingerprint"] = configuration_identity(
        {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
    )
    trial["artifact_fingerprint"] = artifact["artifact_fingerprint"]
    write_json(artifact_path, artifact)
    write_json(record_path, trial)
    block_research(monkeypatch)
    with pytest.raises(
        ManifestError, match=r"context timing|distinct generated signal|decision status"
    ):
        inspect_study(StudyType.PARAMETER_STUDY, root, artifact_root=tmp_path)


class FutureContextProvider(WindowProvider):
    def get_context_at(
        self, requirements: PredictionContextRequirements, *, as_of: datetime
    ) -> MultiTimeframeContext:
        return super().get_context_at(requirements, as_of=as_of + timedelta(minutes=5))


class MissingContextProvider(WindowProvider):
    def get_context_at(
        self, requirements: PredictionContextRequirements, *, as_of: datetime
    ) -> MultiTimeframeContext:
        return cast(MultiTimeframeContext, None)


@pytest.mark.parametrize(
    "mode", ["missing_primary", "future_context", "no_source", "stale", "no_prediction"]
)
def test_valid_skipped_evidence_and_no_prediction_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    original_generate = WindowRule.generate_with_context

    def no_signals(
        self: WindowRule, context: PredictionRuleContext
    ) -> PredictionStrategyOutput:
        return replace(original_generate(self, context), signals=())

    if mode == "no_prediction":
        monkeypatch.setattr(WindowRule, "generate_with_context", no_signals)
    provider = (
        FutureContextProvider()
        if mode == "future_context"
        else MissingContextProvider()
        if mode == "no_source"
        else WindowProvider()
    )
    requirements = _requirements(failure_policy=PredictionContextFailurePolicy.SKIP)
    if mode == "stale":
        requirements = _requirements(
            failure_policy=PredictionContextFailurePolicy.SKIP,
            daily_maximum_age=timedelta(hours=1),
        )
    result = run_window(
        provider,
        requirements=requirements,
        decision_schedule=schedule(
            END + timedelta(minutes=5), END + timedelta(minutes=5)
        )
        if mode == "missing_primary"
        else schedule(),
    )
    window = result.to_primitive()
    expected_status = "no_prediction" if mode == "no_prediction" else "skipped"
    assert all(
        decision["status"] == expected_status
        for decision in cast(list[PrimitiveMapping], window["decisions"])
    )
    path = tmp_path / "window.json"
    write_json(path, window)
    block_research(monkeypatch)
    bundle = inspect_study(StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path)
    assert (
        bundle.provenance.observations.to_primitive()["window_result_id"]
        == result.window_result_id
    )
