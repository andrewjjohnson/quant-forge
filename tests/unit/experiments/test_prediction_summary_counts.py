"""Rehashed sample counts must still describe captured prediction observations."""

from copy import deepcopy
from typing import cast

import pytest

from quantforge.experiments import ManifestError
from quantforge.experiments._json import mapping
from quantforge.experiments._prediction_counts_integrity import (
    validate_prediction_summary_counts,
)
from quantforge.oos import aggregate_prediction, load_oos_source
from quantforge.oos.prediction import (
    PredictionMetricFields,
    summarize_prediction_observations,
)
from quantforge.walk_forward import WalkForwardStudy
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_aggregate_schemas import (
    Captured,
    RecordPath,
    at,
    inspect_record,
)
from tests.unit.oos.conftest import CompletedStudy
from tests.unit.oos.test_prediction import RichFactory, observation
from tests.unit.walk_forward.fixtures import prediction_fixture


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory) -> Captured:
    root = tmp_path_factory.mktemp("prediction-counts")
    config, evaluator = prediction_fixture(root, factory=RichFactory())
    study = WalkForwardStudy(config, evaluator, root / "study")
    study.run()
    source = load_oos_source(config.plan, study.study_path)
    fields = PredictionMetricFields(
        correct="baseline_correct",
        signed_outcome="mfe_percentage",
        mfe="signed_prediction_return",
        baseline_correct="direction_correct",
        baseline_name="custom_baseline",
    )
    aggregate = aggregate_prediction(source, fields)
    return CompletedStudy(study, source, evaluator), aggregate.to_primitive(), root


@pytest.mark.parametrize("window", [False, True], ids=["aggregate", "window"])
@pytest.mark.parametrize(
    ("path", "field"),
    [
        *(
            ((), name)
            for name in (
                "prediction_count",
                "generated_signal_count",
                "excluded_signal_count",
                "scheduled_decisions",
                "labeled_prediction_count",
                "accuracy_sample_count",
                "accuracy_unavailable_count",
            )
        ),
        (("direction_distribution",), "up"),
        (("direction_distribution",), "down"),
        (("accuracy_interval",), "sample_count"),
        (("matched_baseline",), "sample_count"),
        *(
            ((name,), field)
            for name in ("signed_outcome", "mfe", "mae", "event_outcomes")
            for field in ("sample_count", "unavailable_count")
        ),
        (("event_outcomes", "counts"), "target_first"),
    ],
)
def test_rehashed_counts_match_the_captured_rows(
    captured: Captured,
    monkeypatch: pytest.MonkeyPatch,
    window: bool,
    path: RecordPath,
    field: str,
) -> None:
    document = deepcopy(captured[1])
    base: RecordPath = ("summary", "windows", 0, "summary") if window else ("summary",)
    target = at(document, base + path)
    target[field] = cast(int, target[field]) + 1
    if path in (("signed_outcome",), ("mfe",), ("mae",)):
        # Keep the schema internally plausible, isolating source reconciliation.
        target["status"] = "partial" if target["unavailable_count"] else "available"
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="counts differ from captured observations"):
        inspect_record(captured, document)


@pytest.mark.parametrize("window", [False, True], ids=["aggregate", "window"])
def test_rehashed_direction_mix_cannot_preserve_a_false_distribution(
    captured: Captured,
    monkeypatch: pytest.MonkeyPatch,
    window: bool,
) -> None:
    document = deepcopy(captured[1])
    base: RecordPath = ("summary", "windows", 0, "summary") if window else ("summary",)
    directions = at(document, (*base, "direction_distribution"))
    assert directions["up"] != directions["down"]
    directions["up"], directions["down"] = directions["down"], directions["up"]
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="counts differ from captured observations"):
        inspect_record(captured, document)


def test_native_custom_metric_counts_remain_inspectable(
    captured: Captured,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_research(monkeypatch)
    inspect_record(captured, captured[1])


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("scheduled", [0, 8])
def test_missing_labels_exclusions_and_unavailable_metrics_use_native_counts(
    monkeypatch: pytest.MonkeyPatch,
    empty: bool,
    scheduled: int,
) -> None:
    fields = PredictionMetricFields(baseline_correct="baseline", baseline_name="fixed")
    observations = (
        ()
        if empty
        else (
            observation(
                "up",
                {
                    "direction_correct": True,
                    "baseline": False,
                    "signed_prediction_return": "0.1",
                    "label": "event",
                },
            ),
            observation(
                "down",
                {
                    "direction_correct": False,
                    "baseline": True,
                    "label": "event",
                    "available": False,
                },
            ),
            observation("up", {"mfe_percentage": "0.3"}),
            observation("down", None),
            observation("down", {"direction_correct": True}, eligible=False),
        )
    )
    summary = summarize_prediction_observations(observations, scheduled, fields)
    rows = [item.to_primitive() for item in observations]
    original = deepcopy(summary)
    block_research(monkeypatch)
    validate_prediction_summary_counts(summary, rows, scheduled, fields.to_primitive())
    assert summary == original
    assert mapping(summary["direction_distribution"])["up"] == (0 if empty else 2)
