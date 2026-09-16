"""Summary availability follows captured denominators without recalculating rates."""

from copy import deepcopy

import pytest

from quantforge.experiments import ManifestError
from quantforge.experiments._json import mapping
from quantforge.experiments._prediction_counts_integrity import (
    validate_prediction_summary_counts,
)
from quantforge.experiments._prediction_summary_integrity import (
    validate_prediction_summary,
)
from quantforge.oos.prediction import (
    PredictionMetricFields,
    summarize_prediction_observations,
)
from tests.unit.experiments import (
    test_holdout_summary_schema,
    test_prediction_summary_counts,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_aggregate_schemas import Captured, at, inspect_record
from tests.unit.experiments.test_holdout_summary_schema import (
    Captured as CapturedHoldout,
)
from tests.unit.experiments.test_holdout_summary_schema import inspect_summary
from tests.unit.oos.test_prediction import observation

holdout_export = test_holdout_summary_schema.captured
aggregate_export = test_prediction_summary_counts.captured


@pytest.mark.parametrize("window", [False, True], ids=["aggregate", "window"])
@pytest.mark.parametrize("metric", ["prediction_frequency", "accuracy"])
def test_rehashed_prediction_summary_cannot_discard_available_metrics(
    aggregate_export: Captured,
    monkeypatch: pytest.MonkeyPatch,
    window: bool,
    metric: str,
) -> None:
    document = deepcopy(aggregate_export[1])
    summary = at(
        document, ("summary", "windows", 0, "summary") if window else ("summary",)
    )
    assert summary[metric] is not None
    summary[metric] = None
    block_research(monkeypatch)
    with pytest.raises(ManifestError):
        inspect_record(aggregate_export, document)


@pytest.mark.parametrize("metric", ["prediction_frequency", "accuracy"])
def test_rehashed_holdout_summary_cannot_discard_available_metrics(
    holdout_export: CapturedHoldout, monkeypatch: pytest.MonkeyPatch, metric: str
) -> None:
    summary = deepcopy(mapping(holdout_export[3]["summary"]))
    assert summary[metric] is not None
    summary[metric] = None
    block_research(monkeypatch)
    inspect_summary(holdout_export, summary)


@pytest.mark.parametrize("scheduled", [0, 8])
@pytest.mark.parametrize("profile", ["empty", "unavailable", "correct", "incorrect"])
def test_metric_availability_matches_native_zero_and_nonzero_denominators(
    monkeypatch: pytest.MonkeyPatch, scheduled: int, profile: str
) -> None:
    observations = (
        ()
        if profile == "empty"
        else (
            observation(
                "up",
                None
                if profile == "unavailable"
                else {"direction_correct": profile == "correct"},
            ),
        )
    )
    fields = PredictionMetricFields()
    summary = summarize_prediction_observations(observations, scheduled, fields)
    original = deepcopy(summary)
    block_research(monkeypatch)
    validate_prediction_summary(summary)
    assert summary == original
    for metric in ("prediction_frequency", "accuracy"):
        changed = deepcopy(summary)
        changed[metric] = "0" if summary[metric] is None else None
        validate_prediction_summary_counts(
            changed,
            [item.to_primitive() for item in observations],
            scheduled,
            fields.to_primitive(),
        )
        with pytest.raises(ManifestError):
            validate_prediction_summary(changed)
