"""Streaming reduction retains scalar samples only and preserves typed missingness."""

from quantforge.configuration import PrimitiveMapping
from quantforge.oos._records import mapping
from quantforge.oos.prediction import (
    PredictionMetricFields,
    PredictionObservationAccumulator,
)
from tests.unit.oos.test_prediction import observation


def test_streaming_counts_typed_statuses_and_median_buffers() -> None:
    accumulator = PredictionObservationAccumulator(PredictionMetricFields())
    for index in range(5760):
        status = index % 6
        values: PrimitiveMapping | None = (
            None
            if status == 0
            else {
                "direction_correct": None if status == 1 else True,
                "available": status != 1,
                "signed_prediction_return": None if status == 1 else "0.1",
                "label": (
                    "path_incomplete",
                    "ambiguous_same_bar",
                    "target_first",
                    "stop_first",
                    "neither",
                )[status - 1],
            }
        )
        item = observation("up", values, eligible=status != 5).to_primitive()
        item["unused_evidence"] = "x" * 32_000
        accumulator.update(item)
    summary = accumulator.summary(5760)
    assert summary["prediction_count"] == 4800
    assert summary["excluded_signal_count"] == 960
    assert summary["accuracy_sample_count"] == 2880
    assert summary["accuracy_unavailable_count"] == 1920
    assert mapping(summary["event_outcomes"])["counts"] == {
        "ambiguous_same_bar": 960,
        "target_first": 960,
        "stop_first": 960,
    }
    assert mapping(summary["event_outcomes"])["unavailable_count"] == 1920
    assert mapping(summary["signed_outcome"])["mean"] == "0.1"
    assert mapping(summary["signed_outcome"])["median"] == "0.1"
    assert accumulator.numeric == {
        "signed_outcome": ["0.1"] * 2880,
        "mfe": [],
        "mae": [],
    }
    assert "unused_evidence" not in vars(accumulator)
    assert not any(
        isinstance(value, (list, tuple)) for value in vars(accumulator).values()
    )
