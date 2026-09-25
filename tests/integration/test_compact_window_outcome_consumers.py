"""QF-40/QF-9 consume real QF-49/47 typed values without normalization loss."""

from pathlib import Path

import pytest

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.experiments import StudyType, inspect_study, verify_artifacts
from quantforge.oos.prediction import (
    PredictionMetricFields,
    iter_prediction_observations,
    summarize_prediction_observations,
)
from quantforge.prediction.window_compact import CompactPredictionWindowResult
from quantforge.prediction.window_encoding import mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.walk_forward.models import PredictionOOSArtifact
from tests.integration.test_compact_prediction_provenance import make_window
from tests.integration.test_intraday_prediction_provenance import Fixture
from tests.integration.test_intraday_prediction_provenance import fixture as fixture
from tests.unit.prediction.test_compact_prediction_window import write_compact


@pytest.mark.parametrize(
    "outcome_name",
    ["forward10", "forward30", "forward60", "forward120", "excursion", "target_stop"],
)
def test_typed_outcome_consumer_equivalence(
    fixture: Fixture, tmp_path: Path, outcome_name: str
) -> None:
    original = make_window(fixture, outcome_name)
    legacy = PredictionWindowReader.from_snapshot(original.to_primitive())
    compact = write_compact(
        tmp_path / "prediction-window.jsonl",
        CompactPredictionWindowResult.from_window(original),
    )
    old_artifact = PredictionOOSArtifact(
        "frozen",
        original.window_result_id,
        PrimitiveMappingSnapshot.capture(original.to_primitive()),
    )
    new_artifact = PredictionOOSArtifact(
        "frozen",
        str(compact.header()["window_result_id"]),
        PrimitiveMappingSnapshot.capture(
            {
                "schema_version": "2",
                "path": compact.path.name,
                "header": compact.header(),
            }
        ),
    )
    old_rows = tuple(iter_prediction_observations(legacy, old_artifact, "test"))
    new_rows = tuple(iter_prediction_observations(compact, new_artifact, "test"))
    for new, old in zip(new_rows, old_rows, strict=True):
        actual, expected = new.to_primitive(), old.to_primitive()
        assert actual["eligible"] == expected["eligible"]
        assert actual["row"] == expected["row"]
        assert actual["signal"] == expected["signal"]
    assert old_rows
    # Exercise both the default QF-40 fields and concrete stored outcome fields.
    for fields in (
        PredictionMetricFields(),
        PredictionMetricFields(signed_outcome="raw_return"),
    ):
        actual = summarize_prediction_observations(
            iter(new_rows), compact.decision_count, fields
        )
        expected = summarize_prediction_observations(
            iter(old_rows), legacy.decision_count, fields
        )
        assert actual == expected
        assert actual["generated_signal_count"] == len(old_rows)
    inspected = inspect_study(
        StudyType.PREDICTION_WINDOW,
        compact.path,
        artifact_root=tmp_path,
        canonical_metadata=fixture.dataset.metadata,
    )
    verify_artifacts(inspected.index, tmp_path).require_valid()
    assert (
        mapping(compact.manifest()["market_data"])["intraday_provenance"]
        == mapping(legacy.manifest()["market_data"])["intraday_provenance"]
    )
