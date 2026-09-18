"""Custom serialized anchors must survive the same UTC contract as QF-42 readers."""

from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import StudyType, inspect_study
from quantforge.prediction import (
    InvalidPredictionConfigurationError,
    PredictionContextRequirements,
    PredictionDecisionSchedule,
    PredictionSignal,
    run_prediction_window,
)
from quantforge.validation import PartitionRole
from quantforge.walk_forward.partitions import partition
from quantforge.walk_forward.prediction import (
    _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.walk_forward.timestamp_fixtures import (
    PRIMARY,
    MetadataLabeler,
    instant,
    timestamp_fixture,
)


@pytest.mark.parametrize(
    "encoding", ["offset", "z", "space", "microseconds", "canonical"]
)
def test_custom_timestamp_serialization_is_rejected_or_remains_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoding: str
) -> None:
    config, adapter = timestamp_fixture(tmp_path)
    study = adapter.factory.build({"window": 2})
    requirements = cast(
        PredictionContextRequirements, getattr(study.strategy, "context_requirements")
    )
    timestamp = instant(4, "10:00")
    serialized = {
        "offset": timestamp.astimezone(ZoneInfo("America/New_York")).isoformat(),
        "z": timestamp.isoformat().replace("+00:00", "Z"),
        "space": timestamp.isoformat(sep=" "),
        "microseconds": timestamp.isoformat(timespec="microseconds"),
        "canonical": timestamp.isoformat(),
    }[encoding]
    original_serializer = PredictionSignal.prediction_primitive

    def custom_serializer(signal: PredictionSignal) -> PrimitiveMapping:
        return {**original_serializer(signal), "decision_timestamp": serialized}

    monkeypatch.setattr(PredictionSignal, "prediction_primitive", custom_serializer)
    permitted = partition(
        adapter.dataset,
        config.plan,
        0,
        PartitionRole.DEVELOPMENT,
        minimum_observations=1,
    )
    schedule = PredictionDecisionSchedule(PRIMARY, timestamp, timestamp)
    provider = _PermittedContextProvider(
        config.plan, permitted, adapter.series, schedule
    )
    assert provider.get_context_at(requirements, as_of=timestamp).as_of == timestamp

    def run_window() -> PrimitiveMapping:
        return run_prediction_window(
            permitted.dataset,
            study,
            schedule=schedule,
            context_provider=provider,
            dataset_family_fingerprint=adapter.series[0].dataset_reference.family_id,
            context_environment={"fixture": "canonical_timestamp"},
            indicator_backend_environment=adapter.backend.to_primitive(),
        ).to_primitive()

    if encoding != "canonical":

        def forbidden_label(*args: object, **kwargs: object) -> None:
            pytest.fail("noncanonical anchors must fail before outcome labeling")

        monkeypatch.setattr(MetadataLabeler, "label_request", forbidden_label)
        with pytest.raises(InvalidPredictionConfigurationError, match="canonical UTC"):
            run_window()
    else:
        snapshot = run_window()
        path = tmp_path / "window.json"
        write_json(path, snapshot)
        block_research(monkeypatch)
        bundle = inspect_study(
            StudyType.PREDICTION_WINDOW, path, artifact_root=tmp_path
        )
        assert (
            bundle.provenance.observations.to_primitive()["window_result_id"]
            == cast(PrimitiveMapping, snapshot["manifest"])["window_result_id"]
        )
