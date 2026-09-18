"""QF-42 request dispatch and QF-7/QF-9 configuration compatibility hooks."""

from dataclasses import replace
from datetime import timedelta
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import MarketDataset
from quantforge.experiments import ManifestError
from quantforge.experiments._producer_integrity import validate_outcome_contract
from quantforge.prediction import (
    ForwardReturnEvaluator,
    ForwardReturnOutcomeLabeler,
    ForwardReturnValues,
    NextSessionOpenGapOutcomeLabeler,
    NextSessionOpenGapValues,
    OutcomeAnchorKind,
    OutcomeEvaluationRequest,
    OutcomeLabel,
    OutcomeTemporalConfiguration,
    PredictionStudyOutcome,
    outcome_resolution_fields,
)
from quantforge.prediction.signal_feature_models import SignalFeatureSchema
from tests.unit.prediction.test_outcome_temporal import TIMEFRAME, MetadataOnlyLabeler
from tests.unit.prediction.test_prediction_window import run_window, schedule


def typed_contract(*, sessions: int | None = None) -> PrimitiveMapping:
    temporal = (
        OutcomeTemporalConfiguration.elapsed_duration(timedelta(minutes=30), TIMEFRAME)
        if sessions is None
        else OutcomeTemporalConfiguration.exchange_sessions(sessions)
    )
    labeler: PrimitiveMapping = {
        "configuration": {
            "component_name": "temporal_fixture",
            "implementation_version": "1",
            "temporal_configuration": temporal.to_primitive(),
            "required_market_fields": ["close"],
            "result_schema_version": "1",
        },
        "temporal_configuration": temporal.to_primitive(),
        "required_market_fields": ["close"],
        "result_schema_version": "1",
    }
    if sessions is not None:
        labeler["required_future_sessions"] = sessions
    return {
        "outcome_labeler": labeler,
        "evaluator": {
            "configuration": {"result_schema_version": "1"},
            "result_schema_version": "1",
        },
    }


@pytest.mark.parametrize("sessions", [None, 1, 5])
def test_qf9_accepts_explicit_typed_outcome_contract(sessions: int | None) -> None:
    validate_outcome_contract(typed_contract(sessions=sessions))


@pytest.mark.parametrize(
    "change",
    [
        "zero",
        "negative",
        "bool",
        "legacy_count",
        "wrapper_count",
        "wrapper_time",
        "duration",
        "anchor",
        "alignment",
        "policy",
        "fields",
        "schema",
        "timeframe",
    ],
)
def test_qf9_rejects_inconsistent_temporal_configuration(change: str) -> None:
    configuration = typed_contract()
    wrapper = cast(PrimitiveMapping, configuration["outcome_labeler"])
    definition = cast(PrimitiveMapping, wrapper["configuration"])
    temporal = cast(PrimitiveMapping, definition["temporal_configuration"])
    horizon = cast(PrimitiveMapping, temporal["horizon"])
    if change in {"zero", "negative", "bool"}:
        horizon["duration_microseconds"] = {"zero": 0, "negative": -1, "bool": True}[
            change
        ]
    elif change == "legacy_count":
        definition["parameters"] = {"future_sessions": 1}
    elif change == "wrapper_count":
        wrapper["required_future_sessions"] = 0
    elif change == "wrapper_time":
        wrapper["temporal_configuration"] = (
            OutcomeTemporalConfiguration.elapsed_duration(
                timedelta(minutes=60), TIMEFRAME
            ).to_primitive()
        )
    elif change == "duration":
        horizon["duration_microseconds"] = 60 * 60 * 1_000_000
    elif change == "anchor":
        temporal["anchor_kind"] = "exchange_session"
    elif change == "alignment":
        temporal["alignment"] = "skip_missing"
    elif change == "policy":
        temporal["session_policy"] = "next_session"
    elif change == "fields":
        wrapper["required_market_fields"] = ["high"]
    elif change == "schema":
        wrapper["result_schema_version"] = "2"
    else:
        cast(PrimitiveMapping, temporal["observation_timeframe"])[
            "configuration_id"
        ] = "foreign"
    # Rehashing outer identities cannot legalize contradictory inner declarations.
    wrapper["configuration_id"] = configuration_identity(definition)
    with pytest.raises(ManifestError):
        validate_outcome_contract(configuration)


def test_qf9_keeps_positive_session_checks_for_untyped_legacy_contracts() -> None:
    config = typed_contract(sessions=1)
    wrapper = cast(PrimitiveMapping, config["outcome_labeler"])
    definition = cast(PrimitiveMapping, wrapper["configuration"])
    del wrapper["temporal_configuration"]
    del definition["temporal_configuration"]
    validate_outcome_contract(config)
    wrapper["required_future_sessions"] = 0
    with pytest.raises(ManifestError, match="positive integer"):
        validate_outcome_contract(config)


def test_qf42_dispatch_receives_original_exact_decisions_without_changing_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = run_window()
    requests: list[OutcomeEvaluationRequest] = []

    def capture(
        self: NextSessionOpenGapOutcomeLabeler,
        dataset: MarketDataset,
        request: OutcomeEvaluationRequest,
    ) -> OutcomeLabel[NextSessionOpenGapValues] | None:
        requests.append(request)
        return self.label(dataset, request.anchor.signal_session)

    monkeypatch.setattr(
        NextSessionOpenGapOutcomeLabeler, "label_request", capture, raising=False
    )
    result = run_window()
    assert (
        tuple(item.anchor.decision_timestamp for item in requests)
        == schedule().decision_timestamps
    )
    assert all(item.anchor.kind is OutcomeAnchorKind.SESSION for item in requests)
    assert result.to_primitive() == baseline.to_primitive()
    # Consumers can select an exact anchor using the original supplied instant.
    original = requests[0]
    temporal = OutcomeTemporalConfiguration.elapsed_duration(
        timedelta(minutes=30), TIMEFRAME
    )
    exact = replace(
        original,
        anchor=replace(original.anchor, kind=OutcomeAnchorKind.TIMESTAMP),
        temporal_configuration=temporal,
    )
    assert exact.anchor.decision_timestamp == original.anchor.decision_timestamp
    assert exact.request_id != original.request_id


def test_qf7_generic_schema_and_adapter_preserve_temporal_identity() -> None:
    fields = outcome_resolution_fields()
    schema = SignalFeatureSchema("1", "1", fields)
    assert [
        field["field_name"]
        for field in cast(list[PrimitiveMapping], schema.to_primitive()["fields"])
    ] == [field.name for field in fields]

    # Existing adapter configuration captures arbitrary material component metadata.
    # Execution/fixed-candidate timestamp replay is deliberately left to QF-48.
    class DeclaredSessionLabeler(ForwardReturnOutcomeLabeler):
        def configuration(self) -> PrimitiveMapping:
            temporal = OutcomeTemporalConfiguration.exchange_sessions(
                self.required_future_sessions
            )
            return {
                **super().configuration(),
                "temporal_configuration": temporal.to_primitive(),
            }

    from quantforge.prediction import forward_return_outcome

    original = forward_return_outcome(1)
    typed = PredictionStudyOutcome[ForwardReturnValues, ForwardReturnValues].create(
        original.namespace,
        DeclaredSessionLabeler(1),
        ForwardReturnEvaluator(),
        original.fields,
        unavailable_values=original.unavailable_values_snapshot.to_primitive(),
    )
    assert typed.configuration_id != original.configuration_id
    captured = cast(PrimitiveMapping, typed.configuration()["labeler"])
    assert (
        captured["temporal_configuration"]
        == OutcomeTemporalConfiguration.exchange_sessions(1).to_primitive()
    )
    first = MetadataOnlyLabeler().configuration()
    second = MetadataOnlyLabeler(timedelta(minutes=60)).configuration()
    assert configuration_identity(first) != configuration_identity(second)


def test_qf7_csv_and_qf29_parquet_preserve_resolution_metadata() -> None:
    import csv
    from importlib import import_module
    from io import StringIO

    from quantforge.configuration import PrimitiveMappingSnapshot
    from quantforge.prediction import SignalFeatureRow, resolve_future_observation
    from quantforge.prediction.feature_dataset import (
        _render_csv_rows,  # pyright: ignore[reportPrivateUsage]
        _render_parquet_rows,  # pyright: ignore[reportPrivateUsage]
    )
    from tests.unit.prediction.test_outcome_resolution import request, series

    source = series()
    resolutions = (
        resolve_future_observation(request(source), source),
        resolve_future_observation(
            request(source, duration=timedelta(hours=24)), source
        ),
    )
    schema = SignalFeatureSchema(
        "1",
        "1",
        tuple(
            replace(field, name="outcome_temporal_" + field.name)
            for field in outcome_resolution_fields()
        ),
    )
    rows = tuple(
        SignalFeatureRow.capture(
            {
                "row_id": str(index),
                "candidate_id": str(index),
                "signal_session": resolution.request.anchor.signal_session.isoformat(),
                "signal_disposition": "accepted",
                **{
                    "outcome_temporal_" + key: value
                    for key, value in resolution.metadata_primitive().items()
                },
            }
        )
        for index, resolution in enumerate(resolutions)
    )
    csv_rows = list(csv.DictReader(StringIO(_render_csv_rows(schema, rows))))
    assert (
        csv_rows[0]["outcome_temporal_decision_timestamp"]
        == resolutions[0].metadata_primitive()["decision_timestamp"]
    )
    assert csv_rows[1]["outcome_temporal_resolved_observation_timestamp"] == ""
    content = _render_parquet_rows("fixture-dataset", schema, rows)
    pa, pq = import_module("pyarrow"), import_module("pyarrow.parquet")
    table = pq.read_table(pa.BufferReader(content))
    primitive_rows = cast(list[PrimitiveMapping], table.to_pylist())
    assert primitive_rows[0]["outcome_temporal_available"] is True
    assert primitive_rows[1]["outcome_temporal_available"] is False
    assert primitive_rows[1]["outcome_temporal_resolved_observation_timestamp"] is None
    for row, resolution in zip(primitive_rows, resolutions, strict=True):
        assert PrimitiveMappingSnapshot.capture(
            row
        ) == PrimitiveMappingSnapshot.capture(
            {
                "outcome_temporal_" + key: value
                for key, value in resolution.metadata_primitive().items()
            }
        )


@pytest.mark.parametrize("count", [True, 1.0])
def test_qf9_rejects_type_aliases_in_typed_wrapper(count: bool | float) -> None:
    configuration = typed_contract(sessions=1)
    wrapper = cast(PrimitiveMapping, configuration["outcome_labeler"])
    temporal = cast(PrimitiveMapping, wrapper["temporal_configuration"])
    cast(PrimitiveMapping, temporal["horizon"])["count"] = count
    with pytest.raises(ManifestError, match="temporal configuration"):
        validate_outcome_contract(configuration)
