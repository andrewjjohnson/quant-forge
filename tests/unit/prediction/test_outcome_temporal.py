"""QF-46 time contracts and compatibility; no intraday outcome arithmetic."""

import json
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from typing import cast
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import MarketDataset
from quantforge.prediction import (
    ElapsedDurationHorizon,
    ExchangeSessionHorizon,
    ForwardReturnOutcomeLabeler,
    OutcomeAnchor,
    OutcomeAnchorKind,
    OutcomeEvaluationRequest,
    OutcomeLabel,
    OutcomeTemporalConfiguration,
    OutcomeTemporalError,
    evaluate_outcome_request,
    outcome_temporal_configuration,
)
from quantforge.timeframes import (
    BarLabel,
    CrossSessionPolicy,
    ExchangeSessionPolicy,
    IntradayAnchor,
    IntradayInterval,
    Timeframe,
)
from quantforge.validation import OutcomeProvenance, TemporalOffset
from tests.unit.helpers import make_dataset

SESSION = date(2024, 7, 1)
DECISION = datetime(2024, 7, 1, 11, 20, tzinfo=ZoneInfo("America/New_York"))
TIMEFRAME = Timeframe.us_equity(IntradayInterval(timedelta(minutes=5)))


@pytest.mark.parametrize("count", [1, 5])
def test_explicit_session_horizons_and_legacy_round_trip(count: int) -> None:
    labeler = ForwardReturnOutcomeLabeler(count)
    original = PrimitiveMappingSnapshot.capture(labeler.configuration())
    decoded = cast(PrimitiveMapping, json.loads(original.canonical_json))
    typed = outcome_temporal_configuration(decoded)
    assert typed.horizon == ExchangeSessionHorizon(count)
    assert typed.anchor_kind is OutcomeAnchorKind.SESSION
    assert typed.future_temporal_reach == TemporalOffset.sessions(count)
    assert OutcomeTemporalConfiguration.from_primitive(typed.to_primitive()) == typed
    assert PrimitiveMappingSnapshot.capture(decoded) == original
    assert labeler.configuration_id == configuration_identity(decoded)


@pytest.mark.parametrize("minutes", [10, 30, 60, 120, 24 * 60])
def test_elapsed_horizons_are_exact_and_round_trip(minutes: int) -> None:
    config = OutcomeTemporalConfiguration.elapsed_duration(
        timedelta(minutes=minutes), TIMEFRAME
    )
    assert config.horizon.to_primitive() == {
        "kind": "elapsed_duration",
        "duration_microseconds": minutes * 60 * 1_000_000,
    }
    snapshot = PrimitiveMappingSnapshot.capture(config.to_primitive())
    restored = OutcomeTemporalConfiguration.from_primitive(snapshot.to_primitive())
    assert restored == config
    assert restored.configuration_id == config.configuration_id
    assert (
        restored.configuration_id
        != OutcomeTemporalConfiguration.exchange_sessions(1).configuration_id
    )
    assert config.required_future_duration == timedelta(minutes=minutes + 5)
    assert config.future_temporal_reach == TemporalOffset.duration(
        timedelta(minutes=minutes + 5)
    )


@pytest.mark.parametrize("count", [0, -1, True, 1.0, "1"])
def test_session_counts_reject_nonpositive_and_ambiguous_values(count: object) -> None:
    with pytest.raises(OutcomeTemporalError):
        ExchangeSessionHorizon(cast(int, count))


@pytest.mark.parametrize("duration", [timedelta(0), timedelta(minutes=-1), 10, None])
def test_elapsed_horizons_reject_nonpositive_and_bare_integers(
    duration: object,
) -> None:
    with pytest.raises(OutcomeTemporalError):
        ElapsedDurationHorizon(cast(timedelta, duration))


def test_microsecond_precision_and_identity_dimensions() -> None:
    config = OutcomeTemporalConfiguration.elapsed_duration(
        timedelta(microseconds=1), TIMEFRAME
    )
    assert config.horizon.to_primitive()["duration_microseconds"] == 1
    variants = [
        config,
        OutcomeTemporalConfiguration.elapsed_duration(timedelta(minutes=30), TIMEFRAME),
        OutcomeTemporalConfiguration.elapsed_duration(timedelta(minutes=60), TIMEFRAME),
        replace(
            config, observation_timeframe=replace(TIMEFRAME, bar_label=BarLabel.END)
        ),
        replace(
            config,
            observation_timeframe=replace(
                TIMEFRAME,
                interval=IntradayInterval(
                    timedelta(minutes=5), IntradayAnchor.CLOCK, time(9, 32)
                ),
            ),
        ),
    ]
    assert len({item.configuration_id for item in variants}) == len(variants)
    for field, unsupported in (
        ("alignment", "last_before"),
        ("session_policy", "next_session"),
    ):
        changed = config.to_primitive()
        changed[field] = unsupported
        assert configuration_identity(changed) != config.configuration_id
        with pytest.raises(OutcomeTemporalError):
            OutcomeTemporalConfiguration.from_primitive(changed)


def test_anchor_preserves_session_label_and_exact_instant() -> None:
    legacy = OutcomeAnchor(OutcomeAnchorKind.SESSION, SESSION)
    exact = OutcomeAnchor(OutcomeAnchorKind.TIMESTAMP, SESSION, DECISION)
    assert exact.decision_timestamp == DECISION.astimezone(UTC)
    assert OutcomeAnchor.from_primitive(exact.to_primitive()) == exact
    assert OutcomeAnchor.from_primitive(legacy.to_primitive()) == legacy
    assert configuration_identity(exact.to_primitive()) != configuration_identity(
        legacy.to_primitive()
    )
    assert exact == OutcomeAnchor(
        OutcomeAnchorKind.TIMESTAMP, SESSION, DECISION.astimezone(UTC)
    )
    # Trade-date labels are authoritative even for exchanges with overnight sessions.
    overnight = OutcomeAnchor(OutcomeAnchorKind.TIMESTAMP, date(2024, 7, 2), DECISION)
    assert overnight.signal_session == date(2024, 7, 2)
    assert overnight.decision_timestamp == exact.decision_timestamp


@pytest.mark.parametrize(
    "timestamp", [None, datetime(2024, 7, 1, 11, 20), "2024-07-01"]
)
def test_exact_anchor_requires_aware_timestamp(timestamp: object) -> None:
    with pytest.raises(OutcomeTemporalError):
        OutcomeAnchor(OutcomeAnchorKind.TIMESTAMP, SESSION, cast(datetime, timestamp))


@pytest.mark.parametrize("field", ["horizon", "anchor_kind", "schema_version", "extra"])
def test_serialization_rejects_malformed_or_unknown_semantics(field: str) -> None:
    primitive = OutcomeTemporalConfiguration.elapsed_duration(
        timedelta(minutes=30), TIMEFRAME
    ).to_primitive()
    primitive[field] = "invalid"
    with pytest.raises(OutcomeTemporalError):
        OutcomeTemporalConfiguration.from_primitive(primitive)


def test_typed_session_config_cannot_hide_legacy_conflicts() -> None:
    definition: PrimitiveMapping = {
        "temporal_configuration": OutcomeTemporalConfiguration.exchange_sessions(
            5
        ).to_primitive(),
        "parameters": {"future_sessions": 1},
    }
    with pytest.raises(OutcomeTemporalError, match="differs"):
        outcome_temporal_configuration(definition)
    elapsed = OutcomeTemporalConfiguration.elapsed_duration(
        timedelta(hours=24), TIMEFRAME
    )
    definition["temporal_configuration"] = elapsed.to_primitive()
    with pytest.raises(OutcomeTemporalError, match="differs"):
        outcome_temporal_configuration(definition)
    with pytest.raises(OutcomeTemporalError):
        outcome_temporal_configuration({"parameters": {"future_sessions": 0}})
    with pytest.raises(OutcomeTemporalError):
        outcome_temporal_configuration({"parameters": {"future_sessions": True}})


def test_unsupported_timeframes_and_anchor_pairings_are_rejected() -> None:
    with pytest.raises(OutcomeTemporalError):
        OutcomeTemporalConfiguration(
            OutcomeAnchorKind.TIMESTAMP, ExchangeSessionHorizon(1)
        )
    for timeframe in (
        Timeframe(),
        replace(
            TIMEFRAME,
            interval=IntradayInterval(
                timedelta(minutes=5), cross_session_policy=CrossSessionPolicy.PERMITTED
            ),
        ),
        replace(
            TIMEFRAME, session_policy=ExchangeSessionPolicy("XHKG", "Asia/Hong_Kong")
        ),
    ):
        with pytest.raises(OutcomeTemporalError):
            OutcomeTemporalConfiguration.elapsed_duration(
                timedelta(minutes=10), timeframe
            )


class MetadataOnlyValues:
    def __init__(self, request: OutcomeEvaluationRequest) -> None:
        self.request = request

    def to_primitive(self) -> PrimitiveMapping:
        return self.request.to_primitive()


class MetadataOnlyLabeler:
    """Test-only consumer: preserve inputs without reading future prices."""

    name = "metadata_only_outcome"
    implementation_version = "1"

    def __init__(self, duration: timedelta = timedelta(minutes=30)) -> None:
        self.temporal = OutcomeTemporalConfiguration.elapsed_duration(
            duration, TIMEFRAME
        )

    @property
    def required_future_duration(self) -> timedelta:
        return self.temporal.required_future_duration

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "implementation_version": self.implementation_version,
            "temporal_configuration": self.temporal.to_primitive(),
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def label_request(
        self, dataset: MarketDataset, request: OutcomeEvaluationRequest
    ) -> OutcomeLabel[MetadataOnlyValues]:
        assert dataset.metadata.dataset_id == request.dataset_id
        return OutcomeLabel(
            request.anchor.signal_session,
            request.anchor.signal_session,
            MetadataOnlyValues(request),
        )


def test_qf8_timestamp_provenance_uses_conservative_reach_without_membership() -> None:
    component = MetadataOnlyLabeler()
    captured = OutcomeProvenance.capture_timestamp(component)
    assert captured.future_horizon == TemporalOffset.duration(timedelta(minutes=35))
    assert (
        captured.configuration.configuration_snapshot.to_primitive()
        == component.configuration()
    )
    assert (
        captured.configuration_id
        != OutcomeProvenance.capture_timestamp(
            MetadataOnlyLabeler(timedelta(minutes=60))
        ).configuration_id
    )


def test_generic_dispatch_preserves_exact_request_and_rejects_identity_mismatch() -> (
    None
):
    dataset = make_dataset(("100", "100"))
    labeler = MetadataOnlyLabeler()
    request = OutcomeEvaluationRequest(
        OutcomeAnchor(OutcomeAnchorKind.TIMESTAMP, SESSION, DECISION),
        labeler.temporal,
        labeler.configuration_id,
        dataset.metadata.dataset_id,
        dataset.metadata.data_sha256,
    )
    result = evaluate_outcome_request(labeler, dataset, request)
    assert result is not None
    assert result.values.to_primitive() == request.to_primitive()
    for changed in (
        replace(request, dataset_id="another-dataset"),
        replace(request, dataset_fingerprint="another-fingerprint"),
        replace(request, outcome_configuration_id="another-config"),
        replace(
            request,
            temporal_configuration=OutcomeTemporalConfiguration.elapsed_duration(
                timedelta(minutes=60), TIMEFRAME
            ),
        ),
    ):
        assert changed.request_id != request.request_id
        with pytest.raises(OutcomeTemporalError, match="differs"):
            evaluate_outcome_request(labeler, dataset, changed)


def test_legacy_dispatch_does_not_confuse_unavailable_and_valid_zero() -> None:
    dataset = make_dataset(("100", "100"))
    labeler = ForwardReturnOutcomeLabeler(1)
    temporal = outcome_temporal_configuration(labeler.configuration())
    for bar in dataset.bars:
        request = OutcomeEvaluationRequest(
            OutcomeAnchor(OutcomeAnchorKind.SESSION, bar.session_date),
            temporal,
            labeler.configuration_id,
            dataset.metadata.dataset_id,
            dataset.metadata.data_sha256,
        )
        result = evaluate_outcome_request(labeler, dataset, request)
        assert result == labeler.label(dataset, bar.session_date)
        if bar == dataset.bars[0]:
            assert result is not None
            assert result.values.raw_return == 0
        else:
            assert result is None
