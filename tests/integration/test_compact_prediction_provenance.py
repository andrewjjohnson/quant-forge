"""Real bounded QF-52 inputs, QF-48 membership and QF-49/QF-47 outcomes in v2."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.data import MultiTimeframeContext, build_multi_timeframe_context
from quantforge.data.models import BoundedPredictionProvenance
from quantforge.data.prediction_views import bounded_prediction_view
from quantforge.prediction import (
    PredictionContextRequirements,
    PredictionDecisionSchedule,
    PredictionStudy,
    PredictionWindowResult,
    SignalFeatureCandidate,
    intraday_excursion_outcome,
    intraday_forward_return_outcome,
    intraday_target_stop_outcome,
    run_prediction_window,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_compact import (
    CompactPredictionWindowDecision,
    CompactPredictionWindowResult,
    PredictionWindowEvidence,
)
from quantforge.prediction.window_compact_validation import (
    validate_prediction_window_reader,
)
from quantforge.prediction.window_encoding import mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.validation import PredictionMembershipSource
from tests.integration.test_intraday_prediction_provenance import (
    DECISION,
    TWO_MINUTES,
    Fixture,
    study_inputs,
)
from tests.integration.test_intraday_prediction_provenance import fixture as fixture
from tests.unit.prediction.test_compact_prediction_window import (
    rehash_decision,
    write_compact,
)


@dataclass(frozen=True)
class WindowProvider:
    fixture: Fixture

    def get_context_at(
        self, requirements: PredictionContextRequirements, *, as_of: datetime
    ) -> MultiTimeframeContext:
        return build_multi_timeframe_context(
            series=(self.fixture.primary, self.fixture.daily),
            primary_timeframe=requirements.primary.timeframe,
            required_timeframes=requirements.context_timeframe_requirements(),
            completion_policy=requirements.context_completion_policy,
            as_of=as_of,
        )


def make_window(
    fixture: Fixture, outcome_name: str = "forward30", *, cutoff: datetime = DECISION
) -> PredictionWindowResult[Any, Any, Any]:
    if outcome_name.startswith("forward"):
        outcome = intraday_forward_return_outcome(
            timedelta(minutes=int(outcome_name.removeprefix("forward"))),
            fixture.primary,
        )
    elif outcome_name == "excursion":
        outcome = intraday_excursion_outcome(timedelta(minutes=60), fixture.primary)
    else:
        outcome = intraday_target_stop_outcome(
            timedelta(minutes=60), fixture.primary, Decimal("0.003"), Decimal("0.002")
        )
    rule, _ = study_inputs(fixture)
    study = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
        rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
    )
    view = bounded_prediction_view(fixture.dataset, cutoff)
    assert isinstance(view.metadata.intraday_provenance, BoundedPredictionProvenance)
    return run_prediction_window(
        view,
        study,
        schedule=PredictionDecisionSchedule(
            TWO_MINUTES, DECISION, DECISION + timedelta(minutes=4)
        ),
        context_provider=WindowProvider(fixture),
        dataset_family_fingerprint=view.metadata.intraday_provenance.family_id,
        context_environment={"provider": "immutable_synthetic_fixture"},
    )


def verify(
    reader: PredictionWindowReader,
    fixture: Fixture,
    window: PredictionWindowResult[Any, Any, Any],
    *,
    expected_identity: PrimitiveMappingSnapshot | None = None,
    parent: bool = True,
) -> None:
    rule, _ = study_inputs(fixture)
    validate_prediction_window_reader(
        reader,
        expected_identity=expected_identity or window.identity_snapshot,
        schedule=window.schedule,
        outcome_sessions=(),
        strategy_parameters=rule.parameters.to_primitive(),
        canonical_metadata=fixture.dataset.metadata if parent else None,
    )


@pytest.mark.parametrize(
    "outcome_name",
    ["forward10", "forward30", "forward60", "forward120", "excursion", "target_stop"],
)
def test_real_contract_roundtrip_preserves_bounded_ancestry_membership_and_outcomes(
    fixture: Fixture, tmp_path: Path, outcome_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = make_window(fixture, outcome_name)
    compact = CompactPredictionWindowResult.from_window(window)
    reader = write_compact(tmp_path / "window.jsonl", compact)
    shared = reader.shared_evidence().identity_snapshot.to_primitive()
    original_market = window.identity_snapshot.to_primitive()["market_data"]
    assert shared["market_data"] == original_market
    provenance = mapping(mapping(shared["market_data"])["intraday_provenance"])
    assert provenance["canonical_input_id"] == fixture.dataset.metadata.dataset_id
    assert provenance["causal_cutoff"] == DECISION.isoformat()
    assert "source_manifest" not in provenance
    assert mapping(shared["market_data"])["actual_last_session"] == "2024-01-02"
    assert mapping(shared["market_data"])["corporate_actions_complete"] is False
    membership = PredictionMembershipSource.capture(window.schedule, fixture.primary)
    assert reader.header()["schedule_id"] == window.schedule.schedule_id
    assert [
        item.to_primitive()["decision_timestamp"] for item in reader.iterate_decisions()
    ] == [t.isoformat() for t in membership.schedule.decision_timestamps]
    for compact_decision, original in zip(
        reader.iterate_decisions(), window.decisions, strict=True
    ):
        record = compact_decision.to_primitive()
        assert record["prediction_study_id"] == original.result.study_id
        assert (
            mapping(record["prediction_study"])["rows"]
            == original.result.to_primitive()["rows"]
        )
        assert (
            record["generated_signals"] == original.to_primitive()["generated_signals"]
        )
    encoded = compact.serialize()
    assert encoded.count(b'"intraday_provenance"') == 1
    assert encoded.count(b'"canonical_input_id"') == 1

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("offline compact verification must not execute research")

    monkeypatch.setattr(WindowProvider, "get_context_at", forbidden)
    monkeypatch.setattr(
        "quantforge.prediction.window.run_prediction_study_in_session", forbidden
    )
    verify(reader, fixture, window)
    # Both physical formats pass identical scientific verification.
    legacy = tmp_path / "legacy.json"
    legacy.write_bytes(window.serialize())
    verify(PredictionWindowReader.open(legacy), fixture, window)


def test_bounded_verification_requires_independent_ancestor(
    fixture: Fixture, tmp_path: Path
) -> None:
    window = make_window(fixture)
    reader = write_compact(
        tmp_path / "window.jsonl", CompactPredictionWindowResult.from_window(window)
    )
    with pytest.raises(
        InvalidPredictionOutputError, match="independent canonical ancestry"
    ):
        verify(reader, fixture, window, parent=False)


@pytest.mark.parametrize(
    "mutation",
    ["ancestor", "provenance", "future_cutoff", "constituent", "adjustment", "range"],
)
def test_rehashed_shared_provenance_fails_closed(
    fixture: Fixture, tmp_path: Path, mutation: str
) -> None:
    window = make_window(fixture)
    original = CompactPredictionWindowResult.from_window(window)
    identity = original.evidence.identity_snapshot.to_primitive()
    market = mapping(identity["market_data"])
    provenance = mapping(market["intraday_provenance"])
    if mutation == "ancestor":
        provenance["canonical_input_id"] = "intraday-projection-" + "0" * 64
    elif mutation == "provenance":
        provenance["canonical_provenance_id"] = "0" * 64
    elif mutation == "future_cutoff":
        provenance["causal_cutoff"] = (DECISION + timedelta(minutes=1)).isoformat()
    elif mutation == "constituent":
        bar = mapping(
            cast(list[Any], mapping(provenance["session_evidence"])["bars"])[0]
        )
        cast(list[Any], mapping(bar["bar"])["source_bar_ids"])[0] = "0" * 64
    elif mutation == "adjustment":
        market["corporate_actions_complete"] = True
    else:
        market["requested_end"] = "2024-01-03"
    # Recompute the bounded identity when its record still parses. Local hashes
    # cannot prove exact ancestry; the independently retained parent must decide.
    changed_provenance = BoundedPredictionProvenance.from_primitive(provenance)
    market["dataset_id"] = changed_provenance.view_id
    evidence = PredictionWindowEvidence(PrimitiveMappingSnapshot.capture(identity))
    decisions: list[CompactPredictionWindowDecision] = []
    for decision in original.decisions:
        record = decision.to_primitive()
        record["shared_evidence_id"] = evidence.evidence_id
        rehash_decision(record)
        decisions.append(
            CompactPredictionWindowDecision(PrimitiveMappingSnapshot.capture(record))
        )
    changed = CompactPredictionWindowResult(evidence, tuple(decisions))
    reader = write_compact(tmp_path / "corrupt.jsonl", changed)
    reader.verify_integrity()
    with pytest.raises(InvalidPredictionOutputError):
        verify(reader, fixture, window, expected_identity=evidence.identity_snapshot)


def test_cutoff_and_outcome_identity_changes_preserve_membership(
    fixture: Fixture,
) -> None:
    baseline = make_window(fixture)
    earlier = make_window(fixture, cutoff=DECISION - timedelta(minutes=1))
    later_outcome = make_window(fixture, "forward60")
    windows = [
        CompactPredictionWindowResult.from_window(item)
        for item in (baseline, earlier, later_outcome)
    ]
    assert len({item.window_id for item in windows}) == 3
    assert len({item.window_result_id for item in windows}) == 3
    assert all(item.evidence.schedule == baseline.schedule for item in windows)
    assert (
        baseline.decisions[0].result.rows[0].outcome.values.raw_return
        == earlier.decisions[0].result.rows[0].outcome.values.raw_return
    )
    with pytest.raises(InvalidPredictionOutputError):
        replace(windows[0], evidence=windows[1].evidence)
