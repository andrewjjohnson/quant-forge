"""QF-52 causal projection, provenance, offline inspection, and source binding."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import MarketDataset, validate_market_dataset
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import (
    dataset_identity_matches,
    serialize_bars_csv,
    sha256_hex,
)
from quantforge.data.models import (
    BoundedPredictionProvenance,
    IntradayPredictionProvenance,
)
from quantforge.data.prediction_inputs import (
    validate_prediction_provenance,
    validate_prediction_source,
)
from quantforge.data.prediction_views import (
    bounded_prediction_view,
    validate_bounded_prediction_ancestry,
    validate_prediction_view_lineage,
)
from quantforge.experiments._producer_integrity import validate_prediction_manifest
from quantforge.prediction import (
    PredictionStudy,
    SignalFeatureCandidate,
    intraday_excursion_outcome,
    intraday_forward_return_outcome,
    intraday_target_stop_outcome,
    run_prediction_study,
)
from quantforge.prediction.models import PredictionMarketData
from quantforge.walk_forward.partitions import (
    prediction_metadata_prefix,
    project_dataset,
)
from tests.integration.test_intraday_prediction_provenance import (
    DECISION,
    Fixture,
    cached_fixture,
    study_inputs,
)
from tests.integration.test_intraday_prediction_provenance import fixture as fixture


def test_exact_reproducer_and_no_future_evidence(fixture: Fixture) -> None:
    canonical = fixture.dataset
    original = canonical.metadata.intraday_provenance
    assert isinstance(original, IntradayPredictionProvenance)
    snapshot = original.to_primitive()
    assert validate_market_dataset(canonical) == ()
    view = prediction_metadata_prefix(canonical, DECISION)
    assert validate_market_dataset(view) == ()
    assert dataset_identity_matches(view)
    assert view.bars == canonical.bars[:1]
    assert view.metadata.actual_last_session == date(2024, 1, 2)
    assert view.metadata.requested_end == date(2024, 1, 2)
    assert view.metadata.bar_count == 1
    assert view.metadata.retrieved_at == canonical.metadata.retrieved_at
    assert view.metadata.data_sha256 == sha256_hex(serialize_bars_csv(view.bars))
    assert (
        view.metadata.corporate_action_policy
        == canonical.metadata.corporate_action_policy
    )
    assert view.metadata.corporate_actions_complete is False
    bounded = view.metadata.intraday_provenance
    assert isinstance(bounded, BoundedPredictionProvenance)
    assert bounded.causal_cutoff == DECISION
    assert bounded.canonical_input_id == canonical.metadata.dataset_id
    assert bounded.canonical_provenance_id == configuration_identity(snapshot)
    assert bounded.source_dataset_id == original.source_dataset_id
    assert bounded.source_retrieved_at == canonical.metadata.retrieved_at
    assert not hasattr(bounded, "source_manifest")
    assert not hasattr(bounded, "source_bar_evidence")
    assert not hasattr(bounded, "source_raw_snapshot_ids")
    evidence = bounded.session_evidence.to_primitive()
    assert set(evidence) == {"bars"}
    entries = cast(list[PrimitiveMapping], evidence["bars"])
    assert len(entries) == 1
    bar = cast(PrimitiveMapping, entries[0]["bar"])
    assert bar["session_dates"] == ["2024-01-02"]
    assert datetime.fromisoformat(cast(str, bar["end_timestamp"])) <= DECISION
    assert BoundedPredictionProvenance.from_primitive(bounded.to_primitive()) == bounded
    validate_bounded_prediction_ancestry(view, canonical)
    assert original.to_primitive() == snapshot
    assert validate_market_dataset(canonical) == ()


def test_deterministic_cutoffs_and_reprojection(fixture: Fixture) -> None:
    canonical = fixture.dataset
    first = bounded_prediction_view(canonical, DECISION)
    assert first == bounded_prediction_view(canonical, DECISION)
    assert first == bounded_prediction_view(first, DECISION)
    # Even equal visible prices cannot alias different information cutoffs.
    later = bounded_prediction_view(canonical, DECISION + timedelta(minutes=1))
    assert later.bars == first.bars
    assert later.metadata.dataset_id != first.metadata.dataset_id
    close = datetime(2024, 1, 3, 21, tzinfo=UTC)
    before = bounded_prediction_view(canonical, close - timedelta(microseconds=1))
    at = bounded_prediction_view(canonical, close)
    assert len(before.bars) == 1
    assert at.bars == canonical.bars
    assert at.metadata.dataset_id != canonical.metadata.dataset_id
    assert at.metadata.dataset_id != before.metadata.dataset_id
    assert bounded_prediction_view(at, DECISION) == first
    with pytest.raises(ValidationError, match="cannot widen"):
        bounded_prediction_view(first, close)
    with pytest.raises(ValidationError, match="timezone-aware"):
        bounded_prediction_view(canonical, DECISION.replace(tzinfo=None))
    with pytest.raises(ValidationError, match="prior completed"):
        bounded_prediction_view(canonical, datetime(2024, 1, 2, 20, tzinfo=UTC))
    assert project_dataset(canonical, date(2024, 1, 2), date(2024, 1, 3)) == at
    last = bounded_prediction_view(canonical, close, start=date(2024, 1, 3))
    assert last.bars == canonical.bars[1:]
    validate_bounded_prediction_ancestry(last, canonical)
    with pytest.raises(ValidationError, match="visible observed session"):
        bounded_prediction_view(canonical, close, start=date(2024, 1, 1))


@pytest.mark.parametrize(
    "field",
    [
        "canonical_input_id",
        "canonical_provenance_id",
        "source_dataset_id",
        "causal_cutoff",
        "bars_fingerprint",
        "retrieved_at",
        "bar_count",
        "requested_end",
        "corporate_action_policy",
        "family_id",
    ],
)
def test_malformed_bounded_records_reject(fixture: Fixture, field: str) -> None:
    view = bounded_prediction_view(fixture.dataset, DECISION)
    record = PredictionMarketData.from_qf3(view.metadata).to_primitive()
    provenance = cast(PrimitiveMapping, record["intraday_provenance"])
    if field in {
        "retrieved_at",
        "bar_count",
        "requested_end",
        "corporate_action_policy",
    }:
        record[field] = {
            "retrieved_at": "1970-01-01T00:00:00+00:00",
            "bar_count": 2,
            "requested_end": "2024-01-03",
            "corporate_action_policy": (
                "separate_provider_reported_cash_dividends_and_splits"
            ),
        }[field]
    else:
        provenance[field] = "fabricated"
    with pytest.raises((ValidationError, ValueError)):
        validate_prediction_provenance(record)


def test_rehashed_future_evidence_and_false_digest_reject(fixture: Fixture) -> None:
    view = bounded_prediction_view(fixture.dataset, DECISION)
    canonical = fixture.dataset.metadata.intraday_provenance
    original = view.metadata.intraday_provenance
    assert isinstance(canonical, IntradayPredictionProvenance)
    assert isinstance(original, BoundedPredictionProvenance)
    future = cast(PrimitiveMapping, canonical.session_evidence.to_primitive()["bars"])[
        "bars"
    ]
    for changed in (
        replace(
            original,
            session_evidence=PrimitiveMappingSnapshot.capture({"bars": future}),
        ),
        replace(original, bars_fingerprint="0" * 64),
    ):
        record = PredictionMarketData.from_qf3(view.metadata).to_primitive()
        record["intraday_provenance"] = changed.to_primitive()
        record["dataset_id"] = changed.view_id
        with pytest.raises(ValidationError, match=r"cutoff|digest"):
            validate_prediction_provenance(record)


def test_independent_ancestry_rejects_rehashed_local_forgery(fixture: Fixture) -> None:
    view = bounded_prediction_view(fixture.dataset, DECISION)
    original = view.metadata.intraday_provenance
    assert isinstance(original, BoundedPredictionProvenance)
    evidence = original.session_evidence.to_primitive()
    entry = cast(list[PrimitiveMapping], evidence["bars"])[0]
    bar = cast(PrimitiveMapping, entry["bar"])
    bar["close"] = "100.1"
    entry["bar_id"] = configuration_identity(bar)
    bars = (replace(view.bars[0], close=Decimal("100.1")),)
    fingerprint = sha256_hex(serialize_bars_csv(bars))
    changed = replace(
        original,
        session_evidence=PrimitiveMappingSnapshot.capture(evidence),
        bars_fingerprint=fingerprint,
    )
    assert changed.view_id != original.view_id
    forged = MarketDataset(
        bars,
        replace(
            view.metadata,
            dataset_id=changed.view_id,
            intraday_provenance=changed,
            data_sha256=fingerprint,
            raw_sha256=configuration_identity(changed.to_primitive()),
            raw_location=f"raw/{configuration_identity(changed.to_primitive())}.json",
            normalized_location=f"datasets/{changed.view_id}/bars.csv",
            corporate_actions_location=f"datasets/{changed.view_id}/corporate_actions.json",
        ),
    )
    # Local consistency does not claim to authenticate a flat-hashed parent.
    assert validate_market_dataset(forged) == ()
    with pytest.raises(ValidationError, match="canonical ancestry"):
        validate_bounded_prediction_ancestry(forged, fixture.dataset)


def test_different_sources_have_distinct_views(
    fixture: Fixture, tmp_path: Path
) -> None:
    other = cached_fixture(tmp_path, provider="other-provider")
    first = bounded_prediction_view(fixture.dataset, DECISION)
    second = bounded_prediction_view(other.dataset, DECISION)
    assert first.bars == second.bars
    assert first.metadata.dataset_id != second.metadata.dataset_id
    with pytest.raises(ValidationError, match="canonical ancestry"):
        validate_bounded_prediction_ancestry(first, other.dataset)
    with pytest.raises(ValidationError, match="lineage"):
        validate_prediction_source(first, other.primary)


@pytest.mark.parametrize(
    "change", ["cutoff", "source", "provenance", "contents", "evidence"]
)
def test_offline_plan_binding_rejects_rehashed_view_changes(
    fixture: Fixture, change: str
) -> None:
    view = bounded_prediction_view(fixture.dataset, DECISION)
    record = PredictionMarketData.from_qf3(view.metadata).to_primitive()
    provenance = cast(PrimitiveMapping, record["intraday_provenance"])
    if change == "cutoff":
        provenance["causal_cutoff"] = (DECISION + timedelta(minutes=1)).isoformat()
    elif change == "source":
        provenance["canonical_input_id"] = "intraday-projection-" + "0" * 64
    elif change == "provenance":
        provenance["canonical_provenance_id"] = "0" * 64
    else:
        entry = cast(
            list[PrimitiveMapping],
            cast(PrimitiveMapping, provenance["session_evidence"])["bars"],
        )[0]
        bar = cast(PrimitiveMapping, entry["bar"])
        if change == "contents":
            bar["close"] = "100.1"
            fingerprint = sha256_hex(
                serialize_bars_csv((replace(view.bars[0], close=Decimal("100.1")),))
            )
            provenance["bars_fingerprint"] = record["bars_fingerprint"] = fingerprint
        else:
            cast(list[str], bar["source_bar_ids"])[0] = "0" * 64
        entry["bar_id"] = configuration_identity(bar)
    changed = BoundedPredictionProvenance.from_primitive(provenance)
    record["dataset_id"] = changed.view_id
    assert changed.view_id != view.metadata.dataset_id
    assert validate_prediction_provenance(record) == changed
    with pytest.raises(ValidationError, match="canonical plan ancestry or cutoff"):
        validate_prediction_view_lineage(record, fixture.dataset.metadata, DECISION)


@pytest.mark.parametrize("kind", ["return", "excursion", "target_stop"])
def test_outcomes_and_offline_manifests_keep_source_binding(
    fixture: Fixture, kind: str
) -> None:
    view = bounded_prediction_view(fixture.dataset, DECISION)
    rule, context = study_inputs(fixture)
    if kind == "return":
        outcome = intraday_forward_return_outcome(
            timedelta(minutes=30), fixture.primary
        )
    elif kind == "excursion":
        outcome = intraday_excursion_outcome(timedelta(minutes=60), fixture.primary)
    else:
        outcome = intraday_target_stop_outcome(
            timedelta(minutes=60),
            fixture.primary,
            target_percentage=Decimal("0.003"),
            stop_percentage=Decimal("0.002"),
        )
    study = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
        rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
    )
    original = run_prediction_study(fixture.dataset, study, context_provider=context)
    bounded = run_prediction_study(view, study, context_provider=context)
    assert len(original.rows) == len(bounded.rows) > 0
    assert [row.outcome.values.to_primitive() for row in original.rows] == [
        row.outcome.values.to_primitive() for row in bounded.rows
    ]
    validate_prediction_manifest(
        cast(PrimitiveMapping, bounded.to_primitive()["manifest"])
    )
