"""Prepared sources preserve reference labels at adversarial calendar boundaries."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, cast

import pytest

from quantforge.data import IntradayBar, TimeframeBarSeries
from quantforge.prediction import (
    IntradayForwardReturnOutcomeLabeler,
    IntradayPathOutcomeLabeler,
    OutcomeTemporalConfiguration,
    evaluate_outcome_request,
)
from quantforge.prediction.outcome_temporal import OutcomeTemporalError
from quantforge.prediction.prepared_outcomes import PreparedOutcomeSources
from quantforge.prediction.timestamp_execution import bounded_outcome_source
from quantforge.timeframes import BarCompletion, DevelopingBarExposure
from tests.unit.helpers import make_dataset
from tests.unit.prediction.test_intraday_forward_return import TWO_MINUTES
from tests.unit.prediction.test_intraday_path import path_source
from tests.unit.prediction.test_outcome_resolution import SESSION, request, timestamp


def altered(
    source: TimeframeBarSeries, bars: tuple[IntradayBar, ...]
) -> TimeframeBarSeries:
    return TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        source.dataset_reference,
        source.timeframe,
        bars,
        dataset_family_manifest_id=source.dataset_family_manifest_id,
    )


@pytest.mark.parametrize("minutes", [10, 30, 31, 60, 120])
@pytest.mark.parametrize(
    "case",
    [
        "normal",
        "early_close",
        "at_close",
        "overflow",
        "holiday",
        "gap",
        "end",
        "developing",
    ],
)
def test_exact_resolution_bounds_and_labels(minutes: int, case: str) -> None:
    dataset = make_dataset(("999",))
    session = date(2024, 7, 3) if case in ("early_close", "holiday") else SESSION
    decision = timestamp(11, 20, session=session)
    if case == "at_close":
        decision = timestamp(16, 0) - timedelta(minutes=minutes + minutes % 2)
    elif case == "overflow":
        decision = timestamp(15, 58)
    timeframe = replace(
        TWO_MINUTES, developing_bar_exposure=DevelopingBarExposure.INCLUDE
    )
    source = path_source(session=session, timeframe=timeframe)
    if case == "holiday":
        source = altered(
            source,
            cast(
                tuple[IntradayBar, ...],
                source.bars
                + path_source(session=date(2024, 7, 5), timeframe=timeframe).bars,
            ),
        )
    if case in ("gap", "end", "developing"):
        boundary = decision + timedelta(minutes=2)
        if case == "developing":
            boundary = decision + timedelta(minutes=minutes + minutes % 2 - 2)
        bars = cast(tuple[IntradayBar, ...], source.bars)
        if case == "gap":
            bars = tuple(bar for bar in bars if bar.end_timestamp != boundary)
        else:
            bars = tuple(bar for bar in bars if bar.end_timestamp <= boundary)
            if case == "developing":
                bars += (
                    replace(
                        cast(IntradayBar, source.bars[len(bars)]),
                        end_timestamp=boundary + timedelta(minutes=1),
                        completion=BarCompletion.DEVELOPING,
                    ),
                )
        source = altered(source, bars)
    prepared = PreparedOutcomeSources().prepare(dataset, source)
    assert prepared is not None
    for labeler_type in (
        IntradayForwardReturnOutcomeLabeler,
        IntradayPathOutcomeLabeler,
    ):
        labeler = labeler_type(
            OutcomeTemporalConfiguration.elapsed_duration(
                timedelta(minutes=minutes), timeframe
            )
        )
        anchor = replace(
            request(
                source,
                decision=decision,
                session=session,
                duration=timedelta(minutes=minutes),
            ),
            dataset_id=dataset.metadata.dataset_id,
            dataset_fingerprint=dataset.metadata.data_sha256,
            outcome_configuration_id=labeler.configuration_id,
        )
        reference, old_resolution = bounded_outcome_source(source, anchor)
        bounded, new_resolution = bounded_outcome_source(
            prepared.source, anchor, prepared=prepared
        )
        assert bounded == reference
        assert new_resolution.to_primitive() == old_resolution.to_primitive()
        old = evaluate_outcome_request(
            cast(Any, labeler),
            dataset,
            anchor,
            source=reference,
            resolution=old_resolution,
        )
        new = evaluate_outcome_request(
            cast(Any, labeler),
            dataset,
            anchor,
            source=bounded,
            resolution=new_resolution,
        )
        assert old is not None
        assert new is not None
        assert old.values.to_primitive() == new.values.to_primitive()


def test_identity_authenticates_contents_and_immutable_backing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantforge.prediction import prepared_outcomes

    dataset = make_dataset(("999",))
    source = path_source()
    registry = PreparedOutcomeSources()
    prepared = registry.prepare(dataset, source)
    assert prepared is not None

    # Equal independently reconstructed content reuses the scientific key; identity
    # is not the Python object address. Its admission still checks full content.
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("repeated source validation/index construction")

    with monkeypatch.context() as guarded:
        guarded.setattr(prepared_outcomes, "validate_prediction_source", forbidden)
        guarded.setattr(prepared_outcomes, "_build_indexes", forbidden)
        assert registry.prepare(dataset, deepcopy(prepared.source)) is prepared
        assert registry.prepare(dataset, prepared.source) is prepared
    assert prepared.source == source
    assert len(prepared.timestamps) == len(source.bars)
    with pytest.raises(FrozenInstanceError):
        setattr(prepared.source, "bars", ())
    with pytest.raises(TypeError):
        cast(Any, prepared.session_ranges)[SESSION] = (0, 0)
    # A changed price with stale claimed IDs must not alias the earlier source.
    first = cast(IntradayBar, source.bars[0])
    changed = altered(
        source,
        (
            replace(first, high=Decimal("101")),
            *cast(tuple[IntradayBar, ...], source.bars[1:]),
        ),
    )
    second = registry.prepare(dataset, changed)
    assert second is not None
    assert second.compatibility_id != prepared.compatibility_id
    assert PreparedOutcomeSources().prepare(dataset, prepared.source) == prepared


@pytest.mark.parametrize(
    "dimension",
    ["dataset", "fingerprint", "source", "timeframe", "session", "adjustment"],
)
def test_incompatible_request_or_input_rejects(dimension: str) -> None:
    dataset = make_dataset(("999",))
    registry = PreparedOutcomeSources()
    prepared = registry.prepare(dataset, path_source())
    assert prepared is not None
    anchor = replace(
        request(prepared.source),
        dataset_id=dataset.metadata.dataset_id,
        dataset_fingerprint=dataset.metadata.data_sha256,
    )
    if dimension in ("session", "adjustment"):
        metadata = replace(
            dataset.metadata,
            **(
                {"calendar": "24/7"}
                if dimension == "session"
                else {"adjusted_fields_used": True}
            ),
        )
        with pytest.raises(OutcomeTemporalError, match="prediction input differs"):
            registry.prepare(replace(dataset, metadata=metadata), prepared.source)
        return
    if dimension == "dataset":
        anchor = replace(anchor, dataset_id="foreign")
    elif dimension == "fingerprint":
        anchor = replace(anchor, dataset_fingerprint="foreign")
    elif dimension == "source":
        anchor = replace(
            anchor,
            source_reference=replace(
                prepared.source.dataset_reference, dataset_id="foreign"
            ),
        )
    else:
        other = replace(
            TWO_MINUTES, developing_bar_exposure=DevelopingBarExposure.INCLUDE
        )
        anchor = replace(
            anchor,
            temporal_configuration=OutcomeTemporalConfiguration.elapsed_duration(
                timedelta(minutes=30), other
            ),
            source_reference=replace(
                prepared.source.dataset_reference,
                timeframe_configuration_id=other.configuration_id,
            ),
        )
    with pytest.raises(
        OutcomeTemporalError, match=r"differs from request|immutable reference"
    ):
        bounded_outcome_source(prepared.source, anchor, prepared=prepared)


def test_mutable_source_retains_reference_path() -> None:
    from tests.unit.prediction.test_source_sharing import AnnotatedText

    dataset = make_dataset(("999",))
    source = path_source()
    first = cast(IntradayBar, source.bars[0])
    source = altered(
        source,
        (
            replace(
                first,
                provenance=replace(
                    first.provenance, provider_name=AnnotatedText("fixture")
                ),
            ),
            *cast(tuple[IntradayBar, ...], source.bars[1:]),
        ),
    )
    assert PreparedOutcomeSources().prepare(dataset, source) is None


def test_exact_anchor_failure_is_preserved() -> None:
    dataset = make_dataset(("999",))
    source = path_source()
    prepared = PreparedOutcomeSources().prepare(dataset, source)
    assert prepared is not None
    labeler = IntradayForwardReturnOutcomeLabeler(
        OutcomeTemporalConfiguration.elapsed_duration(
            timedelta(minutes=30), source.timeframe
        )
    )
    anchor = replace(
        request(source, decision=timestamp(11, 21)),
        dataset_id=dataset.metadata.dataset_id,
        dataset_fingerprint=dataset.metadata.data_sha256,
        outcome_configuration_id=labeler.configuration_id,
    )
    from quantforge.prediction.errors import InvalidPredictionDataError

    for capability in (None, prepared):
        bounded, resolution = bounded_outcome_source(
            source if capability is None else capability.source,
            anchor,
            prepared=capability,
        )
        with pytest.raises(
            InvalidPredictionDataError, match="exact completed decision"
        ):
            evaluate_outcome_request(
                labeler, dataset, anchor, source=bounded, resolution=resolution
            )


@pytest.mark.parametrize(
    ("high", "low", "expected"),
    [
        ("100.3", "100", "target_first"),
        ("100", "99.8", "stop_first"),
        ("100.3", "99.8", "both_same_bar"),
        ("100", "100", "neither"),
    ],
)
def test_mfe_mae_target_stop_and_ambiguity_are_exact(
    high: str, low: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quantforge.prediction import (
        IntradayExcursionEvaluator,
        IntradayTargetStopEvaluator,
        PredictionDirection,
    )
    from tests.unit.prediction import test_intraday_path
    from tests.unit.prediction.test_feature_outcomes import (
        _candidate,  # pyright: ignore[reportPrivateUsage]
    )

    source = path_source(prices={timestamp(11, 22): (high, low)})
    old = test_intraday_path.path_label(source)
    prepared = PreparedOutcomeSources().prepare(make_dataset(("999",)), source)
    assert prepared is not None

    def indexed(source: Any, anchor: Any) -> Any:
        assert source == prepared.source
        return bounded_outcome_source(prepared.source, anchor, prepared=prepared)

    monkeypatch.setattr(test_intraday_path, "bounded_outcome_source", indexed)
    new = test_intraday_path.path_label(source)
    assert new.to_primitive() == old.to_primitive()
    candidate = _candidate(SESSION, PredictionDirection.UP)
    excursion = IntradayExcursionEvaluator()
    assert excursion.evaluate(candidate, new) == excursion.evaluate(candidate, old)
    target = IntradayTargetStopEvaluator(Decimal("0.003"), Decimal("0.002"))
    assert target.evaluate(candidate, new) == target.evaluate(candidate, old)
    assert target.evaluate(candidate, new).label.value == expected
