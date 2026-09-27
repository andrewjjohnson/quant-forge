"""Count invariant work for 1,000 explicit anchors, without timing thresholds."""

from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from time import perf_counter
from typing import Any, cast

import pytest

from quantforge.data import IntradayBar
from quantforge.prediction import (
    IntradayForwardReturnOutcomeLabeler,
    OutcomeTemporalConfiguration,
    evaluate_outcome_request,
    prepared_outcomes,
)
from quantforge.prediction.prepared_outcomes import (
    PreparedOutcomeSource,
    PreparedOutcomeSources,
)
from quantforge.prediction.timestamp_execution import bounded_outcome_source
from tests.integration.test_intraday_prediction_provenance import cached_fixture
from tests.unit.helpers import SESSIONS
from tests.unit.prediction.test_outcome_resolution import request


def test_dense_explicit_anchors_prepare_once(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    fixture = cached_fixture(tmp_path, session_dates=SESSIONS[:7])
    dataset, source = fixture.dataset, fixture.primary
    labeler = IntradayForwardReturnOutcomeLabeler(
        OutcomeTemporalConfiguration.elapsed_duration(
            timedelta(minutes=30), source.timeframe
        )
    )
    anchors = tuple(
        replace(
            request(source, decision=bar.end_timestamp, session=bar.session_date),
            dataset_id=dataset.metadata.dataset_id,
            dataset_fingerprint=dataset.metadata.data_sha256,
            outcome_configuration_id=labeler.configuration_id,
        )
        for bar in cast(tuple[IntradayBar, ...], source.bars[:1000])
    )
    assert len(anchors) == 1000

    def label(anchor: Any, preparation: PreparedOutcomeSource | None = None) -> Any:
        bounded, resolution = bounded_outcome_source(
            source if preparation is None else preparation.source,
            anchor,
            prepared=preparation,
        )
        result = evaluate_outcome_request(
            labeler, dataset, anchor, source=bounded, resolution=resolution
        )
        assert result is not None
        return result.values.to_primitive()

    reference = tuple(label(anchor) for anchor in anchors)
    counts = {"validation": 0, "indexes": 0, "requests": 0, "labels": 0}
    seconds = {"validation": 0.0, "indexes": 0.0}
    validate, build, checked, calculate = (
        prepared_outcomes.validate_prediction_source,
        prepared_outcomes._build_indexes,  # pyright: ignore[reportPrivateUsage]
        PreparedOutcomeSource.validate_request,
        IntradayForwardReturnOutcomeLabeler.label_request,
    )  # pyright: ignore[reportPrivateUsage]

    def validation(*args: Any, **kwargs: Any) -> Any:
        begin = perf_counter()
        counts["validation"] += 1
        result = validate(*args, **kwargs)
        seconds["validation"] += perf_counter() - begin
        return result

    def indexes(*args: Any, **kwargs: Any) -> Any:
        begin = perf_counter()
        counts["indexes"] += 1
        result = build(*args, **kwargs)
        seconds["indexes"] += perf_counter() - begin
        return result

    def check(*args: Any, **kwargs: Any) -> Any:
        counts["requests"] += 1
        return checked(*args, **kwargs)

    def calculated(*args: Any, **kwargs: Any) -> Any:
        counts["labels"] += 1
        return calculate(*args, **kwargs)

    monkeypatch.setattr(prepared_outcomes, "validate_prediction_source", validation)
    monkeypatch.setattr(prepared_outcomes, "_build_indexes", indexes)
    monkeypatch.setattr(PreparedOutcomeSource, "validate_request", check)
    monkeypatch.setattr(
        IntradayForwardReturnOutcomeLabeler, "label_request", calculated
    )
    begin = perf_counter()
    prepared = PreparedOutcomeSources().prepare(dataset, source)
    preparation_seconds = perf_counter() - begin
    assert prepared is not None
    begin = perf_counter()
    actual = tuple(label(anchor, prepared) for anchor in anchors)
    labeling_seconds = perf_counter() - begin
    assert actual == reference
    assert counts == {"validation": 1, "indexes": 1, "requests": 1000, "labels": 1000}
    for name, measurement in {
        **seconds,
        "preparation": preparation_seconds,
        "dense_total": labeling_seconds,
        "dense_per_label": labeling_seconds / 1000,
    }.items():
        record_property(name + "_seconds", measurement)
    for name, count in counts.items():
        record_property(name + "_count", count)
