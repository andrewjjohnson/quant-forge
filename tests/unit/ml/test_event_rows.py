"""QF-67 row ordering, equal-timestamp ties, identities and input boundaries."""

import random
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from quantforge.ml import (
    EventDatasetIntegrityError,
    EventRow,
    NonAuthoritativeSourceError,
    TargetLabel,
)
from quantforge.ml.dataset import label_summary, logical_rows_sha256
from quantforge.ml.sources import EventSourceWindow, reject_non_authoritative
from quantforge.validation import PartitionRole

STAMP = datetime(2024, 12, 23, 16, 0, tzinfo=UTC)


def event(
    *,
    role: PartitionRole = PartitionRole.SELECTION,
    fold: int | None = 0,
    signal: int = 0,
    identity: str = "a",
    stamp: datetime = STAMP,
    label: TargetLabel | None = None,
) -> EventRow:
    return EventRow(
        source_observation_id=identity,
        source_index=0,
        decision_timestamp=stamp,
        signal_session=date(2024, 12, 23),
        decision_sequence=45,
        signal_index=signal,
        context_id="context",
        prediction_study_id="study",
        direction="up",
        disposition="accepted",
        partition_role=role,
        fold_id=None if fold is None else f"fold-{fold}",
        fold_index=fold,
        features=("1.5", None),
        label=label or TargetLabel(True, "available", "0.01", "outcome"),
    )


def test_equal_timestamps_use_the_documented_stable_tie_break() -> None:
    expected = [
        event(fold=0, role=PartitionRole.SELECTION, signal=0, identity="b"),
        event(fold=0, role=PartitionRole.SELECTION, signal=1, identity="a"),
        event(fold=0, role=PartitionRole.WALK_FORWARD_TEST, identity="a"),
        event(fold=1, role=PartitionRole.DEVELOPMENT, identity="a"),
        event(fold=1, role=PartitionRole.SELECTION, signal=1, identity="a"),
        event(fold=1, role=PartitionRole.SELECTION, signal=1, identity="b"),
        event(fold=None, role=PartitionRole.FINAL_HOLDOUT, identity="a"),
    ]
    earlier = event(
        fold=5, role=PartitionRole.FINAL_HOLDOUT, stamp=STAMP.replace(minute=0, hour=15)
    )
    for seed in range(20):
        shuffled = [earlier, *expected]
        random.Random(seed).shuffle(shuffled)
        assert sorted(shuffled, key=EventRow.sort_key) == [earlier, *expected]


def test_logical_rows_digest_covers_every_row_field_and_order() -> None:
    rows = [event(identity="a"), event(identity="b", signal=1)]
    digest = logical_rows_sha256(rows)
    assert logical_rows_sha256(list(rows)) == digest
    assert logical_rows_sha256(rows[::-1]) != digest
    for changed in (
        replace(rows[0], features=("1.50", None)),
        replace(rows[0], features=("1.5", "0")),
        replace(rows[0], label=TargetLabel(None, "session_overflow", None, "outcome")),
        replace(rows[0], partition_role=PartitionRole.WALK_FORWARD_TEST),
        replace(rows[0], fold_id="fold-9"),
        replace(rows[0], context_id=None),
    ):
        assert logical_rows_sha256([changed, rows[1]]) != digest
    primitive = rows[0].to_primitive(0)
    assert primitive["features"] == ["1.5", None]
    assert primitive["target"] == {
        "value": True,
        "status": "available",
        "source_value": "0.01",
        "outcome_id": "outcome",
    }


def test_label_summary_counts_unavailable_separately_from_negatives() -> None:
    rows = [
        event(identity="a"),
        event(identity="b", label=TargetLabel(False, "available", "0", "o")),
        event(identity="c", label=TargetLabel(None, "session_overflow", None, "o")),
        event(
            identity="d",
            role=PartitionRole.WALK_FORWARD_TEST,
            label=TargetLabel(None, "dataset_end", None, "o"),
        ),
    ]
    summary = label_summary(rows)
    assert summary["overall"] == {
        "rows": 4,
        "positive": 1,
        "negative": 1,
        "unavailable": 2,
        "unavailable_by_status": {"dataset_end": 1, "session_overflow": 1},
    }
    partitions = cast(list[dict[str, Any]], summary["by_partition"])
    assert [(p["partition_role"], p["rows"]) for p in partitions] == [
        ("validation_selection", 3),
        ("walk_forward_test", 1),
    ]


def test_non_authoritative_inputs_are_refused_at_the_boundary() -> None:
    for value in (
        SimpleNamespace(authoritative=False),
        SimpleNamespace(mode="exploratory"),
        Path("reports/exploration/ema.rapid.json"),
        Path("reports/exploration/EMA.RAPID.JSON"),
    ):
        with pytest.raises(NonAuthoritativeSourceError):
            reject_non_authoritative(value)
    for value in (None, Path("reports/walk-forward/study"), SimpleNamespace()):
        reject_non_authoritative(value)


def test_sources_cannot_be_constructed_outside_the_verified_loaders() -> None:
    with pytest.raises(EventDatasetIntegrityError, match="verified"):
        EventSourceWindow(
            PartitionRole.WALK_FORWARD_TEST,
            "fold",
            0,
            cast(Any, None),
            cast(Any, None),
            cast(Any, None),
            "selection",
            cast(Any, None),
            cast(Any, None),
        )
