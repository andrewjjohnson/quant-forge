"""Construction-time immutability of exported partition membership records."""

from dataclasses import replace
from datetime import UTC, date, datetime
from typing import cast

import pytest

from quantforge.data.identity import canonical_json_bytes
from quantforge.timeframes import ExchangeSessionPolicy
from quantforge.validation import (
    ExchangeSessionBoundary,
    PurgedPartitionObservations,
    TimestampBoundary,
    ValidationBoundary,
    ValidationPlanError,
    WindowObservationSelection,
)

COLLECTION_FIELDS = ("retained", "purged", "warm_up_context", "study_observations")
IDENTITY_FIELDS = (
    ("purge", "plan_id"),
    ("purge", "fold_id"),
    ("purge", "source_window_id"),
    ("purge", "protected_window_id"),
    ("purge", "source_timeframe_configuration_id"),
    ("selection", "window_id"),
    ("selection", "source_timeframe_configuration_id"),
    ("selection", "source_family_manifest_id"),
)
SESSION_A = ExchangeSessionBoundary(date(2024, 1, 2))
SESSION_B = ExchangeSessionBoundary(date(2024, 1, 3))
SESSION_C = ExchangeSessionBoundary(date(2024, 1, 4))
TIMESTAMP_A = TimestampBoundary(datetime(2024, 1, 2, 15, tzinfo=UTC))
TIMESTAMP_B = TimestampBoundary(datetime(2024, 1, 3, 15, tzinfo=UTC))
LONDON_B = ExchangeSessionBoundary(
    date(2024, 1, 3), ExchangeSessionPolicy("XLON", "Europe/London")
)


def _result(
    field_name: str, observations: tuple[ValidationBoundary, ...]
) -> PurgedPartitionObservations | WindowObservationSelection:
    identity = "a" * 64
    if field_name in {"retained", "purged"}:
        return PurgedPartitionObservations(
            plan_id=identity,
            fold_id=identity,
            source_window_id=identity,
            protected_window_id=identity,
            retained=observations if field_name == "retained" else (),
            purged=observations if field_name == "purged" else (),
            source_dataset_id=identity,
            source_timeframe_configuration_id=identity,
        )
    return WindowObservationSelection(
        window_id=identity,
        warm_up_context=observations if field_name == "warm_up_context" else (),
        study_observations=observations if field_name == "study_observations" else (),
        source_dataset_id=identity,
        source_timeframe_configuration_id=identity,
    )


def _paired_result(
    kind: str,
    earlier: tuple[ValidationBoundary, ...],
    later: tuple[ValidationBoundary, ...],
) -> PurgedPartitionObservations | WindowObservationSelection:
    result = _result("retained" if kind == "purge" else "warm_up_context", earlier)
    if isinstance(result, PurgedPartitionObservations):
        return replace(result, purged=later)
    return replace(result, study_observations=later)


@pytest.mark.parametrize(("kind", "field_name"), IDENTITY_FIELDS)
@pytest.mark.parametrize(
    "invalid", ["", " ", 1, True, "a" * 63, "a" * 65, "A" * 64, "g" * 64]
)
def test_result_rejects_malformed_provenance_hashes(
    kind: str,
    field_name: str,
    invalid: object,
) -> None:
    result = _paired_result(kind, (), ())
    with pytest.raises(ValidationPlanError, match=field_name):
        replace(result, **{field_name: invalid})


@pytest.mark.parametrize(("kind", "field_name"), IDENTITY_FIELDS[:-1])
def test_result_requires_nonoptional_provenance_hashes(
    kind: str, field_name: str
) -> None:
    with pytest.raises(ValidationPlanError, match=field_name):
        replace(_paired_result(kind, (), ()), **{field_name: None})


@pytest.mark.parametrize("kind", ["purge", "selection"])
@pytest.mark.parametrize("invalid", ["", " \t", None, 1, True])
def test_result_rejects_missing_or_untyped_dataset_identity(
    kind: str, invalid: object
) -> None:
    with pytest.raises(ValidationPlanError, match="source_dataset_id"):
        replace(_paired_result(kind, (), ()), source_dataset_id=cast(str, invalid))


@pytest.mark.parametrize("kind", ["purge", "selection"])
def test_result_preserves_opaque_dataset_ids_and_valid_hashes(kind: str) -> None:
    result = replace(
        _paired_result(kind, (SESSION_A,), (SESSION_B,)),
        source_dataset_id="provider:immutable-snapshot",
    )
    original = canonical_json_bytes(result.to_primitive())
    assert canonical_json_bytes(replace(result).to_primitive()) == original
    assert result.source_dataset_id == "provider:immutable-snapshot"
    if isinstance(result, WindowObservationSelection):
        assert result.source_family_manifest_id is None
        family_result = replace(result, source_family_manifest_id="b" * 64)
        assert family_result.selection_id != result.selection_id
        assert family_result.to_primitive()["source_family_manifest_id"] == "b" * 64


@pytest.mark.parametrize("field_name", COLLECTION_FIELDS)
@pytest.mark.parametrize(
    "observations",
    [
        (SESSION_A, SESSION_A),
        (SESSION_B, SESSION_A),
        (SESSION_A, TIMESTAMP_B),
        (SESSION_A, LONDON_B),
    ],
    ids=["duplicate", "reversed", "mixed-axes", "mixed-session-policies"],
)
def test_result_rejects_invalid_collection_chronology(
    field_name: str,
    observations: tuple[ValidationBoundary, ...],
) -> None:
    with pytest.raises(ValidationPlanError):
        _result(field_name, observations)


@pytest.mark.parametrize("kind", ["purge", "selection"])
@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        ((SESSION_A, SESSION_B), (SESSION_B, SESSION_C)),
        ((SESSION_B,), (SESSION_A,)),
        ((SESSION_A, SESSION_C), (SESSION_B,)),
        ((SESSION_A,), (TIMESTAMP_B,)),
        ((SESSION_A,), (LONDON_B,)),
    ],
    ids=["overlap", "reversed", "interleaved", "mixed-axes", "mixed-session-policies"],
)
def test_result_rejects_incompatible_cross_field_membership(
    kind: str,
    earlier: tuple[ValidationBoundary, ...],
    later: tuple[ValidationBoundary, ...],
) -> None:
    with pytest.raises(ValidationPlanError):
        _paired_result(kind, earlier, later)


@pytest.mark.parametrize("kind", ["purge", "selection"])
@pytest.mark.parametrize("keys", [(SESSION_A, SESSION_B), (TIMESTAMP_A, TIMESTAMP_B)])
def test_result_preserves_valid_ordered_and_empty_membership(
    kind: str,
    keys: tuple[ValidationBoundary, ...],
) -> None:
    for earlier, later in ((keys[:1], keys[1:]), ((), keys), (keys, ()), ((), ())):
        result = _paired_result(kind, earlier, later)
        assert canonical_json_bytes(result.to_primitive()) == canonical_json_bytes(
            _paired_result(kind, earlier, later).to_primitive()
        )


@pytest.mark.parametrize("field_name", COLLECTION_FIELDS)
@pytest.mark.parametrize(
    "collection",
    [[], [ExchangeSessionBoundary(date(2024, 1, 2))], ({"mutable": []},)],
    ids=["empty-list", "boundary-list", "mutable-tuple-member"],
)
def test_result_rejects_mutable_membership(field_name: str, collection: object) -> None:
    with pytest.raises(ValidationPlanError, match="tuple of validation boundaries"):
        _result(field_name, cast(tuple[ValidationBoundary, ...], collection))


@pytest.mark.parametrize("field_name", COLLECTION_FIELDS)
def test_result_tuple_capture_preserves_identity_after_source_list_edits(
    field_name: str,
) -> None:
    caller_keys: list[ValidationBoundary] = [ExchangeSessionBoundary(date(2024, 1, 2))]
    captured_keys = tuple(caller_keys)
    result = _result(field_name, captured_keys)
    original = canonical_json_bytes(result.to_primitive())
    caller_keys.append(ExchangeSessionBoundary(date(2024, 1, 3)))
    caller_keys.pop(0)
    assert getattr(result, field_name) == captured_keys
    assert canonical_json_bytes(result.to_primitive()) == original
    caller_keys.clear()
    assert canonical_json_bytes(result.to_primitive()) == original
    assert (
        canonical_json_bytes(_result(field_name, captured_keys).to_primitive())
        == original
    )
