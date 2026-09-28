"""QF-62 catalogue identity, exact ranges, explicit exceptions and fail-closed use."""

import hashlib
from copy import deepcopy
from datetime import time, timedelta
from typing import Any, cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_membership import (
    CATALOGUE_SEGMENTS_FIELD,
    MembershipCatalogues,
    catalogue_identity,
    membership_binding,
    prediction_context,
)
from quantforge.timeframes import (
    ExchangeSessionPolicy,
    IntradayInterval,
    SessionInterval,
    SessionScope,
    Timeframe,
)

TWO_MINUTES = Timeframe.us_equity(IntradayInterval(timedelta(minutes=2)))
DAILY = Timeframe.us_equity(SessionInterval(1))


def bar_ids(prefix: str, count: int, start: int = 0) -> list[str]:
    return [
        hashlib.sha256(f"{prefix}-{index}".encode()).hexdigest()
        for index in range(start, start + count)
    ]


def timeframe(value: Timeframe) -> PrimitiveMapping:
    return {
        "configuration": value.to_primitive(),
        "configuration_id": value.configuration_id,
    }


def reference(value: Timeframe, dataset: str = "2m") -> PrimitiveMapping:
    return {
        "canonical_source_snapshot_id": "c" * 64,
        "dataset_id": hashlib.sha256(f"dataset-{dataset}".encode()).hexdigest(),
        "family_id": "f" * 64,
        "timeframe_configuration_id": value.configuration_id,
    }


def context(
    primary: list[str],
    daily: list[str],
    *,
    developing: str | None = None,
) -> PrimitiveMapping:
    """QF-45 layout: 2m in four locations and daily in three."""

    def source(value: Timeframe, name: str, ids: list[str]) -> PrimitiveMapping:
        return {
            "requirement": {
                "maximum_age_microseconds": None,
                "timeframe": timeframe(value),
            },
            "dataset_reference": reference(value, name),
            "availability": "available",
            "visible_bar_ids": cast(list[Primitive], ids),
        }

    def indicator(value: Timeframe, name: str, alias: str, ids: list[str]) -> Primitive:
        return {
            "alias": alias,
            "dataset_reference": {
                **reference(value, name),
                "feed_scope": {"coverage": "consolidated"},
            },
            "source_timeframe": timeframe(value),
            "warm_up_bars": 8,
            "visible_bar_ids": cast(list[Primitive], ids),
        }

    daily_source = source(DAILY, "daily", daily)
    if developing is not None:
        daily_source["visible_bar_ids"] = [*daily, developing]
        daily_source["developing_bar"] = {"bar_id": developing, "bar": {}}
    return {
        "status": "available",
        "decision_session": "2025-05-01",
        "source_context": {
            "context_id": "0" * 64,
            "timeframes": [source(TWO_MINUTES, "2m", primary), daily_source],
        },
        "timeframes": [
            {
                "requirement": {"timeframe": timeframe(TWO_MINUTES)},
                "visible_bar_ids": cast(list[Primitive], primary),
                "indicators": [
                    indicator(TWO_MINUTES, "2m", "fast", primary),
                    indicator(TWO_MINUTES, "2m", "slow", primary),
                ],
            },
            {
                "requirement": {"timeframe": timeframe(DAILY)},
                "visible_bar_ids": cast(list[Primitive], daily),
                "indicators": [indicator(DAILY, "daily", "trend", daily)],
            },
        ],
    }


def record(logical: PrimitiveMapping, state: MembershipCatalogues) -> PrimitiveMapping:
    normalized, segments = state.normalize(logical)
    return {
        "prediction_study": {
            "manifest": {"prediction_context": normalized},
            "rows": [],
        },
        CATALOGUE_SEGMENTS_FIELD: segments,
    }


def logical_record(logical: PrimitiveMapping) -> PrimitiveMapping:
    return {
        "prediction_study": {"manifest": {"prediction_context": logical}, "rows": []}
    }


def ranges(normalized: PrimitiveMapping) -> list[Primitive]:
    source = cast(dict[str, Any], normalized["source_context"])
    rules = cast(list[dict[str, Any]], normalized["timeframes"])
    owners = [
        *source["timeframes"],
        *(item for rule in rules for item in (rule, *rule["indicators"])),
    ]
    return [
        owner.get("visible_bar_range", owner.get("visible_bar_ids")) for owner in owners
    ]


def accept_all(
    contexts: list[PrimitiveMapping],
) -> tuple[MembershipCatalogues, list[PrimitiveMapping]]:
    state = MembershipCatalogues()
    records: list[PrimitiveMapping] = []
    for logical in contexts:
        item = record(logical, state)
        view = state.accept(item)
        assert view.expand_record(item) == logical_record(logical)
        records.append(item)
    return state, records


def qf45_sequence() -> list[PrimitiveMapping]:
    """Prefix-growing 2m plus the observed rolling daily start-bound shift."""
    primary = bar_ids("2m", 66)
    daily = bar_ids("daily", 8)
    return [
        context(primary[:62], daily[:5]),
        context(primary[:63], daily[:5]),
        context(primary[:64], daily[1:6]),  # daily rolls: start 0 -> 1
        context(primary[:65], daily[1:7]),
        context(primary[:66], daily[1:8]),
    ]


def test_catalogue_identity_binds_source_timeframe_session_and_family() -> None:
    binding = membership_binding(reference(TWO_MINUTES), timeframe(TWO_MINUTES))
    assert binding is not None
    identity = catalogue_identity(binding)
    assert identity == catalogue_identity(deepcopy(binding))
    indicator_reference: PrimitiveMapping = {
        **reference(TWO_MINUTES),
        "feed_scope": {"coverage": "x"},
    }
    assert membership_binding(indicator_reference, timeframe(TWO_MINUTES)) == binding
    extended = Timeframe(
        TWO_MINUTES.interval,
        ExchangeSessionPolicy(
            TWO_MINUTES.session_policy.calendar_name,
            TWO_MINUTES.session_policy.timezone_name,
            SessionScope.EXTENDED_HOURS,
            time(4),
            time(20),
        ),
    )
    changed = [
        membership_binding(reference(TWO_MINUTES, "other"), timeframe(TWO_MINUTES)),
        membership_binding(
            {**reference(TWO_MINUTES), "family_id": "e" * 64}, timeframe(TWO_MINUTES)
        ),
        membership_binding(
            {**reference(TWO_MINUTES), "canonical_source_snapshot_id": "d" * 64},
            timeframe(TWO_MINUTES),
        ),
        membership_binding(reference(DAILY, "2m"), timeframe(DAILY)),
        membership_binding(reference(extended), timeframe(extended)),
    ]
    identities = {catalogue_identity(cast(PrimitiveMapping, item)) for item in changed}
    assert identity not in identities
    assert len(identities) == len(changed)


@pytest.mark.parametrize(
    "change",
    ["missing", "extra", "blank", "mismatched", "rehashed_configuration", "none"],
)
def test_incomplete_or_inconsistent_binding_cannot_use_a_catalogue(change: str) -> None:
    source: dict[str, Any] = dict(reference(TWO_MINUTES))
    frame: dict[str, Any] = dict(timeframe(TWO_MINUTES))
    if change == "missing":
        del source["family_id"]
    elif change == "extra":
        source["account"] = "x"
    elif change == "blank":
        source["dataset_id"] = ""
    elif change == "mismatched":
        source["timeframe_configuration_id"] = DAILY.configuration_id
    elif change == "rehashed_configuration":
        frame["configuration"] = {**frame["configuration"], "bar_label": "bar_end"}
    else:
        source = cast(dict[str, Any], None)
    assert membership_binding(source, frame) is None


def test_prefix_growth_and_rolling_daily_start_are_exact_ranges() -> None:
    contexts = qf45_sequence()
    state, records = accept_all(contexts)
    catalogues = {
        item["catalogue_id"]: item["bar_count"]
        for item in cast(list[dict[str, Any]], state.summaries())
    }
    assert sorted(catalogues.values()) == [8, 66]
    expected_primary = [(0, 62), (0, 63), (0, 64), (0, 65), (0, 66)]
    expected_daily = [(0, 5), (0, 5), (1, 6), (1, 7), (1, 8)]
    for item, primary, daily in zip(
        records, expected_primary, expected_daily, strict=True
    ):
        found = [
            (
                cast(dict[str, int], value)["start_index"],
                cast(dict[str, int], value)["stop_index"],
            )
            for value in ranges(prediction_context(item))
        ]
        # 2m: source, rule and two indicators; daily: source, rule and indicator.
        assert found == [primary, daily, primary, primary, primary, daily, daily]
    segments = [
        cast(list[dict[str, Any]], item[CATALOGUE_SEGMENTS_FIELD]) for item in records
    ]
    assert [len(item) for item in segments] == [2, 1, 2, 2, 2]
    assert all("binding" in segment for segment in segments[0])
    assert all("binding" not in segment for item in segments[1:] for segment in item)
    # Every ID is physically stored exactly once across the whole sequence.
    stored = [
        bar for item in segments for segment in item for bar in segment["bar_ids"]
    ]
    assert len(stored) == len(set(stored)) == 66 + 8


def test_normalize_is_pure_and_detached() -> None:
    logical = qf45_sequence()[0]
    before = deepcopy(logical)
    state = MembershipCatalogues()
    first = state.normalize(logical)
    assert state.lengths() == {}
    assert state.normalize(logical) == first
    assert logical == before


@pytest.mark.parametrize(
    "members",
    [
        "empty",
        "skipped_bar",
        "reordered",
        "duplicate",
        "new_then_known",
        "developing",
        "unbound",
    ],
)
def test_noncontiguous_or_unbound_membership_stays_explicit(members: str) -> None:
    primary, daily = bar_ids("2m", 10), bar_ids("daily", 5)
    state, _ = accept_all([context(primary[:8], daily[:3])])
    changed = context(primary[:8], daily[:3])
    rule = cast(list[dict[str, Any]], changed["timeframes"])[0]
    explicit: list[str] = {
        "empty": [],
        "skipped_bar": [primary[0], primary[2]],
        "reordered": [primary[1], primary[0]],
        "duplicate": [primary[0], primary[0]],
        "new_then_known": [primary[9], primary[0]],
        "developing": [*primary[:8], "developing-bar"],
        "unbound": primary[:8],
    }[members]
    rule["visible_bar_ids"] = cast(list[Primitive], explicit)
    if members == "developing":
        source = cast(
            list[dict[str, Any]],
            cast(dict[str, Any], changed["source_context"])["timeframes"],
        )[0]
        source["developing_bar"] = {"bar_id": "developing-bar", "bar": {}}
    if members == "unbound":
        rule["requirement"] = {"timeframe": timeframe(DAILY)}
    before = state.lengths()
    normalized, segments = state.normalize(changed)
    kept = cast(list[dict[str, Any]], normalized["timeframes"])[0]
    assert kept["visible_bar_ids"] == explicit
    assert "visible_bar_range" not in kept
    assert segments == []
    assert state.lengths() == before
    item: PrimitiveMapping = {
        "prediction_study": {
            "manifest": {"prediction_context": normalized},
            "rows": [],
        },
        CATALOGUE_SEGMENTS_FIELD: segments,
    }
    assert state.accept(item).expand_record(item) == logical_record(changed)


def test_developing_bar_never_pollutes_later_completed_ranges() -> None:
    primary, daily = bar_ids("2m", 4), bar_ids("daily", 4)
    state, records = accept_all(
        [
            context(primary[:2], daily[:2], developing="developing-1"),
            context(primary[:3], daily[:2], developing="developing-2"),
            context(primary[:4], daily[:3]),
        ]
    )
    source = [
        cast(
            list[dict[str, Any]],
            cast(dict[str, Any], prediction_context(item)["source_context"])[
                "timeframes"
            ],
        )[1]
        for item in records
    ]
    assert source[0]["visible_bar_ids"][-1] == "developing-1"
    assert source[1]["visible_bar_ids"][-1] == "developing-2"
    assert source[2]["visible_bar_range"]["stop_index"] == 3
    stored = [bar for ids in state.view().catalogues.values() for bar in ids]
    assert not {"developing-1", "developing-2"} & set(stored)
    assert sorted(state.lengths().values()) == [3, 4]


def mutate(
    mutation: str, records: list[PrimitiveMapping]
) -> tuple[int, PrimitiveMapping]:
    """Corrupt one normalized record; returns its position in the sequence."""
    index = 0 if mutation in {"first_segment_binding", "truncated_catalogue"} else 2
    item = deepcopy(records[index])
    normalized = prediction_context(item)
    source = cast(
        list[dict[str, Any]],
        cast(dict[str, Any], normalized["source_context"])["timeframes"],
    )
    primary_range = cast(dict[str, Any], source[0]["visible_bar_range"])
    daily_range = cast(dict[str, Any], source[1]["visible_bar_range"])
    segments = cast(list[dict[str, Any]], item[CATALOGUE_SEGMENTS_FIELD])
    primary_segment = next(
        segment
        for segment in segments
        if segment["catalogue_id"] == primary_range["catalogue_id"]
    )
    if mutation == "invalid_catalogue_id":
        primary_range["catalogue_id"] = "0" * 64
    elif mutation == "wrong_timeframe_binding":
        source[0]["visible_bar_range"] = dict(daily_range)
    elif mutation == "wrong_source_binding":
        source[0]["dataset_reference"] = reference(TWO_MINUTES, "other")
    elif mutation == "wrong_session_binding":
        requirement = cast(dict[str, Any], source[0]["requirement"])
        configuration = dict(requirement["timeframe"]["configuration"])
        configuration["session_policy"] = {
            **configuration["session_policy"],
            "scope": "extended_hours",
        }
        requirement["timeframe"] = {
            "configuration": configuration,
            "configuration_id": configuration_identity(configuration),
        }
    elif mutation == "wrong_family_binding":
        source[0]["dataset_reference"] = {
            **reference(TWO_MINUTES),
            "family_id": "e" * 64,
        }
    elif mutation == "negative_start":
        primary_range["start_index"] = -1
    elif mutation == "stop_past_catalogue":
        primary_range["stop_index"] += 1
    elif mutation == "start_after_stop":
        primary_range["start_index"] = primary_range["stop_index"] + 1
    elif mutation == "empty_range":
        primary_range["start_index"] = primary_range["stop_index"]
    elif mutation == "boolean_index":
        daily_range["start_index"] = True
    elif mutation == "extra_reference_field":
        primary_range["length"] = 64
    elif mutation == "truncated_catalogue":
        primary_segment["bar_ids"] = primary_segment["bar_ids"][:-1]
    elif mutation == "duplicate_catalogue_entry":
        primary_segment["bar_ids"] = [*primary_segment["bar_ids"], bar_ids("2m", 1)[0]]
    elif mutation == "future_entry":
        primary_segment["bar_ids"] = [
            *primary_segment["bar_ids"],
            bar_ids("2m", 1, 90)[0],
        ]
    elif mutation == "segment_offset":
        primary_segment["start_index"] -= 1
    elif mutation == "segment_order":
        segments.reverse()
    elif mutation == "foreign_catalogue":
        binding = membership_binding(
            reference(TWO_MINUTES, "foreign"), timeframe(TWO_MINUTES)
        )
        assert binding is not None
        segments.append(
            {
                "catalogue_id": "f" * 64,
                "start_index": 0,
                "bar_ids": ["x"],
                "binding": binding,
            }
        )
    elif mutation == "rebound_foreign_catalogue":
        binding = cast(
            dict[str, Any],
            membership_binding(
                reference(TWO_MINUTES, "foreign"), timeframe(TWO_MINUTES)
            ),
        )
        segments.append(
            {
                "catalogue_id": catalogue_identity(binding),
                "start_index": 0,
                "bar_ids": ["x"],
                "binding": binding,
            }
        )
        segments.sort(key=lambda segment: segment["catalogue_id"])
    elif mutation == "first_segment_binding":
        del segments[0]["binding"]
    elif mutation == "later_segment_binding":
        first = cast(list[dict[str, Any]], records[0][CATALOGUE_SEGMENTS_FIELD])
        primary_segment["binding"] = first[0]["binding"]
    elif mutation == "stray_reference":
        cast(dict[str, Any], normalized["source_context"])["visible_bar_range"] = dict(
            primary_range
        )
    elif mutation == "explicit_and_range":
        source[0]["visible_bar_ids"] = []
    elif mutation == "unneeded_explicit":
        del source[0]["visible_bar_range"]
        source[0]["visible_bar_ids"] = bar_ids("2m", 64)
    elif mutation == "missing_segments":
        del item[CATALOGUE_SEGMENTS_FIELD]
    else:
        raise AssertionError(mutation)
    return index, item


@pytest.mark.parametrize(
    "mutation",
    [
        "invalid_catalogue_id",
        "wrong_timeframe_binding",
        "wrong_source_binding",
        "wrong_session_binding",
        "wrong_family_binding",
        "negative_start",
        "stop_past_catalogue",
        "start_after_stop",
        "empty_range",
        "boolean_index",
        "extra_reference_field",
        "truncated_catalogue",
        "duplicate_catalogue_entry",
        "future_entry",
        "segment_offset",
        "segment_order",
        "foreign_catalogue",
        "rebound_foreign_catalogue",
        "first_segment_binding",
        "later_segment_binding",
        "stray_reference",
        "explicit_and_range",
        "unneeded_explicit",
        "missing_segments",
    ],
)
def test_corrupted_references_and_segments_fail_closed_and_roll_back(
    mutation: str,
) -> None:
    contexts = qf45_sequence()
    _, records = accept_all(contexts)
    index, corrupted = mutate(mutation, records)
    state = MembershipCatalogues()
    for item in records[:index]:
        state.accept(item)
    before = state.lengths()
    with pytest.raises(InvalidPredictionOutputError):
        state.accept(corrupted)
    assert state.lengths() == before
    # A failed record leaves the committed prefix usable for the authentic record.
    for item in records[index:]:
        state.accept(item)


def test_widened_range_is_self_consistent_but_changes_reconstructed_membership() -> (
    None
):
    """Ranges alone cannot vouch for semantics; expanded scientific IDs must."""
    contexts = qf45_sequence()
    _, records = accept_all(contexts)
    item = deepcopy(records[2])
    daily = cast(
        list[dict[str, Any]],
        cast(dict[str, Any], prediction_context(item)["source_context"])["timeframes"],
    )[1]
    daily["visible_bar_range"]["start_index"] = 0  # Reintroduce the rolled-out bar.
    state = MembershipCatalogues()
    for previous in records[:2]:
        state.accept(previous)
    expanded = state.accept(item).expand_record(item)
    changed = cast(
        list[dict[str, Any]],
        cast(dict[str, Any], prediction_context(expanded)["source_context"])[
            "timeframes"
        ],
    )[1]
    original = cast(
        list[dict[str, Any]],
        cast(dict[str, Any], contexts[2]["source_context"])["timeframes"],
    )[1]
    assert changed["visible_bar_ids"] != original["visible_bar_ids"]
    assert configuration_identity(
        prediction_context(expanded)
    ) != configuration_identity(contexts[2])


def test_logical_context_with_references_is_rejected() -> None:
    state = MembershipCatalogues()
    normalized, _ = state.normalize(qf45_sequence()[0])
    with pytest.raises(InvalidPredictionOutputError, match="already contains"):
        state.normalize(normalized)
