"""QF-72: rapid scans reproduce authoritative QF-45 trading semantics exactly.

Both paths call the same ``spy_ema.midday_bullish_cross`` predicate. Recording
its arguments proves that the rapid scan hands the kernel exactly the causal
values the authoritative QF-42/QF-11/QF-63 rule sees at every eligible decision.
Triggers, directions, features and every configured outcome are compared with
the production QF-39 selection window and QF-7 candidate exports.
"""

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples import spy_ema
from quantforge.examples.spy_ema import (
    DAILY,
    EmaParameters,
    EmaSmokeRule,
    configured_outcomes,
)
from quantforge.examples.spy_ema_events import export_candidate_features
from quantforge.prediction import run_prediction_window
from quantforge.prediction.window import PredictionWindowResult
from quantforge.prediction.window_encoding import mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.rapid import RapidScanResult
from quantforge.validation import PartitionRole
from quantforge.walk_forward.partitions import partition
from quantforge.walk_forward.prediction import (
    _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.rapid_scan_fixtures import NEW_YORK, RapidCase, rapid_case
from tests.integration.test_prepared_feature_execution import sentinel_series

type KernelCall = tuple[time, tuple[Decimal | None, ...]]
THIRTY_MINUTES = "intraday_forward_return_1800000000us"
PAIRS = {"8/40": (8, 40), "8/48": (8, 48), "12/60": (12, 60)}


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> Generator[RapidCase]:
    # One QF-65 preparation for the module, as one production run would hold.
    with canonical_preparation():
        yield rapid_case(tmp_path_factory.mktemp("rapid-scan"))


@contextmanager
def recorded_kernel() -> Generator[list[KernelCall]]:
    """Record every call of the shared QF-45 predicate, from either path."""
    calls: list[KernelCall] = []
    original = spy_ema.midday_bullish_cross

    def record(clock: time, *values: Decimal | None) -> bool:
        calls.append((clock, values))
        return original(clock, *values)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(spy_ema, "midday_bullish_cross", record)
        yield calls


def exact(calls: list[KernelCall]) -> list[tuple[time, tuple[object, ...]]]:
    """Values with their exact Decimal representation, not only numeric value."""
    return [
        (clock, tuple(None if v is None else v.as_tuple() for v in values))
        for clock, values in calls
    ]


def eligible_clock(timestamp: datetime) -> bool:
    return time(11) <= timestamp.astimezone(NEW_YORK).time() <= time(14)


def rapid_scan(
    case: RapidCase,
    role: PartitionRole,
    pair: str,
    *,
    outcomes: bool = True,
) -> tuple[RapidScanResult, tuple[datetime, ...], list[KernelCall]]:
    rule = EmaSmokeRule(EmaParameters(*PAIRS[pair]))
    with recorded_kernel() as calls, case.session(role) as session:
        eligible = session.eligible_timestamps(rule)
        result = session.scan(
            rule,
            outcomes=configured_outcomes(case.inputs.primary) if outcomes else (),
        )
    return result, eligible, calls


@pytest.fixture(scope="module")
def selection(
    case: RapidCase,
) -> tuple[PredictionWindowReader, list[KernelCall], Path]:
    """The production QF-39 fixed selection window and its QF-7 exports."""
    output = case.root / "authoritative"
    with recorded_kernel() as calls:
        case.adapter.select(case.config, 0, output)
    reader = PredictionWindowReader.open(next(output.rglob("prediction-window.jsonl")))
    permitted = case.adapter._partition(case.config, 0, test=False)  # pyright: ignore[reportPrivateUsage]
    exports = case.root / "features"
    export_candidate_features(
        case.inputs, case.config, permitted, reader, {"ema_pair": "8/48"}, exports
    )
    return reader, calls, exports


def window_triggers(reader: PredictionWindowReader) -> list[dict[str, Any]]:
    triggers: list[dict[str, Any]] = []
    for receipt in reader.iterate_decision_receipts():
        if receipt.decision is None:
            continue
        record = receipt.decision.to_primitive()
        signals = cast(list[dict[str, Any]], record["generated_signals"])
        if signals:
            rows = cast(
                list[dict[str, Any]], mapping(record["prediction_study"])["rows"]
            )
            triggers.append(
                {
                    "timestamp": record["decision_timestamp"],
                    "directions": [
                        mapping(mapping(s["prediction"])["values"])["direction"]
                        for s in signals
                    ],
                    "features": [s["features"] for s in signals],
                    "evaluations": [
                        mapping(mapping(row["evaluation"])["values"]) for row in rows
                    ],
                }
            )
    return triggers


def rapid_features(result: RapidScanResult) -> list[dict[str, str]]:
    return [
        {name: str(value) for name, value in event.values} for event in result.events
    ]


def test_selection_scan_reproduces_the_authoritative_window_exactly(
    case: RapidCase, selection: tuple[PredictionWindowReader, list[KernelCall], Path]
) -> None:
    reader, authoritative_calls, _ = selection
    result, eligible, rapid_calls = rapid_scan(case, PartitionRole.SELECTION, "8/48")
    schedule = reader.evidence.schedule.decision_timestamps
    # Population: QF-8 retained membership, then the frozen causal clock filter.
    assert result.window.permitted_decisions == len(schedule)
    assert eligible == tuple(t for t in schedule if eligible_clock(t))
    assert result.eligible_decisions == result.evaluated_decisions == len(eligible)
    # The authoritative rule called the shared kernel at every scheduled
    # decision; rapid called it only at eligible ones, with identical inputs.
    assert len(authoritative_calls) == len(schedule)
    assert rapid_calls == [
        call for call in authoritative_calls if time(11) <= call[0] <= time(14)
    ]
    assert exact(rapid_calls) == exact(
        [call for call in authoritative_calls if time(11) <= call[0] <= time(14)]
    )
    triggers = window_triggers(reader)
    assert len(triggers) == result.trigger_count == 3
    assert [t["timestamp"] for t in triggers] == [
        event.decision_timestamp.isoformat() for event in result.events
    ]
    assert [t["directions"] for t in triggers] == [
        [event.direction.value] for event in result.events
    ]
    assert [t["features"][0] for t in triggers] == rapid_features(result)
    for trigger, event in zip(triggers, result.events, strict=True):
        rapid = event.outcome(THIRTY_MINUTES)
        assert rapid == {key: trigger["evaluations"][0][key] for key in rapid}


def test_selection_outcomes_match_every_qf7_configured_outcome(
    case: RapidCase, selection: tuple[PredictionWindowReader, list[KernelCall], Path]
) -> None:
    _, _, exports = selection
    result, _, _ = rapid_scan(case, PartitionRole.SELECTION, "8/48")
    rows = {
        row["decision_timestamp"]: row
        for row in (
            json.loads(path.read_text()) for path in exports.glob("*/rows/*.json")
        )
    }
    assert sorted(rows) == [e.decision_timestamp.isoformat() for e in result.events]
    configurations = {item.namespace for item in result.outcome_configurations}
    assert configurations == {
        outcome.namespace for outcome in configured_outcomes(case.inputs.primary)
    }
    compared = 0
    for event in result.events:
        row = rows[event.decision_timestamp.isoformat()]
        for name, value in event.values:
            assert row[f"feature_{name}"] == str(value)
        assert {item.namespace for item in event.outcomes} == configurations
        for item in event.outcomes:
            for field, value in item.values.to_primitive().items():
                assert row[f"outcome_{item.namespace}_{field}"] == value
                compared += 1
    assert compared > 3 * 6 * 5


def authoritative_window(
    case: RapidCase, role: PartitionRole, pair: str
) -> PredictionWindowResult[Any, Any, Any]:
    """Unchanged QF-8 partition, QF-39 provider and QF-42/QF-11 execution."""
    plan = case.config.plan
    permitted = partition(case.inputs.dataset, plan, 0, role, minimum_observations=1)
    schedule = case.adapter._schedule(permitted)  # pyright: ignore[reportPrivateUsage]
    return run_prediction_window(
        permitted.dataset,
        case.adapter.factory.build({"ema_pair": pair}),
        schedule=schedule,
        context_provider=_PermittedContextProvider(
            plan, permitted, case.adapter.series, schedule
        ),
        dataset_family_fingerprint=case.adapter.series[0].dataset_reference.family_id,
        context_environment=case.adapter._environment(  # pyright: ignore[reportPrivateUsage]
            plan, permitted
        ).to_primitive(),
        indicator_backend_environment=case.adapter.backend.to_primitive(),
    )


@pytest.mark.parametrize("pair", ["8/40", "12/60"])
def test_development_scans_match_authoritative_execution_for_other_pairs(
    case: RapidCase, pair: str
) -> None:
    with recorded_kernel() as authoritative_calls:
        window = authoritative_window(case, PartitionRole.DEVELOPMENT, pair)
    result, eligible, rapid_calls = rapid_scan(
        case, PartitionRole.DEVELOPMENT, pair, outcomes=False
    )
    schedule = tuple(decision.decision_timestamp for decision in window.decisions)
    assert eligible == tuple(t for t in schedule if eligible_clock(t))
    assert exact(rapid_calls) == exact(
        [call for call in authoritative_calls if time(11) <= call[0] <= time(14)]
    )
    authoritative = [
        (decision.decision_timestamp, row)
        for decision in window.decisions
        for row in decision.result.rows
    ]
    assert [t for t, _ in authoritative] == [
        e.decision_timestamp for e in result.events
    ]
    assert len(authoritative) == result.trigger_count == 2
    for (_, row), event in zip(authoritative, result.events, strict=True):
        assert row.signal.direction is event.direction
        assert {item.name: item.value for item in row.signal.strategy_features} == dict(
            event.values
        )
    with_outcomes = rapid_scan(case, PartitionRole.DEVELOPMENT, pair)[0]
    for (_, row), event in zip(authoritative, with_outcomes.events, strict=True):
        values = cast(dict[str, Any], row.evaluation.values.to_primitive())
        rapid = event.outcome(THIRTY_MINUTES)
        assert rapid == {key: values[key] for key in rapid}


def test_calendar_warm_up_and_daily_context_boundaries(case: RapidCase) -> None:
    result, eligible, _ = rapid_scan(case, PartitionRole.SELECTION, "8/48")
    by_session: dict[date, list[time]] = {}
    for timestamp in eligible:
        local = timestamp.astimezone(NEW_YORK)
        by_session.setdefault(local.date(), []).append(local.time())
    # Normal session, 13:00 early close, holiday gap, then the window end.
    assert {day.isoformat(): len(clocks) for day, clocks in by_session.items()} == {
        "2024-12-23": 91,
        "2024-12-24": 61,
        "2024-12-26": 31,
    }
    assert by_session[date(2024, 12, 24)][-1] == time(13)
    assert [e.decision_timestamp.astimezone(NEW_YORK) for e in result.events] == [
        datetime(2024, 12, day, 11, tzinfo=NEW_YORK) for day in (23, 24, 26)
    ]
    # Completed daily context only: the latest close before each decision's
    # session, never the current session's eventual close (holiday skipped).
    closes = {
        bar.end_timestamp.astimezone(NEW_YORK).date(): bar.close
        for bar in case.inputs.daily.bars
    }
    previous = {23: date(2024, 12, 20), 24: date(2024, 12, 23), 26: date(2024, 12, 24)}
    for event in result.events:
        day = event.decision_timestamp.astimezone(NEW_YORK).day
        assert event.value("daily_close") == closes[previous[day]]
        assert event.value("daily_close") != closes[event.signal_session]
    # The 120-minute path on the early-close session ends exactly at 13:00.
    early = result.events[1].outcome("intraday_forward_return_7200000000us")
    assert early["status"] == "available"
    assert early["resolved_observation_timestamp"] == "2024-12-24T18:00:00+00:00"
    window = case.config.plan.folds[0].selection
    assert window is not None
    assert result.window.warm_up == tuple(
        (item.timeframe.configuration_id, item.observations)
        for item in window.warm_up_by_timeframe
    )
    assert dict(result.window.warm_up)[DAILY.configuration_id] == 50


def test_sentinel_future_bars_never_change_earlier_rapid_values(
    case: RapidCase,
) -> None:
    baseline, eligible, baseline_calls = rapid_scan(
        case, PartitionRole.SELECTION, "8/48", outcomes=False
    )
    cutoff = baseline.events[1].decision_timestamp
    absurd = Decimal("999999")
    series = tuple(
        sentinel_series(source, cutoff, absurd)
        for source in (case.inputs.primary, case.inputs.daily)
    )
    rule = EmaSmokeRule(EmaParameters(8, 48))
    with recorded_kernel() as calls, case.session(series=series) as session:
        assert session.eligible_timestamps(rule) == eligible
        sentinel = session.scan(rule)
    visible = sum(t <= cutoff for t in eligible)
    # Every kernel input at or before the cutoff is byte-identical.
    assert exact(calls[:visible]) == exact(baseline_calls[:visible])
    assert [e.to_primitive() for e in sentinel.events[:2]] == [
        e.to_primitive() for e in baseline.events[:2]
    ]
    # Sanity: once absurd bars are visible, the rapid values do change.
    assert exact(calls[visible:]) != exact(baseline_calls[visible:])
