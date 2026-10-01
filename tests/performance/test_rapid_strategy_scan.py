"""QF-72 bounded rapid-scan profile: preparation, scan and marginal sweep cost.

Measurements are recorded as JUnit properties, never thresholds. Invocation
counts pin the architecture: one QF-8 partition and QF-59/QF-63 preparation per
session, one shared-kernel call per eligible decision and none per ineligible
one, outcomes only for triggers, and each unique feature series built once
across the sweep. The authoritative QF-42/QF-11 path runs the same window for
a measured (not asserted) speed comparison.
"""

from collections.abc import Callable, Generator
from time import perf_counter
from typing import Any

import pytest

import quantforge.rapid.session as rapid_session
from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples import spy_ema
from quantforge.examples.spy_ema import EmaParameters, EmaSmokeRule, configured_outcomes
from quantforge.rapid import RapidScanResult
from quantforge.validation import PartitionRole
from tests.integration.rapid_scan_fixtures import RapidCase, rapid_case
from tests.integration.test_rapid_strategy_scan import authoritative_window

PAIRS = ((8, 48), (8, 40), (12, 60))


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> Generator[RapidCase]:
    with canonical_preparation():
        yield rapid_case(tmp_path_factory.mktemp("rapid-performance"))


def test_parameter_grid_preparation_scan_and_marginal_costs(
    case: RapidCase,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    counts = {"partition": 0, "kernel": 0}
    partition = rapid_session.partition
    kernel = spy_ema.midday_bullish_cross

    def counted_partition(*args: Any, **kwargs: Any) -> Any:
        counts["partition"] += 1
        return partition(*args, **kwargs)

    def counted_kernel(*args: Any) -> bool:
        counts["kernel"] += 1
        return kernel(*args)

    monkeypatch.setattr(rapid_session, "partition", counted_partition)
    monkeypatch.setattr(spy_ema, "midday_bullish_cross", counted_kernel)
    outcomes = configured_outcomes(case.inputs.primary)
    began = perf_counter()
    with case.session(PartitionRole.DEVELOPMENT) as session:
        opened = perf_counter() - began
        eligible = len(session.eligible_timestamps(EmaSmokeRule()))
        scan_seconds: list[float] = []
        results: list[RapidScanResult] = []
        for pair in PAIRS:
            began = perf_counter()
            results.append(
                session.scan(EmaSmokeRule(EmaParameters(*pair)), outcomes=outcomes)
            )
            scan_seconds.append(perf_counter() - began)
        statistics = session.statistics()
        preparation = session.preparation_seconds
    monkeypatch.undo()
    permitted = results[0].window.permitted_decisions
    assert counts["partition"] == 1
    assert counts["kernel"] == len(PAIRS) * eligible < len(PAIRS) * permitted
    assert all(item.evaluated_decisions == eligible for item in results)
    assert statistics["eligibility_builds"] == 1
    assert statistics["position_builds"] == 2
    assert statistics["anchor_contexts"] <= 2 * len(PAIRS)
    assert statistics["qf63_unique_series"] == statistics["qf63_series_built"]
    assert statistics["outcome_requests"] == sum(
        item.trigger_count * len(outcomes) for item in results
    )

    began = perf_counter()
    window = authoritative_window(case, PartitionRole.DEVELOPMENT, "8/48")
    authoritative_seconds = perf_counter() - began
    assert len(window.decisions) == permitted
    measurements: dict[str, object] = {
        "permitted_decisions": permitted,
        "eligible_decisions": eligible,
        "session_preparation_seconds": opened,
        **{f"preparation_{k}_seconds": v for k, v in preparation.items()},
        **{
            f"scan_{fast}_{slow}_seconds": seconds
            for (fast, slow), seconds in zip(PAIRS, scan_seconds, strict=True)
        },
        "marginal_scan_seconds_mean": sum(scan_seconds[1:]) / (len(PAIRS) - 1),
        "rapid_decisions_per_second": eligible / min(scan_seconds),
        "authoritative_window_seconds": authoritative_seconds,
        "authoritative_decisions_per_second": permitted / authoritative_seconds,
        **{f"statistic_{name}": value for name, value in statistics.items()},
    }
    for name, value in measurements.items():
        record_property(name, value)
    print(measurements)
