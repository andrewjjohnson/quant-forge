"""QF-72 boundaries: holdout isolation, admission, reuse and non-authoritative output.

Rapid scans may read only permitted QF-8 development/selection windows, refuse
every unreviewed rule, feature or outcome instead of approximating it, reuse
preparation across a parameter sweep, persist nothing, and produce results that
no authoritative QF-9/QF-42/QF-41 path accepts.
"""

import inspect
import json
import os
from collections.abc import Generator
from dataclasses import replace
from datetime import date, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

import quantforge.prediction.study as study_module
import quantforge.prediction.window as window_module
import quantforge.rapid.session as rapid_session
from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples import spy_ema
from quantforge.examples.spy_ema import (
    DAILY,
    EmaParameters,
    EmaSmokeRule,
    EmaStudyFactory,
    configured_outcomes,
)
from quantforge.experiments import ArtifactType, ManifestError, index_artifact
from quantforge.indicators import (
    TALIB_INDICATOR_BACKEND,
    ExponentialMovingAverage,
    ExponentialMovingAverageParameters,
)
from quantforge.oos import HoldoutLedger
from quantforge.prediction import (
    PredictionDirection,
    PredictionIndicatorRequirement,
    intraday_forward_return_outcome,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.intraday_path_evaluation import IntradayExcursionEvaluator
from quantforge.prediction.window import PredictionWindowResult
from quantforge.prediction.window_incremental import IncrementalPredictionWindowWriter
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.rapid import (
    RAPID_EXPORT_SUFFIX,
    RapidAdmissionError,
    RapidHoldoutError,
    RapidIndicatorInput,
    RapidRuleSpecification,
    RapidScanError,
    RapidScanResult,
    RapidScopeError,
    export_rapid_scan,
    promote_strategy_configuration,
    rapid_research_session,
)
from quantforge.rapid.models import RapidValue
from quantforge.validation import PartitionRole, TimestampBoundary
from tests.integration.rapid_scan_fixtures import RapidCase, rapid_case, reserve_scope
from tests.integration.test_prepared_feature_execution import sentinel_series


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> Generator[RapidCase]:
    # One QF-65 preparation for the module, as one production run would hold.
    with canonical_preparation():
        yield rapid_case(tmp_path_factory.mktemp("rapid-boundaries"))


@pytest.fixture(scope="module")
def result(case: RapidCase) -> RapidScanResult:
    with case.session() as session:
        return session.scan(
            EmaSmokeRule(EmaParameters(8, 48)),
            outcomes=configured_outcomes(case.inputs.primary),
        )


def files(root: Path) -> list[tuple[str, bytes]]:
    return sorted(
        (str(path.relative_to(root)), path.read_bytes())
        for path in root.rglob("*")
        if path.is_file() and path.name != ".lock"
    )


# -- research scope and holdout ----------------------------------------------


@pytest.mark.parametrize("role", [PartitionRole.DEVELOPMENT, PartitionRole.SELECTION])
def test_development_and_selection_windows_are_permitted(
    case: RapidCase, role: PartitionRole
) -> None:
    with case.session(role) as session:
        scanned = session.scan(EmaSmokeRule(EmaParameters(8, 48)))
    window = scanned.window
    assert window.role is role
    assert window.research_plan_id == case.config.plan.plan_id
    holdout = case.config.plan.final_holdout.window.interval
    assert (
        window.reserved_holdout_start
        == cast(TimestampBoundary, holdout.start).timestamp
    )
    assert window.footprint_last_session < date(2024, 12, 30)
    primitive = cast(dict[str, Any], scanned.to_primitive()["window"])
    assert primitive["reserved_holdout"]["read"] is False


def test_walk_forward_test_windows_are_refused(case: RapidCase) -> None:
    with pytest.raises(RapidScopeError, match="authoritative OOS"):
        with case.session(PartitionRole.WALK_FORWARD_TEST):
            pass


def test_final_holdout_is_refused_as_exploration(case: RapidCase) -> None:
    with pytest.raises(RapidHoldoutError, match="authoritative-only"):
        with case.session(PartitionRole.FINAL_HOLDOUT):
            pass


@pytest.mark.parametrize(
    ("name", "start", "end", "symbol", "refused"),
    [
        ("partial-overlap", date(2024, 12, 26), date(2024, 12, 27), "SPY", True),
        ("holdout-only", date(2024, 12, 23), date(2024, 12, 26), "SPY", True),
        ("warm-up-footprint", date(2024, 12, 10), date(2024, 12, 10), "SPY", True),
        ("adjacent-after", date(2024, 12, 27), date(2024, 12, 31), "SPY", False),
        ("other-symbol", date(2024, 12, 23), date(2024, 12, 26), "QQQ", False),
    ],
)
def test_reserved_ledger_holdouts_are_never_scanned(
    case: RapidCase,
    name: str,
    start: date,
    end: date,
    symbol: str,
    refused: bool,
) -> None:
    ledger = case.ledger(f"ledger-{name}")
    reserve_scope(ledger, "a" * 64, start=start, end=end, symbol=symbol)
    before = files(ledger.root)
    if refused:
        with pytest.raises(RapidHoldoutError, match="reserved"):
            with case.session(ledger=ledger):
                pass
    else:
        with case.session(ledger=ledger) as session:
            assert session.scan(EmaSmokeRule(EmaParameters(8, 48))).trigger_count == 3
            assert session.research_window.ledger_exposure_scopes_checked == (
                1 if symbol == "SPY" else 0
            )
    assert files(ledger.root) == before


def test_a_permanent_ledger_is_required_and_there_is_no_override(
    case: RapidCase,
) -> None:
    with pytest.raises(RapidScopeError, match="holdout ledger"):
        with rapid_research_session(
            plan=case.config.plan,
            fold_index=0,
            role=PartitionRole.SELECTION,
            dataset=case.inputs.dataset,
            series=(case.inputs.primary, case.inputs.daily),
            holdout_ledger=cast(HoldoutLedger, None),
        ):
            pass
    parameters = set(inspect.signature(rapid_research_session).parameters)
    assert parameters == {
        "plan",
        "fold_index",
        "role",
        "dataset",
        "series",
        "holdout_ledger",
    }
    scan = set(inspect.signature(rapid_session.RapidResearchSession.scan).parameters)
    assert scan == {"self", "rule", "outcomes", "record_values"}


# -- admission ---------------------------------------------------------------


class LookalikeEma(ExponentialMovingAverage):
    """Identical values, but not a reviewed exact prefix-stable type."""


class LookalikeRule(EmaSmokeRule):
    @staticmethod
    def _ema(alias: str, period: int) -> PredictionIndicatorRequirement:
        return PredictionIndicatorRequirement(
            alias,
            LookalikeEma(
                ExponentialMovingAverageParameters(period),
                backend_id=TALIB_INDICATOR_BACKEND,
            ),
        )


class UndeclaredAliasRule(EmaSmokeRule):
    def rapid_specification(self) -> RapidRuleSpecification:
        specification = super().rapid_specification()
        return replace(
            specification,
            inputs=(
                *specification.inputs[:-1],
                RapidIndicatorInput(
                    "daily_ema50", DAILY, "missing", "exponential_moving_average"
                ),
            ),
        )


class StaleLimitedRule(EmaSmokeRule):
    def __init__(self) -> None:
        super().__init__()
        requirements = self.context_requirements
        self.context_requirements = replace(
            requirements,
            contextual=(
                replace(requirements.contextual[0], maximum_age=timedelta(days=5)),
            ),
        )


class OpaqueRule:
    """A rule with authoritative requirements but no rapid capability."""

    def __init__(self) -> None:
        self.context_requirements = EmaSmokeRule().context_requirements


def _down(clock: time, values: tuple[RapidValue, ...]) -> PredictionDirection | None:
    previous_fast, previous_slow, fast, slow, close, ema50 = values
    if spy_ema.midday_bullish_cross(
        clock, previous_fast, previous_slow, fast, slow, close, ema50
    ):
        return PredictionDirection.DOWN
    return None


class DisagreeingRule(EmaSmokeRule):
    def rapid_specification(self) -> RapidRuleSpecification:
        return replace(super().rapid_specification(), decide=_down)


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        (OpaqueRule(), "no reviewed rapid specification"),
        (LookalikeRule(), "prefix-stable"),
        (UndeclaredAliasRule(), "undeclared indicator"),
        (StaleLimitedRule(), "staleness"),
        (EmaSmokeRule(EmaParameters(8, 70)), "warm-up"),
    ],
    ids=["no-specification", "unreviewed-indicator", "alias", "stale", "warm-up"],
)
def test_unsupported_rules_and_features_are_refused(
    case: RapidCase, rule: object, message: str
) -> None:
    with case.session() as session:
        with pytest.raises(RapidAdmissionError, match=message):
            session.scan(rule)


def test_kernel_and_candidate_disagreement_fails_closed(case: RapidCase) -> None:
    with case.session() as session, pytest.raises(RapidScanError, match="disagrees"):
        session.scan(DisagreeingRule())


def test_unreviewed_or_out_of_scope_outcomes_are_refused(case: RapidCase) -> None:
    primary = case.inputs.primary
    forward = intraday_forward_return_outcome(timedelta(minutes=30), primary)
    other_source = sentinel_series(
        primary, primary.bars[-1].end_timestamp - timedelta(days=1), Decimal(1)
    )
    with case.session() as session:
        rule = EmaSmokeRule(EmaParameters(8, 48))
        for outcomes, message in (
            ((object(),), "QF-7 study outcomes"),
            ((replace(forward, evaluator=IntradayExcursionEvaluator()),), "reviewed"),
            (
                (intraday_forward_return_outcome(timedelta(minutes=30), other_source),),
                "canonical primary source",
            ),
            (
                (intraday_forward_return_outcome(timedelta(minutes=180), primary),),
                "purge horizon",
            ),
            ((forward, forward), "unique"),
        ):
            with pytest.raises(RapidAdmissionError, match=message):
                session.scan(rule, outcomes=cast(tuple[object, ...], outcomes))


# -- reuse, determinism and no authoritative machinery -----------------------


def test_parameter_sweep_reuses_preparation_and_feature_series(
    case: RapidCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    partitions: list[object] = []
    original = rapid_session.partition

    def counted(*args: Any, **kwargs: Any) -> Any:
        partitions.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(rapid_session, "partition", counted)
    outcomes = configured_outcomes(case.inputs.primary)
    with case.session() as session:
        scans = [
            session.scan(EmaSmokeRule(EmaParameters(*pair)), outcomes=outcomes)
            for pair in ((8, 48), (8, 40), (12, 60))
        ]
        repeat = session.scan(EmaSmokeRule(EmaParameters(8, 48)), outcomes=outcomes)
        statistics = session.statistics()
    assert len(partitions) == 1
    assert statistics["eligibility_builds"] == 1
    assert statistics["eligibility_reuses"] == 3
    assert statistics["position_builds"] == 2
    assert statistics["position_reuses"] == 6
    # Unique (timeframe, period, input start) series: 2m EMA 8/48/40/12/60 once
    # each; daily EMA50 once per QF-63 rolling start (before/after the first
    # in-window close). Later scans build only their new periods.
    assert [dict(item.profile.counts)["qf63_series_built"] for item in scans] == [
        4,
        1,
        2,
    ]
    assert dict(repeat.profile.counts)["qf63_series_built"] == 0
    assert statistics["qf63_unique_series"] == 7
    assert statistics["qf63_series_verification_fallbacks"] == 0
    assert statistics["qf63_reference_indicator_fallbacks"] == 0
    # Deterministic: a repeated scan and an independent session agree exactly.
    assert repeat.scientific_primitive() == scans[0].scientific_primitive()
    with case.session() as other:
        independent = other.scan(EmaSmokeRule(EmaParameters(8, 48)), outcomes=outcomes)
    assert independent.scientific_primitive() == scans[0].scientific_primitive()


def test_scans_build_no_receipts_journals_checkpoints_or_study_identities(
    case: RapidCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("authoritative persistence/identity machinery was used")

    identities: list[object] = []
    stable_id = study_module._stable_id  # pyright: ignore[reportPrivateUsage]

    def counted_identity(value: Any) -> str:
        identities.append(value)
        return stable_id(value)

    kernel_calls: list[object] = []
    kernel = spy_ema.midday_bullish_cross

    def counted_kernel(*args: Any) -> bool:
        kernel_calls.append(args)
        return kernel(*args)

    before = files(case.root)
    with case.session() as session:
        rule = EmaSmokeRule(EmaParameters(8, 48))
        eligible = session.eligible_timestamps(rule)
        for owner, name in (
            (window_module, "run_prediction_study_in_session"),
            (study_module, "run_prediction_study_in_session"),
            (IncrementalPredictionWindowWriter, "append"),
            (IncrementalPredictionWindowWriter, "finalize"),
            (os, "fsync"),
        ):
            monkeypatch.setattr(owner, name, forbidden)
        monkeypatch.setattr(study_module, "_stable_id", counted_identity)
        monkeypatch.setattr(spy_ema, "midday_bullish_cross", counted_kernel)
        scanned = session.scan(rule, outcomes=configured_outcomes(case.inputs.primary))
    monkeypatch.undo()
    assert files(case.root) == before
    # One shared-kernel call per eligible decision, none per ineligible one;
    # the only hashes are the transient QF-11 outcome objects for triggers.
    assert len(kernel_calls) == len(eligible) == scanned.evaluated_decisions
    assert len(identities) == scanned.trigger_count * len(
        scanned.outcome_configurations
    )


def test_closed_sessions_release_state_and_refuse_scans(case: RapidCase) -> None:
    with case.session() as session:
        pass
    with pytest.raises(RapidScanError, match="closed"):
        session.scan(EmaSmokeRule(EmaParameters(8, 48)))


# -- non-authoritative result boundary, export and promotion ------------------


def keys(value: object) -> set[str]:
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        return set(mapping) | {k for item in mapping.values() for k in keys(item)}
    if isinstance(value, list):
        return {k for item in cast(list[object], value) for k in keys(item)}
    return set()


def test_rapid_results_are_structurally_non_authoritative(
    result: RapidScanResult,
) -> None:
    assert type(result).__mro__ == (RapidScanResult, object)
    assert not isinstance(result, PredictionWindowResult)
    assert result.authoritative is False
    assert result.mode == "exploratory"
    with pytest.raises(AttributeError):
        setattr(result, "authoritative", True)
    primitive = result.to_primitive()
    assert primitive["authoritative"] is False
    assert primitive["authoritative_reproduction_required"] is True
    assert "NON-AUTHORITATIVE" in cast(str, primitive["notice"])
    assert not keys(primitive) & {
        "result_id",
        "window_result_id",
        "study_id",
        "prediction_study_id",
        "context_id",
        "decision_id",
        "receipt",
        "checkpoint",
        "selection_id",
        "holdout_id",
        "artifact_id",
    }
    assert result.trigger_count == 3


def test_exports_are_unmistakable_and_rejected_by_authoritative_paths(
    result: RapidScanResult, tmp_path: Path
) -> None:
    with pytest.raises(RapidScanError, match=RAPID_EXPORT_SUFFIX):
        export_rapid_scan(result, tmp_path / "scan.json")
    path = export_rapid_scan(result, tmp_path / "exploration" / "scan.rapid.json")
    document = json.loads(path.read_text())
    assert document["authoritative"] is False
    assert document["mode"] == "exploratory"
    assert document["schema_version"] == "1"
    assert "NON-AUTHORITATIVE" in document["notice"]
    assert document["strategy"]["configuration_id"] == result.strategy.configuration_id
    assert len(document["events"]) == result.trigger_count
    with pytest.raises(ManifestError, match="non_authoritative_artifact"):
        index_artifact(
            path.parent,
            path=path.name,
            artifact_type=ArtifactType.PREDICTION_RESULT,
            schema_version="1",
            producer_study_id="rapid",
            producer_artifact_id="scan",
        )
    with pytest.raises(InvalidPredictionOutputError):
        PredictionWindowReader.open(path)
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    study = tmp_path / "study"
    study.mkdir()
    (study / "manifest.json").write_text("{}")
    for destination in (
        ledger.root / "scan.rapid.json",
        study / "x" / "scan.rapid.json",
    ):
        with pytest.raises(RapidScanError, match="authoritative study directory"):
            export_rapid_scan(result, destination)


def test_promotion_freezes_configuration_for_authoritative_reproduction(
    case: RapidCase, result: RapidScanResult
) -> None:
    promoted = promote_strategy_configuration(result)
    primitive = promoted.to_primitive()
    assert primitive["status"] == "requires_authoritative_reproduction"
    assert primitive["authoritative"] is False
    assert not keys(primitive) & {"events", "outcomes", "exploratory_outcome_summaries"}
    # The authoritative factory's own rule for the same parameters matches.
    authoritative = EmaStudyFactory(case.inputs.primary).build({"ema_pair": "8/48"})
    promoted.require_same_rule(authoritative.strategy)
    promoted.require_same_rule(
        EmaSmokeRule(EmaParameters.from_primitive(promoted.parameters))
    )
    with pytest.raises(RapidScanError, match="differs"):
        promoted.require_same_rule(EmaSmokeRule(EmaParameters(8, 40)))
    with pytest.raises(RapidScanError, match="only rapid scan results"):
        promote_strategy_configuration(cast(RapidScanResult, object()))
