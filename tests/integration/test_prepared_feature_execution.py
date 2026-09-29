"""QF-63 prepared context/feature execution against the unchanged reference path.

Every comparison uses the same validated QF-45 composition and QF-39 provider;
the reference provider only disables the QF-63 scope, keeping QF-59 selection.
"""

from copy import deepcopy
from dataclasses import fields, replace
from datetime import datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest
import talib

import quantforge.prediction.prepared_outcomes as prepared_outcomes
from quantforge.data import IntradayBar, TimeframeBarSeries
from quantforge.data.lineage import AdjustmentBasis
from quantforge.data.multi_timeframe import (
    ContextCompletionPolicy,
    MultiTimeframeContext,
    MultiTimeframeContextError,
)
from quantforge.examples.spy_ema import (
    DAILY,
    TWO_MINUTES,
    EmaSmokeRule,
    configured_outcomes,
)
from quantforge.indicators import (
    TALIB_INDICATOR_BACKEND,
    ExponentialMovingAverage,
    ExponentialMovingAverageParameters,
    TimeframeIndicatorOutput,
)
from quantforge.indicators.backends import NATIVE_INDICATOR_BACKEND
from quantforge.indicators.exceptions import InvalidIndicatorBackendError
from quantforge.prediction import (
    PredictionContextRequirements,
    PredictionIndicatorRequirement,
    PredictionTimeframeRequirement,
    build_signal_feature_dataset,
)
from quantforge.prediction.context import (
    PredictionContextError,
    PredictionIndicatorOutputCache,
    PredictionRuleContext,
    build_prediction_rule_context,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.prepared_features import (
    PreparedContextScope,
    PreparedFeatureIntegrityError,
)
from quantforge.prediction.window import (
    PredictionWindowResult,
    _DecisionContextProvider,  # pyright: ignore[reportPrivateUsage]
    run_prediction_window_in_session,
)
from quantforge.prediction.window_compact import CompactPredictionWindowResult
from quantforge.prediction.window_encoding import canonical
from quantforge.prediction.window_execution import (
    run_incremental_prediction_window_in_session,
)
from quantforge.prediction.window_incremental import IncrementalPredictionWindowWriter
from quantforge.walk_forward import FoldStatus, WalkForwardStudy
from quantforge.walk_forward.prediction import (
    _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.prepared_feature_fixtures import (
    SmokeScope,
    smoke_inputs,
    smoke_scope,
)

PAIRS = ("8/40", "8/48", "12/60")
NEW_YORK = ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def scope(tmp_path_factory: pytest.TempPathFactory) -> SmokeScope:
    root = tmp_path_factory.mktemp("qf63-prepared")
    return smoke_scope(smoke_inputs(root / "cache"), root / "plan", fixed=False)


@pytest.fixture(scope="module")
def references(scope: SmokeScope) -> dict[str, PredictionWindowResult[Any, Any, Any]]:
    return {
        pair: run_prediction_window_in_session(
            scope.session(),
            scope.study(pair),
            **scope.arguments(scope.provider(reference=True)),
        )
        for pair in PAIRS
    }


def scope_of(provider: _PermittedContextProvider) -> PreparedContextScope:
    prepared = provider._prepared_features  # pyright: ignore[reportPrivateUsage]
    assert prepared is not None
    return prepared


def requirements_of(scope: SmokeScope, pair: str = "8/48") -> Any:
    return scope.study(pair).strategy.context_requirements


def rule_context(
    scope: SmokeScope,
    provider: _PermittedContextProvider,
    requirements: PredictionContextRequirements,
    as_of: datetime,
    **kwargs: Any,
) -> PredictionRuleContext:
    metadata = scope.permitted.dataset.metadata
    return build_prediction_rule_context(
        requirements,
        provider.get_context_at(requirements, as_of=as_of),
        prediction_dataset_id=metadata.dataset_id,
        symbol=metadata.canonical_symbol,
        prediction_adjustment_basis=AdjustmentBasis(
            metadata.adjustment_mode,
            metadata.ohlc_basis,
            metadata.volume_basis,
            metadata.corporate_action_policy,
            metadata.adjusted_fields_used,
        ),
        **kwargs,
    )


def record_rule_values(
    monkeypatch: pytest.MonkeyPatch, into: list[tuple[datetime, bytes]]
) -> None:
    original = EmaSmokeRule.generate_with_context

    def recorded(self: EmaSmokeRule, context: PredictionRuleContext) -> Any:
        # Complete rule input, including every bar and indicator value.
        into.append((context.as_of, canonical(context.values_primitive())))
        return original(self, context)

    monkeypatch.setattr(EmaSmokeRule, "generate_with_context", recorded)


def sentinel_series(
    source: TimeframeBarSeries, cutoff: datetime, price: Decimal
) -> TimeframeBarSeries:
    """Same artifact binding; every bar ending after ``cutoff`` has absurd prices."""
    bars = tuple(
        bar
        if bar.end_timestamp <= cutoff
        else replace(bar, open=price, high=price, low=price, close=price)
        for bar in source.bars
    )
    return TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        source.dataset_reference,
        source.timeframe,
        bars,
        dataset_family_manifest_id=source.dataset_family_manifest_id,
        developing_source_evidence=source._developing_source_evidence,  # pyright: ignore[reportPrivateUsage]
    )


# -- exact equivalence --------------------------------------------------------


def test_every_decision_and_rule_input_equals_reference(
    scope: SmokeScope,
    references: dict[str, PredictionWindowResult[Any, Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = scope.study("8/48")
    reference_values: list[tuple[datetime, bytes]] = []
    prepared_values: list[tuple[datetime, bytes]] = []
    with monkeypatch.context() as patched:
        record_rule_values(patched, reference_values)
        reference = run_prediction_window_in_session(
            scope.session(), study, **scope.arguments(scope.provider(reference=True))
        )
    provider = scope.provider()
    with monkeypatch.context() as patched:
        record_rule_values(patched, prepared_values)
        prepared = run_prediction_window_in_session(
            scope.session(), study, **scope.arguments(provider)
        )
    # Timestamps, candidate/no-candidate results, context/study/prediction/
    # outcome/evaluation IDs, features and every serialized byte.
    assert prepared.serialize() == reference.serialize()
    assert reference.serialize() == references["8/48"].serialize()
    for version in ("2", "3"):
        assert CompactPredictionWindowResult.from_window(
            prepared, schema_version=version
        ).serialize() == (
            CompactPredictionWindowResult.from_window(
                reference, schema_version=version
            ).serialize()
        )
    # Complete rule inputs: all visible bars and all indicator values.
    assert prepared_values == reference_values
    assert len(prepared_values) == len(scope.schedule.decision_timestamps) == 64
    candidates = [d for d in prepared.decisions if d.result.signals]
    assert [d.decision_timestamp.astimezone(NEW_YORK).time() for d in candidates] == [
        time(11)
    ]
    assert {
        d.decision_timestamp.astimezone(NEW_YORK).date() for d in prepared.decisions
    } == {
        scope.schedule.decision_sessions[0],
        scope.schedule.decision_sessions[-1],
    }
    statistics = scope_of(provider).statistics()
    assert statistics["contexts"] == 64
    # 2m EMA8, 2m EMA48, and daily EMA50 from two distinct rolling starts.
    assert statistics["unique_series"] == statistics["series_built"] == 4
    assert statistics["series_verified"] == 4
    assert statistics["series_verification_fallbacks"] == 0
    assert statistics["reference_indicator_fallbacks"] == 0
    assert statistics["series_reused"] == 3 * 64 - 4


def test_contexts_are_exact_causal_prefixes_with_rolling_daily_start(
    scope: SmokeScope,
) -> None:
    requirements = requirements_of(scope)
    prepared_provider = scope.provider()
    reference_provider = scope.provider(reference=True)
    first_daily: list[str] = []
    for as_of in scope.schedule.decision_timestamps:
        prepared = prepared_provider.get_context_at(requirements, as_of=as_of)
        reference = reference_provider.get_context_at(requirements, as_of=as_of)
        assert prepared.prepared_evidence is not None
        assert reference.prepared_evidence is None
        assert prepared == reference
        assert prepared.to_primitive() == reference.to_primitive()
        primary = prepared.bars_for(TWO_MINUTES)
        daily = prepared.bars_for(DAILY)
        assert primary[-1].end_timestamp == as_of
        assert all(bar.end_timestamp <= as_of for bar in (*primary, *daily))
        session = as_of.astimezone(NEW_YORK).date()
        close = as_of.astimezone(NEW_YORK).time() == time(16)
        # The current session's eventual daily bar is visible only at its close.
        assert (cast(Any, daily[-1]).session_dates[-1] == session) is close
        first_daily.append(daily[0].bar_id)
    # QF-45's rolling daily start: anchor + 50 before the first in-window close,
    # exactly 50 afterwards. The start shifts once, at the 16:00 decision.
    shift = next(i for i, bar in enumerate(first_daily) if bar != first_daily[0])
    assert scope.schedule.decision_timestamps[shift].astimezone(NEW_YORK).time() == (
        time(16)
    )
    assert len(set(first_daily)) == 2
    # Prepared runs physically end at the window's last permitted cutoff.
    runs = scope_of(prepared_provider).runs
    last = scope.schedule.decision_timestamps[-1]
    assert all(run.end_timestamps[-1] <= last for run in runs.values())


def test_sentinel_future_prices_never_reach_earlier_prepared_features(
    scope: SmokeScope,
) -> None:
    requirements = requirements_of(scope)
    stamps = scope.schedule.decision_timestamps
    cutoff = stamps[20]
    absurd = Decimal("999999")
    sentinel = scope.provider(
        series=tuple(
            sentinel_series(source, cutoff, absurd) for source in scope.adapter.series
        )
    )
    original = scope.provider()
    # The sentinel run physically contains the absurd later bars.
    run = scope_of(sentinel).runs[TWO_MINUTES.configuration_id]
    assert run.bars[-1].close == absurd
    for as_of in stamps[:21]:
        assert canonical(
            rule_context(scope, sentinel, requirements, as_of).values_primitive()
        ) == canonical(
            rule_context(scope, original, requirements, as_of).values_primitive()
        )
    # Sanity: once a sentinel bar is at/before the cutoff, it is visible.
    changed = rule_context(scope, sentinel, requirements, stamps[21])
    assert changed.latest_bar_for(TWO_MINUTES).close == absurd
    assert changed.indicator_for(TWO_MINUTES, "fast").values_for(
        "exponential_moving_average"
    ) != (
        rule_context(scope, original, requirements, stamps[21])
        .indicator_for(TWO_MINUTES, "fast")
        .values_for("exponential_moving_average")
    )


# -- reuse --------------------------------------------------------------------


def test_trials_share_series_and_outcome_backing_with_exact_results(
    scope: SmokeScope,
    references: dict[str, PredictionWindowResult[Any, Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authentications: list[int] = []
    original = prepared_outcomes._source_identity  # pyright: ignore[reportPrivateUsage]

    def counted(*args: Any, **kwargs: Any) -> str:
        authentications.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(prepared_outcomes, "_source_identity", counted)
    provider = scope.provider()
    session = scope.session()  # QF-32 shares one dataset session across trials.
    for pair in PAIRS:
        window = run_prediction_window_in_session(
            session, scope.study(pair), **scope.arguments(provider)
        )
        assert window.serialize() == references[pair].serialize()
    statistics = scope_of(provider).statistics()
    # 2m EMA8 is shared by 8/40 and 8/48; daily EMA50 by all trials (two starts).
    # Distinct parameters (8, 12, 40, 48, 60) remain distinct series.
    assert statistics["unique_series"] == statistics["series_built"] == 7
    assert statistics["series_reused"] == 3 * 64 * 3 - 7
    assert statistics["series_verification_fallbacks"] == 0
    # The outcome source is authenticated once, not per trial or per decision.
    assert len(authentications) == 1


def test_parameters_backends_and_scopes_never_share_series(scope: SmokeScope) -> None:
    provider = scope.provider()
    prepared = scope_of(provider)
    as_of = scope.schedule.decision_timestamps[30]

    def requirements(period: int, backend_id: str) -> PredictionContextRequirements:
        return PredictionContextRequirements(
            PredictionTimeframeRequirement(
                TWO_MINUTES,
                scope.adapter.series[0].dataset_reference.feed_scope,
                (
                    PredictionIndicatorRequirement(
                        "value",
                        ExponentialMovingAverage(
                            ExponentialMovingAverageParameters(period),
                            backend_id=backend_id,
                        ),
                    ),
                ),
            ),
            (
                PredictionTimeframeRequirement(
                    DAILY, scope.adapter.series[0].dataset_reference.feed_scope
                ),
            ),
        )

    outputs: dict[tuple[int, str], TimeframeIndicatorOutput] = {}
    for period, backend_id in (
        (8, TALIB_INDICATOR_BACKEND),
        (9, TALIB_INDICATOR_BACKEND),
        (8, NATIVE_INDICATOR_BACKEND),
        (8, TALIB_INDICATOR_BACKEND),
    ):
        context = rule_context(scope, provider, requirements(period, backend_id), as_of)
        outputs[(period, backend_id)] = context.indicator_for(TWO_MINUTES, "value")
        reference = rule_context(
            scope,
            scope.provider(reference=True),
            requirements(period, backend_id),
            as_of,
        )
        assert outputs[(period, backend_id)] == reference.indicator_for(
            TWO_MINUTES, "value"
        )
    statistics = prepared.statistics()
    assert statistics["unique_series"] == statistics["series_built"] == 3
    assert statistics["series_reused"] == 1  # only the repeated talib EMA8
    assert (
        outputs[(8, TALIB_INDICATOR_BACKEND)].backend_identity
        != outputs[(8, NATIVE_INDICATOR_BACKEND)].backend_identity
    )
    # Every provider owns its scope; series never cross providers (and hence
    # never cross folds, roles or windows; see the two-fold QF-39 test).
    other = scope.provider()
    assert scope_of(other) is not prepared
    assert scope_of(other).statistics()["unique_series"] == 0


def nondefault_compatibility() -> int:
    return 1


def nonzero_unstable_period(function_name: str) -> int:
    return 5


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        (
            "get_compatibility",
            nondefault_compatibility,
            "default compatibility and zero unstable period",
        ),
        (
            "get_unstable_period",
            nonzero_unstable_period,
            "default compatibility and zero unstable period",
        ),
        (
            "__ta_version__",
            b"0.7.2 (different runtime)",
            "backend result metadata changed during calculation",
        ),
    ],
    ids=["compatibility", "unstable_period", "runtime_identity"],
)
def test_talib_state_drift_after_caching_fails_exactly_like_reference(
    scope: SmokeScope,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: object,
    message: str,
) -> None:
    """Reusing a cached talib_v1 series still applies compute()'s process checks."""
    provider = scope.provider()
    requirements = requirements_of(scope)
    stamps = scope.schedule.decision_timestamps
    rule_context(scope, provider, requirements, stamps[10])
    before = scope_of(provider).statistics()
    assert before["series_built"] == before["series_verified"] == 3
    with monkeypatch.context() as drifted:
        drifted.setattr(talib, attribute, value)
        # Every series is cached: the prepared path must fail as a new
        # computation (the reference path) does, not silently reuse.
        for candidate in (provider, scope.provider(reference=True)):
            with pytest.raises(InvalidIndicatorBackendError, match=message):
                rule_context(scope, candidate, requirements, stamps[11])
    after = scope_of(provider).statistics()
    assert after["series_reused"] == before["series_reused"]
    # Restored state: the same cached series serve again and stay exact.
    restored = rule_context(scope, provider, requirements, stamps[11])
    reference = rule_context(
        scope, scope.provider(reference=True), requirements, stamps[11]
    )
    assert restored.values_primitive() == reference.values_primitive()
    assert scope_of(provider).statistics()["series_built"] == 3


# -- mutation safety ------------------------------------------------------------


def isolated(scope: SmokeScope) -> _PermittedContextProvider:
    """A provider over private deep copies: mutation tests never touch fixtures."""
    return scope.provider(series=deepcopy(scope.adapter.series))


@pytest.mark.parametrize("target", ["bar", "provenance"])
def test_shared_backing_mutation_fails_closed_for_later_decisions_and_trials(
    scope: SmokeScope, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    provider = isolated(scope)
    original = EmaSmokeRule.generate_with_context
    stamps = scope.schedule.decision_timestamps

    def malicious(self: EmaSmokeRule, context: PredictionRuleContext) -> Any:
        output = original(self, context)
        if context.as_of == stamps[5]:
            bar = cast(IntradayBar, context.bars_for(TWO_MINUTES)[0])
            if target == "bar":
                object.__setattr__(bar, "close", Decimal("999999"))
            else:
                object.__setattr__(bar.provenance, "provider_name", "tampered")
        return output

    monkeypatch.setattr(EmaSmokeRule, "generate_with_context", malicious)
    with pytest.raises(
        InvalidPredictionOutputError,
        match="prediction rule context changed while the prediction evaluator",
    ):
        run_prediction_window_in_session(
            scope.session(), scope.study("8/48"), **scope.arguments(provider)
        )
    monkeypatch.setattr(EmaSmokeRule, "generate_with_context", original)
    # The scope refuses every later decision and every other trial.
    with pytest.raises(PreparedFeatureIntegrityError):
        provider.get_context_at(requirements_of(scope), as_of=stamps[6])
    with pytest.raises(PreparedFeatureIntegrityError):
        run_prediction_window_in_session(
            scope.session(), scope.study("12/60"), **scope.arguments(provider)
        )


def test_reference_path_rejects_the_same_mutation(
    scope: SmokeScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = scope.provider(series=deepcopy(scope.adapter.series), reference=True)
    original = EmaSmokeRule.generate_with_context

    def malicious(self: EmaSmokeRule, context: PredictionRuleContext) -> Any:
        output = original(self, context)
        object.__setattr__(context.bars_for(TWO_MINUTES)[0], "close", Decimal("9"))
        return output

    monkeypatch.setattr(EmaSmokeRule, "generate_with_context", malicious)
    with pytest.raises(
        InvalidPredictionOutputError,
        match="prediction rule context changed while the prediction evaluator",
    ):
        run_prediction_window_in_session(
            scope.session(), scope.study("8/48"), **scope.arguments(provider)
        )


def test_decision_owned_shell_mutation_is_rejected_without_leaking(
    scope: SmokeScope,
) -> None:
    provider = isolated(scope)
    requirements = requirements_of(scope)
    stamps = scope.schedule.decision_timestamps
    context = rule_context(scope, provider, requirements, stamps[10])
    context.validate_values_unchanged()
    fast = context.indicator_for(TWO_MINUTES, "fast")
    object.__setattr__(fast, "fields", ())
    with pytest.raises(PredictionContextError, match="changed after construction"):
        context.validate_values_unchanged()
    # Decision-owned shells are fresh per decision; shared state is untouched.
    later = rule_context(scope, provider, requirements, stamps[11])
    later.validate_values_unchanged()
    reference = rule_context(
        scope, scope.provider(reference=True), requirements, stamps[11]
    )
    assert canonical(later.values_primitive()) == canonical(
        reference.values_primitive()
    )
    rerun = rule_context(scope, provider, requirements, stamps[10])
    assert rerun.indicator_for(TWO_MINUTES, "fast").fields
    assert (
        rerun.values_primitive()
        == rule_context(
            scope, scope.provider(reference=True), requirements, stamps[10]
        ).values_primitive()
    )


def test_rule_facing_context_holds_only_causally_visible_data(
    scope: SmokeScope,
) -> None:
    provider = scope.provider()
    as_of = scope.schedule.decision_timestamps[3]
    context = rule_context(scope, provider, requirements_of(scope), as_of)
    run = scope_of(provider).runs[TWO_MINUTES.configuration_id]
    future = {id(bar) for bar in run.bars if bar.end_timestamp > as_of}
    assert future
    seen: set[int] = set()
    stack: list[object] = [context]
    while stack:
        item = stack.pop()
        if id(item) in seen or isinstance(item, type) or callable(item):
            continue
        seen.add(id(item))
        assert id(item) not in future
        assert not isinstance(item, (PreparedContextScope, MultiTimeframeContext))
        if isinstance(item, tuple | list):
            stack.extend(cast(tuple[object, ...], tuple(cast(Any, item))))
        elif hasattr(item, "__dataclass_fields__"):
            stack.extend(getattr(item, entry.name) for entry in fields(cast(Any, item)))
        elif hasattr(item, "__slots__"):
            stack.extend(
                getattr(item, name)
                for name in cast(tuple[str, ...], getattr(type(item), "__slots__"))
                if hasattr(item, name)
            )
    last = context.indicator_for(TWO_MINUTES, "fast")
    assert last.bar_end_timestamps[-1] == as_of


# -- explicit fallback -----------------------------------------------------------


class LookalikeEma(ExponentialMovingAverage):
    """A subclass is not an exact reviewed indicator type."""


class EvaluatingCache:
    def resolve(
        self,
        requirement: PredictionIndicatorRequirement,
        context: MultiTimeframeContext,
        timeframe: Any,
        completion_policy: ContextCompletionPolicy,
    ) -> TimeframeIndicatorOutput:
        return requirement.evaluate(context, timeframe, completion_policy)


def test_unsupported_components_use_the_equivalent_reference_path(
    scope: SmokeScope,
) -> None:
    feed = scope.adapter.series[0].dataset_reference.feed_scope
    requirements = PredictionContextRequirements(
        PredictionTimeframeRequirement(
            TWO_MINUTES,
            feed,
            (
                PredictionIndicatorRequirement(
                    "custom",
                    LookalikeEma(
                        ExponentialMovingAverageParameters(8),
                        backend_id=TALIB_INDICATOR_BACKEND,
                    ),
                ),
            ),
        ),
        (PredictionTimeframeRequirement(DAILY, feed),),
    )
    provider = scope.provider()
    as_of = scope.schedule.decision_timestamps[40]
    prepared = rule_context(scope, provider, requirements, as_of)
    reference = rule_context(scope, scope.provider(reference=True), requirements, as_of)
    assert prepared.values_primitive() == reference.values_primitive()
    statistics = scope_of(provider).statistics()
    assert statistics["reference_indicator_fallbacks"] == 1
    assert statistics["unique_series"] == 0
    # An explicit indicator-output cache keeps the complete reference path.
    cached = rule_context(
        scope,
        provider,
        requirements,
        as_of,
        indicator_output_cache=cast(PredictionIndicatorOutputCache, EvaluatingCache()),
    )
    assert not cached.values_guarded
    assert cached.values_primitive() == reference.values_primitive()
    # Developing-bar contexts never use prepared runs; both fail identically here
    # because the derived 2m series carries no developing-source evidence.
    developing = replace(
        requirements,
        contextual=(
            PredictionTimeframeRequirement(
                DAILY,
                feed,
                completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
            ),
        ),
    )
    errors: list[str] = []
    for candidate in (provider, scope.provider(reference=True)):
        with pytest.raises(MultiTimeframeContextError) as raised:
            candidate.get_context_at(developing, as_of=as_of)
        errors.append(str(raised.value))
    assert errors[0] == errors[1]


def test_custom_source_records_disable_preparation(scope: SmokeScope) -> None:
    class TaggedBar(IntradayBar):
        __slots__ = ()

    source = next(
        item for item in scope.adapter.series if item.timeframe == TWO_MINUTES
    )
    tagged = object.__new__(TaggedBar)
    for entry in fields(IntradayBar):
        object.__setattr__(tagged, entry.name, getattr(source.bars[-1], entry.name))
    custom = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        source.dataset_reference,
        source.timeframe,
        (*source.bars[:-1], tagged),
        dataset_family_manifest_id=source.dataset_family_manifest_id,
    )
    provider = _PermittedContextProvider(
        scope.config.plan,
        scope.permitted,
        tuple(custom if item is source else item for item in scope.adapter.series),
        scope.schedule,
    )
    assert provider._prepared_features is None  # pyright: ignore[reportPrivateUsage]
    as_of = scope.schedule.decision_timestamps[0]
    context = provider.get_context_at(requirements_of(scope), as_of=as_of)
    assert context.prepared_evidence is None
    assert context == scope.provider(reference=True).get_context_at(
        requirements_of(scope), as_of=as_of
    )


# -- resume, export and QF-32/QF-39 -----------------------------------------------


def test_restart_rebuilds_preparation_without_recomputing_committed_decisions(
    scope: SmokeScope,
    references: dict[str, PredictionWindowResult[Any, Any, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "prediction-window.jsonl"
    study = scope.study("8/48")
    append = IncrementalPredictionWindowWriter.append

    def interrupt(self: IncrementalPredictionWindowWriter, decision: Any) -> None:
        append(self, decision)
        if self.completed_count == 5:
            raise KeyboardInterrupt("bounded prepared prefix")

    def execute(provider: _PermittedContextProvider) -> Any:
        with scope.adapter.preparation_scope():
            return run_incremental_prediction_window_in_session(
                scope.session(),
                study,
                path=path,
                canonical_metadata=scope.inputs.dataset.metadata,
                schema_version="3",
                projection_registry=scope.adapter._projection_registry,  # pyright: ignore[reportPrivateUsage]
                projection_scope=scope.adapter._projection_scope(  # pyright: ignore[reportPrivateUsage]
                    scope.config.plan, scope.permitted
                ),
                **scope.arguments(provider),
            )

    with monkeypatch.context() as stopped:
        stopped.setattr(IncrementalPredictionWindowWriter, "append", interrupt)
        with pytest.raises(KeyboardInterrupt):
            execute(scope.provider())
    prefix = (
        path.with_name(path.name + ".in-progress") / "decisions.jsonl"
    ).read_bytes()
    calls: list[datetime] = []
    get_context_at = _PermittedContextProvider.get_context_at

    def tracked(self: Any, requirements: Any, *, as_of: datetime) -> Any:
        calls.append(as_of)
        return get_context_at(self, requirements, as_of=as_of)

    monkeypatch.setattr(_PermittedContextProvider, "get_context_at", tracked)
    fresh = scope.provider()  # A new process: preparation is rebuilt, not loaded.
    execute(fresh)
    assert calls == list(scope.schedule.decision_timestamps[5:])
    final = path.read_bytes()
    assert final == (
        CompactPredictionWindowResult.from_window(
            references["8/48"], schema_version="3"
        ).serialize()
    )
    assert prefix in final
    calls.clear()
    execute(scope.provider())
    assert calls == []


def test_candidate_feature_export_is_equivalent_and_reuses_preparation(
    scope: SmokeScope,
    references: dict[str, PredictionWindowResult[Any, Any, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = next(d for d in references["8/48"].decisions if d.result.signals)
    authentications: list[int] = []
    original = prepared_outcomes._source_identity  # pyright: ignore[reportPrivateUsage]

    def counted(*args: Any, **kwargs: Any) -> str:
        authentications.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(prepared_outcomes, "_source_identity", counted)
    rule = cast(EmaSmokeRule, scope.study("8/48").strategy)
    results: list[Any] = []
    for name, provider in (
        ("reference", scope.provider(reference=True)),
        ("prepared", scope.provider()),
    ):
        authentications.clear()
        results.append(
            build_signal_feature_dataset(
                dataset=scope.permitted.dataset,
                prediction_study=scope.study("8/48"),
                contextual_features=(),
                multi_timeframe_features=rule.multi_timeframe_feature_requests,
                context_provider=_DecisionContextProvider(
                    provider,
                    candidate.decision_timestamp,
                    scope.adapter.series[0].dataset_reference.family_id,
                ),
                outcomes=configured_outcomes(scope.inputs.primary),
                output_root=tmp_path / name,
            )
        )
        # Configured outcomes reuse one retained backing, not one per outcome.
        assert len(authentications) <= 2
    reference, prepared = results
    assert prepared.dataset_id == reference.dataset_id
    assert [row.to_primitive() for row in prepared.rows] == [
        row.to_primitive() for row in reference.rows
    ]
    assert len(prepared.rows) == 1
    assert prepared.manifest_primitive() == reference.manifest_primitive()
    for relative in sorted(
        path.relative_to(tmp_path / "reference")
        for path in (tmp_path / "reference").rglob("*")
        if path.is_file() and path.suffix in {".csv", ".json"}
    ):
        assert (tmp_path / "prepared" / relative).read_bytes() == (
            tmp_path / "reference" / relative
        ).read_bytes()


def reference_capture(cls: type, runs: object) -> None:
    """Disable QF-63 preparation only; QF-59 indexed selection stays active."""
    return None


def test_two_fold_walk_forward_selection_ranking_and_oos_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.walk_forward.test_incremental_prediction import compact_adapter
    from tests.unit.walk_forward.timestamp_fixtures import timestamp_fixture

    config, original = timestamp_fixture(tmp_path)
    adapter = compact_adapter(original)
    scopes: list[PreparedContextScope] = []
    capture = PreparedContextScope.capture.__func__  # pyright: ignore[reportFunctionMemberAccess]

    def record(cls: type, runs: Any) -> PreparedContextScope:
        prepared = capture(cls, runs)
        assert prepared is not None
        scopes.append(prepared)
        return prepared

    with monkeypatch.context() as recorded:
        recorded.setattr(
            PreparedContextScope, "capture", cast(Any, classmethod(record))
        )
        prepared = WalkForwardStudy(config, adapter, tmp_path / "prepared").run()
    with monkeypatch.context() as reference_mode:
        reference_mode.setattr(
            PreparedContextScope, "capture", cast(Any, classmethod(reference_capture))
        )
        reference = WalkForwardStudy(config, adapter, tmp_path / "reference").run()
    assert [fold.status for fold in prepared.folds] == [FoldStatus.COMPLETED] * 2
    # Frozen selections (ranking, tie-breaking, chosen configuration), OOS
    # artifacts and failure evidence are identical for both folds.
    assert prepared == reference
    for kind in ("selection.json", "summary.json", "prediction-window.jsonl"):
        left = sorted((tmp_path / "prepared").rglob(kind))
        right = sorted((tmp_path / "reference").rglob(kind))
        assert len(left) == len(right) > 0
        for a, b in zip(left, right, strict=True):
            assert a.relative_to(tmp_path / "prepared") == b.relative_to(
                tmp_path / "reference"
            )
            assert a.read_bytes() == b.read_bytes()
    # Each fold/role invocation owns a separate scope; nothing crosses folds.
    used = [item for item in scopes if item.statistics()["contexts"]]
    assert len(used) >= 4
    assert len({id(item) for item in scopes}) == len(scopes)
    assert all(item.statistics()["series_built"] > 0 for item in used)
    assert all(item.statistics()["series_verification_fallbacks"] == 0 for item in used)
