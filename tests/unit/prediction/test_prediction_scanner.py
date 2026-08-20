import ast
import concurrent.futures
import io
import json
import os
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from scripts import export_spy_multi_timeframe_context as spy_example

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    ContextAvailability,
    ContextCompletionPolicy,
    ContextTimeframeRequirement,
    FeedScope,
    MultiTimeframeContext,
    TimeframeContext,
    build_multi_timeframe_context,
)
from quantforge.indicators import (
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    TALIB_INDICATOR_BACKEND,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
)
from quantforge.prediction import (
    AlertDeduplicationPolicy,
    AlertDeduplicationStore,
    AlertPersistenceError,
    ConsolePredictionAlertSink,
    HistoricalPredictionStudyReference,
    HistoricalStudyMismatchError,
    InMemoryAlertDeduplicationStore,
    JsonFileAlertDeduplicationStore,
    JsonFilePredictionAlertSink,
    PredictionAlert,
    PredictionContextError,
    PredictionContextFailurePolicy,
    PredictionContextRequirements,
    PredictionDirection,
    PredictionIndicatorRequirement,
    PredictionRuleContext,
    PredictionScanner,
    PredictionScannerError,
    PredictionScannerRule,
    PredictionScannerRuleBinding,
    PredictionScannerSnapshot,
    PredictionTimeframeRequirement,
    PublishedAlertDeduplication,
    SignalFeatureCandidateOutput,
    TechnicalCondition,
    TechnicalConditionOperand,
    TechnicalConditionOperator,
    TechnicalConfluenceEvaluation,
    TechnicalConfluenceParameters,
    TechnicalConfluencePredictionRule,
    build_prediction_rule_context,
)
from quantforge.timeframes import BarCompletion, Timeframe
from tests.unit.indicators import test_timeframe_evaluation as timeframe_fixtures
from tests.unit.prediction import test_technical_confluence as rule_fixtures

HISTORICAL_DATASET_FINGERPRINT = "a" * 64


class _CapturingSink:
    def __init__(self) -> None:
        self.alerts: list[PredictionAlert] = []

    def emit(self, alert: PredictionAlert) -> None:
        self.alerts.append(alert)


@dataclass
class _FixtureSource:
    context: MultiTimeframeContext
    dataset_id: str
    adjustment_basis: AdjustmentBasis
    symbol: str = "SPY"

    def __post_init__(self) -> None:
        self.refresh_values: list[bool] = []

    def prepare_context(
        self,
        requirements: PredictionContextRequirements,
        *,
        as_of: datetime,
        refresh: bool,
    ) -> PredictionScannerSnapshot:
        del requirements
        self.refresh_values.append(refresh)
        if self.context.as_of != as_of:
            raise AssertionError("test source received an unexpected as-of")
        return PredictionScannerSnapshot(
            self.context,
            self.dataset_id,
            self.symbol,
            self.adjustment_basis,
        )


class _SelectiveFixtureSource:
    def __init__(
        self,
        delegate: _FixtureSource,
        unavailable_requirements: PredictionContextRequirements,
    ) -> None:
        self.delegate = delegate
        self.unavailable_requirements_id = configuration_identity(
            unavailable_requirements.to_primitive()
        )
        self.requirement_ids: list[str] = []

    def prepare_context(
        self,
        requirements: PredictionContextRequirements,
        *,
        as_of: datetime,
        refresh: bool,
    ) -> PredictionScannerSnapshot:
        requirements_id = configuration_identity(requirements.to_primitive())
        self.requirement_ids.append(requirements_id)
        if requirements_id == self.unavailable_requirements_id:
            raise PredictionContextError("required current context is unavailable")
        return self.delegate.prepare_context(
            requirements,
            as_of=as_of,
            refresh=refresh,
        )


@dataclass
class _MalformedOutputRule:
    delegate: TechnicalConfluencePredictionRule
    output_contract_version: str = "1"
    candidate_strategy_identity: tuple[str, str, str] | None = None
    configuration_id_override: str | None = None
    name: str = field(init=False)
    implementation_version: str = field(init=False)
    context_requirements: PredictionContextRequirements = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.delegate.name
        self.implementation_version = self.delegate.implementation_version
        self.context_requirements = self.delegate.context_requirements

    @property
    def configuration_id(self) -> str:
        return self.configuration_id_override or self.delegate.configuration_id

    def configuration(self) -> PrimitiveMapping:
        return self.delegate.configuration()

    def evaluate(self, context: PredictionRuleContext) -> TechnicalConfluenceEvaluation:
        return self.delegate.evaluate(context)

    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> SignalFeatureCandidateOutput:
        output = self.delegate.generate_with_context(context)
        if self.candidate_strategy_identity is None:
            return replace(output, contract_version=self.output_contract_version)
        strategy_id, implementation_version, configuration_id = (
            self.candidate_strategy_identity
        )
        candidate = replace(
            output.signals[0],
            strategy_id=strategy_id,
            strategy_implementation_version=implementation_version,
            strategy_configuration_id=configuration_id,
        )
        return replace(
            output,
            signals=(candidate,),
            contract_version=self.output_contract_version,
        )


@dataclass
class _RuleMutatingFixtureSource:
    delegate: _FixtureSource
    rule: _MalformedOutputRule

    def prepare_context(
        self,
        requirements: PredictionContextRequirements,
        *,
        as_of: datetime,
        refresh: bool,
    ) -> PredictionScannerSnapshot:
        snapshot = self.delegate.prepare_context(
            requirements,
            as_of=as_of,
            refresh=refresh,
        )
        self.rule.configuration_id_override = "b" * 64
        return snapshot


def _rule(
    *,
    feed_scope: FeedScope = FeedScope.consolidated(),
    completion_policy: ContextCompletionPolicy = (
        ContextCompletionPolicy.COMPLETED_BARS_ONLY
    ),
    backend_id: str = TALIB_INDICATOR_BACKEND,
    maximum_age: timedelta | None = None,
    failure_policy: PredictionContextFailurePolicy = (
        PredictionContextFailurePolicy.FAIL
    ),
) -> TechnicalConfluencePredictionRule:
    five_minute, four_hour, daily, weekly = timeframe_fixtures._timeframes()  # pyright: ignore[reportPrivateUsage]

    def requirement(timeframe: Timeframe) -> PredictionTimeframeRequirement:
        return PredictionTimeframeRequirement(
            timeframe,
            feed_scope,
            (
                PredictionIndicatorRequirement(
                    "trend",
                    SimpleMovingAverage(
                        SimpleMovingAverageParameters(2),
                        backend_id=backend_id,
                    ),
                ),
            ),
            completion_policy,
            maximum_age,
        )

    requirements = PredictionContextRequirements(
        PredictionTimeframeRequirement(five_minute, feed_scope),
        (requirement(four_hour), requirement(daily), requirement(weekly)),
        failure_policy,
    )
    trend = TechnicalConditionOperand.indicator(
        "trend", SIMPLE_MOVING_AVERAGE_OUTPUT, "price_per_share"
    )

    def condition(
        name: str,
        timeframe_name: str,
        timeframe: Timeframe,
        operator: TechnicalConditionOperator,
    ) -> TechnicalCondition:
        return TechnicalCondition(
            name,
            timeframe_name,
            timeframe,
            trend,
            operator,
            Decimal("10"),
        )

    up = tuple(
        condition(
            f"up_{name}_trend",
            name,
            timeframe,
            TechnicalConditionOperator.GREATER_THAN,
        )
        for name, timeframe in (
            ("weekly", weekly),
            ("daily", daily),
            ("four_hour", four_hour),
        )
    )
    down = tuple(
        condition(
            f"down_{name}_trend",
            name,
            timeframe,
            TechnicalConditionOperator.LESS_THAN,
        )
        for name, timeframe in (
            ("weekly", weekly),
            ("daily", daily),
            ("four_hour", four_hour),
        )
    )
    return TechnicalConfluencePredictionRule(
        TechnicalConfluenceParameters(up, down, "scanner_fixture_v1"),
        requirements,
    )


def _scanner(
    rule: PredictionScannerRule,
    source: _FixtureSource,
    sink: _CapturingSink,
    *,
    historical_rule: PredictionScannerRule | None = None,
    deduplication_policy: AlertDeduplicationPolicy = (
        AlertDeduplicationPolicy.EXACT_CONTEXT
    ),
    store: AlertDeduplicationStore | None = None,
) -> PredictionScanner:
    historical = historical_rule or rule
    reference = HistoricalPredictionStudyReference.capture(
        study_id="validated-study-1",
        rule=historical,
        validated_symbols=(source.symbol,),
        historical_dataset_fingerprint=HISTORICAL_DATASET_FINGERPRINT,
        adjustment_basis=source.adjustment_basis,
        summary={"validation_period": "untouched_holdout"},
        sample_count=42,
    )
    return PredictionScanner(
        source,
        (PredictionScannerRuleBinding(rule, reference),),
        (sink,),
        InMemoryAlertDeduplicationStore() if store is None else store,
        deduplication_policy,
    )


def _case_context(case_index: int) -> MultiTimeframeContext:
    return rule_fixtures._context(rule_fixtures._cases()[case_index])  # pyright: ignore[reportPrivateUsage]


def _source(context: MultiTimeframeContext) -> _FixtureSource:
    canonical_source_ids = {
        reference.canonical_source_snapshot_id
        for timeframe_context in context.timeframes
        if (reference := timeframe_context.dataset_reference) is not None
    }
    assert len(canonical_source_ids) == 1
    return _FixtureSource(
        context,
        next(iter(canonical_source_ids)),
        timeframe_fixtures._adjustment_basis(),  # pyright: ignore[reportPrivateUsage]
    )


def _context_with_requirements(
    context: MultiTimeframeContext,
    requirements: PredictionContextRequirements,
    *,
    as_of: datetime,
) -> MultiTimeframeContext:
    aligned: list[TimeframeContext] = []
    for prediction_requirement in requirements.all_timeframes:
        timeframe = prediction_requirement.timeframe
        original = context.metadata_for(timeframe)
        bars = context.bars_for(timeframe)
        context_requirement = (
            ContextTimeframeRequirement(timeframe)
            if prediction_requirement is requirements.primary
            else ContextTimeframeRequirement(
                timeframe, prediction_requirement.maximum_age
            )
        )
        age = as_of - bars[-1].end_timestamp
        availability = (
            ContextAvailability.STALE
            if context_requirement.maximum_age is not None
            and age > context_requirement.maximum_age
            else ContextAvailability.AVAILABLE
        )
        completed = tuple(
            bar for bar in bars if bar.completion is not BarCompletion.DEVELOPING
        )
        aligned.append(
            TimeframeContext._from_aligned_series(  # pyright: ignore[reportPrivateUsage]
                requirement=context_requirement,
                dataset_reference=original.dataset_reference,
                availability=availability,
                bars=bars,
                latest_completed_bar_timestamp=completed[-1].end_timestamp,
                age=age,
            )
        )
    return MultiTimeframeContext._from_aligned_timeframes(  # pyright: ignore[reportPrivateUsage]
        as_of=as_of,
        primary_timeframe=requirements.primary.timeframe,
        required_timeframes=requirements.context_timeframe_requirements(),
        completion_policy=requirements.context_completion_policy,
        source_consistency=context.source_consistency,
        timeframes=tuple(aligned),
    )


def test_accepted_alert_has_complete_causal_payload_and_dry_run_sinks(
    tmp_path: Path,
) -> None:
    rule = _rule()
    context = _case_context(0)
    source = _source(context)
    capture = _CapturingSink()
    console = io.StringIO()
    output_directory = tmp_path / "alerts"
    scanner = PredictionScanner(
        source,
        (
            PredictionScannerRuleBinding(
                rule,
                HistoricalPredictionStudyReference.capture(
                    study_id="validated-study-1",
                    rule=rule,
                    validated_symbols=(source.symbol,),
                    historical_dataset_fingerprint=HISTORICAL_DATASET_FINGERPRINT,
                    adjustment_basis=source.adjustment_basis,
                    summary={"accuracy": "descriptive_only"},
                    sample_count=42,
                ),
            ),
        ),
        (
            capture,
            ConsolePredictionAlertSink(console),
            JsonFilePredictionAlertSink(output_directory),
        ),
        JsonFileAlertDeduplicationStore(tmp_path / "dedup"),
    )

    result = scanner.scan(as_of=context.as_of, dry_run=True)

    assert source.refresh_values == [False]
    assert len(result.alerts) == len(capture.alerts) == 1
    alert = result.alerts[0]
    payload = alert.to_primitive()
    assert alert.direction is PredictionDirection.UP
    assert payload["as_of"] == context.as_of.isoformat()
    assert cast(str, payload["decision_timestamp"]) <= cast(str, payload["as_of"])
    assert payload["context_id"] == context.context_id
    assert payload["disclaimer"]
    historical_study = cast(PrimitiveMapping, payload["historical_study"])
    assert historical_study["sample_count"] == 42
    assert (
        historical_study["historical_dataset_fingerprint"]
        == HISTORICAL_DATASET_FINGERPRINT
    )
    assert historical_study["adjustment_basis"] == (
        source.adjustment_basis.to_primitive()
    )
    assert len(cast(list[object], payload["conditions"])) == 6
    indicators = cast(list[PrimitiveMapping], payload["indicators"])
    assert len(indicators) == 3
    assert all(item["normalized_values"] for item in indicators)
    assert all(
        cast(PrimitiveMapping, item["backend"])["backend_id"] == TALIB_INDICATOR_BACKEND
        for item in indicators
    )
    source_bars = cast(list[PrimitiveMapping], payload["source_bars"])
    assert {cast(str, item["completion"]) for item in source_bars} == {"completed"}
    assert all(item["dataset_reference"] for item in source_bars)
    assert all(item["feed_scope"] for item in source_bars)
    assert json.loads(console.getvalue())["alert_id"] == alert.alert_id
    assert (
        json.loads(
            (output_directory / f"{alert.alert_id}.json").read_text(encoding="utf-8")
        )
        == payload
    )
    assert "outcome_label" not in json.dumps(payload, sort_keys=True)


def test_scanner_rejects_an_unsupported_candidate_output_contract() -> None:
    rule = _MalformedOutputRule(_rule(), output_contract_version="2")
    context = _case_context(0)
    source = _source(context)
    sink = _CapturingSink()

    with pytest.raises(PredictionScannerError, match="incompatible current-data"):
        _scanner(rule, source, sink).scan(as_of=context.as_of)

    assert source.refresh_values == [True]
    assert sink.alerts == []


@pytest.mark.parametrize("identity_index", [0, 1, 2])
def test_scanner_rejects_mismatched_candidate_strategy_identity(
    identity_index: int,
) -> None:
    current_rule = _rule()
    candidate_identity = [
        current_rule.name,
        current_rule.implementation_version,
        current_rule.configuration_id,
    ]
    candidate_identity[identity_index] = (
        "b" * 64 if identity_index == 2 else "mismatched-identity"
    )
    rule = _MalformedOutputRule(
        current_rule,
        candidate_strategy_identity=(
            candidate_identity[0],
            candidate_identity[1],
            candidate_identity[2],
        ),
    )
    context = _case_context(0)
    source = _source(context)
    sink = _CapturingSink()

    with pytest.raises(PredictionScannerError, match="candidate identity"):
        _scanner(rule, source, sink).scan(as_of=context.as_of)

    assert source.refresh_values == [True]
    assert sink.alerts == []


def test_json_alert_sink_syncs_temporary_content_before_atomic_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rule = _rule()
    context = _case_context(0)
    capture = _CapturingSink()
    alert = (
        _scanner(rule, _source(context), capture).scan(as_of=context.as_of).alerts[0]
    )
    output_directory = tmp_path / "alerts"
    content_was_synced = False
    original_fsync = os.fsync
    original_link = os.link

    def track_fsync(descriptor: int) -> None:
        nonlocal content_was_synced
        content_was_synced = True
        original_fsync(descriptor)

    def verify_atomic_install(source: Path, destination: Path) -> None:
        assert content_was_synced
        assert not destination.exists()
        assert source.read_bytes() == alert.serialize()
        original_link(source, destination)

    monkeypatch.setattr(os, "fsync", track_fsync)
    monkeypatch.setattr(os, "link", verify_atomic_install)

    JsonFilePredictionAlertSink(output_directory).emit(alert)

    assert (output_directory / f"{alert.alert_id}.json").read_bytes() == (
        alert.serialize()
    )
    assert tuple(output_directory.glob(".*.tmp")) == ()


def test_historical_metadata_change_produces_a_distinct_idempotent_artifact(
    tmp_path: Path,
) -> None:
    rule = _rule()
    context = _case_context(0)
    source = _source(context)
    output_directory = tmp_path / "alerts"
    sink = JsonFilePredictionAlertSink(output_directory)
    store = InMemoryAlertDeduplicationStore()
    references = tuple(
        HistoricalPredictionStudyReference.capture(
            study_id="validated-study-1",
            rule=rule,
            validated_symbols=(source.symbol,),
            historical_dataset_fingerprint=HISTORICAL_DATASET_FINGERPRINT,
            adjustment_basis=source.adjustment_basis,
            summary={"validation_note": validation_note},
            sample_count=sample_count,
        )
        for validation_note, sample_count in (("original", 42), ("corrected", 43))
    )

    alerts = tuple(
        PredictionScanner(
            source,
            (PredictionScannerRuleBinding(rule, reference),),
            (sink,),
            store,
            AlertDeduplicationPolicy.DECISION_BAR,
        )
        .scan(as_of=context.as_of)
        .alerts[0]
        for reference in references
    )

    assert alerts[0].alert_id != alerts[1].alert_id
    assert alerts[0].serialize() != alerts[1].serialize()
    assert alerts[0].identity_primitive()["historical_study"] == (
        references[0].to_primitive()
    )
    assert {path.name for path in output_directory.glob("*.json")} == {
        f"{alert.alert_id}.json" for alert in alerts
    }


def test_no_prediction_is_audited_without_emitting_an_alert() -> None:
    rule = _rule()
    context = _case_context(2)
    source = _source(context)
    sink = _CapturingSink()

    result = _scanner(rule, source, sink).scan(as_of=context.as_of)

    assert source.refresh_values == [True]
    assert result.alerts == ()
    assert sink.alerts == []
    assert not result.rule_results[0].accepted
    evaluation = result.rule_results[0].evaluation
    assert evaluation is not None
    assert evaluation.to_primitive()["outcome"] == "no_prediction"


def test_one_scan_can_evaluate_multiple_independently_validated_rules() -> None:
    rules = (_rule(), _rule(backend_id="native_v1"))
    context = _case_context(0)
    source = _source(context)
    sink = _CapturingSink()
    scanner = PredictionScanner(
        source,
        tuple(
            PredictionScannerRuleBinding(
                rule,
                HistoricalPredictionStudyReference.capture(
                    study_id=f"validated-study-{index}",
                    rule=rule,
                    validated_symbols=(source.symbol,),
                    historical_dataset_fingerprint=HISTORICAL_DATASET_FINGERPRINT,
                    adjustment_basis=source.adjustment_basis,
                ),
            )
            for index, rule in enumerate(rules)
        ),
        (sink,),
    )

    result = scanner.scan(as_of=context.as_of, dry_run=True)

    assert len(result.alerts) == len(sink.alerts) == 2
    assert source.refresh_values == [False, False]
    assert {alert.direction for alert in result.alerts} == {PredictionDirection.UP}
    assert len({alert.alert_id for alert in result.alerts}) == 2


def test_repeated_unchanged_context_is_deduplicated_across_file_store_instances(
    tmp_path: Path,
) -> None:
    rule = _rule()
    context = _case_context(0)
    state = tmp_path / "dedup"
    first_sink = _CapturingSink()
    second_sink = _CapturingSink()

    first = _scanner(
        rule,
        _source(context),
        first_sink,
        store=JsonFileAlertDeduplicationStore(state),
    ).scan(as_of=context.as_of)
    second = _scanner(
        rule,
        _source(context),
        second_sink,
        store=JsonFileAlertDeduplicationStore(state),
    ).scan(as_of=context.as_of)

    assert len(first.alerts) == 1
    assert second.alerts == ()
    assert second.rule_results[0].duplicate_alert_id == first.alerts[0].alert_id
    assert len(first_sink.alerts) == 1
    assert second_sink.alerts == []


@pytest.mark.parametrize("lifecycle_state", [None, "pending"])
def test_file_store_recovers_abandoned_pending_claims(
    tmp_path: Path, lifecycle_state: str | None
) -> None:
    state_directory = tmp_path / "dedup"
    state_directory.mkdir()
    payload = {
        "alert_id": "interrupted-alert",
        "deduplication_key": "recoverable-key",
    }
    if lifecycle_state is not None:
        payload["state"] = lifecycle_state
    (state_directory / "recoverable-key.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    store = JsonFileAlertDeduplicationStore(state_directory)
    recovered = store.claim("recoverable-key", "interrupted-alert")

    assert not isinstance(recovered, PublishedAlertDeduplication)
    recovered.publish()
    published = store.claim("recoverable-key", "later-alert")
    assert isinstance(published, PublishedAlertDeduplication)
    assert published.alert_id == "interrupted-alert"


def test_file_store_rejects_pending_claim_with_a_different_alert_id(
    tmp_path: Path,
) -> None:
    state_directory = tmp_path / "dedup"
    state_directory.mkdir()
    state_path = state_directory / "decision-key.json"
    pending = {
        "alert_id": "interrupted-alert",
        "deduplication_key": "decision-key",
        "state": "pending",
    }
    state_path.write_text(json.dumps(pending), encoding="utf-8")
    store = JsonFileAlertDeduplicationStore(state_directory)

    with pytest.raises(AlertPersistenceError, match="different alert identity"):
        store.claim("decision-key", "newer-context-alert")

    assert json.loads(state_path.read_text(encoding="utf-8")) == pending
    recovered = store.claim("decision-key", "interrupted-alert")
    assert not isinstance(recovered, PublishedAlertDeduplication)
    recovered.release()


def test_file_store_blocks_an_active_claim_and_reuses_a_released_key(
    tmp_path: Path,
) -> None:
    first_store = JsonFileAlertDeduplicationStore(tmp_path / "dedup")
    second_store = JsonFileAlertDeduplicationStore(tmp_path / "dedup")
    active = first_store.claim("active-key", "first-alert")

    assert not isinstance(active, PublishedAlertDeduplication)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        waiting_claim = executor.submit(
            second_store.claim, "active-key", "replacement-alert"
        )
        with pytest.raises(concurrent.futures.TimeoutError):
            waiting_claim.result(timeout=0.05)

        active.release()
        replacement = waiting_claim.result(timeout=1)

    assert not isinstance(replacement, PublishedAlertDeduplication)
    replacement.publish()


def test_file_store_transition_never_truncates_the_live_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_directory = tmp_path / "dedup"
    store = JsonFileAlertDeduplicationStore(state_directory)
    claim = store.claim("atomic-key", "atomic-alert")
    assert not isinstance(claim, PublishedAlertDeduplication)
    state_path = state_directory / "atomic-key.json"
    pending_content = state_path.read_bytes()

    def interrupt_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("simulated interruption before atomic replace")

    monkeypatch.setattr(os, "replace", interrupt_replace)

    with pytest.raises(AlertPersistenceError, match="deduplication state"):
        claim.publish()

    assert state_path.read_bytes() == pending_content
    assert json.loads(pending_content)["state"] == "pending"
    assert tuple(state_directory.glob(".*.tmp")) == ()
    claim.release()


def test_historical_backend_or_configuration_mismatch_fails_before_data_access() -> (
    None
):
    historical_rule = _rule()
    current_rule = _rule(backend_id="native_v1")
    reference = HistoricalPredictionStudyReference.capture(
        study_id="validated-study-1",
        rule=historical_rule,
        validated_symbols=("SPY",),
        historical_dataset_fingerprint=HISTORICAL_DATASET_FINGERPRINT,
        adjustment_basis=timeframe_fixtures._adjustment_basis(),  # pyright: ignore[reportPrivateUsage]
    )

    with pytest.raises(HistoricalStudyMismatchError, match="does not match"):
        PredictionScannerRuleBinding(current_rule, reference)


def test_historical_study_reference_record_round_trips() -> None:
    rule = _rule()
    reference = HistoricalPredictionStudyReference.capture(
        study_id="validated-study-1",
        rule=rule,
        validated_symbols=("SPY",),
        historical_dataset_fingerprint=HISTORICAL_DATASET_FINGERPRINT,
        adjustment_basis=timeframe_fixtures._adjustment_basis(),  # pyright: ignore[reportPrivateUsage]
        summary={"validated": True},
        sample_count=42,
    )

    assert (
        HistoricalPredictionStudyReference.from_primitive(reference.to_primitive())
        == reference
    )
    assert (
        reference.to_primitive()["historical_dataset_fingerprint"]
        == HISTORICAL_DATASET_FINGERPRINT
    )
    assert reference.to_primitive()["validated_symbols"] == ["SPY"]


def test_historical_study_reference_requires_a_dataset_fingerprint() -> None:
    rule = _rule()
    reference = HistoricalPredictionStudyReference.capture(
        study_id="validated-study-1",
        rule=rule,
        validated_symbols=("SPY",),
        historical_dataset_fingerprint=HISTORICAL_DATASET_FINGERPRINT,
        adjustment_basis=timeframe_fixtures._adjustment_basis(),  # pyright: ignore[reportPrivateUsage]
    )
    record = dict(reference.to_primitive())
    del record["historical_dataset_fingerprint"]

    with pytest.raises(PredictionScannerError, match="record is invalid"):
        HistoricalPredictionStudyReference.from_primitive(record)


def test_historical_study_reference_requires_a_validated_universe() -> None:
    rule = _rule()
    reference = HistoricalPredictionStudyReference.capture(
        study_id="validated-study-1",
        rule=rule,
        validated_symbols=("SPY",),
        historical_dataset_fingerprint=HISTORICAL_DATASET_FINGERPRINT,
        adjustment_basis=timeframe_fixtures._adjustment_basis(),  # pyright: ignore[reportPrivateUsage]
    )
    record = dict(reference.to_primitive())
    del record["validated_symbols"]

    with pytest.raises(PredictionScannerError, match="record is invalid"):
        HistoricalPredictionStudyReference.from_primitive(record)


def test_historical_adjustment_mismatch_fails_before_rule_evaluation() -> None:
    rule = _rule()
    context = _case_context(0)
    historical_adjustment = timeframe_fixtures._adjustment_basis()  # pyright: ignore[reportPrivateUsage]
    current_adjustment = AdjustmentBasis(
        AdjustmentMode.SPLIT_ADJUSTED,
        "split_adjusted",
        "split_adjusted",
        "split_adjusted_prices_without_dividend_adjustment",
        True,
    )
    source = _source(context)
    source.adjustment_basis = current_adjustment
    reference = HistoricalPredictionStudyReference.capture(
        study_id="validated-study-1",
        rule=rule,
        validated_symbols=("SPY",),
        historical_dataset_fingerprint=HISTORICAL_DATASET_FINGERPRINT,
        adjustment_basis=historical_adjustment,
    )
    sink = _CapturingSink()
    scanner = PredictionScanner(
        source,
        (PredictionScannerRuleBinding(rule, reference),),
        (sink,),
    )

    with pytest.raises(HistoricalStudyMismatchError, match="adjustment basis"):
        scanner.scan(as_of=context.as_of)

    assert source.refresh_values == [True]
    assert sink.alerts == []


def test_current_symbol_must_belong_to_the_historical_study_universe() -> None:
    rule = _rule()
    context = _case_context(0)
    source = _source(context)
    source.symbol = "QQQ"
    reference = HistoricalPredictionStudyReference.capture(
        study_id="validated-study-1",
        rule=rule,
        validated_symbols=("SPY",),
        historical_dataset_fingerprint=HISTORICAL_DATASET_FINGERPRINT,
        adjustment_basis=source.adjustment_basis,
    )
    sink = _CapturingSink()
    scanner = PredictionScanner(
        source,
        (PredictionScannerRuleBinding(rule, reference),),
        (sink,),
    )

    with pytest.raises(HistoricalStudyMismatchError, match="validated universe"):
        scanner.scan(as_of=context.as_of)

    assert source.refresh_values == [True]
    assert sink.alerts == []


def test_rule_is_revalidated_after_context_preparation() -> None:
    rule = _MalformedOutputRule(_rule())
    context = _case_context(0)
    delegate = _source(context)
    source = _RuleMutatingFixtureSource(delegate, rule)
    reference = HistoricalPredictionStudyReference.capture(
        study_id="validated-study-1",
        rule=rule,
        validated_symbols=(delegate.symbol,),
        historical_dataset_fingerprint=HISTORICAL_DATASET_FINGERPRINT,
        adjustment_basis=delegate.adjustment_basis,
    )
    sink = _CapturingSink()
    scanner = PredictionScanner(
        source,
        (PredictionScannerRuleBinding(rule, reference),),
        (sink,),
    )

    with pytest.raises(HistoricalStudyMismatchError, match="does not match"):
        scanner.scan(as_of=context.as_of)

    assert delegate.refresh_values == [True]
    assert sink.alerts == []


def test_snapshot_dataset_id_must_match_context_lineage() -> None:
    rule = _rule()
    context = _case_context(0)
    source = _source(context)
    source.dataset_id = "unrelated-dataset-id"
    sink = _CapturingSink()

    with pytest.raises(PredictionScannerError, match="context lineage"):
        _scanner(rule, source, sink).scan(as_of=context.as_of)

    assert source.refresh_values == [True]
    assert sink.alerts == []


def test_stale_data_fails_before_rule_evaluation() -> None:
    rule = _rule(maximum_age=timedelta(days=1))
    base_context = _case_context(0)
    stale_context = _context_with_requirements(
        base_context,
        rule.context_requirements,
        as_of=base_context.as_of + timedelta(days=10),
    )

    with pytest.raises(PredictionContextError, match="stale"):
        _scanner(rule, _source(stale_context), _CapturingSink()).scan(
            as_of=stale_context.as_of
        )


def test_stale_data_is_audited_when_context_policy_skips() -> None:
    rule = _rule(
        maximum_age=timedelta(days=1),
        failure_policy=PredictionContextFailurePolicy.SKIP,
    )
    base_context = _case_context(0)
    stale_context = _context_with_requirements(
        base_context,
        rule.context_requirements,
        as_of=base_context.as_of + timedelta(days=10),
    )
    source = _source(stale_context)
    sink = _CapturingSink()

    result = _scanner(rule, source, sink).scan(as_of=stale_context.as_of)

    rule_result = result.rule_results[0]
    assert rule_result.skipped
    assert not rule_result.accepted
    assert rule_result.context_id is None
    assert rule_result.evaluation is None
    assert rule_result.context_failure is not None
    failure = rule_result.context_failure.to_primitive()
    assert failure["status"] == "skipped"
    assert "stale" in cast(str, failure["reason"])
    assert failure["source_context"] == stale_context.to_primitive()
    assert source.refresh_values == [True]
    assert sink.alerts == []


def test_skipped_context_preparation_continues_to_later_rules() -> None:
    skipped_rule = _rule(failure_policy=PredictionContextFailurePolicy.SKIP)
    available_rule = _rule(backend_id="native_v1")
    context = _case_context(0)
    delegate = _source(context)
    source = _SelectiveFixtureSource(delegate, skipped_rule.context_requirements)
    sink = _CapturingSink()
    bindings = tuple(
        PredictionScannerRuleBinding(
            rule,
            HistoricalPredictionStudyReference.capture(
                study_id=f"validated-study-{index}",
                rule=rule,
                validated_symbols=(delegate.symbol,),
                historical_dataset_fingerprint=HISTORICAL_DATASET_FINGERPRINT,
                adjustment_basis=delegate.adjustment_basis,
            ),
        )
        for index, rule in enumerate((skipped_rule, available_rule))
    )

    result = PredictionScanner(source, bindings, (sink,)).scan(as_of=context.as_of)

    assert result.rule_results[0].skipped
    assert result.rule_results[0].context_failure is not None
    assert "unavailable" in cast(
        str, result.rule_results[0].context_failure.to_primitive()["reason"]
    )
    assert not result.rule_results[1].skipped
    assert result.rule_results[1].alert is not None
    assert len(result.alerts) == len(sink.alerts) == 1
    assert len(source.requirement_ids) == 2
    assert delegate.refresh_values == [True]


def test_future_context_is_rejected_and_no_alert_is_emitted() -> None:
    rule = _rule()
    future_context = timeframe_fixtures._all_completed_context()  # pyright: ignore[reportPrivateUsage]
    sink = _CapturingSink()

    with pytest.raises(PredictionContextError, match="decision boundary"):
        _scanner(rule, _source(future_context), sink).scan(as_of=future_context.as_of)

    assert sink.alerts == []


def test_historical_and_current_fixed_context_have_identical_values_and_decision() -> (
    None
):
    rule = _rule()
    context = _case_context(0)
    dataset = rule_fixtures._dataset()  # pyright: ignore[reportPrivateUsage]
    historical_context = build_prediction_rule_context(
        rule.context_requirements,
        context,
        prediction_dataset_id=dataset.metadata.dataset_id,
        symbol="SPY",
        prediction_adjustment_basis=timeframe_fixtures._adjustment_basis(),  # pyright: ignore[reportPrivateUsage]
    )
    historical_evaluation = rule.evaluate(historical_context)

    current = _scanner(rule, _source(context), _CapturingSink()).scan(
        as_of=context.as_of, dry_run=True
    )
    alert = current.alerts[0]

    assert alert.direction.value == historical_evaluation.outcome.value
    current_indicators = {
        (
            item.to_primitive()["timeframe_configuration_id"],
            item.to_primitive()["alias"],
        ): item.to_primitive()["normalized_values"]
        for item in alert.indicators
    }
    historical_indicators = {
        (
            item.requirement.timeframe.configuration_id,
            named.alias,
        ): {
            field.name: (
                None if field.values[-1] is None else str(field.values[-1].normalize())
            )
            for field in named.output.fields
        }
        for item in historical_context.timeframes
        for named in item.indicators
    }
    assert current_indicators == historical_indicators


def test_developing_context_can_alert_again_only_under_explicit_context_policy(
    tmp_path: Path,
) -> None:
    example = spy_example.build_example(
        spy_example.DEFAULT_FIXTURE_PATH,
        tmp_path / "cache",
    )
    source_timeframe, four_hour, daily, weekly = spy_example._timeframes()  # pyright: ignore[reportPrivateUsage]
    feed_scope = example.datasets.source.request.feed_scope
    rule = _rule(
        feed_scope=feed_scope,
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
    )
    requirements = tuple(
        ContextTimeframeRequirement(timeframe)
        for timeframe in (four_hour, daily, weekly)
    )
    series = spy_example._series(example.datasets)  # pyright: ignore[reportPrivateUsage]
    first_as_of = datetime(2024, 7, 10, 16, 0, tzinfo=UTC)
    second_as_of = first_as_of + timedelta(minutes=1)
    contexts = tuple(
        build_multi_timeframe_context(
            as_of=as_of,
            primary_timeframe=source_timeframe,
            required_timeframes=requirements,
            series=series,
            completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
        )
        for as_of in (first_as_of, second_as_of)
    )
    assert any(
        bar.completion is BarCompletion.DEVELOPING
        for bar in contexts[0].timeframes[-1].bars
    )

    exact_store = InMemoryAlertDeduplicationStore()
    exact_alert_ids: list[str] = []
    for context in contexts:
        result = _scanner(
            rule,
            _FixtureSource(
                context,
                example.datasets.source.metadata.dataset_id,
                example.datasets.source.request.adjustment_basis,
            ),
            _CapturingSink(),
            store=exact_store,
        ).scan(as_of=context.as_of, dry_run=True)
        exact_alert_ids.append(result.alerts[0].alert_id)
    assert len(set(exact_alert_ids)) == 2

    bar_store = InMemoryAlertDeduplicationStore()
    bar_results = tuple(
        _scanner(
            rule,
            _FixtureSource(
                context,
                example.datasets.source.metadata.dataset_id,
                example.datasets.source.request.adjustment_basis,
            ),
            _CapturingSink(),
            store=bar_store,
            deduplication_policy=AlertDeduplicationPolicy.DECISION_BAR,
        ).scan(as_of=context.as_of, dry_run=True)
        for context in contexts
    )
    assert len(bar_results[0].alerts) == 1
    assert bar_results[1].alerts == ()
    assert exact_alert_ids[0] != exact_alert_ids[1]
    assert (
        bar_results[1].rule_results[0].duplicate_alert_id
        == bar_results[0].alerts[0].alert_id
    )

    state_directory = tmp_path / "pending-decision-bar"
    first_sink = _CapturingSink()
    first_result = _scanner(
        rule,
        _FixtureSource(
            contexts[0],
            example.datasets.source.metadata.dataset_id,
            example.datasets.source.request.adjustment_basis,
        ),
        first_sink,
        store=JsonFileAlertDeduplicationStore(state_directory),
        deduplication_policy=AlertDeduplicationPolicy.DECISION_BAR,
    ).scan(as_of=contexts[0].as_of, dry_run=True)
    state_path = next(state_directory.glob("*.json"))
    pending_state = json.loads(state_path.read_text(encoding="utf-8"))
    pending_state["state"] = "pending"
    state_path.write_text(json.dumps(pending_state), encoding="utf-8")

    second_sink = _CapturingSink()
    with pytest.raises(AlertPersistenceError, match="different alert identity"):
        _scanner(
            rule,
            _FixtureSource(
                contexts[1],
                example.datasets.source.metadata.dataset_id,
                example.datasets.source.request.adjustment_basis,
            ),
            second_sink,
            store=JsonFileAlertDeduplicationStore(state_directory),
            deduplication_policy=AlertDeduplicationPolicy.DECISION_BAR,
        ).scan(as_of=contexts[1].as_of, dry_run=True)

    preserved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert first_sink.alerts == list(first_result.alerts)
    assert second_sink.alerts == []
    assert preserved_state["state"] == "pending"
    assert preserved_state["alert_id"] == first_result.alerts[0].alert_id


def test_scanner_and_alert_module_has_no_direct_talib_import() -> None:
    module_path = Path(__file__).parents[3] / "src/quantforge/prediction/scanner.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }

    assert all("talib" not in name.casefold() for name in imported_modules)
