import ast
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from scripts import export_spy_multi_timeframe_context as spy_example

from quantforge.configuration import PrimitiveMapping
from quantforge.data import (
    AdjustmentBasis,
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
    ConsolePredictionAlertSink,
    HistoricalPredictionStudyReference,
    HistoricalStudyMismatchError,
    InMemoryAlertDeduplicationStore,
    JsonFileAlertDeduplicationStore,
    JsonFilePredictionAlertSink,
    PredictionAlert,
    PredictionContextError,
    PredictionContextRequirements,
    PredictionDirection,
    PredictionIndicatorRequirement,
    PredictionScanner,
    PredictionScannerRuleBinding,
    PredictionScannerSnapshot,
    PredictionTimeframeRequirement,
    TechnicalCondition,
    TechnicalConditionOperand,
    TechnicalConditionOperator,
    TechnicalConfluenceParameters,
    TechnicalConfluencePredictionRule,
    build_prediction_rule_context,
)
from quantforge.timeframes import BarCompletion, Timeframe
from tests.unit.indicators import test_timeframe_evaluation as timeframe_fixtures
from tests.unit.prediction import test_technical_confluence as rule_fixtures


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
            "refreshed_provider_data" if refresh else "replayed_local_cache",
        )


def _rule(
    *,
    feed_scope: FeedScope = FeedScope.consolidated(),
    completion_policy: ContextCompletionPolicy = (
        ContextCompletionPolicy.COMPLETED_BARS_ONLY
    ),
    backend_id: str = TALIB_INDICATOR_BACKEND,
    maximum_age: timedelta | None = None,
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
    rule: TechnicalConfluencePredictionRule,
    source: _FixtureSource,
    sink: _CapturingSink,
    *,
    historical_rule: TechnicalConfluencePredictionRule | None = None,
    deduplication_policy: AlertDeduplicationPolicy = (
        AlertDeduplicationPolicy.EXACT_CONTEXT
    ),
    store: AlertDeduplicationStore | None = None,
) -> PredictionScanner:
    historical = historical_rule or rule
    reference = HistoricalPredictionStudyReference.capture(
        study_id="validated-study-1",
        rule=historical,
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
    return _FixtureSource(
        context,
        rule_fixtures._dataset().metadata.dataset_id,  # pyright: ignore[reportPrivateUsage]
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
    assert cast(PrimitiveMapping, payload["historical_study"])["sample_count"] == 42
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
    assert result.rule_results[0].evaluation.to_primitive()["outcome"] == (
        "no_prediction"
    )


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
                    study_id=f"validated-study-{index}", rule=rule
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


def test_historical_backend_or_configuration_mismatch_fails_before_data_access() -> (
    None
):
    historical_rule = _rule()
    current_rule = _rule(backend_id="native_v1")
    reference = HistoricalPredictionStudyReference.capture(
        study_id="validated-study-1", rule=historical_rule
    )

    with pytest.raises(HistoricalStudyMismatchError, match="does not match"):
        PredictionScannerRuleBinding(current_rule, reference)


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
    assert bar_results[1].rule_results[0].duplicate_alert_id is not None


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
