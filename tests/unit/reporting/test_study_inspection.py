import json
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest
import scripts.export_spy_multi_timeframe_context as spy_context_script
import scripts.scan_spy_predictions as spy_scanner_script

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    configuration_identity,
)
from quantforge.data import ContextCompletionPolicy
from quantforge.indicators import (
    NATIVE_INDICATOR_BACKEND,
    TIMEFRAME_INDICATOR_CONTRACT_VERSION,
    DevelopingBarSupport,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
)
from quantforge.prediction import (
    HistoricalPredictionStudyReference,
    PredictionContextFailurePolicy,
    PredictionRuleContext,
    TechnicalConfluenceEvaluation,
    TechnicalConfluenceParameters,
    TechnicalConfluencePredictionRule,
    build_prediction_rule_context,
    create_reference_technical_confluence_rule,
)
from quantforge.prediction.multi_timeframe_features import (
    capture_multi_timeframe_features,
)
from quantforge.reporting import (
    FutureOutcomeRegion,
    StudyInspectionReportConfig,
    StudyInspectionReportError,
    StudyInspectionSelection,
    build_study_inspection_report,
    export_study_inspection_report,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPOSITORY_ROOT / "examples" / "spy_multi_timeframe" / "fixture.json"


@dataclass(frozen=True, slots=True)
class _InspectionFixture:
    datasets: spy_context_script.ExampleDatasets
    rule: TechnicalConfluencePredictionRule
    context: PredictionRuleContext
    evaluation: TechnicalConfluenceEvaluation
    historical_study: HistoricalPredictionStudyReference


@pytest.fixture(scope="module")
def inspection_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> _InspectionFixture:
    fixture = spy_context_script.load_fixture(FIXTURE_PATH)
    datasets = spy_context_script.build_datasets(
        fixture, tmp_path_factory.mktemp("qf34-cache")
    )
    as_of = next(
        item.as_of for item in fixture.decision_scenarios if item.name == "midweek"
    )
    rule = spy_scanner_script.create_fixture_parity_rule(
        datasets, ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
    )
    snapshot = spy_scanner_script.CachedSpyScannerSource(datasets).prepare_context(
        rule.context_requirements,
        as_of=as_of,
        refresh=False,
    )
    context = build_prediction_rule_context(
        rule.context_requirements,
        snapshot.context,
        prediction_dataset_id=snapshot.prediction_dataset_id,
        symbol=snapshot.symbol,
        prediction_adjustment_basis=snapshot.adjustment_basis,
    )
    evaluation = rule.evaluate(context)
    historical_study = spy_scanner_script.load_historical_study_reference(
        ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
    )
    return _InspectionFixture(datasets, rule, context, evaluation, historical_study)


def _selection(fixture: _InspectionFixture) -> StudyInspectionSelection:
    decision = fixture.context.latest_bar_for(
        fixture.context.requirements.primary.timeframe
    ).end_timestamp
    return StudyInspectionSelection(
        "fixed_spy_prediction",
        fixture.context,
        fixture.rule,
        fixture.evaluation,
        fixture.historical_study,
        fixture.datasets.family,
        FutureOutcomeRegion(
            decision,
            decision + timedelta(days=1),
            "One-session inspection window",
            None,
        ),
    )


def _manifest_selection(report_bytes: bytes) -> PrimitiveMapping:
    manifest = cast(PrimitiveMapping, json.loads(report_bytes))
    return cast(PrimitiveMapping, cast(list[Primitive], manifest["selections"])[0])


def test_report_selects_exact_causal_series_and_timestamps(
    inspection_fixture: _InspectionFixture,
) -> None:
    report = build_study_inspection_report(
        (_selection(inspection_fixture),),
        config=StudyInspectionReportConfig(max_bars_per_panel=4),
    )
    selection = _manifest_selection(report.manifest_bytes())
    panels = cast(list[PrimitiveMapping], selection["panels"])

    assert [panel["name"] for panel in panels] == [
        "weekly",
        "daily",
        "four_hour",
        "primary_5m",
    ]
    for panel in panels:
        timeframe = cast(PrimitiveMapping, panel["timeframe"])
        timeframe_id = cast(str, timeframe["configuration_id"])
        source = next(
            item
            for item in inspection_fixture.context.timeframes
            if item.requirement.timeframe.configuration_id == timeframe_id
        )
        expected_bars = [
            {"bar_id": bar.bar_id, **bar.to_primitive()} for bar in source.bars[-4:]
        ]
        assert panel["bars"] == expected_bars
        reference = cast(PrimitiveMapping, panel["dataset_reference"])
        assert reference["family_id"] == inspection_fixture.datasets.family.family_id
        for indicator in cast(list[PrimitiveMapping], panel["indicators"]):
            named = next(
                item for item in source.indicators if item.alias == indicator["alias"]
            )
            assert indicator["rows"] == named.output.to_rows()[-4:]
            backend = named.output.backend_identity
            assert backend is not None
            assert indicator["backend"] == backend.to_primitive()


def test_report_marks_developing_bars_and_separates_future_outcomes(
    inspection_fixture: _InspectionFixture,
) -> None:
    report = build_study_inspection_report((_selection(inspection_fixture),))
    selection = _manifest_selection(report.manifest_bytes())
    panels = cast(list[PrimitiveMapping], selection["panels"])
    html_text = report.html_bytes().decode("utf-8")

    contextual_panels = [panel for panel in panels if panel["name"] != "primary_5m"]
    assert all(
        cast(list[PrimitiveMapping], panel["bars"])[-1]["completion"] == "developing"
        for panel in contextual_panels
    )
    assert 'class="developing"' in html_text
    assert "POST-DECISION OUTCOME — NOT AVAILABLE AT DECISION TIME" in html_text
    future = cast(PrimitiveMapping, selection["future_outcome"])
    assert future["availability"] == "post_decision_only"


def test_report_latest_values_match_qf29_feature_capture_exactly(
    inspection_fixture: _InspectionFixture,
) -> None:
    report = build_study_inspection_report((_selection(inspection_fixture),))
    selection = _manifest_selection(report.manifest_bytes())
    panels = {
        cast(str, panel["name"]): panel
        for panel in cast(list[PrimitiveMapping], selection["panels"])
    }
    requests = inspection_fixture.rule.multi_timeframe_feature_requests
    captured = {
        item.name: item.value
        for item in capture_multi_timeframe_features(
            requests, inspection_fixture.context
        )
    }

    for request in requests:
        panel = panels[request.timeframe_name]
        indicator = next(
            item
            for item in cast(list[PrimitiveMapping], panel["indicators"])
            if item["alias"] == request.indicator_alias
        )
        latest = cast(list[PrimitiveMapping], indicator["rows"])[-1]
        assert latest[request.normalized_output_name] == str(captured[request.name])
        metadata = cast(PrimitiveMapping, captured[request.metadata_name])
        assert (
            indicator["indicator_configuration_id"]
            == metadata["indicator_configuration_id"]
        )
        assert (
            indicator["timeframe_indicator_configuration_id"]
            == metadata["timeframe_indicator_configuration_id"]
        )
        assert indicator["backend"] == metadata["indicator_backend"]
        reference = cast(PrimitiveMapping, panel["dataset_reference"])
        assert reference["family_id"] == metadata["dataset_family_id"]
        assert reference["dataset_id"] == metadata["source_dataset_id"]


def test_report_supports_multiple_condition_aliases_for_one_timeframe(
    inspection_fixture: _InspectionFixture,
) -> None:
    original_rule = inspection_fixture.rule
    daily_id = next(
        condition.timeframe.configuration_id
        for condition in original_rule.parameters.conditions
        if condition.timeframe_name == "daily"
    )
    renamed_up_conditions = tuple(
        replace(condition, timeframe_name="daily_trend")
        if condition.timeframe.configuration_id == daily_id
        else condition
        for condition in original_rule.parameters.up_conditions
    )
    rule = TechnicalConfluencePredictionRule(
        TechnicalConfluenceParameters(
            renamed_up_conditions,
            original_rule.parameters.down_conditions,
            original_rule.parameters.reference_name,
        ),
        original_rule.context_requirements,
    )
    study = HistoricalPredictionStudyReference.capture(
        study_id="qf34_multiple_timeframe_aliases_study_v1",
        rule=rule,
        validated_symbols=(inspection_fixture.context.symbol,),
        historical_dataset_fingerprint="c" * 64,
        adjustment_basis=inspection_fixture.context.adjustment_basis,
    )
    report = build_study_inspection_report(
        (
            StudyInspectionSelection(
                "multiple_aliases",
                inspection_fixture.context,
                rule,
                rule.evaluate(inspection_fixture.context),
                study,
                inspection_fixture.datasets.family,
            ),
        )
    )
    selection = _manifest_selection(report.manifest_bytes())

    panels = cast(list[PrimitiveMapping], selection["panels"])
    assert [panel["name"] for panel in panels] == [
        "weekly",
        "daily / daily_trend",
        "four_hour",
        "primary_5m",
    ]


def test_full_reference_suite_renders_normalized_standard_and_volume_outputs(
    inspection_fixture: _InspectionFixture,
) -> None:
    datasets = inspection_fixture.datasets
    primary, four_hour, daily, weekly = spy_scanner_script._timeframes(  # pyright: ignore[reportPrivateUsage]
        datasets
    )
    rule = create_reference_technical_confluence_rule(
        primary_timeframe=primary,
        four_hour_timeframe=four_hour,
        daily_timeframe=daily,
        weekly_timeframe=weekly,
        feed_scope=datasets.source.request.feed_scope,
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
    )
    as_of = inspection_fixture.context.as_of
    snapshot = spy_scanner_script.CachedSpyScannerSource(datasets).prepare_context(
        rule.context_requirements,
        as_of=as_of,
        refresh=False,
    )
    context = build_prediction_rule_context(
        rule.context_requirements,
        snapshot.context,
        prediction_dataset_id=snapshot.prediction_dataset_id,
        symbol=snapshot.symbol,
        prediction_adjustment_basis=snapshot.adjustment_basis,
    )
    study = HistoricalPredictionStudyReference.capture(
        study_id="qf34_full_indicator_fixture_study_v1",
        rule=rule,
        validated_symbols=("SPY",),
        historical_dataset_fingerprint="a" * 64,
        adjustment_basis=snapshot.adjustment_basis,
    )
    report = build_study_inspection_report(
        (
            StudyInspectionSelection(
                "full_normalized_suite",
                context,
                rule,
                rule.evaluate(context),
                study,
                datasets.family,
            ),
        )
    )
    selection = _manifest_selection(report.manifest_bytes())
    indicator_names = {
        cast(str, indicator["indicator_name"])
        for panel in cast(list[PrimitiveMapping], selection["panels"])
        for indicator in cast(list[PrimitiveMapping], panel["indicators"])
    }

    assert {
        "simple_moving_average",
        "exponential_moving_average",
        "bollinger_bands",
        "moving_average_convergence_divergence",
        "stochastic_oscillator",
        "volume_moving_average",
        "relative_volume",
    }.issubset(indicator_names)
    assert "indicator_configuration_id" in report.html_bytes().decode("utf-8")


def test_indicator_backend_changes_produce_distinct_provenance_and_require_a_study(
    inspection_fixture: _InspectionFixture,
) -> None:
    datasets = inspection_fixture.datasets
    native_rule = spy_scanner_script.create_fixture_parity_rule(
        datasets, ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
    )
    talib_report = build_study_inspection_report((_selection(inspection_fixture),))

    # Rebuild the small parity rule with the same logic and a different backend.
    native_requirements = replace(
        native_rule.context_requirements,
        contextual=tuple(
            replace(
                timeframe,
                indicators=(
                    replace(
                        timeframe.indicators[0],
                        indicator=SimpleMovingAverage(
                            SimpleMovingAverageParameters(2),
                            backend_id=NATIVE_INDICATOR_BACKEND,
                        ),
                    ),
                ),
            )
            for timeframe in native_rule.context_requirements.contextual
        ),
    )
    native_rule = TechnicalConfluencePredictionRule(
        native_rule.parameters, native_requirements
    )
    snapshot = spy_scanner_script.CachedSpyScannerSource(datasets).prepare_context(
        native_rule.context_requirements,
        as_of=inspection_fixture.context.as_of,
        refresh=False,
    )
    native_context = build_prediction_rule_context(
        native_rule.context_requirements,
        snapshot.context,
        prediction_dataset_id=snapshot.prediction_dataset_id,
        symbol=snapshot.symbol,
        prediction_adjustment_basis=snapshot.adjustment_basis,
    )
    with pytest.raises(StudyInspectionReportError, match="historical-study"):
        StudyInspectionSelection(
            "native",
            native_context,
            native_rule,
            native_rule.evaluate(native_context),
            inspection_fixture.historical_study,
            datasets.family,
        )
    native_study = HistoricalPredictionStudyReference.capture(
        study_id="qf34_native_backend_fixture_study_v1",
        rule=native_rule,
        validated_symbols=("SPY",),
        historical_dataset_fingerprint="b" * 64,
        adjustment_basis=snapshot.adjustment_basis,
    )
    native_report = build_study_inspection_report(
        (
            StudyInspectionSelection(
                "native",
                native_context,
                native_rule,
                native_rule.evaluate(native_context),
                native_study,
                datasets.family,
            ),
        )
    )

    assert native_report.report_id != talib_report.report_id


def test_report_rejects_indicator_alignment_or_family_tampering(
    inspection_fixture: _InspectionFixture,
) -> None:
    timeframe = next(
        item for item in inspection_fixture.context.timeframes if item.indicators
    )
    named = timeframe.indicators[0]
    corrupt_output = replace(named.output, configuration_id="foreign-config")
    corrupt_timeframe = replace(
        timeframe, indicators=(replace(named, output=corrupt_output),)
    )
    corrupt_context = replace(
        inspection_fixture.context,
        timeframes=tuple(
            corrupt_timeframe if item is timeframe else item
            for item in inspection_fixture.context.timeframes
        ),
    )
    with pytest.raises(StudyInspectionReportError, match="provenance"):
        StudyInspectionSelection(
            "corrupt",
            corrupt_context,
            inspection_fixture.rule,
            inspection_fixture.evaluation,
            inspection_fixture.historical_study,
            inspection_fixture.datasets.family,
        )

    foreign_family = replace(inspection_fixture.datasets.family, canonical_symbol="QQQ")
    with pytest.raises(StudyInspectionReportError, match="symbol"):
        StudyInspectionSelection(
            "foreign",
            inspection_fixture.context,
            inspection_fixture.rule,
            inspection_fixture.evaluation,
            inspection_fixture.historical_study,
            foreign_family,
        )


def test_report_rejects_developing_bar_support_tampering(
    inspection_fixture: _InspectionFixture,
) -> None:
    timeframe = next(
        item for item in inspection_fixture.context.timeframes if item.indicators
    )
    named = timeframe.indicators[0]
    requirement = next(
        item for item in timeframe.requirement.indicators if item.alias == named.alias
    )
    assert named.output.developing_bar_support is DevelopingBarSupport.DEVELOPING_AS_OF
    altered_support = DevelopingBarSupport.COMPLETED_ONLY
    altered_configuration: PrimitiveMapping = {
        "component_type": "timeframe_indicator",
        "contract_version": TIMEFRAME_INDICATOR_CONTRACT_VERSION,
        "indicator": {
            "configuration_id": requirement.configuration_id,
            "configuration": requirement.indicator.configuration(),
        },
        "source": {
            "timeframe": {
                "configuration_id": timeframe.requirement.timeframe.configuration_id,
                "configuration": timeframe.requirement.timeframe.to_primitive(),
            },
            "fields": [item.value for item in named.output.source_fields],
            "completion_policy": named.output.completion_policy.value,
            "developing_bar_support": altered_support.value,
            "observation_unit": "bar",
            "warm_up_bars": named.output.warm_up_bars,
            "aggregation_provenance": named.output.dataset_reference.to_primitive(
                include_feed_scope=True
            ),
            "feed_scope": named.output.feed_scope.to_primitive(),
        },
    }
    corrupt_output = replace(
        named.output,
        configuration_id=configuration_identity(altered_configuration),
        developing_bar_support=altered_support,
    )
    corrupt_timeframe = replace(
        timeframe, indicators=(replace(named, output=corrupt_output),)
    )
    corrupt_context = replace(
        inspection_fixture.context,
        timeframes=tuple(
            corrupt_timeframe if item is timeframe else item
            for item in inspection_fixture.context.timeframes
        ),
    )

    with pytest.raises(StudyInspectionReportError, match="provenance"):
        StudyInspectionSelection(
            "corrupt_support",
            corrupt_context,
            inspection_fixture.rule,
            inspection_fixture.evaluation,
            inspection_fixture.historical_study,
            inspection_fixture.datasets.family,
        )


def test_report_snapshots_validated_serialization_inputs(
    inspection_fixture: _InspectionFixture,
) -> None:
    rule = TechnicalConfluencePredictionRule(
        inspection_fixture.rule.parameters,
        inspection_fixture.rule.context_requirements,
    )
    study = HistoricalPredictionStudyReference.capture(
        study_id="qf34_report_snapshot_study_v1",
        rule=rule,
        validated_symbols=(inspection_fixture.context.symbol,),
        historical_dataset_fingerprint="d" * 64,
        adjustment_basis=inspection_fixture.context.adjustment_basis,
    )
    report = build_study_inspection_report(
        (
            StudyInspectionSelection(
                "snapshot",
                inspection_fixture.context,
                rule,
                rule.evaluate(inspection_fixture.context),
                study,
                inspection_fixture.datasets.family,
            ),
        )
    )
    expected = (report.report_id, report.manifest_bytes(), report.html_bytes())

    rule.context_requirements = replace(
        rule.context_requirements,
        failure_policy=PredictionContextFailurePolicy.SKIP,
    )

    assert (report.report_id, report.manifest_bytes(), report.html_bytes()) == expected
    with pytest.raises(StudyInspectionReportError, match="historical-study"):
        build_study_inspection_report(report.selections)


def test_static_export_is_reproducible_immutable_and_viewable_offline(
    inspection_fixture: _InspectionFixture, tmp_path: Path
) -> None:
    report = build_study_inspection_report((_selection(inspection_fixture),))

    created, created_status = export_study_inspection_report(report, tmp_path)
    repeated, repeated_status = export_study_inspection_report(report, tmp_path)

    assert created == repeated
    assert created_status == "created_immutable_report"
    assert repeated_status == "reused_immutable_report"
    assert (created / "manifest.json").read_bytes() == report.manifest_bytes()
    report_html = (created / "report.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in report_html
    assert "https://" not in report_html
    assert "<script src=" not in report_html

    (created / "report.html").write_text("changed", encoding="utf-8")
    with pytest.raises(StudyInspectionReportError, match="content differs"):
        export_study_inspection_report(report, tmp_path)


def test_future_outcome_cannot_overlap_causal_decision_state(
    inspection_fixture: _InspectionFixture,
) -> None:
    decision = inspection_fixture.context.latest_bar_for(
        inspection_fixture.context.requirements.primary.timeframe
    ).end_timestamp
    with pytest.raises(StudyInspectionReportError, match="cannot begin before"):
        StudyInspectionSelection(
            "overlap",
            inspection_fixture.context,
            inspection_fixture.rule,
            inspection_fixture.evaluation,
            inspection_fixture.historical_study,
            inspection_fixture.datasets.family,
            FutureOutcomeRegion(
                decision - timedelta(minutes=5),
                decision + timedelta(minutes=5),
                "invalid overlap",
            ),
        )
