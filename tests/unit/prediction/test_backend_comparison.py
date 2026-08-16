from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from quantforge.configuration import (
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.models import MarketDataset
from quantforge.indicators import (
    NATIVE_INDICATOR_BACKEND,
    TALIB_INDICATOR_BACKEND,
    IndicatorBackendIdentity,
    IndicatorBackendRegistry,
    IndicatorComparisonTolerances,
    IndicatorComputationRequest,
    IndicatorComputationResult,
    NativeIndicatorBackend,
    StandardIndicatorDefinition,
    TalibIndicatorBackend,
)
from quantforge.prediction import (
    PREDICTION_BACKEND_COMPARISON_ARTIFACT_FILENAMES,
    InvalidPredictionConfigurationError,
    InvalidPredictionOutputError,
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    compare_prediction_backends,
    export_overnight_gap_backend_comparison,
    run_overnight_gap_backend_comparison,
    run_prediction_analysis,
    validate_overnight_gap_backend_comparison_export,
)

from ..helpers import make_dataset


class CountingBackend:
    def __init__(
        self,
        delegate: NativeIndicatorBackend | TalibIndicatorBackend,
    ) -> None:
        self._delegate = delegate
        self.compute_count = 0

    @property
    def backend_id(self) -> str:
        return self._delegate.backend_id

    def identity_for(
        self, definition: StandardIndicatorDefinition
    ) -> IndicatorBackendIdentity:
        return self._delegate.identity_for(definition)

    def compute(
        self, request: IndicatorComputationRequest
    ) -> IndicatorComputationResult:
        self.compute_count += 1
        return self._delegate.compute(request)


def _overnight_dataset() -> MarketDataset:
    closes = (
        "100",
        "102",
        "101",
        "103",
        "99",
        "101",
        "100",
        "102",
        "98",
        "100",
        "99",
        "101",
        "100",
        "102",
        "101",
    )
    opens = tuple(str(int(value) - 1) for value in closes)
    highs = tuple(str(int(value) + 1) for value in closes)
    lows = tuple(str(int(value) - 2) for value in closes)
    return make_dataset(closes, opens=opens, highs=highs, lows=lows)


def test_implicit_overnight_strategy_remains_the_legacy_native_path() -> None:
    default = OvernightGapPredictionStrategy(OvernightGapPredictionParameters())
    explicit_native = OvernightGapPredictionStrategy(
        OvernightGapPredictionParameters(), backend_id=NATIVE_INDICATOR_BACKEND
    )

    assert all(
        "backend" not in indicator.configuration()
        for indicator in default.required_indicators
    )
    assert all(
        "backend" in indicator.configuration()
        for indicator in explicit_native.required_indicators
    )
    assert default.configuration()["contract_version"] == "1"


def test_custom_backend_registry_requires_an_explicit_backend_id() -> None:
    registry = IndicatorBackendRegistry((NativeIndicatorBackend(),))

    with pytest.raises(
        InvalidPredictionConfigurationError,
        match="custom indicator backend registry requires an explicit backend_id",
    ):
        OvernightGapPredictionStrategy(
            OvernightGapPredictionParameters(), backend_registry=registry
        )

    explicit = OvernightGapPredictionStrategy(
        OvernightGapPredictionParameters(),
        backend_id=NATIVE_INDICATOR_BACKEND,
        backend_registry=registry,
    )
    for indicator in explicit.required_indicators:
        backend_configuration = indicator.configuration().get("backend")
        assert isinstance(backend_configuration, dict)
        assert backend_configuration.get("backend_id") == NATIVE_INDICATOR_BACKEND


def test_prediction_comparison_rejects_backend_labels_not_bound_to_results() -> None:
    dataset = _overnight_dataset()
    parameters = OvernightGapPredictionParameters()
    native_result = run_prediction_analysis(
        dataset,
        OvernightGapPredictionStrategy(parameters, backend_id=NATIVE_INDICATOR_BACKEND),
    )
    talib_result = run_prediction_analysis(
        dataset,
        OvernightGapPredictionStrategy(parameters, backend_id=TALIB_INDICATOR_BACKEND),
    )

    with pytest.raises(
        InvalidPredictionOutputError,
        match="backend id does not match analyzed configuration: backend A",
    ):
        compare_prediction_backends(
            native_result,
            talib_result,
            backend_a_id=TALIB_INDICATOR_BACKEND,
            backend_b_id=NATIVE_INDICATOR_BACKEND,
        )


@pytest.mark.parametrize(
    "configuration_difference",
    ["signal_timestamp", "warm_up_observations", "required_indicators"],
)
def test_prediction_comparison_rejects_non_backend_configuration_difference(
    configuration_difference: str,
) -> None:
    dataset = _overnight_dataset()
    parameters = OvernightGapPredictionParameters()
    native_result = run_prediction_analysis(
        dataset,
        OvernightGapPredictionStrategy(parameters, backend_id=NATIVE_INDICATOR_BACKEND),
    )
    talib_result = run_prediction_analysis(
        dataset,
        OvernightGapPredictionStrategy(parameters, backend_id=TALIB_INDICATOR_BACKEND),
    )
    changed_configuration = talib_result.strategy_configuration
    if configuration_difference == "signal_timestamp":
        changed_configuration["signal_timestamp"] = "after_session_open"
    elif configuration_difference == "warm_up_observations":
        changed_configuration["warm_up_observations"] = 999
    else:
        required_indicators = changed_configuration["required_indicators"]
        assert isinstance(required_indicators, list)
        changed_configuration["required_indicators"] = required_indicators[:-1]
    changed_snapshot = PrimitiveMappingSnapshot.capture(changed_configuration)
    changed_result = replace(
        talib_result,
        strategy_configuration_id=configuration_identity(changed_configuration),
        strategy_configuration_snapshot=changed_snapshot,
    )

    with pytest.raises(
        InvalidPredictionOutputError,
        match="same dataset and logical rule",
    ):
        compare_prediction_backends(
            native_result,
            changed_result,
            backend_a_id=NATIVE_INDICATOR_BACKEND,
            backend_b_id=TALIB_INDICATOR_BACKEND,
        )


def test_prediction_comparison_uses_complete_signals_from_analyzed_runs() -> None:
    dataset = _overnight_dataset()
    parameters = OvernightGapPredictionParameters()
    native_strategy = OvernightGapPredictionStrategy(
        parameters, backend_id=NATIVE_INDICATOR_BACKEND
    )
    talib_strategy = OvernightGapPredictionStrategy(
        parameters, backend_id=TALIB_INDICATOR_BACKEND
    )
    native_result = run_prediction_analysis(dataset, native_strategy)
    talib_result = run_prediction_analysis(dataset, talib_strategy)
    comparison = compare_prediction_backends(
        native_result,
        talib_result,
        backend_a_id=NATIVE_INDICATOR_BACKEND,
        backend_b_id=TALIB_INDICATOR_BACKEND,
    )

    assert comparison.backend_a_prediction_count == len(native_result.generated_signals)
    assert comparison.backend_b_prediction_count == len(talib_result.generated_signals)
    assert len(native_result.generated_signals) > len(native_result.rows)
    assert len(talib_result.generated_signals) > len(talib_result.rows)


def test_prediction_comparison_rejects_truncated_retained_signal_set() -> None:
    dataset = _overnight_dataset()
    parameters = OvernightGapPredictionParameters()
    native_strategy = OvernightGapPredictionStrategy(
        parameters, backend_id=NATIVE_INDICATOR_BACKEND
    )
    talib_strategy = OvernightGapPredictionStrategy(
        parameters, backend_id=TALIB_INDICATOR_BACKEND
    )
    native_result = run_prediction_analysis(dataset, native_strategy)
    talib_result = run_prediction_analysis(dataset, talib_strategy)
    truncated_result = replace(
        native_result, generated_signals=native_result.generated_signals[:-1]
    )

    with pytest.raises(
        InvalidPredictionOutputError,
        match="does not match analyzed run: backend A",
    ):
        compare_prediction_backends(
            truncated_result,
            talib_result,
            backend_a_id=NATIVE_INDICATOR_BACKEND,
            backend_b_id=TALIB_INDICATOR_BACKEND,
        )


def test_prediction_comparison_rejects_altered_retained_labeled_signal() -> None:
    dataset = _overnight_dataset()
    parameters = OvernightGapPredictionParameters()
    native_strategy = OvernightGapPredictionStrategy(
        parameters, backend_id=NATIVE_INDICATOR_BACKEND
    )
    talib_strategy = OvernightGapPredictionStrategy(
        parameters, backend_id=TALIB_INDICATOR_BACKEND
    )
    native_result = run_prediction_analysis(dataset, native_strategy)
    talib_result = run_prediction_analysis(dataset, talib_strategy)
    labeled_session = native_result.rows[0].signal_session
    altered_signals = tuple(
        replace(signal, reason=f"{signal.reason} altered")
        if signal.signal_session == labeled_session
        else signal
        for signal in native_result.generated_signals
    )

    with pytest.raises(
        InvalidPredictionOutputError,
        match="does not match analyzed run: backend A",
    ):
        compare_prediction_backends(
            replace(native_result, generated_signals=altered_signals),
            talib_result,
            backend_a_id=NATIVE_INDICATOR_BACKEND,
            backend_b_id=TALIB_INDICATOR_BACKEND,
        )


def test_overnight_gap_report_quantifies_value_signal_and_metric_impact() -> None:
    dataset = _overnight_dataset()

    result = run_overnight_gap_backend_comparison(
        dataset,
        tolerances=IndicatorComparisonTolerances(
            Decimal("0.000000000001"), Decimal("0.000000000001")
        ),
    )
    prediction = result.prediction_comparison

    assert tuple(item.definition.name for item in result.indicator_comparisons) == (
        "wilder_relative_strength_index",
        "wilder_directional_movement",
    )
    assert all(
        item.backend_a_identity.backend_id == NATIVE_INDICATOR_BACKEND
        and item.backend_b_identity.backend_id == TALIB_INDICATOR_BACKEND
        for item in result.indicator_comparisons
    )
    assert all(
        item.backend_b_identity.library_version == "0.7.1"
        for item in result.indicator_comparisons
    )
    assert prediction.backend_a_prediction_count == (
        len(prediction.backend_a_only_prediction_dates)
        + prediction.shared_prediction_date_count
    )
    assert prediction.backend_b_prediction_count == (
        len(prediction.backend_b_only_prediction_dates)
        + prediction.shared_prediction_date_count
    )
    assert tuple(item.metric_name for item in prediction.metrics) == (
        "accuracy",
        "average_signed_return",
    )
    assert dataset.metadata.data_sha256 in result.source_snapshot.canonical_json
    assert prediction.matched_prediction_count >= 0


def test_overnight_gap_report_reuses_each_compared_backend_computation() -> None:
    native = CountingBackend(NativeIndicatorBackend())
    talib = CountingBackend(TalibIndicatorBackend())

    run_overnight_gap_backend_comparison(
        _overnight_dataset(),
        backend_registry=IndicatorBackendRegistry((native, talib)),
    )

    assert native.compute_count == 2
    assert talib.compute_count == 2


def test_overnight_gap_backend_artifacts_are_deterministic(tmp_path: Path) -> None:
    result = run_overnight_gap_backend_comparison(
        _overnight_dataset(),
        tolerances=IndicatorComparisonTolerances(
            Decimal("0.000000000001"), Decimal("0.000000000001")
        ),
    )

    first = export_overnight_gap_backend_comparison(result, tmp_path / "first")
    second = export_overnight_gap_backend_comparison(result, tmp_path / "second")

    assert {item.name for item in first.iterdir()} == set(
        PREDICTION_BACKEND_COMPARISON_ARTIFACT_FILENAMES
    )
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes()
        for name in PREDICTION_BACKEND_COMPARISON_ARTIFACT_FILENAMES
    )
    assert validate_overnight_gap_backend_comparison_export(result, first) == first
    summary = (first / "summary.txt").read_text(encoding="utf-8")
    assert "Matched predictions:" in summary
    assert "Changed directions:" in summary
    assert "Comparison only; native studies remain unchanged." in summary
