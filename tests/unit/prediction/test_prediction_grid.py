import json
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.indicators import (
    IndicatorBackendIdentity,
    SimpleMovingAverageParameters,
)
from quantforge.optimization import IntegerValues, ParameterSearchSpace
from quantforge.optimization.models import (
    StabilityClassification,
    StabilityConfig,
    ThresholdOperator,
    TrialStatus,
)
from quantforge.prediction import (
    InvalidPredictionGridConfigurationError,
    InvalidPredictionGridParametersError,
    PredictionContextEnvironment,
    PredictionContextError,
    PredictionGridConfig,
    PredictionGridPersistenceError,
    PredictionGridStudy,
    PredictionIndicatorBackendEnvironment,
    PredictionMetricConstraint,
    PredictionRankingConfig,
    PredictionStudy,
    PredictionTrialAnalysis,
)
from quantforge.prediction.grid import PredictionGridExecutionCache
from quantforge.prediction.study import PredictionStudyResult
from tests.unit.prediction import test_multi_timeframe_study as fixtures


class FixtureStudyFactory:
    name = "fixture_prediction_grid_factory"
    version = "1"
    parameter_order = ("window",)
    required_parameter_names = frozenset(parameter_order)

    def configuration(self) -> PrimitiveMapping:
        return {
            "name": self.name,
            "version": self.version,
            "parameter_order": list(self.parameter_order),
        }

    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        window = cast(int, parameters["window"])
        if window < 1:
            raise InvalidPredictionGridParametersError(
                "window must be positive api_key=excluded-secret "
                "account_id=excluded-account"
            )
        rule = fixtures.FixtureMultiTimeframeRule(
            fixtures._requirements(window=window)  # pyright: ignore[reportPrivateUsage]
        )
        return cast(
            PredictionStudy[Any, Any, Any],
            fixtures._study(rule),  # pyright: ignore[reportPrivateUsage]
        )


class InterruptingStudyFactory(FixtureStudyFactory):
    def __init__(self) -> None:
        self.should_interrupt = True

    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        if cast(int, parameters["window"]) == 3 and self.should_interrupt:
            self.should_interrupt = False
            raise KeyboardInterrupt("fixture candidate construction interruption")
        return super().build(parameters)


class FixtureAnalyzer:
    name = "fixture_prediction_grid_analyzer"
    version = "1"

    def configuration(self) -> PrimitiveMapping:
        return {"name": self.name, "version": self.version}

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def analyze(
        self, result: PredictionStudyResult[Any, Any, Any]
    ) -> PredictionTrialAnalysis:
        value = cast(int, result.to_primitive()["window"])
        return PredictionTrialAnalysis.create(
            prediction_count=10 + value,
            metrics={"accuracy": f"0.{value + 4}", "quality": value},
            period_comparisons=({"period": "development", "accuracy": "0.5"},),
            weekday_comparisons=({"weekday": 1, "accuracy": "0.6"},),
            matched_baseline_comparisons=(
                {
                    "baseline_name": "always_up",
                    "scope": "matched_prediction_sessions",
                    "accuracy_delta": "0.1",
                },
            ),
            artifacts={"comparison_schema_version": "1"},
        )


class MutableAnalyzer(FixtureAnalyzer):
    name = "fixture_prediction_grid_mutable_analyzer"

    def __init__(self) -> None:
        self.configuration_state: PrimitiveMapping = {"mode": "initial"}

    def configuration(self) -> PrimitiveMapping:
        return self.configuration_state

    def analyze(
        self, result: PredictionStudyResult[Any, Any, Any]
    ) -> PredictionTrialAnalysis:
        analysis = super().analyze(result)
        self.configuration_state["mode"] = "mutated"
        return analysis


class SpikeAnalyzer(FixtureAnalyzer):
    name = "fixture_prediction_grid_spike_analyzer"

    def analyze(
        self, result: PredictionStudyResult[Any, Any, Any]
    ) -> PredictionTrialAnalysis:
        value = cast(int, result.to_primitive()["window"])
        accuracy = "1.0" if value == 3 else "0.1"
        return PredictionTrialAnalysis.create(
            prediction_count=10 + value,
            metrics={"accuracy": accuracy, "quality": value},
            period_comparisons=({"period": "development", "accuracy": accuracy},),
            weekday_comparisons=({"weekday": 1, "accuracy": accuracy},),
            matched_baseline_comparisons=(
                {
                    "baseline_name": "always_up",
                    "scope": "matched_prediction_sessions",
                    "accuracy_delta": accuracy,
                },
            ),
        )


class FakeResult:
    def __init__(self, window: int) -> None:
        self.window = window
        self.study_id = f"fixture-prediction-study-{window}"

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "manifest": {"study_id": self.study_id},
            "rows": [{"row_id": f"fixture-row-{self.window}"}],
            "window": self.window,
        }


def _backend_environment(
    *, configuration: PrimitiveMapping | None = None
) -> PredictionIndicatorBackendEnvironment:
    requirement = fixtures._requirements().primary.indicators[0]  # pyright: ignore[reportPrivateUsage]
    identity = requirement.backend_identity
    assert isinstance(identity, IndicatorBackendIdentity)
    return PredictionIndicatorBackendEnvironment.create(
        backend_id=identity.backend_id,
        library_name=identity.library_name,
        library_version=identity.library_version,
        contract_version=identity.contract_version,
        runtime_library_name=identity.runtime_library_name,
        runtime_library_version=identity.runtime_library_version,
        configuration=configuration,
    )


def _grid(
    output_root: Path,
    *,
    backend_configuration: PrimitiveMapping | None = None,
    retry_failed: bool = False,
    analyzer: FixtureAnalyzer | None = None,
    stability: StabilityConfig | None = None,
    factory: FixtureStudyFactory | None = None,
) -> PredictionGridStudy:
    context = fixtures._prediction_context()  # pyright: ignore[reportPrivateUsage]
    family_id = context.source_consistency.family_id
    assert family_id is not None
    return PredictionGridStudy(
        dataset=fixtures._prediction_dataset(),  # pyright: ignore[reportPrivateUsage]
        dataset_family_fingerprint=family_id,
        study_factory=FixtureStudyFactory() if factory is None else factory,
        analyzer=FixtureAnalyzer() if analyzer is None else analyzer,
        context_provider=fixtures.FixtureContextProvider(context),
        context_environment=PredictionContextEnvironment.create(
            "fixture_context_provider", "1", {"dataset_family": "fixture"}
        ),
        indicator_backend=_backend_environment(configuration=backend_configuration),
        config=PredictionGridConfig(
            label="fixture deterministic prediction grid",
            search_space=ParameterSearchSpace({"window": IntegerValues((0, 2, 3, 4))}),
            ranking=PredictionRankingConfig(
                "accuracy",
                "always_up",
                minimum_prediction_count=12,
                outcome_quality_constraints=(
                    PredictionMetricConstraint(
                        "quality", ThresholdOperator.GREATER_THAN_OR_EQUAL, Decimal(2)
                    ),
                ),
            ),
            stability=(
                StabilityConfig(minimum_eligible_neighbors=1)
                if stability is None
                else stability
            ),
            output_root=output_root,
            retry_failed=retry_failed,
        ),
    )


def _window(study: PredictionStudy[Any, Any, Any]) -> int:
    rule = cast(fixtures.FixtureMultiTimeframeRule, study.strategy)
    requirement = rule.context_requirements.primary.indicators[0]
    parameters = cast(SimpleMovingAverageParameters, requirement.indicator.parameters)
    return parameters.window


def test_grid_is_deterministic_resumable_and_retains_comparisons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake_run(
        prepared: object,
        study: PredictionStudy[Any, Any, Any],
        **kwargs: object,
    ) -> PredictionStudyResult[Any, Any, Any]:
        del prepared, kwargs
        window = _window(study)
        calls.append(window)
        if window == 3:
            raise RuntimeError(
                "fixture trial failure api_key=super-secret account_id=broker-123"
            )
        return cast(PredictionStudyResult[Any, Any, Any], FakeResult(window))

    monkeypatch.setattr(
        "quantforge.prediction.grid.run_prediction_study_in_session", fake_run
    )
    grid = _grid(tmp_path)
    first = grid.run()

    assert calls == [2, 3, 4]
    assert tuple(item.status for item in first.trials) == (
        TrialStatus.EXCLUDED,
        TrialStatus.SUCCEEDED,
        TrialStatus.FAILED,
        TrialStatus.SUCCEEDED,
    )
    assert tuple(item.objective_value for item in first.rankings) == (
        Decimal("0.8"),
        Decimal("0.6"),
    )
    succeeded = first.trials[1]
    assert succeeded.analysis is not None
    assert succeeded.analysis.period_comparisons[0].to_primitive()["period"] == (
        "development"
    )
    assert succeeded.analysis.weekday_comparisons[0].to_primitive()["weekday"] == 1
    assert (
        succeeded.analysis.matched_baseline_comparisons[0].to_primitive()["scope"]
        == "matched_prediction_sessions"
    )
    assert first.stability
    assert "not validated profitability" in first.limitations[0]
    assert "multiple-comparison correction" in first.warnings[0]
    failed = first.trials[2]
    assert failed.failure_message is not None
    assert "super-secret" not in failed.failure_message
    assert "broker-123" not in failed.failure_message
    assert "raw exception text was not persisted" in failed.failure_message
    excluded = first.trials[0]
    assert excluded.exclusion_reason is not None
    assert "excluded-secret" not in excluded.exclusion_reason
    assert "excluded-account" not in excluded.exclusion_reason
    assert "raw exception text was not persisted" in excluded.exclusion_reason

    resumed = grid.resume()
    assert calls == [2, 3, 4]
    assert resumed.summary_primitive()["counts"] == first.summary_primitive()["counts"]
    assert tuple(item.trial_id for item in resumed.trials) == tuple(
        item.trial_id for item in first.trials
    )


def test_equivalent_grids_have_stable_order_and_backend_configuration_changes_identity(
    tmp_path: Path,
) -> None:
    first = _grid(tmp_path / "first")
    equivalent = _grid(tmp_path / "equivalent")
    changed_backend = _grid(
        tmp_path / "changed", backend_configuration={"rounding_policy": "alternate"}
    )

    assert first.study_id == equivalent.study_id
    assert first.study_id != changed_backend.study_id


def test_component_configuration_mutation_fails_before_success_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(
        prepared: object,
        study: PredictionStudy[Any, Any, Any],
        **kwargs: object,
    ) -> PredictionStudyResult[Any, Any, Any]:
        del prepared, kwargs
        return cast(PredictionStudyResult[Any, Any, Any], FakeResult(_window(study)))

    monkeypatch.setattr(
        "quantforge.prediction.grid.run_prediction_study_in_session", fake_run
    )
    analyzer = MutableAnalyzer()
    grid = _grid(tmp_path, analyzer=analyzer)

    with pytest.raises(
        InvalidPredictionGridConfigurationError,
        match="factory or analyzer changed",
    ):
        grid.run()

    manifest_path = tmp_path / grid.study_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["analyzer"]["configuration"] == {"mode": "initial"}
    assert analyzer.configuration_state == {"mode": "mutated"}
    trial_documents = tuple((tmp_path / grid.study_id / "trials").glob("*.json"))
    assert any(
        json.loads(path.read_text(encoding="utf-8"))["status"] == "running"
        for path in trial_documents
    )


def test_candidate_construction_interruption_preserves_enumerated_trials_for_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(
        prepared: object,
        study: PredictionStudy[Any, Any, Any],
        **kwargs: object,
    ) -> PredictionStudyResult[Any, Any, Any]:
        del prepared, kwargs
        return cast(PredictionStudyResult[Any, Any, Any], FakeResult(_window(study)))

    monkeypatch.setattr(
        "quantforge.prediction.grid.run_prediction_study_in_session", fake_run
    )
    factory = InterruptingStudyFactory()
    grid = _grid(tmp_path, factory=factory)

    with pytest.raises(KeyboardInterrupt, match="candidate construction"):
        grid.run()

    study_path = tmp_path / grid.study_id
    persisted_before_resume = tuple((study_path / "trials").glob("*.json"))
    assert (study_path / "manifest.json").is_file()
    assert len(persisted_before_resume) == 2

    resumed = grid.resume()

    assert len(resumed.trials) == 4
    assert all(
        item.status in (TrialStatus.EXCLUDED, TrialStatus.SUCCEEDED, TrialStatus.FAILED)
        for item in resumed.trials
    )


def test_context_and_normalized_indicator_cache_reuses_only_compatible_identity() -> (
    None
):
    requirements = fixtures._requirements(window=2)  # pyright: ignore[reportPrivateUsage]
    context = fixtures._prediction_context()  # pyright: ignore[reportPrivateUsage]
    family_id = context.source_consistency.family_id
    assert family_id is not None
    provider = fixtures.FixtureContextProvider(context)
    cache = PredictionGridExecutionCache(
        dataset_family_fingerprint=family_id,
        backend=_backend_environment(),
        context_environment=PredictionContextEnvironment.create(
            "fixture_context_provider", "1", {"dataset_family": "fixture"}
        ),
    )

    assert cache.context(requirements, provider) is context
    assert cache.context(requirements, provider) is context
    indicator = requirements.primary.indicators[0]
    first = cache.resolve(
        indicator,
        context,
        requirements.primary.timeframe,
        requirements.primary.completion_policy,
    )
    second = cache.resolve(
        indicator,
        context,
        requirements.primary.timeframe,
        requirements.primary.completion_policy,
    )

    assert first is second
    assert cache.statistics.context_hits == 1
    assert cache.statistics.context_misses == 1
    assert cache.statistics.indicator_hits == 1
    assert cache.statistics.indicator_misses == 1

    incompatible = PredictionGridExecutionCache(
        dataset_family_fingerprint="different-family",
        backend=_backend_environment(),
        context_environment=PredictionContextEnvironment.create(
            "fixture_context_provider", "1", {"dataset_family": "fixture"}
        ),
    )
    with pytest.raises(PredictionContextError, match="dataset family"):
        incompatible.context(requirements, provider)
    assert incompatible.statistics.context_hits == 0
    assert incompatible.statistics.context_misses == 0

    object.__setattr__(
        indicator.indicator,
        "_parameters",
        SimpleMovingAverageParameters(3),
    )
    with pytest.raises(PredictionContextError, match="declared indicator changed"):
        cache.resolve(
            indicator,
            context,
            requirements.primary.timeframe,
            requirements.primary.completion_policy,
        )


def test_retry_preserves_the_failed_attempt_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: dict[int, int] = {}

    def fake_run(
        prepared: object,
        study: PredictionStudy[Any, Any, Any],
        **kwargs: object,
    ) -> PredictionStudyResult[Any, Any, Any]:
        del prepared, kwargs
        window = _window(study)
        attempts[window] = attempts.get(window, 0) + 1
        if window == 3 and attempts[window] == 1:
            raise RuntimeError("temporary credential token=do-not-persist")
        return cast(PredictionStudyResult[Any, Any, Any], FakeResult(window))

    monkeypatch.setattr(
        "quantforge.prediction.grid.run_prediction_study_in_session", fake_run
    )
    grid = _grid(tmp_path, retry_failed=True)

    first = grid.run()
    assert first.trials[2].status is TrialStatus.FAILED
    retried = grid.resume()

    succeeded = retried.trials[2]
    assert succeeded.status is TrialStatus.SUCCEEDED
    assert succeeded.failure_type is None
    assert succeeded.failure_message is None
    assert len(succeeded.failed_attempts) == 1
    archived = succeeded.failed_attempts[0]
    assert archived.failure_type == "RuntimeError"
    assert "do-not-persist" not in archived.failure_message
    assert archived.started_at
    assert archived.finished_at


@pytest.mark.parametrize(
    "corruption", ["truncated", "changed_analysis", "changed_prediction_study"]
)
def test_load_result_rejects_corrupt_success_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    def fake_run(
        prepared: object,
        study: PredictionStudy[Any, Any, Any],
        **kwargs: object,
    ) -> PredictionStudyResult[Any, Any, Any]:
        del prepared, kwargs
        return cast(PredictionStudyResult[Any, Any, Any], FakeResult(_window(study)))

    monkeypatch.setattr(
        "quantforge.prediction.grid.run_prediction_study_in_session", fake_run
    )
    grid = _grid(tmp_path)
    result = grid.run()
    succeeded = next(
        item for item in result.trials if item.status is TrialStatus.SUCCEEDED
    )
    assert succeeded.artifact_location is not None
    artifact_path = tmp_path / grid.study_id / succeeded.artifact_location
    if corruption == "truncated":
        artifact_path.write_text('{"schema_version":', encoding="utf-8")
    elif corruption == "changed_analysis":
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["analysis"]["prediction_count"] = 999
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    else:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["prediction_study"]["rows"] = []
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(PredictionGridPersistenceError, match="artifact"):
        grid.load_result()


def test_load_result_rejects_trial_whose_filename_mismatches_its_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(
        prepared: object,
        study: PredictionStudy[Any, Any, Any],
        **kwargs: object,
    ) -> PredictionStudyResult[Any, Any, Any]:
        del prepared, kwargs
        return cast(PredictionStudyResult[Any, Any, Any], FakeResult(_window(study)))

    monkeypatch.setattr(
        "quantforge.prediction.grid.run_prediction_study_in_session", fake_run
    )
    grid = _grid(tmp_path)
    grid.run()
    trials_path = tmp_path / grid.study_id / "trials"
    source = next(trials_path.glob("*.json"))
    (trials_path / "renamed.json").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(PredictionGridPersistenceError, match="artifact path"):
        grid.load_result()


def test_stability_marks_an_isolated_center_as_fragile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(
        prepared: object,
        study: PredictionStudy[Any, Any, Any],
        **kwargs: object,
    ) -> PredictionStudyResult[Any, Any, Any]:
        del prepared, kwargs
        return cast(PredictionStudyResult[Any, Any, Any], FakeResult(_window(study)))

    monkeypatch.setattr(
        "quantforge.prediction.grid.run_prediction_study_in_session", fake_run
    )
    grid = _grid(
        tmp_path,
        analyzer=SpikeAnalyzer(),
        stability=StabilityConfig(
            minimum_eligible_neighbors=2,
            isolated_peak_top_fraction=Decimal("0.5"),
            isolated_peak_absolute_drop=Decimal("0.5"),
            isolated_peak_relative_drop=Decimal("0.5"),
            isolated_peak_maximum_constraint_pass_fraction=Decimal("1"),
        ),
    )

    result = grid.run()
    spike = next(item for item in result.stability if item.objective_rank == 1)

    assert spike.objective_value == Decimal("1.0")
    assert spike.median_neighbor_objective == Decimal("0.1")
    assert spike.center_to_neighbor_difference == Decimal("0.9")
    assert spike.relative_center_to_neighbor_difference == Decimal("0.9")
    assert spike.is_isolated_peak
    assert spike.isolation_reason is not None
    assert spike.classification is StabilityClassification.FRAGILE


def test_prediction_grid_orchestration_has_no_direct_talib_import() -> None:
    source = Path("src/quantforge/prediction/grid.py").read_text(encoding="utf-8")
    assert "import talib" not in source
    assert "from talib" not in source
