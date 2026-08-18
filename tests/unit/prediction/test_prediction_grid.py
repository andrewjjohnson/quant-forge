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
    StabilityConfig,
    ThresholdOperator,
    TrialStatus,
)
from quantforge.prediction import (
    InvalidPredictionGridParametersError,
    PredictionContextEnvironment,
    PredictionGridConfig,
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
            raise InvalidPredictionGridParametersError("window must be positive")
        rule = fixtures.FixtureMultiTimeframeRule(
            fixtures._requirements(window=window)  # pyright: ignore[reportPrivateUsage]
        )
        return cast(
            PredictionStudy[Any, Any, Any],
            fixtures._study(rule),  # pyright: ignore[reportPrivateUsage]
        )


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


class FakeResult:
    def __init__(self, window: int) -> None:
        self.window = window

    def to_primitive(self) -> PrimitiveMapping:
        return {"window": self.window}


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
) -> PredictionGridStudy:
    return PredictionGridStudy(
        dataset=fixtures._prediction_dataset(),  # pyright: ignore[reportPrivateUsage]
        dataset_family_fingerprint="family-spy-fixture",
        study_factory=FixtureStudyFactory(),
        analyzer=FixtureAnalyzer(),
        context_provider=fixtures.FixtureContextProvider(
            fixtures._prediction_context()  # pyright: ignore[reportPrivateUsage]
        ),
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
            stability=StabilityConfig(minimum_eligible_neighbors=1),
            output_root=output_root,
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
            raise RuntimeError("fixture trial failure")
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
    assert tuple(first._trial_id(item) for item in first._candidates) == tuple(  # pyright: ignore[reportPrivateUsage]
        equivalent._trial_id(item)  # pyright: ignore[reportPrivateUsage]
        for item in equivalent._candidates  # pyright: ignore[reportPrivateUsage]
    )
    assert first.study_id != changed_backend.study_id
    assert tuple(first._trial_id(item) for item in first._candidates) != tuple(  # pyright: ignore[reportPrivateUsage]
        changed_backend._trial_id(item)  # pyright: ignore[reportPrivateUsage]
        for item in changed_backend._candidates  # pyright: ignore[reportPrivateUsage]
    )


def test_context_and_normalized_indicator_cache_reuses_only_compatible_identity() -> (
    None
):
    requirements = fixtures._requirements(window=2)  # pyright: ignore[reportPrivateUsage]
    provider = fixtures.FixtureContextProvider(
        fixtures._prediction_context()  # pyright: ignore[reportPrivateUsage]
    )
    cache = PredictionGridExecutionCache(
        dataset_family_fingerprint="family-spy-fixture",
        backend=_backend_environment(),
        context_environment=PredictionContextEnvironment.create(
            "fixture_context_provider", "1", {"dataset_family": "fixture"}
        ),
    )

    context = cache.context(requirements, provider)
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
    assert incompatible.context(requirements, provider) is context
    assert incompatible.statistics.context_hits == 0
    assert incompatible.statistics.context_misses == 1


def test_prediction_grid_orchestration_has_no_direct_talib_import() -> None:
    source = Path("src/quantforge/prediction/grid.py").read_text(encoding="utf-8")
    assert "import talib" not in source
    assert "from talib" not in source
