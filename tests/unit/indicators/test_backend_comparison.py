from decimal import Decimal
from pathlib import Path

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.indicators import (
    INDICATOR_COMPARISON_ARTIFACT_FILENAMES,
    NATIVE_INDICATOR_BACKEND,
    TALIB_INDICATOR_BACKEND,
    IndicatorBackendIdentity,
    IndicatorBackendRegistry,
    IndicatorComparisonSource,
    IndicatorComparisonTolerances,
    IndicatorComputationRequest,
    IndicatorComputationResult,
    IndicatorFieldOutput,
    MarketField,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
    StandardIndicatorDefinition,
    compare_indicator_backends,
    compare_standard_indicator_backends,
    export_indicator_backend_comparison,
    validate_indicator_backend_comparison_export,
)

from ..helpers import make_dataset


class FixtureComparisonBackend:
    def __init__(
        self,
        backend_id: str,
        *,
        leading_unavailable: int,
        offsets: dict[str, Decimal],
        reverse_fields: bool = False,
    ) -> None:
        self._backend_id = backend_id
        self._leading_unavailable = leading_unavailable
        self._offsets = offsets
        self._reverse_fields = reverse_fields

    @property
    def backend_id(self) -> str:
        return self._backend_id

    def identity_for(
        self, definition: StandardIndicatorDefinition
    ) -> IndicatorBackendIdentity:
        return IndicatorBackendIdentity(
            backend_id=self.backend_id,
            library_name=f"fixture-{self.backend_id}",
            library_version="1.0",
            function_name=f"mapped_{definition.name}",
        )

    def compute(
        self, request: IndicatorComputationRequest
    ) -> IndicatorComputationResult:
        names = request.definition.output_fields
        if self._reverse_fields:
            names = tuple(reversed(names))
        fields = tuple(
            IndicatorFieldOutput(
                name,
                tuple(
                    None
                    if index < self._leading_unavailable
                    else bar.close + self._offsets[name]
                    for index, bar in enumerate(request.bars)
                ),
            )
            for name in names
        )
        return IndicatorComputationResult(
            definition_name=request.definition.name,
            backend_identity=self.identity_for(request.definition),
            normalized_parameters=request.definition.parameters,
            normalized_input_fields=request.definition.input_fields,
            fields=fields,
            observation_count=len(request.bars),
        )


def _definition() -> StandardIndicatorDefinition:
    return StandardIndicatorDefinition(
        name="comparison_fixture",
        parameters=PrimitiveMappingSnapshot.capture({"period": 2}),
        input_fields=(MarketField.CLOSE,),
        output_fields=("alpha", "beta"),
    )


def _registry() -> IndicatorBackendRegistry:
    return IndicatorBackendRegistry(
        (
            FixtureComparisonBackend(
                "fixture_a",
                leading_unavailable=1,
                offsets={"alpha": Decimal(0), "beta": Decimal(0)},
            ),
            FixtureComparisonBackend(
                "fixture_b",
                leading_unavailable=2,
                offsets={"alpha": Decimal(0), "beta": Decimal(1)},
                reverse_fields=True,
            ),
        )
    )


def test_native_and_talib_exact_match_uses_one_normalized_configuration() -> None:
    dataset = make_dataset(("1", "2", "3", "4"))
    indicator = SimpleMovingAverage(SimpleMovingAverageParameters(2))

    result = compare_indicator_backends(dataset, indicator)
    field = result.field_comparisons[0]

    assert result.definition == indicator.standard_definition
    assert result.backend_a_identity.backend_id == NATIVE_INDICATOR_BACKEND
    assert result.backend_b_identity.backend_id == TALIB_INDICATOR_BACKEND
    assert field.backend_a_first_valid_timestamp == "2024-07-02"
    assert field.backend_b_first_valid_timestamp == "2024-07-02"
    assert field.overlapping_valid_count == 3
    assert field.maximum_absolute_difference == 0
    assert field.divergences == ()


def test_warm_up_is_separate_from_named_multi_output_numerical_differences() -> None:
    dataset = make_dataset(("10", "11", "12", "13"))

    result = compare_standard_indicator_backends(
        IndicatorComparisonSource.from_market_dataset(dataset),
        _definition(),
        backend_a_id="fixture_a",
        backend_b_id="fixture_b",
        tolerances=IndicatorComparisonTolerances(Decimal("0.5"), Decimal(0)),
        backend_registry=_registry(),
    )
    alpha, beta = result.field_comparisons

    assert alpha.output_name == "alpha"
    assert alpha.backend_a_first_valid_timestamp == "2024-07-02"
    assert alpha.backend_b_first_valid_timestamp == "2024-07-03"
    assert alpha.backend_a_only_valid_timestamps == ("2024-07-02",)
    assert alpha.overlapping_valid_count == 2
    assert alpha.divergences == ()
    assert beta.output_name == "beta"
    assert beta.overlapping_valid_count == 2
    assert beta.maximum_absolute_difference == 1
    assert tuple(item.timestamp for item in beta.divergences) == (
        "2024-07-03",
        "2024-07-05",
    )
    assert all(item.output_name == "beta" for item in beta.divergences)


def test_appending_future_bars_preserves_historical_comparison_rows() -> None:
    historical_dataset = make_dataset(("10", "11", "12", "13"))
    extended_dataset = make_dataset(("10", "11", "12", "13", "14", "15"))
    tolerances = IndicatorComparisonTolerances(Decimal("0.5"), Decimal(0))

    historical = compare_standard_indicator_backends(
        IndicatorComparisonSource.from_market_dataset(historical_dataset),
        _definition(),
        backend_a_id="fixture_a",
        backend_b_id="fixture_b",
        tolerances=tolerances,
        backend_registry=_registry(),
    )
    extended = compare_standard_indicator_backends(
        IndicatorComparisonSource.from_market_dataset(extended_dataset),
        _definition(),
        backend_a_id="fixture_a",
        backend_b_id="fixture_b",
        tolerances=tolerances,
        backend_registry=_registry(),
    )

    cutoff = historical_dataset.bars[-1].session_date.isoformat()
    for historical_field, extended_field in zip(
        historical.field_comparisons, extended.field_comparisons, strict=True
    ):
        historical_rows = tuple(
            item.to_primitive() for item in historical_field.divergences
        )
        extended_rows = tuple(
            item.to_primitive()
            for item in extended_field.divergences
            if item.timestamp <= cutoff
        )
        assert extended_rows == historical_rows


def test_comparison_artifacts_are_deterministic_and_complete(tmp_path: Path) -> None:
    dataset = make_dataset(("10", "11", "12", "13"))
    result = compare_standard_indicator_backends(
        IndicatorComparisonSource.from_market_dataset(dataset),
        _definition(),
        backend_a_id="fixture_a",
        backend_b_id="fixture_b",
        tolerances=IndicatorComparisonTolerances(Decimal("0.5"), Decimal("0.01")),
        backend_registry=_registry(),
    )

    first = export_indicator_backend_comparison(result, tmp_path / "first")
    second = export_indicator_backend_comparison(result, tmp_path / "second")

    assert {item.name for item in first.iterdir()} == set(
        INDICATOR_COMPARISON_ARTIFACT_FILENAMES
    )
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes()
        for name in INDICATOR_COMPARISON_ARTIFACT_FILENAMES
    )
    assert validate_indicator_backend_comparison_export(result, first) == first
    comparison = (first / "comparison.json").read_text(encoding="utf-8")
    assert dataset.metadata.data_sha256 in comparison
    assert '"timeframe"' in comparison
    assert '"library_version": "1.0"' in comparison
    assert '"period": 2' in comparison
    assert '"absolute": "0.5"' in comparison
