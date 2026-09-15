"""Rehashed QF-40 aggregates still require complete, typed producer records."""

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, inspect_validation, verify_artifacts
from quantforge.experiments._json import mapping
from quantforge.oos import aggregate_backtest, aggregate_prediction
from quantforge.oos.prediction import PredictionMetricFields
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.oos.conftest import CompletedStudy, complete_study

type Captured = tuple[CompletedStudy, PrimitiveMapping, Path]
type RecordPath = tuple[str | int, ...]


@pytest.fixture(scope="module")
def exports(tmp_path_factory: pytest.TempPathFactory) -> dict[bool, Captured]:
    captured: dict[bool, Captured] = {}
    for prediction in (False, True):
        root = tmp_path_factory.mktemp("aggregate-schema")
        completed = complete_study(root, prediction=prediction)
        aggregate = (aggregate_prediction if prediction else aggregate_backtest)(
            completed.source
        )
        captured[prediction] = completed, aggregate.to_primitive(), root
    return captured


@pytest.fixture(autouse=True)
def no_research(
    exports: dict[bool, Captured],
    custom_export: Captured,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_research(monkeypatch)


def at(document: PrimitiveMapping, path: RecordPath) -> PrimitiveMapping:
    current: Primitive = document
    for key in path:
        current = (
            cast(list[Primitive], current)[key]
            if isinstance(key, int)
            else mapping(current)[key]
        )
    return mapping(current)


def inspect_record(captured: Captured, document: PrimitiveMapping) -> None:
    completed, _, root = captured
    path = root / "oos" / f"{configuration_identity(document)}.json"
    path.parent.mkdir(exist_ok=True)
    write_json(path, document)
    before = path.read_bytes()
    try:
        result = inspect_validation(
            completed.source,
            completed.study.study_path,
            artifact_root=root,
            aggregate_path=path,
        )
        assert verify_artifacts(result.index, root).valid
    finally:
        assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("prediction", "path"),
    [
        (False, ("normalized_equity", 0)),
        (False, ("summary",)),
        (False, ("summary", "completeness")),
        (True, ("summary",)),
        (True, ("summary", "windows", 0)),
        (True, ("summary", "windows", 0, "summary")),
        *(
            (True, ("summary", section))
            for section in (
                "direction_distribution",
                "accuracy_interval",
                "matched_baseline",
                "signed_outcome",
                "mfe",
                "mae",
                "event_outcomes",
                "metric_fields",
                "completeness",
                "window_accuracy_consistency",
                "window_signed_outcome_consistency",
            )
        ),
        *(
            (prediction, path)
            for prediction in (False, True)
            for path in (("stability",), ("stability", "windows", 0))
        ),
    ],
)
def test_all_producer_fields_are_required(
    exports: dict[bool, Captured], prediction: bool, path: RecordPath
) -> None:
    captured = exports[prediction]
    for field in at(captured[1], path):
        document = deepcopy(captured[1])
        at(document, path).pop(field)
        with pytest.raises(ManifestError):
            inspect_record(captured, document)
    document = deepcopy(captured[1])
    at(document, path)["unexpected"] = 0
    with pytest.raises(ManifestError):
        inspect_record(captured, document)


@pytest.mark.parametrize(
    "field",
    [
        "window_start_index",
        "equity_index",
        "benchmark_index",
        "drawdown",
        "benchmark_drawdown",
    ],
)
@pytest.mark.parametrize("invalid", [None, True, 1, "NaN", "invalid", "-2"])
def test_normalized_equity_requires_finite_decimal_domains(
    exports: dict[bool, Captured], field: str, invalid: Primitive
) -> None:
    captured = exports[False]
    document = deepcopy(captured[1])
    at(document, ("normalized_equity", 0))[field] = invalid
    with pytest.raises(ManifestError):
        inspect_record(captured, document)


@pytest.mark.parametrize(
    "field",
    [
        "ending_index",
        "total_return",
        "maximum_drawdown",
        "benchmark_total_return",
        "benchmark_maximum_drawdown",
        "win_rate",
        "profit_factor",
        "gross_profit",
        "gross_loss",
        "profitable_window_fraction",
        "exposure",
        "native_commissions",
        "native_fees",
        "native_slippage_cost",
        "native_dividend_income",
    ],
)
@pytest.mark.parametrize("invalid", [True, 1, "NaN", "invalid"])
def test_backtest_summary_decimal_fields(
    exports: dict[bool, Captured], field: str, invalid: Primitive
) -> None:
    captured = exports[False]
    document = deepcopy(captured[1])
    at(document, ("summary",))[field] = invalid
    with pytest.raises(ManifestError):
        inspect_record(captured, document)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("stitching", "foreign"),
        ("starting_index", 1),
        ("profitable_window_denominator", "all_windows"),
        ("trade_count", True),
        ("open_trade_count", -1),
        ("winning_trades", "1"),
        ("losing_trades", None),
        ("oos_session_count", 1.5),
        ("ending_index", "-1"),
        ("total_return", "-2"),
        ("maximum_drawdown", "0.1"),
        ("benchmark_total_return", "-2"),
        ("benchmark_maximum_drawdown", "-2"),
        ("win_rate", "2"),
        ("profit_factor", "-1"),
        ("gross_profit", "-1"),
        ("gross_loss", "1"),
        ("profitable_window_fraction", "2"),
        ("exposure", "2"),
        ("native_commissions", "-1"),
        ("native_fees", "-1"),
        ("native_slippage_cost", "-1"),
        ("warnings", "warning"),
        ("warnings", cast(Primitive, [1])),
        ("completeness", None),
    ],
)
def test_backtest_summary_domains(
    exports: dict[bool, Captured], field: str, invalid: Primitive
) -> None:
    captured = exports[False]
    document = deepcopy(captured[1])
    at(document, ("summary",))[field] = invalid
    with pytest.raises(ManifestError):
        inspect_record(captured, document)


@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize(
    ("path", "field", "invalid"),
    [
        ((), "prediction_count", True),
        ((), "generated_signal_count", -1),
        ((), "accuracy_sample_count", "1"),
        ((), "accuracy", "2"),
        ((), "prediction_frequency", "NaN"),
        (("direction_distribution",), "up", True),
        (("accuracy_interval",), "confidence_level", "invalid"),
        (("accuracy_interval",), "lower_bound", "-1"),
        (("accuracy_interval",), "method", "unknown"),
        (("matched_baseline",), "status", "partial"),
        (("matched_baseline",), "accuracy_difference", "2"),
        (("matched_baseline",), "name", 1),
        (("signed_outcome",), "status", "unknown"),
        (("signed_outcome",), "mean", "NaN"),
        (("mfe",), "sample_count", True),
        (("mae",), "unavailable_count", -1),
        (("event_outcomes",), "counts", {"event": True}),
        (("event_outcomes",), "rates", {"event": "NaN"}),
    ],
)
def test_prediction_summary_domains(
    exports: dict[bool, Captured],
    window: bool,
    path: RecordPath,
    field: str,
    invalid: Primitive,
) -> None:
    captured = exports[True]
    document = deepcopy(captured[1])
    base: RecordPath = ("summary", "windows", 0, "summary") if window else ("summary",)
    at(document, base + path)[field] = invalid
    with pytest.raises(ManifestError):
        inspect_record(captured, document)


@pytest.mark.parametrize("prediction", [False, True])
@pytest.mark.parametrize("invalid", [None, {}, [], "stability"])
def test_stability_requires_a_complete_record(
    exports: dict[bool, Captured], prediction: bool, invalid: Primitive
) -> None:
    captured = exports[prediction]
    document = deepcopy(captured[1])
    document["stability"] = invalid
    with pytest.raises(ManifestError):
        inspect_record(captured, document)


@pytest.mark.parametrize("prediction", [False, True])
@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("windows", None),
        ("windows", cast(Primitive, [{}])),
        ("comparable_transitions", True),
        ("configuration_changes", -1),
        ("repeat_selections", "1"),
        ("configuration_change_frequency", "2"),
        ("repeat_selection_frequency", "NaN"),
        ("selection_counts", {"candidate": True}),
        ("parameter_change_counts", {"parameter": -1}),
        ("interpretation", "foreign"),
    ],
)
def test_stability_domains(
    exports: dict[bool, Captured], prediction: bool, field: str, invalid: Primitive
) -> None:
    captured = exports[prediction]
    document = deepcopy(captured[1])
    at(document, ("stability",))[field] = invalid
    with pytest.raises(ManifestError):
        inspect_record(captured, document)


@pytest.mark.parametrize("prediction", [False, True])
def test_native_aggregate_schemas_remain_readable(
    exports: dict[bool, Captured], prediction: bool
) -> None:
    captured = exports[prediction]
    inspect_record(captured, captured[1])


@pytest.fixture(scope="module")
def custom_export(tmp_path_factory: pytest.TempPathFactory) -> Captured:
    root = tmp_path_factory.mktemp("custom-aggregate-schema")
    completed = complete_study(root, prediction=True)
    aggregate = aggregate_prediction(
        completed.source,
        PredictionMetricFields(
            mfe="signed_prediction_return",
            mae="signed_prediction_return",
            baseline_correct="direction_correct",
            baseline_name="captured_correctness",
        ),
    )
    return completed, aggregate.to_primitive(), root


def test_custom_metric_fields_and_available_baseline_remain_readable(
    custom_export: Captured,
) -> None:
    summary = at(custom_export[1], ("summary",))
    assert mapping(summary["matched_baseline"])["status"] == "available"
    assert mapping(summary["mfe"])["sample_count"] != 0
    inspect_record(custom_export, custom_export[1])
