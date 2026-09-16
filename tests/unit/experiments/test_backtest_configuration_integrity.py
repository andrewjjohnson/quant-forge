"""Rehashed QF-5 exports still require complete execution assumptions."""

from copy import deepcopy
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from quantforge.backtesting.config import EvaluationInterval
from quantforge.backtesting.costs import (
    BasisPointCommission,
    BasisPointFees,
    BasisPointSlippage,
    ExplicitZeroFees,
    FixedCommission,
    PerShareCommission,
)
from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments._json import mapping
from tests.unit.experiments.test_backtest_manifest_schema import export as export
from tests.unit.experiments.test_backtest_manifest_schema import inspect_manifest
from tests.unit.experiments.test_backtest_sizing_integrity import reidentify
from tests.unit.experiments.test_grid_integrity import read_record


def inspect_configuration(
    export: Path, configuration: PrimitiveMapping, *, reject: bool = True
) -> Path:
    manifest = read_record(export / "manifest.json")
    manifest["backtest_configuration"] = configuration
    benchmark = mapping(mapping(manifest["benchmark"])["configuration"])
    for field in (
        "initial_capital",
        "commission",
        "fees",
        "slippage",
        "dividend_policy",
        "split_policy",
    ):
        benchmark[field] = configuration.get(field)
    if "evaluation_interval" in configuration:
        benchmark["start"] = "first_evaluation_session_open"
        benchmark["evaluation_interval"] = configuration["evaluation_interval"]
    else:
        benchmark["start"] = "first_dataset_session_open"
        benchmark.pop("evaluation_interval", None)
    renamed = export.rename(export.with_name(reidentify(manifest)))
    if not reject:
        from tests.unit.experiments.backtest_fixture_records import (
            refresh_export_records,
        )

        refresh_export_records(renamed, manifest)
    inspect_manifest(renamed, manifest, reject=reject)
    return renamed


@pytest.mark.parametrize(
    "section",
    [None, "execution", "sizing", "split_policy", "arithmetic", "evaluation_interval"],
)
def test_execution_configuration_requires_complete_fixed_records(
    export: Path, section: str | None
) -> None:
    original = mapping(read_record(export / "manifest.json")["backtest_configuration"])
    if section == "evaluation_interval":
        original[section] = EvaluationInterval(
            date(2024, 1, 2), date(2024, 1, 3)
        ).to_primitive()
    target = original if section is None else mapping(original[section])
    for field in (*target, "undeclared"):
        configuration = deepcopy(original)
        changed = configuration if section is None else mapping(configuration[section])
        if field == "undeclared":
            changed[field] = None
        else:
            del changed[field]
        export = inspect_configuration(export, configuration)


@pytest.mark.parametrize(
    "field",
    [
        "execution",
        "split_policy",
        "arithmetic",
        "commission",
        "fees",
        "slippage",
        "evaluation_interval",
    ],
)
@pytest.mark.parametrize("invalid", [None, {}, [], False, "invalid"])
def test_execution_nested_records_reject_malformed_values(
    export: Path, field: str, invalid: Primitive
) -> None:
    configuration = mapping(
        read_record(export / "manifest.json")["backtest_configuration"]
    )
    configuration[field] = invalid
    inspect_configuration(export, configuration)


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("initial_capital",), "0"),
        (("initial_capital",), "NaN"),
        (("initial_capital",), 1000),
        (("annual_risk_free_rate",), "-1"),
        (("annual_risk_free_rate",), "Infinity"),
        (("annual_risk_free_rate",), False),
        (("annualization_factor",), 0),
        (("annualization_factor",), True),
        (("annualization_factor",), "252"),
        (("long_only",), 1),
        (("long_only",), False),
        (("forced_liquidation",), 0),
        (("forced_liquidation",), True),
        (("engine_version",), "5"),
        (("result_schema_version",), "4"),
        (("dividend_policy",), "unknown"),
        (("dividend_entitlement",), "current_open"),
        (("dividend_credit_timing",), "before_open"),
        (("trade_dividend_attribution",), "price_only"),
        (("execution", "timing"), "same_close"),
        (("execution", "price_field"), "close"),
        (("execution", "order_type"), "limit"),
        (("split_policy", "maximum_split_ratio_denominator"), True),
        (("arithmetic", "decimal_precision"), 28),
        (("arithmetic", "clamp"), False),
        (("commission", "implementation_version"), ""),
        (("fees", "buy_cost_is_non_decreasing_by_quantity"), 1),
        (("commission", "parameters", "amount"), "-1"),
        (("fees", "parameters", "extra"), "0"),
        (("slippage", "parameters", "basis_points"), "10000"),
        (("slippage", "parameters", "basis_points"), False),
    ],
)
def test_execution_configuration_enforces_fixed_semantics_and_domains(
    export: Path, path: tuple[str, ...], invalid: Primitive
) -> None:
    configuration = mapping(
        read_record(export / "manifest.json")["backtest_configuration"]
    )
    target = configuration
    for key in path[:-1]:
        target = mapping(target[key])
    target[path[-1]] = invalid
    inspect_configuration(export, configuration)


@pytest.mark.parametrize("field", ["commission", "fees", "slippage"])
def test_cost_records_require_their_complete_native_fields(
    export: Path, field: str
) -> None:
    original = mapping(read_record(export / "manifest.json")["backtest_configuration"])
    for section in ((), ("parameters",)):
        target = mapping(original[field])
        if section:
            target = mapping(target[section[0]])
        for key in (*target, "undeclared"):
            if not section and key == "model":
                # Generic QF-5 costs need not serialize a model name; losing it
                # cannot be distinguished from a custom producer-owned record.
                continue
            configuration = deepcopy(original)
            changed = mapping(configuration[field])
            if section:
                changed = mapping(changed[section[0]])
            if key == "undeclared":
                changed[key] = None
            else:
                del changed[key]
            export = inspect_configuration(export, configuration)


@pytest.mark.parametrize("bounded", [False, True])
def test_native_cost_variants_and_historical_interval_absence_remain_valid(
    export: Path, bounded: bool
) -> None:
    configuration = mapping(
        read_record(export / "manifest.json")["backtest_configuration"]
    )
    if bounded:
        configuration["evaluation_interval"] = EvaluationInterval(
            date(2024, 1, 2), date(2024, 1, 3)
        ).to_primitive()
    for commission in (
        FixedCommission(Decimal(1)),
        PerShareCommission(Decimal("0.01"), Decimal(1)),
        BasisPointCommission(Decimal(5)),
    ):
        configuration["commission"] = commission.configuration()
        configuration["fees"] = (
            BasisPointFees(Decimal(2)) if bounded else ExplicitZeroFees()
        ).configuration()
        configuration["slippage"] = BasisPointSlippage(Decimal(3)).configuration()
        export = inspect_configuration(export, configuration, reject=False)


def test_custom_cost_configuration_keeps_its_producer_owned_shape(export: Path) -> None:
    configuration = mapping(
        read_record(export / "manifest.json")["backtest_configuration"]
    )
    for field in ("commission", "fees", "slippage"):
        custom: PrimitiveMapping = {
            "implementation_version": "1",
            "custom_schedule": [{"threshold": 10, "charge": "0.25"}],
        }
        if field != "slippage":
            custom["buy_cost_is_non_decreasing_by_quantity"] = True
        configuration[field] = custom
    inspect_configuration(export, configuration, reject=False)
