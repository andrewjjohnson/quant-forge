"""Validate recorded QF-5 configuration without constructing an executable run."""

from datetime import date
from decimal import Decimal

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text


def _decimal(value: object, minimum: int, *, strict: bool = False) -> Decimal:
    number = Decimal(text(value))
    if not number.is_finite() or number < minimum or (strict and number == minimum):
        raise ValueError("decimal is outside the configuration domain")
    return number


def _cost(value: object, category: str) -> None:
    configuration = mapping(value)
    version = text(configuration.get("implementation_version"))
    guarantee = "buy_cost_is_non_decreasing_by_quantity"
    if category != "slippage" and configuration.get(guarantee) is not True:
        raise ValueError("cost configuration lacks its monotonicity guarantee")
    native = {
        "fixed_per_fill": ("commission", {"amount"}),
        "per_share": ("commission", {"amount_per_share", "minimum"}),
        "basis_points": ("commission", {"basis_points"}),
        "explicit_zero_fees": ("fees", set[str]()),
        "basis_point_fees": ("fees", {"basis_points"}),
        "adverse_basis_points": ("slippage", {"basis_points"}),
    }
    model = configuration.get("model")
    if version != "1" or not isinstance(model, str) or model not in native:
        # Custom models own their primitive configuration. QF-5 guarantees only
        # an explicit implementation version and (for buy costs) monotonicity.
        return
    expected_category, parameters = native[model]
    keys = {"model", "implementation_version", "parameters"}
    if category != "slippage":
        keys.add(guarantee)
    recorded = mapping(configuration.get("parameters"))
    if (
        expected_category != category
        or set(configuration) != keys
        or set(recorded) != parameters
    ):
        raise ValueError("native cost configuration differs from its schema")
    for value in recorded.values():
        amount = _decimal(value, 0)
        if category == "slippage" and amount >= 10_000:
            raise ValueError("slippage must be below 10000 basis points")


def validate_backtest_configuration(configuration: PrimitiveMapping) -> None:
    """Check supported fixed metadata and native/custom cost recording contracts."""
    from quantforge.backtesting._arithmetic import arithmetic_configuration
    from quantforge.backtesting.config import (
        DiscreteTargetWeightSizing,
        DividendPolicy,
        EvaluationInterval,
        NextSessionOpenExecution,
        SplitAccountingPolicy,
    )

    expected: PrimitiveMapping = {
        "execution": NextSessionOpenExecution().to_primitive(),
        "sizing": DiscreteTargetWeightSizing().to_primitive(),
        "split_policy": SplitAccountingPolicy().to_primitive(),
        "arithmetic": arithmetic_configuration(),
        "engine_version": "4",
        "result_schema_version": "3",
        "long_only": True,
        "forced_liquidation": False,
        "dividend_entitlement": "previous_session_close_shares",
        "dividend_credit_timing": "after_open_execution_before_close_mark",
        "trade_dividend_attribution": "total_economic_pnl_separate_from_price_pnl",
    }
    keys = set(expected) | {
        "initial_capital",
        "commission",
        "fees",
        "slippage",
        "dividend_policy",
        "annual_risk_free_rate",
        "annualization_factor",
    }
    try:
        if "evaluation_interval" in configuration:
            interval = mapping(configuration["evaluation_interval"])
            expected["evaluation_interval"] = EvaluationInterval(
                date.fromisoformat(text(interval.get("start_session"))),
                date.fromisoformat(text(interval.get("end_session"))),
            ).to_primitive()
            keys.add("evaluation_interval")
        if set(configuration) != keys or configuration_identity(
            {key: configuration.get(key) for key in expected}
        ) != configuration_identity(expected):
            raise ValueError("fixed execution metadata differs from producer schema")
        _decimal(configuration["initial_capital"], 0, strict=True)
        _decimal(configuration["annual_risk_free_rate"], -1, strict=True)
        factor = configuration["annualization_factor"]
        if type(factor) is not int or factor <= 0:
            raise ValueError("annualization factor must be a positive integer")
        if configuration["dividend_policy"] not in tuple(DividendPolicy):
            raise ValueError("unsupported dividend policy")
        for category in ("commission", "fees", "slippage"):
            _cost(configuration[category], category)
    except (ValueError, ArithmeticError) as error:
        raise ManifestError("backtest execution configuration is invalid") from error
