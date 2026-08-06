"""Deterministic raw-price corporate-action accounting primitives."""

from datetime import date
from decimal import Decimal

from quantforge.backtesting._arithmetic import arithmetic
from quantforge.backtesting.config import DividendPolicy
from quantforge.backtesting.errors import PortfolioAccountingError
from quantforge.backtesting.models import (
    DividendAccountingSummary,
    DividendCashflowRecord,
    ReturnBasis,
    SplitAdjustmentRecord,
)
from quantforge.configuration import configuration_identity
from quantforge.data.models import (
    CashDividend,
    CorporateAction,
    MarketDataset,
    StockSplit,
)

PRICE_RETURN_ONLY_WARNING = (
    "PRICE-RETURN-ONLY: reported strategy and benchmark performance excludes "
    "cash dividends; long-period returns are understated relative to total return."
)


def actions_by_session(
    dataset: MarketDataset,
) -> dict[date, tuple[CorporateAction, ...]]:
    """Index already-validated immutable actions by effective session."""
    indexed: dict[date, list[CorporateAction]] = {}
    for action in dataset.corporate_actions:
        session = (
            action.ex_dividend_session
            if isinstance(action, CashDividend)
            else action.effective_session
        )
        indexed.setdefault(session, []).append(action)
    return {session: tuple(actions) for session, actions in indexed.items()}


def apply_split_action(
    *,
    run_id: str,
    account_id: str,
    action: StockSplit,
    shares: int,
    total_cost_basis: Decimal,
    cash: Decimal,
) -> tuple[int, SplitAdjustmentRecord]:
    """Multiply shares by Tiingo's shares-after/shares-before factor."""
    with arithmetic():
        exact_shares_after = Decimal(shares) * action.split_factor
        integral_shares_after = exact_shares_after.to_integral_value()
        if exact_shares_after != integral_shares_after:
            raise PortfolioAccountingError(
                "stock split would create fractional shares; "
                "cash-in-lieu is unsupported"
            )
        shares_after = int(integral_shares_after)
        average_before = None if shares == 0 else total_cost_basis / Decimal(shares)
        average_after = (
            None if shares_after == 0 else total_cost_basis / Decimal(shares_after)
        )
    adjustment_id = configuration_identity(
        {
            "run_id": run_id,
            "account_id": account_id,
            "record_type": "split_adjustment",
            "corporate_action_id": action.action_id,
        }
    )
    return shares_after, SplitAdjustmentRecord(
        split_adjustment_id=adjustment_id,
        run_id=run_id,
        account_id=account_id,
        corporate_action_id=action.action_id,
        symbol=action.symbol,
        effective_session=action.effective_session,
        split_factor=action.split_factor,
        shares_before=shares,
        shares_after=shares_after,
        average_entry_cost_before=average_before,
        average_entry_cost_after=average_after,
        total_cost_basis_before=total_cost_basis,
        total_cost_basis_after=total_cost_basis,
        resulting_cash_balance=cash,
        source_dataset_id=action.source_dataset_id,
    )


def credit_dividend_action(
    *,
    run_id: str,
    account_id: str,
    action: CashDividend,
    entitled_shares: int,
    cash: Decimal,
) -> tuple[Decimal, DividendCashflowRecord]:
    """Credit one ex-date cashflow without changing price-trade proceeds."""
    with arithmetic():
        total_cash = Decimal(entitled_shares) * action.amount_per_share
        resulting_cash = cash + total_cash
    cashflow_id = configuration_identity(
        {
            "run_id": run_id,
            "account_id": account_id,
            "record_type": "dividend_cashflow",
            "corporate_action_id": action.action_id,
        }
    )
    return resulting_cash, DividendCashflowRecord(
        dividend_cashflow_id=cashflow_id,
        run_id=run_id,
        account_id=account_id,
        corporate_action_id=action.action_id,
        symbol=action.symbol,
        ex_dividend_session=action.ex_dividend_session,
        entitled_share_quantity=entitled_shares,
        amount_per_share=action.amount_per_share,
        total_dividend_cash=total_cash,
        resulting_cash_balance=resulting_cash,
        source_dataset_id=action.source_dataset_id,
    )


def apply_dividend_policy(
    *,
    policy: DividendPolicy,
    run_id: str,
    account_id: str,
    action: CashDividend,
    entitled_shares: int,
    cash: Decimal,
) -> tuple[Decimal, DividendCashflowRecord | None, Decimal]:
    """Apply or disclose one dividend without changing split semantics."""
    if policy is DividendPolicy.CASH_DIVIDENDS:
        resulting_cash, cashflow = credit_dividend_action(
            run_id=run_id,
            account_id=account_id,
            action=action,
            entitled_shares=entitled_shares,
            cash=cash,
        )
        return resulting_cash, cashflow, Decimal(0)
    if policy is DividendPolicy.PRICE_RETURN_ONLY:
        with arithmetic():
            estimated_ignored_cash = Decimal(entitled_shares) * action.amount_per_share
        return cash, None, estimated_ignored_cash
    raise PortfolioAccountingError(
        "reject-if-dividends policy reached dividend accounting unexpectedly"
    )


def summarize_dividend_accounting(
    *,
    policy: DividendPolicy,
    corporate_action_snapshot_id: str,
    actions: tuple[CashDividend, ...],
    cashflows: tuple[DividendCashflowRecord, ...],
    estimated_ignored_cash: Decimal,
) -> DividendAccountingSummary:
    """Describe applied and intentionally excluded dividend economics."""
    with arithmetic():
        total_cash_credited = sum(
            (item.total_dividend_cash for item in cashflows), start=Decimal(0)
        )
        ignored_amount_per_share = (
            sum((item.amount_per_share for item in actions), start=Decimal(0))
            if policy is DividendPolicy.PRICE_RETURN_ONLY
            else Decimal(0)
        )
    price_only = policy is not DividendPolicy.CASH_DIVIDENDS
    events_present = len(actions)
    return DividendAccountingSummary(
        dividend_policy=policy,
        return_basis=(
            ReturnBasis.PRICE_RETURN
            if price_only
            else ReturnBasis.TOTAL_RETURN_WITH_CASH_DIVIDENDS
        ),
        corporate_action_snapshot_id=corporate_action_snapshot_id,
        dividend_events_present=events_present,
        dividend_events_credited=len(cashflows),
        dividend_events_ignored=(
            events_present if policy is DividendPolicy.PRICE_RETURN_ONLY else 0
        ),
        total_dividend_cash_credited=total_cash_credited,
        total_ignored_dividend_amount_per_share=ignored_amount_per_share,
        estimated_ignored_dividend_cash=(
            estimated_ignored_cash
            if policy is DividendPolicy.PRICE_RETURN_ONLY
            else Decimal(0)
        ),
        warning=(
            PRICE_RETURN_ONLY_WARNING
            if policy is DividendPolicy.PRICE_RETURN_ONLY and actions
            else None
        ),
    )
