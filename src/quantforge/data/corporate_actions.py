"""Stable normalized corporate-action construction and identity."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data.exceptions import ValidationError
from quantforge.data.models import CashDividend, CorporateAction, StockSplit

CORPORATE_ACTION_SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class CashDividendSeed:
    """Normalized dividend awaiting its source QF-3 dataset identity."""

    symbol: str
    ex_dividend_session: date
    amount_per_share: Decimal
    provider_name: str


@dataclass(frozen=True, slots=True)
class StockSplitSeed:
    """Normalized split awaiting its source QF-3 dataset identity."""

    symbol: str
    effective_session: date
    split_factor: Decimal
    provider_name: str


type CorporateActionSeed = CashDividendSeed | StockSplitSeed


def _seed_primitive(seed: CorporateActionSeed) -> PrimitiveMapping:
    if isinstance(seed, CashDividendSeed):
        return {
            "action_type": "cash_dividend",
            "symbol": seed.symbol,
            "ex_dividend_session": seed.ex_dividend_session.isoformat(),
            "amount_per_share": str(seed.amount_per_share),
            "provider_name": seed.provider_name,
        }
    return {
        "action_type": "stock_split",
        "symbol": seed.symbol,
        "effective_session": seed.effective_session.isoformat(),
        "split_factor": str(seed.split_factor),
        "provider_name": seed.provider_name,
    }


def _seed_key(seed: CorporateActionSeed) -> tuple[date, str, str]:
    if isinstance(seed, CashDividendSeed):
        return seed.ex_dividend_session, "cash_dividend", seed.symbol
    return seed.effective_session, "stock_split", seed.symbol


def validate_action_seeds(
    seeds: tuple[CorporateActionSeed, ...],
) -> tuple[CorporateActionSeed, ...]:
    """Return unique chronological seeds after validating economic values."""
    ordered = tuple(sorted(seeds, key=_seed_key))
    keys = tuple(_seed_key(seed) for seed in ordered)
    if len(keys) != len(set(keys)):
        raise ValidationError("duplicate corporate actions")
    for seed in ordered:
        if isinstance(seed, CashDividendSeed):
            if not seed.amount_per_share.is_finite() or seed.amount_per_share <= 0:
                raise ValidationError(
                    "cash-dividend actions must have a positive finite amount"
                )
        elif not seed.split_factor.is_finite() or seed.split_factor <= 0:
            raise ValidationError("stock-split actions must have a positive factor")
        elif seed.split_factor == Decimal(1):
            raise ValidationError("neutral split factors must not create actions")
    return ordered


def corporate_action_snapshot_id(
    seeds: tuple[CorporateActionSeed, ...],
) -> str:
    """Fingerprint ordered action economics independently of dataset paths."""
    ordered = validate_action_seeds(seeds)
    return configuration_identity(
        {
            "component": "quantforge_corporate_action_snapshot",
            "schema_version": CORPORATE_ACTION_SCHEMA_VERSION,
            "actions": [_seed_primitive(seed) for seed in ordered],
        }
    )


def bind_corporate_actions(
    seeds: tuple[CorporateActionSeed, ...],
    *,
    dataset_id: str,
    snapshot_id: str,
) -> tuple[CorporateAction, ...]:
    """Bind normalized seeds to their immutable source dataset."""
    ordered = validate_action_seeds(seeds)
    if corporate_action_snapshot_id(ordered) != snapshot_id:
        raise ValidationError("corporate-action snapshot identity mismatch")
    actions: list[CorporateAction] = []
    for seed in ordered:
        source = _seed_primitive(seed)
        action_id = configuration_identity(
            {
                "component": "quantforge_corporate_action",
                "schema_version": CORPORATE_ACTION_SCHEMA_VERSION,
                "corporate_action_snapshot_id": snapshot_id,
                "source_dataset_id": dataset_id,
                "action": source,
            }
        )
        if isinstance(seed, CashDividendSeed):
            actions.append(
                CashDividend(
                    action_id=action_id,
                    symbol=seed.symbol,
                    ex_dividend_session=seed.ex_dividend_session,
                    amount_per_share=seed.amount_per_share,
                    provider_name=seed.provider_name,
                    source_dataset_id=dataset_id,
                )
            )
        else:
            actions.append(
                StockSplit(
                    action_id=action_id,
                    symbol=seed.symbol,
                    effective_session=seed.effective_session,
                    split_factor=seed.split_factor,
                    provider_name=seed.provider_name,
                    source_dataset_id=dataset_id,
                )
            )
    return tuple(actions)


def action_seeds_from_records(
    actions: tuple[CorporateAction, ...],
) -> tuple[CorporateActionSeed, ...]:
    """Recover snapshot inputs from bound public corporate-action records."""
    seeds: list[CorporateActionSeed] = []
    for action in actions:
        if isinstance(action, CashDividend):
            seeds.append(
                CashDividendSeed(
                    action.symbol,
                    action.ex_dividend_session,
                    action.amount_per_share,
                    action.provider_name,
                )
            )
        else:
            seeds.append(
                StockSplitSeed(
                    action.symbol,
                    action.effective_session,
                    action.split_factor,
                    action.provider_name,
                )
            )
    return tuple(seeds)


def corporate_actions_primitive(
    actions: tuple[CorporateAction, ...], snapshot_id: str
) -> PrimitiveMapping:
    """Return the canonical persisted corporate-action snapshot."""
    return {
        "schema_version": CORPORATE_ACTION_SCHEMA_VERSION,
        "corporate_action_snapshot_id": snapshot_id,
        "actions": [action.to_primitive() for action in actions],
    }


def corporate_actions_list(
    actions: tuple[CorporateAction, ...],
) -> list[Primitive]:
    """Return detached primitive action rows for result provenance."""
    return [action.to_primitive() for action in actions]
