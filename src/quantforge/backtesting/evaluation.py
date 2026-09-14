"""Select execution membership before accounting, retaining immutable provenance."""

from dataclasses import asdict, replace
from datetime import UTC, datetime

from quantforge.backtesting.config import EvaluationInterval
from quantforge.backtesting.corporate_actions import actions_by_session
from quantforge.backtesting.errors import InvalidMarketDataError
from quantforge.configuration import configuration_identity
from quantforge.data.corporate_actions import (
    action_seeds_from_records,
    bind_corporate_actions,
    corporate_action_snapshot_id,
)
from quantforge.data.identity import (
    calculate_dataset_id,
    serialize_bars_csv,
    sha256_hex,
)
from quantforge.data.models import CashDividend, MarketDataset, StockSplit


def _causal_history_metadata(history: MarketDataset) -> MarketDataset:
    """Rebind raw prefix metadata without exposing full-source snapshot facts.

    The view is synthetic and in-memory: epoch is an unavailable retrieval-time
    sentinel, and its raw digest identifies the projection rather than provider
    bytes. Canonical QF-3 identities/paths support defensive input validation;
    no corresponding cache artifacts are created. Source provenance stays on the
    enclosing backtest result, never in the strategy-facing view or action IDs.
    """
    final_session = history.bars[-1].session_date
    seeds = action_seeds_from_records(history.corporate_actions)
    snapshot_id = corporate_action_snapshot_id(seeds)
    data_digest = sha256_hex(serialize_bars_csv(history.bars))
    raw_digest = configuration_identity(
        {
            "component": "quantforge_backtest_causal_prefix",
            "contract_version": "1",
            "data_sha256": data_digest,
            "corporate_action_snapshot_id": snapshot_id,
        }
    )
    metadata = replace(
        history.metadata,
        retrieved_at=datetime(1970, 1, 1, tzinfo=UTC),
        requested_end=final_session,
        actual_last_session=final_session,
        bar_count=len(history.bars),
        missing_sessions=tuple(
            session
            for session in history.metadata.missing_sessions
            if session <= final_session
        ),
        split_sessions=tuple(
            action.effective_session
            for action in history.corporate_actions
            if isinstance(action, StockSplit)
        ),
        dividend_sessions=tuple(
            action.ex_dividend_session
            for action in history.corporate_actions
            if isinstance(action, CashDividend)
        ),
        corporate_action_count=len(seeds),
        split_count=sum(
            isinstance(action, StockSplit) for action in history.corporate_actions
        ),
        dividend_count=sum(
            isinstance(action, CashDividend) for action in history.corporate_actions
        ),
        corporate_action_snapshot_id=snapshot_id,
        raw_sha256=raw_digest,
        data_sha256=data_digest,
        adapter_version="qf43-causal-prefix-v1",
    )
    metadata_values = asdict(metadata)
    for field in (
        "raw_location",
        "normalized_location",
        "corporate_actions_location",
        "raw_sha256",
        "data_sha256",
        "dataset_id",
        "schema_version",
    ):
        del metadata_values[field]
    dataset_id = calculate_dataset_id(
        metadata_values,
        raw_sha256=raw_digest,
        data_sha256=data_digest,
        schema_version=metadata.schema_version,
    )
    metadata = replace(
        metadata,
        dataset_id=dataset_id,
        raw_location=f"raw/{raw_digest}.json",
        normalized_location=f"datasets/{dataset_id}/bars.csv",
        corporate_actions_location=f"datasets/{dataset_id}/corporate_actions.json",
    )
    return MarketDataset(
        history.bars,
        metadata,
        bind_corporate_actions(seeds, dataset_id=dataset_id, snapshot_id=snapshot_id),
    )


def select_backtest_datasets(
    dataset: MarketDataset, interval: EvaluationInterval | None
) -> tuple[MarketDataset, MarketDataset]:
    """Return strategy history and evaluation views of an already validated source.

    Strategy history has self-consistent, prefix-only metadata/action identities.
    The accounting view retains source provenance and is not an independently
    valid QF-3 dataset. Neither view creates cache artifacts.
    The caller must validate the original dataset before selecting membership.
    """
    if interval is None:
        return dataset, dataset
    sessions = {bar.session_date for bar in dataset.bars}
    if interval.start_session not in sessions or interval.end_session not in sessions:
        raise InvalidMarketDataError(
            "both evaluation endpoints must be observed source exchange sessions"
        )
    indexed_actions = actions_by_session(dataset)
    history_bars = tuple(
        bar for bar in dataset.bars if bar.session_date <= interval.end_session
    )
    evaluation_bars = tuple(
        bar for bar in history_bars if bar.session_date >= interval.start_session
    )
    history = replace(
        dataset,
        bars=history_bars,
        corporate_actions=tuple(
            action
            for bar in history_bars
            for action in indexed_actions.get(bar.session_date, ())
        ),
    )
    evaluation = replace(
        dataset,
        bars=evaluation_bars,
        corporate_actions=tuple(
            action
            for bar in evaluation_bars
            for action in indexed_actions.get(bar.session_date, ())
        ),
    )
    return _causal_history_metadata(history), evaluation
