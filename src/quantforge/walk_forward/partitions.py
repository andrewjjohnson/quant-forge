"""Adapt QF-8 membership to bounded, independently valid evaluator inputs."""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from typing import Protocol, cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import (
    CashDividend,
    MarketDataset,
    StockSplit,
    validate_market_dataset,
)
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
from quantforge.timeframes import resolve_exchange_session
from quantforge.validation import (
    BoundaryAxis,
    DatasetProvenance,
    ExchangeSessionBoundary,
    PartitionRole,
    PurgedPartitionObservations,
    TimestampBoundary,
    ValidationBoundary,
    ValidationPlan,
    ValidationWindow,
    WindowObservationSelection,
    purge_partition_observations,
    select_window_observations,
)
from quantforge.walk_forward.models import WalkForwardError


class EvaluationPartition(Protocol):
    """Validated boundary shared by test and explicitly consumed holdout adapters."""

    @property
    def window(self) -> ValidationWindow: ...

    @property
    def dataset(self) -> MarketDataset: ...

    @property
    def sessions(self) -> tuple[date, ...]: ...

    def to_primitive(self) -> PrimitiveMapping: ...


@dataclass(frozen=True, slots=True)
class PermittedPartition:
    window: ValidationWindow
    membership: WindowObservationSelection
    purge: PurgedPartitionObservations
    dataset: MarketDataset
    sessions: tuple[date, ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "window": self.window.to_primitive(),
            "membership": self.membership.to_primitive(),
            "purge": self.purge.to_primitive(),
            "evaluation_sessions": [session.isoformat() for session in self.sessions],
            "bounded_dataset_id": self.dataset.metadata.dataset_id,
            "bounded_data_sha256": self.dataset.metadata.data_sha256,
        }


def observation_keys(
    dataset: MarketDataset, plan: ValidationPlan
) -> tuple[ValidationBoundary, ...]:
    timeframe = plan.environment.outcome_dataset.standalone_timeframe
    if timeframe is None:
        raise WalkForwardError("the existing evaluators require a QF-3 study dataset")
    if plan.axis is BoundaryAxis.EXCHANGE_SESSION:
        return tuple(
            ExchangeSessionBoundary(bar.session_date, timeframe.session_policy)
            for bar in dataset.bars
        )
    return tuple(
        TimestampBoundary(
            resolve_exchange_session(
                bar.session_date, timeframe.session_policy
            ).close_timestamp
        )
        for bar in dataset.bars
    )


def partition(
    dataset: MarketDataset,
    plan: ValidationPlan,
    fold_index: int,
    role: PartitionRole,
    *,
    minimum_observations: int,
) -> PermittedPartition:
    """Delegate all boundary/purge/context decisions to QF-8, then project rows."""
    if role is PartitionRole.FINAL_HOLDOUT:
        raise WalkForwardError("QF-39 never consumes the reserved final holdout")
    if (
        DatasetProvenance.from_market_dataset(dataset)
        != plan.environment.outcome_dataset
    ):
        raise WalkForwardError("dataset does not match the validation plan")
    fold = plan.folds[fold_index]
    window = {
        PartitionRole.DEVELOPMENT: fold.development,
        PartitionRole.SELECTION: fold.selection,
        PartitionRole.WALK_FORWARD_TEST: fold.test,
    }[role]
    if window is None:
        raise WalkForwardError("this fold has no selection partition")
    observations = observation_keys(dataset, plan)
    membership = select_window_observations(
        window,
        observations,
        source=dataset,
        source_timeframe=plan.environment.outcome_dataset.standalone_timeframe,
    )
    purged = purge_partition_observations(
        plan,
        fold_index,
        role,
        observations,
        source=dataset,
    )
    if len(purged.retained) < minimum_observations:
        raise WalkForwardError(f"insufficient {role.value} observations after purging")
    start_key = (membership.warm_up_context or purged.retained)[0]
    start = observations.index(start_key)
    final_decision = observations.index(purged.retained[-1])
    # QF-8 already certified that this maximum horizon ends before protection.
    horizon = plan.purge_policy.label_horizon.exchange_sessions or 0
    end = final_decision + horizon
    if end >= len(observations):
        raise WalkForwardError("insufficient outcome observations")
    sessions = tuple(
        dataset.bars[observations.index(key)].session_date for key in purged.retained
    )
    return PermittedPartition(
        window,
        membership,
        purged,
        project_dataset(
            dataset, dataset.bars[start].session_date, dataset.bars[end].session_date
        ),
        sessions,
    )


def project_dataset(dataset: MarketDataset, start: date, end: date) -> MarketDataset:
    """QF-3 in-memory projection, with no protected metadata or price references.

    Bars are copied unchanged. Only range/content/action provenance is rebound,
    using QF-3 serialization. Original immutable lineage lives on the QF-39 study.
    This is neither an ingestion operation nor a new execution model.
    """
    bars = tuple(bar for bar in dataset.bars if start <= bar.session_date <= end)
    if not bars or bars[0].session_date != start or bars[-1].session_date != end:
        raise WalkForwardError("projection endpoints must be observed sessions")
    actions = tuple(
        action
        for action in dataset.corporate_actions
        if start
        <= (
            action.ex_dividend_session
            if isinstance(action, CashDividend)
            else action.effective_session
        )
        <= end
    )
    seeds = action_seeds_from_records(actions)
    snapshot_id = corporate_action_snapshot_id(seeds)
    digest = sha256_hex(serialize_bars_csv(bars))
    raw_digest = configuration_identity(
        {
            "component": "quantforge_walk_forward_projection",
            "version": "1",
            "data_sha256": digest,
            "corporate_action_snapshot_id": snapshot_id,
        }
    )
    metadata = replace(
        dataset.metadata,
        requested_start=start,
        requested_end=end,
        actual_first_session=start,
        actual_last_session=end,
        bar_count=len(bars),
        retrieved_at=datetime(1970, 1, 1, tzinfo=UTC),
        missing_sessions=tuple(
            s for s in dataset.metadata.missing_sessions if start <= s <= end
        ),
        split_sessions=tuple(
            a.effective_session for a in actions if isinstance(a, StockSplit)
        ),
        dividend_sessions=tuple(
            a.ex_dividend_session for a in actions if isinstance(a, CashDividend)
        ),
        corporate_action_count=len(actions),
        split_count=sum(isinstance(a, StockSplit) for a in actions),
        dividend_count=sum(isinstance(a, CashDividend) for a in actions),
        corporate_action_snapshot_id=snapshot_id,
        raw_sha256=raw_digest,
        data_sha256=digest,
        adapter_version="qf39-bounded-projection-v1",
    )
    values = asdict(metadata)
    for name in (
        "raw_location",
        "normalized_location",
        "corporate_actions_location",
        "raw_sha256",
        "data_sha256",
        "dataset_id",
        "schema_version",
    ):
        del values[name]
    dataset_id = calculate_dataset_id(
        cast(dict[str, object], values),
        raw_sha256=raw_digest,
        data_sha256=digest,
        schema_version=metadata.schema_version,
    )
    projected = MarketDataset(
        bars,
        replace(
            metadata,
            dataset_id=dataset_id,
            raw_location=f"raw/{raw_digest}.json",
            normalized_location=f"datasets/{dataset_id}/bars.csv",
            corporate_actions_location=f"datasets/{dataset_id}/corporate_actions.json",
        ),
        bind_corporate_actions(seeds, dataset_id=dataset_id, snapshot_id=snapshot_id),
    )
    validate_market_dataset(projected)
    return projected
