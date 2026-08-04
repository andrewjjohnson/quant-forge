"""Small valid QF-3 datasets for deterministic unit tests."""

from datetime import UTC, date, datetime
from decimal import Decimal

from quantforge.data.models import (
    AdjustmentMode,
    DailyBar,
    DatasetMetadata,
    MarketDataset,
)

SESSIONS = (
    date(2024, 7, 1),
    date(2024, 7, 2),
    date(2024, 7, 3),
    date(2024, 7, 5),
    date(2024, 7, 8),
    date(2024, 7, 9),
    date(2024, 7, 10),
    date(2024, 7, 11),
    date(2024, 7, 12),
)


def make_dataset(
    closes: tuple[str, ...],
    *,
    sessions: tuple[date, ...] | None = None,
    dataset_id: str = "synthetic-dataset",
    adjustment_mode: AdjustmentMode = AdjustmentMode.UNADJUSTED,
) -> MarketDataset:
    """Build canonical in-memory daily bars with explicit QF-3 metadata."""
    selected_sessions = SESSIONS[: len(closes)] if sessions is None else sessions
    assert len(selected_sessions) == len(closes)
    bars: list[DailyBar] = []
    for session, close_text in zip(selected_sessions, closes, strict=True):
        close = Decimal(close_text)
        comparison_close = close if close.is_finite() else Decimal(100)
        bars.append(
            DailyBar(
                "SPY",
                session,
                comparison_close,
                comparison_close + Decimal(1),
                comparison_close,
                close,
                Decimal(1000),
            )
        )
    metadata = DatasetMetadata(
        canonical_symbol="SPY",
        provider_name="synthetic",
        provider_symbol="SPY",
        retrieved_at=datetime(2024, 7, 15, tzinfo=UTC),
        requested_start=selected_sessions[0],
        requested_end=selected_sessions[-1],
        actual_first_session=selected_sessions[0],
        actual_last_session=selected_sessions[-1],
        calendar="XNYS",
        provider_timezone="America/New_York",
        adjustment_mode=adjustment_mode,
        raw_location="raw/synthetic.json",
        normalized_location="datasets/synthetic/bars.csv",
        dataset_id=dataset_id,
        schema_version="1",
        bar_count=len(bars),
        missing_sessions=(),
        adapter_version="test-1",
    )
    return MarketDataset(tuple(bars), metadata)
