"""Small valid QF-3 datasets for deterministic unit tests."""

from datetime import UTC, date, datetime
from decimal import Decimal

from quantforge.data.identity import (
    calculate_dataset_id,
    canonical_json_bytes,
    serialize_bars_csv,
    sha256_hex,
)
from quantforge.data.models import (
    SCHEMA_VERSION,
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
    calendar: str = "XNYS",
    requested_start: date | None = None,
    requested_end: date | None = None,
    missing_sessions: tuple[date, ...] = (),
    split_sessions: tuple[date, ...] = (),
    dividend_sessions: tuple[date, ...] = (),
) -> MarketDataset:
    """Build identity-bound daily bars; ``dataset_id`` is a raw-fixture seed."""
    selected_sessions = SESSIONS[: len(closes)] if sessions is None else sessions
    assert len(selected_sessions) == len(closes)
    selected_start = (
        selected_sessions[0] if requested_start is None else requested_start
    )
    selected_end = selected_sessions[-1] if requested_end is None else requested_end
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
    typed_bars = tuple(bars)
    retrieved_at = datetime(2024, 7, 15, tzinfo=UTC)
    metadata_values: dict[str, object] = {
        "canonical_symbol": "SPY",
        "provider_name": "synthetic",
        "provider_symbol": "SPY",
        "retrieved_at": retrieved_at,
        "requested_start": selected_start,
        "requested_end": selected_end,
        "actual_first_session": selected_sessions[0],
        "actual_last_session": selected_sessions[-1],
        "calendar": calendar,
        "provider_timezone": "America/New_York",
        "adjustment_mode": adjustment_mode,
        "bar_count": len(bars),
        "missing_sessions": missing_sessions,
        "split_sessions": split_sessions,
        "dividend_sessions": dividend_sessions,
        "adapter_version": "test-1",
    }
    raw_sha256 = sha256_hex(canonical_json_bytes({"synthetic_raw_fixture": dataset_id}))
    data_sha256 = sha256_hex(serialize_bars_csv(typed_bars))
    calculated_dataset_id = calculate_dataset_id(
        metadata_values,
        raw_sha256=raw_sha256,
        data_sha256=data_sha256,
        schema_version=SCHEMA_VERSION,
    )
    metadata = DatasetMetadata(
        canonical_symbol="SPY",
        provider_name="synthetic",
        provider_symbol="SPY",
        retrieved_at=retrieved_at,
        requested_start=selected_start,
        requested_end=selected_end,
        actual_first_session=selected_sessions[0],
        actual_last_session=selected_sessions[-1],
        calendar=calendar,
        provider_timezone="America/New_York",
        adjustment_mode=adjustment_mode,
        raw_location=f"raw/{raw_sha256}.json",
        normalized_location=f"datasets/{calculated_dataset_id}/bars.csv",
        raw_sha256=raw_sha256,
        data_sha256=data_sha256,
        dataset_id=calculated_dataset_id,
        schema_version=SCHEMA_VERSION,
        bar_count=len(bars),
        missing_sessions=missing_sessions,
        split_sessions=split_sessions,
        dividend_sessions=dividend_sessions,
        adapter_version="test-1",
    )
    return MarketDataset(typed_bars, metadata)
