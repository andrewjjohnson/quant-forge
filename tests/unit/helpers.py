"""Small valid QF-3 datasets for deterministic unit tests."""

from datetime import UTC, date, datetime
from decimal import Decimal

from quantforge.data.corporate_actions import (
    CashDividendSeed,
    StockSplitSeed,
    bind_corporate_actions,
    corporate_action_snapshot_id,
)
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
    date(2024, 7, 15),
    date(2024, 7, 16),
    date(2024, 7, 17),
    date(2024, 7, 18),
    date(2024, 7, 19),
    date(2024, 7, 22),
)


def make_dataset(
    closes: tuple[str, ...],
    *,
    sessions: tuple[date, ...] | None = None,
    dataset_id: str = "synthetic-dataset",
    adjustment_mode: AdjustmentMode = AdjustmentMode.UNADJUSTED,
    adjusted_fields_used: bool = False,
    calendar: str = "XNYS",
    requested_start: date | None = None,
    requested_end: date | None = None,
    missing_sessions: tuple[date, ...] = (),
    split_sessions: tuple[date, ...] = (),
    dividend_sessions: tuple[date, ...] = (),
    dividends: tuple[tuple[date, str], ...] = (),
    splits: tuple[tuple[date, str], ...] = (),
    corporate_actions_complete: bool = True,
    opens: tuple[str, ...] | None = None,
    highs: tuple[str, ...] | None = None,
    lows: tuple[str, ...] | None = None,
    volumes: tuple[str, ...] | None = None,
) -> MarketDataset:
    """Build identity-bound daily bars; ``dataset_id`` is a raw-fixture seed."""
    selected_sessions = SESSIONS[: len(closes)] if sessions is None else sessions
    assert len(selected_sessions) == len(closes)
    assert opens is None or len(opens) == len(closes)
    assert highs is None or len(highs) == len(closes)
    assert lows is None or len(lows) == len(closes)
    assert volumes is None or len(volumes) == len(closes)
    selected_start = (
        selected_sessions[0] if requested_start is None else requested_start
    )
    selected_end = selected_sessions[-1] if requested_end is None else requested_end
    bars: list[DailyBar] = []
    for index, (session, close_text) in enumerate(
        zip(selected_sessions, closes, strict=True)
    ):
        close = Decimal(close_text)
        comparison_close = close if close.is_finite() else Decimal(100)
        session_open = comparison_close if opens is None else Decimal(opens[index])
        session_high = (
            comparison_close + Decimal(1) if highs is None else Decimal(highs[index])
        )
        session_low = comparison_close if lows is None else Decimal(lows[index])
        bars.append(
            DailyBar(
                "SPY",
                session,
                session_open,
                session_high,
                session_low,
                close,
                Decimal(1000) if volumes is None else Decimal(volumes[index]),
            )
        )
    typed_bars = tuple(bars)
    retrieved_at = datetime(2024, 7, 15, tzinfo=UTC)
    if dividends and dividend_sessions:
        raise ValueError("use dividends or dividend_sessions, not both")
    if splits and split_sessions:
        raise ValueError("use splits or split_sessions, not both")
    dividend_values = dividends or tuple(
        (session, "1") for session in dividend_sessions
    )
    split_values = splits or tuple((session, "2") for session in split_sessions)
    corporate_action_seeds = tuple(
        [
            CashDividendSeed("SPY", session, Decimal(amount), "synthetic")
            for session, amount in dividend_values
        ]
        + [
            StockSplitSeed("SPY", session, Decimal(factor), "synthetic")
            for session, factor in split_values
        ]
    )
    snapshot_id = corporate_action_snapshot_id(corporate_action_seeds)
    normalized_dividend_sessions = tuple(session for session, _ in dividend_values)
    normalized_split_sessions = tuple(session for session, _ in split_values)
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
        "split_sessions": normalized_split_sessions,
        "dividend_sessions": normalized_dividend_sessions,
        "corporate_actions_complete": corporate_actions_complete,
        "corporate_action_count": len(corporate_action_seeds),
        "dividend_count": len(dividend_values),
        "split_count": len(split_values),
        "corporate_action_snapshot_id": snapshot_id,
        "ohlc_basis": (
            "raw_provider"
            if adjustment_mode is AdjustmentMode.UNADJUSTED
            else "split_adjusted"
        ),
        "volume_basis": (
            "raw_provider"
            if adjustment_mode is AdjustmentMode.UNADJUSTED
            else "split_adjusted"
        ),
        "adjusted_fields_used": adjusted_fields_used,
        "corporate_action_policy": (
            "separate_provider_reported_cash_dividends_and_splits"
        ),
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
    corporate_actions = bind_corporate_actions(
        corporate_action_seeds,
        dataset_id=calculated_dataset_id,
        snapshot_id=snapshot_id,
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
        corporate_actions_location=(
            f"datasets/{calculated_dataset_id}/corporate_actions.json"
        ),
        raw_sha256=raw_sha256,
        data_sha256=data_sha256,
        dataset_id=calculated_dataset_id,
        schema_version=SCHEMA_VERSION,
        bar_count=len(bars),
        missing_sessions=missing_sessions,
        split_sessions=normalized_split_sessions,
        dividend_sessions=normalized_dividend_sessions,
        corporate_actions_complete=corporate_actions_complete,
        corporate_action_count=len(corporate_actions),
        dividend_count=len(dividend_values),
        split_count=len(split_values),
        corporate_action_snapshot_id=snapshot_id,
        ohlc_basis=(
            "raw_provider"
            if adjustment_mode is AdjustmentMode.UNADJUSTED
            else "split_adjusted"
        ),
        volume_basis=(
            "raw_provider"
            if adjustment_mode is AdjustmentMode.UNADJUSTED
            else "split_adjusted"
        ),
        adjusted_fields_used=adjusted_fields_used,
        corporate_action_policy=(
            "separate_provider_reported_cash_dividends_and_splits"
        ),
        adapter_version="test-1",
    )
    return MarketDataset(typed_bars, metadata, corporate_actions)
