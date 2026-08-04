import json
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from io import BytesIO
from pathlib import Path

import pytest

from quantforge.data.cache import MarketDataCache
from quantforge.data.exceptions import CacheError, ProviderError, ValidationError
from quantforge.data.models import (
    SCHEMA_VERSION,
    AdjustmentMode,
    DailyBar,
    ProviderResponse,
)
from quantforge.data.normalize import (
    normalize_response,
    normalize_response_with_corporate_action_sessions,
    normalize_response_with_split_sessions,
    normalize_symbol,
)
from quantforge.data.providers import AlphaVantageProvider
from quantforge.data.service import MarketDataService
from quantforge.data.validate import validate_bars


class FakeProvider:
    name = "fake"

    def __init__(self, records: tuple[dict[str, str], ...]) -> None:
        self.records = records
        self.calls = 0

    def fetch_daily_bars(
        self, symbol: str, start: date, end: date, adjustment: AdjustmentMode
    ) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse(
            self.name,
            symbol,
            datetime(2024, 7, 6, tzinfo=UTC),
            "America/New_York",
            adjustment,
            self.records,
            {"fixture": "three XNYS sessions"},
            "test-1",
        )


def record(
    session: str,
    *,
    open_price: str = "100",
    high: str = "102",
    low: str = "99",
    close: str = "101",
    volume: str = "1000",
    dividend: str = "0",
    split: str = "1",
) -> dict[str, str]:
    return {
        "session_date": session,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "dividend_amount": dividend,
        "split_coefficient": split,
    }


VALID = (record("2024-07-03"), record("2024-07-01"), record("2024-07-02"))


def response(
    records: tuple[dict[str, str], ...],
    adjustment: AdjustmentMode = AdjustmentMode.UNADJUSTED,
) -> ProviderResponse:
    return ProviderResponse(
        "fake",
        "SPY",
        datetime(2024, 1, 1, tzinfo=UTC),
        "America/New_York",
        adjustment,
        records,
        {},
        "1",
    )


def test_alpha_vantage_preserves_dividend_amount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "Meta Data": {"2. Symbol": "SPY"},
        "Time Series (Daily)": {
            "2024-07-01": {
                "1. open": "100",
                "2. high": "102",
                "3. low": "99",
                "4. close": "101",
                "5. adjusted close": "100.25",
                "6. volume": "1000",
                "7. dividend amount": "0.75",
                "8. split coefficient": "1",
            }
        },
    }

    def fake_urlopen(url: str, timeout: float) -> BytesIO:
        del url, timeout
        return BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("quantforge.data.providers.alpha_vantage.urlopen", fake_urlopen)
    provider = AlphaVantageProvider("test-api-key")
    provider_response = provider.fetch_daily_bars(
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 1),
        AdjustmentMode.UNADJUSTED,
    )

    assert provider.adapter_version == "2"
    assert provider_response.records[0]["dividend_amount"] == "0.75"


def test_normalizes_symbol_and_stably_orders_rows() -> None:
    bars = normalize_response(response(VALID), normalize_symbol(" spy "))
    assert [bar.session_date.day for bar in bars] == [1, 2, 3]
    assert {bar.symbol for bar in bars} == {"SPY"}


def test_synthetic_two_for_one_split_adjusts_all_ohlc_and_volume() -> None:
    records = (
        record(
            "2024-07-01",
            open_price="200",
            high="204",
            low="198",
            close="202",
            volume="500",
        ),
        record("2024-07-02", split="2"),
    )
    bars, split_sessions = normalize_response_with_split_sessions(
        response(records, AdjustmentMode.SPLIT_ADJUSTED), "SPY"
    )
    assert bars[0] == DailyBar(
        "SPY",
        date(2024, 7, 1),
        Decimal("100"),
        Decimal("102"),
        Decimal("99"),
        Decimal("101"),
        Decimal("1000"),
    )
    assert bars[1].open == Decimal("100")
    assert split_sessions == (date(2024, 7, 2),)


def test_requires_split_coefficient_for_verified_provenance() -> None:
    incomplete = record("2024-07-01")
    del incomplete["split_coefficient"]

    with pytest.raises(ValidationError, match="split_coefficient"):
        normalize_response(response((incomplete,)), "SPY")


def test_requires_dividend_amount_for_verified_provenance() -> None:
    incomplete = record("2024-07-01")
    del incomplete["dividend_amount"]

    with pytest.raises(ValidationError, match="dividend_amount"):
        normalize_response(response((incomplete,)), "SPY")


@pytest.mark.parametrize("dividend", ["-0.01", "NaN", "Infinity"])
def test_rejects_invalid_dividend_amount(dividend: str) -> None:
    with pytest.raises(ValidationError, match="dividend amount"):
        normalize_response(response((record("2024-07-01", dividend=dividend),)), "SPY")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"open": "0"}, "positive"),
        ({"volume": "0"}, "positive"),
        ({"high": "98"}, "high"),
        ({"low": "101.5"}, "low"),
        ({"close": "NaN"}, "finite"),
        ({"close": "Infinity"}, "finite"),
    ],
)
def test_rejects_invalid_numeric_values(change: dict[str, str], message: str) -> None:
    invalid = record("2024-07-01") | change
    bars = normalize_response(response((invalid,)), "SPY")
    with pytest.raises(ValidationError, match=message):
        validate_bars(bars, "SPY", date(2024, 7, 1), date(2024, 7, 1), "XNYS")


def test_rejects_duplicate_null_empty_and_out_of_range() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        validate_bars(
            normalize_response(
                response((record("2024-07-01"), record("2024-07-01"))), "SPY"
            ),
            "SPY",
            date(2024, 7, 1),
            date(2024, 7, 1),
            "XNYS",
        )
    with pytest.raises(ValidationError, match="invalid"):
        normalize_response(response((record("2024-07-01") | {"close": None},)), "SPY")  # type: ignore[dict-item]
    with pytest.raises(ValidationError, match="no bars"):
        validate_bars((), "SPY", date(2024, 7, 1), date(2024, 7, 1), "XNYS")
    with pytest.raises(ValidationError, match="outside"):
        validate_bars(
            normalize_response(response((record("2024-07-02"),)), "SPY"),
            "SPY",
            date(2024, 7, 1),
            date(2024, 7, 1),
            "XNYS",
        )


def test_calendar_ignores_weekend_and_holiday_but_detects_missing_session() -> None:
    bars = normalize_response(
        response((record("2024-07-03"), record("2024-07-05"))), "SPY"
    )
    assert (
        len(validate_bars(bars, "SPY", date(2024, 7, 3), date(2024, 7, 7), "XNYS")) == 2
    )
    with pytest.raises(ValidationError) as caught:
        validate_bars(bars, "SPY", date(2024, 7, 2), date(2024, 7, 7), "XNYS")
    assert caught.value.missing_sessions == ("2024-07-02",)


def test_service_persists_metadata_and_reuses_cache_across_instances(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(VALID)
    cache = MarketDataCache(tmp_path)
    first = MarketDataService(provider, cache).get_daily_bars(
        " spy ", date(2024, 7, 1), date(2024, 7, 3)
    )
    second = MarketDataService(provider, MarketDataCache(tmp_path)).get_daily_bars(
        "SPY", date(2024, 7, 1), date(2024, 7, 3)
    )
    assert provider.calls == 1
    assert first == second
    assert first.metadata.dataset_id
    assert first.metadata.schema_version == SCHEMA_VERSION == "3"
    assert first.metadata.bar_count == 3
    assert first.metadata.adjustment_mode is AdjustmentMode.SPLIT_ADJUSTED
    assert first.metadata.split_sessions == ()
    assert first.metadata.dividend_sessions == ()
    assert first.metadata.raw_location.startswith("raw/")


def test_service_persists_and_reloads_verified_split_sessions(tmp_path: Path) -> None:
    provider = FakeProvider(
        (
            record(
                "2024-07-01",
                open_price="200",
                high="204",
                low="198",
                close="202",
            ),
            record("2024-07-02", split="2"),
        )
    )
    cache = MarketDataCache(tmp_path)
    dataset = MarketDataService(provider, cache).get_daily_bars(
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 2),
        AdjustmentMode.UNADJUSTED,
    )

    assert dataset.metadata.split_sessions == (date(2024, 7, 2),)
    assert cache.load(dataset.metadata.dataset_id) == dataset


def test_service_persists_and_reloads_verified_dividend_sessions(
    tmp_path: Path,
) -> None:
    records = (
        record("2024-07-01"),
        record("2024-07-02", dividend="0.75"),
    )
    bars, split_sessions, dividend_sessions = (
        normalize_response_with_corporate_action_sessions(response(records), "SPY")
    )
    assert len(bars) == 2
    assert split_sessions == ()
    assert dividend_sessions == (date(2024, 7, 2),)

    cache = MarketDataCache(tmp_path)
    dataset = MarketDataService(FakeProvider(records), cache).get_daily_bars(
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 2),
        AdjustmentMode.UNADJUSTED,
    )

    assert dataset.metadata.dividend_sessions == (date(2024, 7, 2),)
    assert cache.load(dataset.metadata.dataset_id) == dataset


def test_cache_rejects_legacy_manifest_without_split_provenance(
    tmp_path: Path,
) -> None:
    cache = MarketDataCache(tmp_path)
    dataset = MarketDataService(FakeProvider(VALID), cache).get_daily_bars(
        "SPY", date(2024, 7, 1), date(2024, 7, 3)
    )
    manifest_path = (
        tmp_path / "datasets" / dataset.metadata.dataset_id / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["split_sessions"]
    manifest["schema_version"] = "1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CacheError, match="incomplete or corrupt"):
        cache.load(dataset.metadata.dataset_id)


def test_cache_rejects_v2_manifest_without_dividend_provenance(
    tmp_path: Path,
) -> None:
    cache = MarketDataCache(tmp_path)
    dataset = MarketDataService(FakeProvider(VALID), cache).get_daily_bars(
        "SPY", date(2024, 7, 1), date(2024, 7, 3)
    )
    manifest_path = (
        tmp_path / "datasets" / dataset.metadata.dataset_id / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["dividend_sessions"]
    manifest["schema_version"] = "2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CacheError, match="incomplete or corrupt"):
        cache.load(dataset.metadata.dataset_id)


def test_cache_detects_corrupted_data_and_never_overwrites(tmp_path: Path) -> None:
    provider = FakeProvider(VALID)
    cache = MarketDataCache(tmp_path)
    dataset = MarketDataService(provider, cache).get_daily_bars(
        "SPY", date(2024, 7, 1), date(2024, 7, 3)
    )
    path = tmp_path / dataset.metadata.normalized_location
    path.write_text("corrupt")
    with pytest.raises(CacheError, match="checksum"):
        cache.load(dataset.metadata.dataset_id)


def test_provider_exception_is_translated(tmp_path: Path) -> None:
    class BrokenProvider(FakeProvider):
        def fetch_daily_bars(
            self, symbol: str, start: date, end: date, adjustment: AdjustmentMode
        ) -> ProviderResponse:
            raise RuntimeError("SDK detail")

    with pytest.raises(ProviderError, match="provider failed"):
        MarketDataService(
            BrokenProvider(VALID), MarketDataCache(tmp_path)
        ).get_daily_bars("SPY", date(2024, 7, 1), date(2024, 7, 3))


def test_rejects_provider_adjustment_mismatch(tmp_path: Path) -> None:
    class WrongProvider(FakeProvider):
        def fetch_daily_bars(
            self, symbol: str, start: date, end: date, adjustment: AdjustmentMode
        ) -> ProviderResponse:
            return replace(
                super().fetch_daily_bars(symbol, start, end, adjustment),
                adjustment_mode=AdjustmentMode.UNADJUSTED,
            )

    with pytest.raises(ProviderError, match="different adjustment"):
        MarketDataService(
            WrongProvider(VALID), MarketDataCache(tmp_path)
        ).get_daily_bars("SPY", date(2024, 7, 1), date(2024, 7, 3))
