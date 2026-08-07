import argparse
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
import scripts.run_spy_gap_prediction as spy_gap_script

from quantforge.data import (
    AdjustmentMode,
    DatasetMetadata,
    MarketDataCache,
    MarketDataset,
    RequestError,
)
from quantforge.data.providers import TiingoProvider
from quantforge.prediction import (
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    PredictionExportError,
    run_prediction_analysis,
)

from .helpers import make_dataset


def _spy_dataset() -> MarketDataset:
    closes = (
        "100",
        "102",
        "101",
        "103",
        "99",
        "101",
        "100",
        "102",
        "98",
        "100",
        "99",
        "101",
        "100",
        "102",
        "101",
    )
    return make_dataset(
        closes,
        opens=tuple(str(int(value) - 1) for value in closes),
        highs=tuple(str(int(value) + 1) for value in closes),
        lows=tuple(str(int(value) - 2) for value in closes),
    )


def _maintained_spy_dataset() -> MarketDataset:
    dataset = _spy_dataset()
    return replace(
        dataset,
        metadata=replace(
            dataset.metadata,
            provider_name=TiingoProvider.name,
            requested_start=spy_gap_script.REQUESTED_START,
            requested_end=spy_gap_script.REQUESTED_END,
            adjustment_mode=AdjustmentMode.UNADJUSTED,
        ),
    )


def test_tiingo_is_the_default_fixed_spy_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = _maintained_spy_dataset()

    class FakeMarketDataService:
        def __init__(self, provider: object, cache: object) -> None:
            assert isinstance(provider, TiingoProvider)
            assert isinstance(cache, MarketDataCache)

        def get_daily_bars(
            self,
            symbol: str,
            start: object,
            end: object,
            adjustment: AdjustmentMode,
            *,
            refresh: bool,
        ) -> MarketDataset:
            assert symbol == spy_gap_script.SYMBOL
            assert start == spy_gap_script.REQUESTED_START
            assert end == spy_gap_script.REQUESTED_END
            assert adjustment is AdjustmentMode.UNADJUSTED
            assert refresh is True
            return expected

    monkeypatch.setenv("TIINGO_API_KEY", "test-key-not-a-secret")
    monkeypatch.setattr(spy_gap_script, "MarketDataService", FakeMarketDataService)
    arguments = argparse.Namespace(
        cache_root=tmp_path / "cache",
        dataset_id=None,
        output_root=tmp_path / "reports",
        refresh=True,
    )

    dataset, source = spy_gap_script.load_dataset(arguments)

    assert dataset is expected
    assert source == "Tiingo End-of-Day"


def test_cached_dataset_matching_the_maintained_request_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = _maintained_spy_dataset()

    class FakeCache:
        def __init__(self, root: Path) -> None:
            assert root == tmp_path / "cache"

        def load(self, dataset_id: str) -> MarketDataset:
            assert dataset_id == "maintained-dataset"
            return expected

    monkeypatch.setattr(spy_gap_script, "MarketDataCache", FakeCache)
    arguments = argparse.Namespace(
        cache_root=tmp_path / "cache",
        dataset_id="maintained-dataset",
        output_root=tmp_path / "reports",
        refresh=False,
    )

    dataset, source = spy_gap_script.load_dataset(arguments)

    assert dataset is expected
    assert source == "cached QF-3 dataset"


@pytest.mark.parametrize(
    "metadata",
    [
        replace(_maintained_spy_dataset().metadata, canonical_symbol="QQQ"),
        replace(_maintained_spy_dataset().metadata, provider_name="other"),
        replace(
            _maintained_spy_dataset().metadata,
            adjustment_mode=AdjustmentMode.SPLIT_ADJUSTED,
        ),
        replace(
            _maintained_spy_dataset().metadata,
            requested_start=date(2021, 1, 1),
        ),
        replace(
            _maintained_spy_dataset().metadata,
            requested_end=date(2024, 12, 31),
        ),
    ],
)
def test_cached_dataset_must_match_the_maintained_request(
    metadata: DatasetMetadata,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = replace(_maintained_spy_dataset(), metadata=metadata)

    class FakeCache:
        def __init__(self, root: Path) -> None:
            assert root == tmp_path / "cache"

        def load(self, dataset_id: str) -> MarketDataset:
            return expected

    monkeypatch.setattr(spy_gap_script, "MarketDataCache", FakeCache)
    arguments = argparse.Namespace(
        cache_root=tmp_path / "cache",
        dataset_id="different-experiment",
        output_root=tmp_path / "reports",
        refresh=False,
    )

    with pytest.raises(RequestError, match="does not match the maintained"):
        spy_gap_script.load_dataset(arguments)


def test_missing_tiingo_key_has_a_safe_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    arguments = argparse.Namespace(
        cache_root=tmp_path / "cache",
        dataset_id=None,
        output_root=tmp_path / "reports",
        refresh=False,
    )

    with pytest.raises(RequestError, match="TIINGO_API_KEY is required"):
        spy_gap_script.load_dataset(arguments)


def test_spy_prediction_export_can_be_safely_reused(tmp_path: Path) -> None:
    result = run_prediction_analysis(
        _spy_dataset(),
        OvernightGapPredictionStrategy(OvernightGapPredictionParameters()),
    )
    output_root = tmp_path / "reports"

    artifact_path, status = spy_gap_script.export_result(result, output_root)
    assert status == "created"
    assert spy_gap_script.export_result(result, output_root) == (
        artifact_path,
        "reused existing immutable export",
    )

    (artifact_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PredictionExportError, match="expected immutable result"):
        spy_gap_script.export_result(result, output_root)
