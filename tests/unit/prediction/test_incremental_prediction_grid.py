"""Compact QF-32 trials preserve analysis, rankings and candidate isolation."""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.optimization import TrialStatus
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.grid import (
    PredictionContextEnvironment,
    PredictionGridStudy,
    PredictionTrialAnalysis,
)
from quantforge.prediction.window_encoding import mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from tests.unit.prediction.test_multi_timeframe_study import (
    _prediction_dataset,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_grid import (
    _backend_environment,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_window import (
    WindowAnalyzer,
    WindowFactory,
    WindowProvider,
    grid,
    schedule,
)


class CompactWindowAnalyzer(WindowAnalyzer):
    """Same fixture math/comparisons, reading normalized rows without expansion."""

    def analyze_compact_window(
        self, reader: PredictionWindowReader
    ) -> PredictionTrialAnalysis:
        count = correct = 0
        row_ids: list[str] = []
        for decision in reader.iterate_decisions():
            rows = cast(
                list[PrimitiveMapping],
                mapping(decision.to_primitive()["prediction_study"])["rows"],
            )
            for row in rows:
                count += 1
                correct += int(
                    cast(
                        bool,
                        mapping(mapping(row["evaluation"])["values"])[
                            "direction_correct"
                        ],
                    )
                )
                row_ids.append(cast(str, row["row_id"]))
        return PredictionTrialAnalysis.create(
            prediction_count=count,
            metrics={
                "accuracy": str(Decimal(correct) / count if count else Decimal(0))
            },
            period_comparisons=({"period": "fixture", "count": count},),
            weekday_comparisons=({"weekday": 3, "count": count},),
            matched_baseline_comparisons=(
                {"baseline_name": "always_up", "count": count},
            ),
            artifacts=cast(PrimitiveMapping, {"row_ids": row_ids}),
        )


def compact_grid(
    root: Path, provider: WindowProvider, *, retry_failed: bool = False
) -> PredictionGridStudy:
    base = grid(root, provider)
    return PredictionGridStudy(
        dataset=_prediction_dataset(),
        dataset_family_fingerprint=provider.family.family_id,
        study_factory=WindowFactory(),
        analyzer=CompactWindowAnalyzer(),
        context_provider=provider,
        context_environment=PredictionContextEnvironment.create("fixture", "1", {}),
        indicator_backend=_backend_environment(),
        decision_schedule=schedule(),
        config=replace(
            base._config,  # pyright: ignore[reportPrivateUsage]
            window_schema_version="2",
            retry_failed=retry_failed,
        ),
    )


def test_grid_resume_reuses_completed_trial_and_incomplete_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = WindowProvider()
    study = compact_grid(tmp_path / "compact", provider)
    legacy = grid(tmp_path / "legacy", WindowProvider()).run()
    calls = 0
    original = provider.get_context_at

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        if calls == 6:
            raise KeyboardInterrupt
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(provider, "get_context_at", interrupted)
    with pytest.raises(KeyboardInterrupt):
        study.run()
    complete = list((tmp_path / "compact").rglob("prediction-window.jsonl"))
    assert len(complete) == 1
    first_bytes = complete[0].read_bytes()
    stages = list((tmp_path / "compact").rglob("*.in-progress"))
    assert len(stages) == 1
    assert len((stages[0] / "decisions.jsonl").read_bytes().splitlines()) == 2
    assert len(list(stages[0].iterdir())) == 3
    monkeypatch.setattr(provider, "get_context_at", original)
    provider.requests.clear()
    result = study.resume()
    assert provider.requests == list(schedule().decision_timestamps[2:])
    assert complete[0].read_bytes() == first_bytes
    windows = list((tmp_path / "compact").rglob("prediction-window.jsonl"))
    assert len(windows) == 2
    assert (
        len(
            {
                cast(str, PredictionWindowReader.open(p).header()["window_id"])
                for p in windows
            }
        )
        == 2
    )
    assert all(record.status is TrialStatus.SUCCEEDED for record in result.trials)
    for compact, embedded in zip(result.trials, legacy.trials, strict=True):
        assert compact.analysis is not None
        assert embedded.analysis is not None
        left, right = compact.analysis.to_primitive(), embedded.analysis.to_primitive()
        left.pop("artifacts")
        right.pop("artifacts")
        assert left == right
    assert [r.combination_id for r in result.rankings] == [
        r.combination_id for r in legacy.rankings
    ]
    provider.requests.clear()
    study.resume()
    assert not provider.requests
    assert result.cache_statistics.context_misses == 0
    assert result.cache_statistics.indicator_misses == 0


def test_compact_trial_corruption_is_not_ranked(tmp_path: Path) -> None:
    study = compact_grid(tmp_path, WindowProvider())
    study.run()
    path = next(tmp_path.rglob("prediction-window.jsonl"))
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(InvalidPredictionOutputError):
        study.resume()


def test_failed_trial_reports_exact_decision_and_retries_its_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = WindowProvider()
    study = compact_grid(tmp_path, provider, retry_failed=True)
    original = provider.get_context_at
    calls = 0

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("secret-token-must-not-be-persisted")
        return original(*args, **kwargs)

    monkeypatch.setattr(provider, "get_context_at", fail_once)
    result = study.run()
    failed = result.trials[0]
    assert failed.status is TrialStatus.FAILED
    assert failed.failure_message is not None
    assert "sequence=2" in failed.failure_message
    assert schedule().decision_timestamps[2].isoformat() in failed.failure_message
    assert "secret-token" not in failed.failure_message
    assert result.trials[1].status is TrialStatus.SUCCEEDED
    monkeypatch.setattr(provider, "get_context_at", original)
    provider.requests.clear()
    resumed = study.resume()
    assert provider.requests == list(schedule().decision_timestamps[2:])
    assert all(t.status is TrialStatus.SUCCEEDED for t in resumed.trials)
    assert len(resumed.trials[0].failed_attempts) == 1
