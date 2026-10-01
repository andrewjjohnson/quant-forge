"""QF-67 event dataset scale on SYNTHETIC schema-4 windows (not evidence).

About 2,500 synthetic QF-45-shaped observations among about 7,400 scheduled
decisions are assembled, written (Parquet, optional CSV) and read back. The
report separates the QF-64 reader's own observation cost from row assembly,
writing and offline reading. No timing or memory threshold is asserted; the
structural assertions are that only observations become rich objects and rows,
labels keep their four states, and the artifact round-trips exactly. See
``docs/event-ml-datasets.md`` for the measured real and synthetic figures.
"""

import json
import resource
import sys
from datetime import time
from pathlib import Path
from time import perf_counter
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples.spy_ema_ml_dataset import (
    ema_event_feature_schema,
    ema_forward_return_target,
)
from quantforge.ml import (
    ROWS_CSV,
    ROWS_PARQUET,
    DispositionPolicy,
    EventPopulation,
    assemble_event_dataset,
    export_event_dataset,
    read_event_dataset,
)
from quantforge.ml.sources import (
    _VERIFIED,  # pyright: ignore[reportPrivateUsage]
    EventSourceWindow,
    HoldoutIsolation,
    universe_candidate,
)
from quantforge.oos import load_oos_source
from quantforge.prediction.window_compact import CompactPredictionWindowDecision
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.validation import PartitionRole
from tests.integration.event_dataset_fixtures import event_inputs, run_event_study
from tests.performance.event_dataset_scale import (
    at,
    long_plan,
    write_synthetic_window,
)

WINDOWS = {
    role: (at(first, time(9, 32)), at(last, time(16)))
    for role, first, last in (
        (PartitionRole.DEVELOPMENT, "2024-10-02", "2024-10-31"),
        (PartitionRole.SELECTION, "2024-11-01", "2024-12-20"),
        (PartitionRole.WALK_FORWARD_TEST, "2024-12-23", "2024-12-27"),
        (PartitionRole.FINAL_HOLDOUT, "2024-12-30", "2024-12-31"),
    )
}


def test_thousands_of_event_rows_assemble_write_and_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report: dict[str, object] = {"fixture": "synthetic_schema4_not_evidence"}
    with canonical_preparation():
        inputs = event_inputs(tmp_path)
        study = run_event_study(tmp_path, inputs, schema="4", fixed=True)
        template = PredictionWindowReader.open(
            next(study.study_path.glob("folds/*/selection/*/artifacts/*/*.jsonl"))
        )
        config, _ = long_plan(inputs, tmp_path / "long", WINDOWS)
        plan = config.plan
        fold = plan.folds[0]
        candidate = universe_candidate(
            load_oos_source(study.config.plan, study.study_path),
            study.combination("8/48"),
        )
    started = perf_counter()
    windows = {
        role: write_synthetic_window(
            tmp_path / role.value / "prediction-window.jsonl",
            template,
            plan,
            window,
            every=3,
        )
        for role, window in (
            (PartitionRole.SELECTION, fold.selection),
            (PartitionRole.WALK_FORWARD_TEST, fold.test),
        )
        if window is not None
    }
    report["fixture_write_seconds"] = round(perf_counter() - started, 2)
    sources = tuple(
        EventSourceWindow(
            role,
            fold.fold_id,
            0,
            fold.selection if role is PartitionRole.SELECTION else fold.test,  # type: ignore[arg-type]
            PrimitiveMappingSnapshot.capture(
                {"retained_decision_count": item.decisions, "synthetic": True}
            ),
            candidate,
            "synthetic-selection",
            PrimitiveMappingSnapshot.capture({"path": item.path.name}),
            PredictionWindowReader.open(item.path),
            _VERIFIED,
        )
        for role, item in windows.items()
    )
    events = sum(item.events for item in windows.values())
    decisions = sum(item.decisions for item in windows.values())
    report.update(
        scheduled_decisions=decisions,
        observations=events,
        window_bytes=sum(item.bytes for item in windows.values()),
    )
    started = perf_counter()
    observed = sum(
        1 for source in sources for _ in source.reader.iterate_observations()
    )
    report["reader_only_seconds"] = round(perf_counter() - started, 3)
    assert observed == events
    constructed: list[int] = []
    validate = CompactPredictionWindowDecision.__post_init__

    def counted(self: CompactPredictionWindowDecision) -> None:
        constructed.append(1)
        validate(self)

    monkeypatch.setattr(CompactPredictionWindowDecision, "__post_init__", counted)
    started = perf_counter()
    dataset = assemble_event_dataset(
        sources,
        plan=plan,
        population=EventPopulation.capture(
            plan, candidate, DispositionPolicy.ACCEPTED_ONLY
        ),
        feature_schema=ema_event_feature_schema(),
        target=ema_forward_return_target(),
        isolation=HoldoutIsolation.capture(plan, (), "SPY"),
    )
    assembled = perf_counter() - started
    monkeypatch.undo()
    # Only observations are expanded (checked, then given their catalogue view).
    assert len(constructed) == 2 * events
    assert dataset.row_count == events > 2_000
    summary = cast(dict[str, object], dataset.summaries.to_primitive()["overall"])
    quarter = events // 4
    assert all(
        quarter <= cast(int, summary[key]) + 1 for key in ("positive", "unavailable")
    )
    assert cast(int, summary["negative"]) >= 2 * quarter
    started = perf_counter()
    path = export_event_dataset(dataset, tmp_path / "datasets", include_csv=True)
    written = perf_counter() - started
    started = perf_counter()
    loaded = read_event_dataset(path)
    read = perf_counter() - started
    assert loaded == dataset
    sizes = {item.name: item.stat().st_size for item in path.iterdir()}
    report.update(
        rows=dataset.row_count,
        assembly_seconds=round(assembled, 3),
        assembly_rows_per_second=round(dataset.row_count / assembled),
        export_seconds_including_staged_validation=round(written, 3),
        read_and_validate_seconds=round(read, 3),
        artifact_bytes=sizes,
        parquet_bytes_per_row=round(sizes[ROWS_PARQUET] / dataset.row_count, 1),
        csv_bytes_per_row=round(sizes[ROWS_CSV] / dataset.row_count, 1),
        max_rss_mb=round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1 if sys.platform == "darwin" else 1 / 1024)
            / 1024
            / 1024,
            1,
        ),
    )
    assert sizes[ROWS_PARQUET] / dataset.row_count < 2_048
    print(json.dumps(report, indent=2, sort_keys=True))
