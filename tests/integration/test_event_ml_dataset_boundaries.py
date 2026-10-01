"""QF-67 input/output boundaries: rapid refusal, holdout lifecycle, corruption.

Uses the synthetic QF-45-composition fixture (``event_dataset_fixtures``) with
schema-4 windows. The final holdout (2024-12-30..31) is consumed only through
the explicit QF-40 ledger workflow in a dedicated fixture workspace.
"""

import hashlib
import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import date
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples.spy_ema import EmaParameters, EmaSmokeRule, configured_outcomes
from quantforge.examples.spy_ema_ml_dataset import ema_event_feature_schema
from quantforge.ml import (
    MANIFEST_FILE,
    ROWS_CSV,
    ROWS_PARQUET,
    DispositionPolicy,
    EventDataset,
    EventDatasetError,
    EventDatasetIntegrityError,
    EventHoldoutError,
    EventPopulation,
    EventPopulationError,
    ForwardReturnBinaryTarget,
    NonAuthoritativeSourceError,
    assemble_event_dataset,
    build_event_dataset,
    export_event_dataset,
    read_event_dataset,
    validate_event_dataset,
)
from quantforge.ml.dataset import label_summary, logical_rows_sha256
from quantforge.ml.sources import (
    EventSourceWindow,
    HoldoutIsolation,
    load_study_event_sources,
    universe_candidate,
)
from quantforge.oos import HoldoutEvaluation, HoldoutState
from quantforge.prediction.window_encoding import canonical
from quantforge.rapid import RapidScanResult, export_rapid_scan, rapid_research_session
from quantforge.validation import PartitionRole
from tests.integration.event_dataset_fixtures import (
    EventStudy,
    event_inputs,
    run_event_study,
)
from tests.integration.rapid_scan_fixtures import reserve_scope

pa: Any = import_module("pyarrow")
pq: Any = import_module("pyarrow.parquet")


@pytest.fixture(scope="module", autouse=True)
def prepared() -> Iterator[None]:
    """One QF-65 load session for direct source loads (operational only)."""
    with canonical_preparation():
        yield


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory, prepared: None) -> EventStudy:
    del prepared
    root = tmp_path_factory.mktemp("qf67-event-boundaries")
    return run_event_study(root, event_inputs(root), schema="4")


@pytest.fixture(scope="module")
def dataset(study: EventStudy) -> EventDataset:
    return study.build()


def rapid_scan(study: EventStudy) -> RapidScanResult:
    with rapid_research_session(
        plan=study.config.plan,
        fold_index=0,
        role=PartitionRole.SELECTION,
        dataset=study.inputs.dataset,
        series=(study.inputs.primary, study.inputs.daily),
        workspace=study.workspace("rapid"),
    ) as session:
        return session.scan(
            EmaSmokeRule(EmaParameters(8, 48)),
            outcomes=configured_outcomes(study.inputs.primary)[1:2],
        )


def test_non_authoritative_rapid_inputs_are_refused(
    study: EventStudy, tmp_path: Path
) -> None:
    result = rapid_scan(study)
    assert result.authoritative is False
    assert result.trigger_count == 3
    plan = study.config.plan
    sources = load_study_event_sources(
        plan,
        study.study_path,
        combination_id=study.combination("8/48"),
        roles=frozenset({PartitionRole.SELECTION}),
    )
    population = EventPopulation.capture(
        plan, sources.windows[0].candidate, DispositionPolicy.ACCEPTED_ONLY
    )
    for exploratory in (result, result.events[0]):
        with pytest.raises(NonAuthoritativeSourceError, match="QF-32/QF-39/QF-40"):
            assemble_event_dataset(
                cast(list[EventSourceWindow], [exploratory]),
                plan=plan,
                population=population,
                feature_schema=ema_event_feature_schema(),
                target=ForwardReturnBinaryTarget(),
                isolation=HoldoutIsolation.capture(plan, (), "SPY"),
            )
    exported = export_rapid_scan(result, tmp_path / "exploration" / "ema.rapid.json")
    with pytest.raises(NonAuthoritativeSourceError, match="rapid"):
        study.build(study_path=exported)
    with pytest.raises(NonAuthoritativeSourceError):
        study.build(final_holdout=result)
    # Sources exist only through the verified loaders.
    with pytest.raises(EventDatasetIntegrityError, match="verified"):
        replace(sources.windows[0], _token=None)
    with pytest.raises(EventDatasetError, match="verified EventSourceWindow"):
        assemble_event_dataset(
            cast(list[EventSourceWindow], [object()]),
            plan=plan,
            population=population,
            feature_schema=ema_event_feature_schema(),
            target=ForwardReturnBinaryTarget(),
            isolation=HoldoutIsolation.capture(plan, (), "SPY"),
        )


def test_permanent_ledger_is_required_and_protected_scopes_are_refused(
    study: EventStudy,
) -> None:
    missing = study.root / "no-ledger-workspace"
    with pytest.raises(EventHoldoutError, match="never created"):
        study.build(workspace=missing)
    assert not (missing / "reports").exists()
    with pytest.raises(EventHoldoutError, match="final_holdout"):
        study.build(roles=frozenset({PartitionRole.FINAL_HOLDOUT}))
    cases = (
        ("test-session", date(2024, 12, 27), date(2024, 12, 27), "SPY", True),
        ("selection-session", date(2024, 12, 20), date(2024, 12, 23), "SPY", True),
        ("other-symbol", date(2024, 12, 23), date(2024, 12, 27), "QQQ", False),
        ("adjacent-after", date(2024, 12, 30), date(2024, 12, 31), "SPY", False),
    )
    for name, start, end, symbol, refused in cases:
        workspace = study.workspace(f"scope-{name}")
        reserve_scope(
            study.ledger(workspace),
            f"foreign-{name}",
            start=start,
            end=end,
            symbol=symbol,
        )
        if refused:
            with pytest.raises(EventHoldoutError, match="permanent ledger"):
                study.build(workspace=workspace)
        else:
            assert study.build(workspace=workspace).row_count == 4


def test_holdout_rows_require_explicit_ledger_consumption(
    study: EventStudy, dataset: EventDataset
) -> None:
    workspace = study.workspace("holdout")
    ledger = study.ledger(workspace)
    source = study.source()
    assert ledger.reserve(source).state is HoldoutState.RESERVED
    evaluation = HoldoutEvaluation.prepare(
        source, study.adapter, selection_fold_id=source.folds[0].fold_id
    )
    # Rows physically exist only after evaluation; a reservation never
    # authorizes them, and building never consumes the holdout.
    with pytest.raises(EventHoldoutError, match="consumed"):
        study.build(workspace=workspace, final_holdout=evaluation)
    assert ledger.state(source).state is HoldoutState.RESERVED
    assert not any((ledger.root / "exposures").iterdir())
    fold_only = study.build(workspace=workspace)
    assert fold_only.dataset_id == dataset.dataset_id
    consumed = ledger.consume(evaluation, run_id="qf67-fixture-explicit-consumption")
    assert consumed.state is HoldoutState.CONSUMED
    with_holdout = study.build(workspace=workspace, final_holdout=evaluation)
    holdout = with_holdout.rows_for(PartitionRole.FINAL_HOLDOUT)
    assert holdout == (4, 5)
    assert with_holdout.rows[:4] == dataset.rows
    assert [
        (row.fold_id, row.fold_index, row.label.value)
        for row in (with_holdout.rows[i] for i in holdout)
    ] == [(None, None, True), (None, None, False)]
    assert with_holdout.dataset_id != dataset.dataset_id
    summary = cast(
        list[PrimitiveMapping], with_holdout.summaries.to_primitive()["by_partition"]
    )
    assert [item["partition_role"] for item in summary] == [
        "validation_selection",
        "walk_forward_test",
        "final_holdout",
    ]
    # The consumed holdout evaluated 8/48; it cannot join another population.
    with pytest.raises(EventPopulationError, match="another candidate"):
        study.build(workspace=workspace, pair="8/40", final_holdout=evaluation)
    assert study.build(workspace=workspace).dataset_id == dataset.dataset_id


def corrupt_copy(
    study: EventStudy, root: Path, mutate: Callable[[Path, str], None]
) -> Path:
    copy = root / study.study_path.name
    shutil.copytree(study.study_path, copy)
    frozen = study.source().folds[0].selection
    assert frozen is not None
    mutate(copy, cast(str, frozen.snapshot.to_primitive()["selected_trial_id"]))
    return copy


def trial_dir(study_path: Path, trial_id: str) -> Path:
    """The population's (frozen 8/48) QF-32 trial artifact directory."""
    return next(study_path.glob(f"folds/*/selection/*/artifacts/{trial_id}"))


def edit_line(path: Path, index: int, old: str, new: str) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    assert old.encode() in lines[index]
    lines[index] = lines[index].replace(old.encode(), new.encode(), 1)
    path.write_bytes(b"".join(lines))


def rich_line(path: Path) -> int:
    return next(
        index
        for index, line in enumerate(path.read_bytes().splitlines())
        if b'"status":"evaluated"' in line
        and b'"record_type":"decision_receipt"' in line
    )


def _feature_edit(copy: Path, trial_id: str) -> None:
    window = trial_dir(copy, trial_id) / "prediction-window.jsonl"
    index = rich_line(window)
    line = window.read_bytes().splitlines()[index].decode()
    value = json.loads(line)["decision"]["generated_signals"][0]["features"][
        "daily_close"
    ]
    edit_line(window, index, f'"daily_close":"{value}"', f'"daily_close":"{value}1"')


def _wrapper_edit(copy: Path, trial_id: str) -> None:
    wrapper = trial_dir(copy, trial_id) / "prediction-window.json"
    record = json.loads(wrapper.read_text())
    record["analysis"]["prediction_count"] = 99
    wrapper.write_text(json.dumps(record, indent=2, sort_keys=True))


def _trial_status_edit(copy: Path, trial_id: str) -> None:
    trial = next(copy.glob(f"folds/*/selection/*/trials/{trial_id}.json"))
    record = json.loads(trial.read_text())
    record["status"] = "failed"
    trial.write_text(json.dumps(record, indent=2, sort_keys=True))


def _test_truncation(copy: Path, trial_id: str) -> None:
    del trial_id
    window = next(copy.glob("folds/*/test/prediction-window.jsonl"))
    window.write_bytes(b"".join(window.read_bytes().splitlines(keepends=True)[:-1]))


def _finder_metadata(copy: Path, trial_id: str) -> None:
    del trial_id
    (copy / "folds" / ".DS_Store").write_bytes(b"\x00")


@pytest.mark.parametrize(
    "mutate",
    [
        _feature_edit,
        _wrapper_edit,
        _trial_status_edit,
        _test_truncation,
        _finder_metadata,
    ],
    ids=["window-feature", "trial-wrapper", "trial-status", "test-truncated", "stray"],
)
def test_corrupted_source_evidence_fails_closed(
    study: EventStudy, tmp_path: Path, mutate: Callable[[Path, str], None]
) -> None:
    copy = corrupt_copy(study, tmp_path, mutate)
    with pytest.raises(EventDatasetIntegrityError):
        study.build(study_path=copy)


def consume_rows(dataset: EventDataset) -> dict[str, object]:
    """A generic consumer: only feature columns, target and membership."""
    matrix = dataset.numeric_feature_rows()
    target = dataset.target
    split: dict[str, list[int]] = {}
    for item in dataset.partition_membership:
        if target.values[item.row_index] is None:
            continue  # Unavailable labels are excluded explicitly, never zeroed.
        split.setdefault(item.role.value, []).append(item.row_index)
    return {
        "columns": dataset.feature_columns,
        "train_x": [matrix[i] for i in split.get("validation_selection", [])],
        "train_y": [target.values[i] for i in split.get("validation_selection", [])],
        "test_y": [target.values[i] for i in split.get("walk_forward_test", [])],
    }


def test_artifact_round_trip_and_generic_consumer(
    dataset: EventDataset, tmp_path: Path
) -> None:
    path = export_event_dataset(dataset, tmp_path / "datasets", include_csv=True)
    assert path.name == dataset.dataset_id
    assert sorted(item.name for item in path.iterdir()) == [
        MANIFEST_FILE,
        ROWS_CSV,
        ROWS_PARQUET,
    ]
    loaded = read_event_dataset(path)
    assert loaded == dataset
    assert validate_event_dataset(path) == dataset.dataset_id
    assert export_event_dataset(dataset, tmp_path / "datasets") == path
    consumed = consume_rows(loaded)
    assert consumed["columns"] == dataset.feature_columns
    assert consumed["train_y"] == [True, False]
    assert consumed["test_y"] == [False]
    table = pq.read_table(path / ROWS_PARQUET)
    groups = {
        field.name: field.metadata[b"quantforge_column_group"].decode()
        for field in table.schema
    }
    assert [name for name, group in groups.items() if group == "feature"] == list(
        dataset.feature_columns
    )
    assert str(table.schema.field("ema_fast").type) == "string"
    assert table.column("target").to_pylist() == [True, None, False, False]
    header = (path / ROWS_CSV).read_text().splitlines()[0].split(",")
    assert header == list(groups)
    with pytest.raises(EventHoldoutError):
        export_event_dataset(dataset, tmp_path / "reports" / "holdout-ledger" / "x")


def rewrite(
    path: Path, edit: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
) -> None:
    """Rewrite rows.parquet and rehash its manifest entry and fingerprint."""
    table = pq.read_table(path / ROWS_PARQUET)
    rows = edit(table.to_pylist())
    sink = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table.schema),
        sink,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
    )
    content = sink.getvalue().to_pybytes()
    (path / ROWS_PARQUET).write_bytes(content)
    envelope = json.loads((path / MANIFEST_FILE).read_text())
    payload = envelope["payload"]
    payload["files"][ROWS_PARQUET] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }
    write_manifest(path, payload)


def write_manifest(path: Path, payload: PrimitiveMapping) -> None:
    (path / MANIFEST_FILE).write_bytes(
        canonical({"payload": payload, "fingerprint": configuration_identity(payload)})
        + b"\n"
    )


def _flip_byte(path: Path) -> None:
    content = bytearray((path / ROWS_PARQUET).read_bytes())
    content[len(content) // 2] ^= 0xFF
    (path / ROWS_PARQUET).write_bytes(bytes(content))


def _manifest_unhashed(path: Path) -> None:
    envelope = json.loads((path / MANIFEST_FILE).read_text())
    envelope["payload"]["summaries"]["overall"]["positive"] = 2
    (path / MANIFEST_FILE).write_bytes(canonical(envelope) + b"\n")


def _summary_rehashed(path: Path) -> None:
    payload = json.loads((path / MANIFEST_FILE).read_text())["payload"]
    payload["summaries"]["overall"]["unavailable"] = 0
    write_manifest(path, payload)


def _label_flipped(path: Path) -> None:
    def edit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows[2]["target"] = True  # source value "0": a negative
        return rows

    rewrite(path, edit)


def _unavailable_as_negative(path: Path) -> None:
    def edit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows[1]["target"] = False
        return rows

    rewrite(path, edit)


def _feature_changed(path: Path) -> None:
    def edit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows[0]["ema_fast"] = "1"
        return rows

    rewrite(path, edit)


def _rows_reordered(path: Path) -> None:
    def edit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows[0], rows[1] = rows[1], rows[0]
        rows[0]["row_index"], rows[1]["row_index"] = 0, 1
        return rows

    rewrite(path, edit)


def _membership_relabelled(path: Path) -> None:
    def edit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows[3]["partition_role"] = "validation_selection"
        return rows

    rewrite(path, edit)


def _unlisted_csv(path: Path) -> None:
    (path / ROWS_CSV).write_text("row_index\n0\n")


def _renamed(path: Path) -> None:
    path.rename(path.with_name("0" * 64))


@pytest.mark.parametrize(
    "corrupt",
    [
        _flip_byte,
        _manifest_unhashed,
        _summary_rehashed,
        _label_flipped,
        _unavailable_as_negative,
        _feature_changed,
        _rows_reordered,
        _membership_relabelled,
        _unlisted_csv,
        _renamed,
    ],
)
def test_corrupted_artifacts_fail_closed(
    dataset: EventDataset, tmp_path: Path, corrupt: Callable[[Path], None]
) -> None:
    path = export_event_dataset(dataset, tmp_path)
    corrupt(path)
    target = path if path.exists() else path.with_name("0" * 64)
    with pytest.raises(EventDatasetIntegrityError):
        read_event_dataset(target)


def test_fully_rehashed_label_forgery_is_still_detected(
    dataset: EventDataset, tmp_path: Path
) -> None:
    """Rewriting identities consistently cannot turn unavailable into negative."""
    rows = list(dataset.rows)
    forged_label = replace(rows[1].label, value=False)
    rows[1] = replace(rows[1], label=forged_label)

    scientific = dataset.scientific.to_primitive()
    cast(dict[str, Any], scientific["rows"])["logical_rows_sha256"] = (
        logical_rows_sha256(rows)
    )
    forged = replace(
        dataset,
        dataset_id=configuration_identity(scientific),
        scientific=PrimitiveMappingSnapshot.capture(scientific),
        summaries=PrimitiveMappingSnapshot.capture(label_summary(rows)),
        rows=tuple(rows),
    )
    with pytest.raises(EventDatasetIntegrityError, match="target label"):
        export_event_dataset(forged, tmp_path)
    assert not any(tmp_path.iterdir())


def test_wrong_population_assembly_is_refused(study: EventStudy) -> None:
    plan = study.config.plan
    sources = load_study_event_sources(
        plan,
        study.study_path,
        combination_id=study.combination("8/40"),
        roles=frozenset({PartitionRole.SELECTION}),
    )
    population = EventPopulation.capture(
        plan,
        universe_candidate(sources.source, study.combination("12/60")),
        DispositionPolicy.ACCEPTED_ONLY,
    )
    with pytest.raises(EventPopulationError):
        assemble_event_dataset(
            sources.windows,
            plan=plan,
            population=population,
            feature_schema=ema_event_feature_schema(),
            target=ForwardReturnBinaryTarget(),
            isolation=HoldoutIsolation.capture(plan, (), "SPY"),
        )
    with pytest.raises(EventPopulationError):
        build_event_dataset(
            plan=plan,
            study_path=study.study_path,
            workspace=study.workspace(),
            combination_id="0" * 64,
            feature_schema=ema_event_feature_schema(),
            target=ForwardReturnBinaryTarget(),
        )
