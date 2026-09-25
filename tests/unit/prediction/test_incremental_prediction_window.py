"""Durable exact prefixes and unchanged QF-11 execution/QF-55 final bytes."""

import json
import weakref
from dataclasses import fields
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import (
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.study import (
    PredictionStudyResult,
    prepare_prediction_study_dataset,
)
from quantforge.prediction.window import PredictionWindowResult
from quantforge.prediction.window_compact import (
    CompactPredictionWindowDecision,
    CompactPredictionWindowResult,
)
from quantforge.prediction.window_compact_validation import (
    PredictionWindowDecisionValidator,
)
from quantforge.prediction.window_encoding import canonical, mapping
from quantforge.prediction.window_execution import (
    run_incremental_prediction_window_in_session,
)
from quantforge.prediction.window_incremental import IncrementalPredictionWindowWriter
from quantforge.prediction.window_reader import PredictionWindowReader
from tests.unit.prediction.test_compact_prediction_window import rehash_decision, verify
from tests.unit.prediction.test_multi_timeframe_study import (
    FixtureParameters,
    _prediction_dataset,  # pyright: ignore[reportPrivateUsage]
    _requirements,  # pyright: ignore[reportPrivateUsage]
    _study,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_window import (
    START,
    WindowProvider,
    WindowRule,
    run_window,
    schedule,
)


@pytest.fixture(scope="module")
def window() -> PredictionWindowResult[Any, Any, Any]:
    return run_window()


def validator(
    window: PredictionWindowResult[Any, Any, Any],
) -> PredictionWindowDecisionValidator:
    return PredictionWindowDecisionValidator(
        expected_identity=window.identity_snapshot,
        schedule=window.schedule,
        outcome_sessions=tuple(bar.session_date for bar in _prediction_dataset().bars),
        strategy_parameters=FixtureParameters().to_primitive(),
    )


def execute(path: Path, provider: WindowProvider) -> PredictionWindowReader:
    return run_incremental_prediction_window_in_session(
        prepare_prediction_study_dataset(_prediction_dataset()),
        _study(WindowRule(_requirements())),
        path=path,
        schedule=schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment={"provider": "immutable_fixture", "version": "1"},
    )


@pytest.mark.parametrize("prefix", [0, 1, 3, 4])
def test_lifecycle_resume_and_canonical_final_bytes(
    tmp_path: Path,
    window: PredictionWindowResult[Any, Any, Any],
    prefix: int,
) -> None:
    compact = CompactPredictionWindowResult.from_window(window)
    path = tmp_path / "window.jsonl"
    writer = IncrementalPredictionWindowWriter.open(path, validator=validator(window))
    assert writer.checkpoint()["completed_count"] == 0
    assert not path.exists()
    for decision in compact.decisions[:prefix]:
        writer.append(decision)
    assert writer.completed_count == prefix
    if prefix < 4:
        with pytest.raises(InvalidPredictionOutputError, match="incomplete"):
            writer.finalize()
    resumed = IncrementalPredictionWindowWriter.open(path, validator=validator(window))
    assert resumed.checkpoint() == writer.checkpoint()
    for decision in compact.decisions[prefix:]:
        resumed.append(decision)
    reader = resumed.finalize()
    assert path.read_bytes() == compact.serialize()
    verify(reader, window)
    assert not writer.staging_path.exists()
    with pytest.raises(InvalidPredictionOutputError, match="finalized"):
        resumed.append(compact.decisions[0])
    before = path.stat().st_mtime_ns
    existing = IncrementalPredictionWindowWriter.open(path, validator=validator(window))
    assert existing.finalized
    assert existing.finalize().header() == reader.header()
    assert path.stat().st_mtime_ns == before


@pytest.mark.parametrize("indexes", [(1,), (0, 2), (0, 1, 3), (0, 0), (0, 1, 1)])
def test_prefix_rejects_skip_reorder_and_duplicate(
    tmp_path: Path,
    window: PredictionWindowResult[Any, Any, Any],
    indexes: tuple[int, ...],
) -> None:
    compact = CompactPredictionWindowResult.from_window(window)
    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "window.jsonl", validator=validator(window)
    )
    for index in indexes[:-1]:
        writer.append(compact.decisions[index])
    with pytest.raises(InvalidPredictionOutputError, match="reference/order"):
        writer.append(compact.decisions[indexes[-1]])
    assert writer.completed_count == len(indexes) - 1


def test_foreign_evidence_and_rehashed_scientific_corruption_reject(
    tmp_path: Path,
    window: PredictionWindowResult[Any, Any, Any],
) -> None:
    compact = CompactPredictionWindowResult.from_window(window)
    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "window.jsonl", validator=validator(window)
    )
    for decision in compact.decisions[:3]:
        writer.append(decision)
    foreign = compact.decisions[3].to_primitive()
    foreign["shared_evidence_id"] = "foreign"
    rehash_decision(foreign)
    with pytest.raises(InvalidPredictionOutputError, match="reference/order"):
        writer.append(
            CompactPredictionWindowDecision(PrimitiveMappingSnapshot.capture(foreign))
        )
    corrupt = compact.decisions[3].to_primitive()
    corrupt["prediction_study_id"] = "foreign"
    rehash_decision(corrupt)
    with pytest.raises(InvalidPredictionOutputError):
        writer.append(
            CompactPredictionWindowDecision(PrimitiveMappingSnapshot.capture(corrupt))
        )
    assert writer.completed_count == 3


@pytest.mark.parametrize(
    "change",
    [
        "truncated",
        "checkpoint",
        "count",
        "last",
        "evidence",
        "schedule",
        "version",
        "gap",
        "decision",
    ],
)
def test_corruption_fails_closed(
    tmp_path: Path,
    window: PredictionWindowResult[Any, Any, Any],
    change: str,
) -> None:
    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "window.jsonl", validator=validator(window)
    )
    for decision in CompactPredictionWindowResult.from_window(window).decisions[:3]:
        writer.append(decision)
    checkpoint_path = writer.staging_path / "checkpoint.json"
    path = (
        writer.journal_path
        if change in ("truncated", "gap", "decision")
        else checkpoint_path
    )
    if change == "truncated":
        path.write_bytes(path.read_bytes()[:-4])
    elif change in ("gap", "decision"):
        lines = path.read_bytes().splitlines(keepends=True)
        if change == "gap":
            del lines[1]
        else:
            record = mapping(json.loads(lines[2]))
            record["decision_timestamp"] = START.isoformat()
            rehash_decision(record)
            lines[2] = canonical(record) + b"\\n"
        path.write_bytes(b"".join(lines))
    else:
        envelope = mapping(json.loads(path.read_bytes()))
        checkpoint = mapping(envelope["checkpoint"])
        if change == "checkpoint":
            envelope["fingerprint"] = "corrupt"
        else:
            key, value = {
                "count": ("completed_count", 2),
                "last": ("last_decision_id", "foreign"),
                "evidence": ("shared_evidence_id", "foreign"),
                "schedule": ("schedule_id", "foreign"),
                "version": ("schema_version", "999"),
            }[change]
            checkpoint[key] = value
            envelope["fingerprint"] = configuration_identity(checkpoint)
        path.write_bytes(canonical(envelope) + b"\\n")
    before = path.read_bytes()
    with pytest.raises(InvalidPredictionOutputError):
        IncrementalPredictionWindowWriter.open(writer.path, validator=validator(window))
    assert not writer.path.exists()
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    ["rule", "combination", "source", "schedule", "order", "validation", "outcome"],
)
def test_material_incompatibility_preserves_prefix(
    tmp_path: Path,
    window: PredictionWindowResult[Any, Any, Any],
    change: str,
) -> None:
    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "window.jsonl", validator=validator(window)
    )
    writer.append(CompactPredictionWindowResult.from_window(window).decisions[0])
    identity = window.identity_snapshot.to_primitive()
    changed_schedule = window.schedule
    if change in ("rule", "combination"):
        mapping(
            mapping(mapping(identity["configuration"])["prediction_rule"])[
                "configuration"
            ]
        )["parameters"] = {"threshold": change}
    elif change == "source":
        mapping(identity["market_data"])["dataset_id"] = "other-source"
    elif change == "validation":
        mapping(identity["context_environment"])["validation_window"] = "other"
    elif change == "outcome":
        mapping(mapping(identity["configuration"])["outcome_labeler"])["version"] = (
            "other"
        )
    elif change == "schedule":
        changed_schedule = schedule(
            START + timedelta(minutes=5), window.schedule.end_timestamp
        )
        identity["schedule"] = changed_schedule.to_primitive()
    else:
        mapping(identity["schedule"])["decision_timestamps"] = list(
            reversed(
                cast(list[Any], mapping(identity["schedule"])["decision_timestamps"])
            )
        )
    before = writer.journal_path.read_bytes()

    def open_changed() -> None:
        other = PredictionWindowDecisionValidator(
            expected_identity=PrimitiveMappingSnapshot.capture(identity),
            schedule=changed_schedule,
            outcome_sessions=(),
            strategy_parameters=FixtureParameters().to_primitive(),
        )
        IncrementalPredictionWindowWriter.open(writer.path, validator=other)

    with pytest.raises(InvalidPredictionOutputError):
        open_changed()
    assert writer.journal_path.read_bytes() == before


@pytest.mark.parametrize("prefix", [0, 1, 3])
def test_execution_interruption_reuses_prefix(
    tmp_path: Path,
    window: PredictionWindowResult[Any, Any, Any],
    prefix: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = WindowProvider()
    original = provider.get_context_at
    calls: list[object] = []

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        if len(calls) == prefix:
            raise KeyboardInterrupt
        calls.append(kwargs["as_of"])
        return original(*args, **kwargs)

    monkeypatch.setattr(provider, "get_context_at", interrupted)
    path = tmp_path / "window.jsonl"
    with pytest.raises(KeyboardInterrupt):
        execute(path, provider)
    writer = IncrementalPredictionWindowWriter.open(path, validator=validator(window))
    assert writer.completed_count == prefix
    resumed_calls: list[object] = []

    def resumed(*args: Any, **kwargs: Any) -> Any:
        resumed_calls.append(kwargs["as_of"])
        return original(*args, **kwargs)

    monkeypatch.setattr(provider, "get_context_at", resumed)
    reader = execute(path, provider)
    assert resumed_calls == list(window.schedule.decision_timestamps[prefix:])
    assert (
        reader.header()["window_result_id"]
        == CompactPredictionWindowResult.from_window(window).window_result_id
    )
    resumed_calls.clear()
    execute(path, provider)
    assert not resumed_calls


@pytest.mark.parametrize(
    "stage", ["context", "execution", "validation", "conversion", "persistence"]
)
def test_failure_preserves_prefix_and_retries_exact_decision(
    tmp_path: Path,
    window: PredictionWindowResult[Any, Any, Any],
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = {
        "context": "quantforge.prediction.window._DecisionContextProvider.get_context",
        "execution": "quantforge.prediction.window.run_prediction_study_in_session",
        "validation": (
            "quantforge.prediction.window_compact_validation."
            "PredictionWindowDecisionValidator.validate"
        ),
        "conversion": (
            "quantforge.prediction.window_execution."
            "CompactPredictionWindowDecision.from_embedded"
        ),
        "persistence": "quantforge.prediction.window_incremental._write_record",
    }
    module_name, name = targets[stage].rsplit(".", 1)
    import importlib

    if stage == "context":
        from quantforge.prediction.window import (
            _DecisionContextProvider,  # pyright: ignore[reportPrivateUsage]
        )

        owner: Any = _DecisionContextProvider
    elif stage == "validation":
        owner = PredictionWindowDecisionValidator
    elif stage == "conversion":
        owner = CompactPredictionWindowDecision
    else:
        owner = importlib.import_module(module_name)
    original = getattr(owner, name)
    calls = 0

    def failed(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        # Initialization writes shared evidence and the zero checkpoint.
        if calls == (5 if stage == "persistence" else 3):
            raise OSError("injected")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, failed)
    path = tmp_path / "window.jsonl"
    with pytest.raises(InvalidPredictionOutputError, match=r"sequence=2.*timestamp="):
        execute(path, WindowProvider())
    monkeypatch.setattr(owner, name, original)
    writer = IncrementalPredictionWindowWriter.open(path, validator=validator(window))
    assert writer.completed_count == 2
    assert not writer.finalized
    assert not path.exists()
    execute(path, WindowProvider())
    assert (
        path.read_bytes()
        == CompactPredictionWindowResult.from_window(window).serialize()
    )


def test_full_results_released_before_next_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quantforge.prediction import window as execution

    class TrackedResult(PredictionStudyResult[Any, Any, Any]):
        pass

    original = execution.run_prediction_study_in_session
    references: list[weakref.ReferenceType[TrackedResult]] = []

    def tracked(*args: Any, **kwargs: Any) -> Any:
        assert all(reference() is None for reference in references)
        result = original(*args, **kwargs)
        values = {field.name: getattr(result, field.name) for field in fields(result)}
        observed = TrackedResult(**values)
        references.append(weakref.ref(observed))
        return observed

    monkeypatch.setattr(execution, "run_prediction_study_in_session", tracked)
    execute(tmp_path / "window.jsonl", WindowProvider())
    assert len(references) == 4
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("tail", [b'{"partial":', b"complete but uncommitted\n"])
def test_recovery_truncates_only_uncommitted_tail(
    tmp_path: Path,
    window: PredictionWindowResult[Any, Any, Any],
    tail: bytes,
) -> None:
    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "window.jsonl", validator=validator(window)
    )
    writer.append(CompactPredictionWindowResult.from_window(window).decisions[0])
    before = writer.journal_path.read_bytes()
    with writer.journal_path.open("ab") as stream:
        stream.write(tail)
    resumed = IncrementalPredictionWindowWriter.open(
        writer.path, validator=validator(window)
    )
    assert resumed.completed_count == 1
    assert resumed.journal_path.read_bytes() == before


def test_invalid_checkpoint_never_truncates_evidence(
    tmp_path: Path,
    window: PredictionWindowResult[Any, Any, Any],
) -> None:
    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "window.jsonl", validator=validator(window)
    )
    writer.append(CompactPredictionWindowResult.from_window(window).decisions[0])
    with writer.journal_path.open("ab") as stream:
        stream.write(b"uncommitted tail")
    before = writer.journal_path.read_bytes()
    path = writer.staging_path / "checkpoint.json"
    checkpoint = mapping(json.loads(path.read_bytes()))
    mapping(checkpoint["checkpoint"])["completed_count"] = 0
    path.write_bytes(canonical(checkpoint) + b"\n")
    with pytest.raises(InvalidPredictionOutputError):
        IncrementalPredictionWindowWriter.open(writer.path, validator=validator(window))
    assert writer.journal_path.read_bytes() == before


@pytest.mark.parametrize("after_publish", [False, True])
def test_checkpoint_interruption_uses_atomic_commit_boundary(
    tmp_path: Path,
    window: PredictionWindowResult[Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    after_publish: bool,
) -> None:
    from quantforge.prediction import window_incremental as persistence

    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "window.jsonl", validator=validator(window)
    )
    compact = CompactPredictionWindowResult.from_window(window)
    writer.append(compact.decisions[0])
    original = persistence._publish  # pyright: ignore[reportPrivateUsage]

    def interrupt(*args: Any, **kwargs: Any) -> None:
        if after_publish:
            original(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(persistence, "_publish", interrupt)
    with pytest.raises(KeyboardInterrupt):
        writer.append(compact.decisions[1])
    assert writer.completed_count == 1
    monkeypatch.setattr(persistence, "_publish", original)
    resumed = IncrementalPredictionWindowWriter.open(
        writer.path, validator=validator(window)
    )
    assert resumed.completed_count == (2 if after_publish else 1)
    for decision in compact.decisions[resumed.completed_count :]:
        resumed.append(decision)
    assert resumed.finalize().path.read_bytes() == compact.serialize()


@pytest.mark.parametrize("after_publish", [False, True])
def test_finalization_failure_preserves_recoverable_work(
    tmp_path: Path,
    window: PredictionWindowResult[Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    after_publish: bool,
) -> None:
    from quantforge.prediction import window_incremental as persistence

    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "window.jsonl", validator=validator(window)
    )
    compact = CompactPredictionWindowResult.from_window(window)
    for decision in compact.decisions:
        writer.append(decision)
    original = persistence._publish  # pyright: ignore[reportPrivateUsage]

    def failed(*args: Any, **kwargs: Any) -> None:
        if after_publish:
            original(*args, **kwargs)
        raise OSError("injected")

    monkeypatch.setattr(persistence, "_publish", failed)
    with pytest.raises(OSError, match="injected"):
        writer.finalize()
    monkeypatch.setattr(persistence, "_publish", original)
    recovered = IncrementalPredictionWindowWriter.open(
        writer.path, validator=validator(window)
    )
    assert recovered.completed_count == 4
    assert recovered.finalized is after_publish
    assert recovered.finalize().path.read_bytes() == compact.serialize()


def test_empty_schedule_finalizes(tmp_path: Path) -> None:
    window = run_window(
        decision_schedule=schedule(
            START + timedelta(seconds=1), START + timedelta(seconds=2)
        )
    )
    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "window.jsonl", validator=validator(window)
    )
    reader = writer.finalize()
    assert reader.decision_count == 0
    assert (
        reader.path.read_bytes()
        == CompactPredictionWindowResult.from_window(window).serialize()
    )
