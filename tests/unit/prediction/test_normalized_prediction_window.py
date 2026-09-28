"""QF-62 schema 3: exact logical equivalence, explicit versions and QF-56 resume."""

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import ContextCompletionPolicy
from quantforge.indicators import NATIVE_INDICATOR_BACKEND
from quantforge.prediction import (
    PredictionContextFailurePolicy,
    PredictionRuleContext,
    PredictionStrategyOutput,
    PredictionWindowResult,
    run_prediction_window,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.study import prepare_prediction_study_dataset
from quantforge.prediction.window_compact import (
    CompactPredictionWindowDecision,
    CompactPredictionWindowResult,
    PredictionWindowEvidence,
)
from quantforge.prediction.window_compact_validation import (
    validate_prediction_window_reader,
)
from quantforge.prediction.window_encoding import canonical, mapping
from quantforge.prediction.window_execution import (
    run_incremental_prediction_window_in_session,
)
from quantforge.prediction.window_incremental import IncrementalPredictionWindowWriter
from quantforge.prediction.window_membership import (
    CATALOGUE_SEGMENTS_FIELD,
    MembershipCatalogues,
    MembershipView,
    prediction_context,
)
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.prediction.window_validation import validate_prediction_window_snapshot
from tests.unit.prediction.test_compact_prediction_window import (
    records,
    rehash_decision,
    verify,
    write_compact,
    write_records,
)
from tests.unit.prediction.test_incremental_prediction_window import validator
from tests.unit.prediction.test_multi_timeframe_study import (
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
from tests.unit.prediction.test_technical_confluence import (
    _rule as confluence_rule,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_technical_confluence import (
    _study as confluence_study,  # pyright: ignore[reportPrivateUsage]
)

PHYSICAL = ("decision_id", "shared_evidence_id")


@pytest.fixture(scope="module")
def window() -> PredictionWindowResult[Any, Any, Any]:
    return run_window()


def normalized(
    window: PredictionWindowResult[Any, Any, Any],
) -> CompactPredictionWindowResult:
    return CompactPredictionWindowResult.from_window(window, schema_version="3")


def logical(record: PrimitiveMapping) -> PrimitiveMapping:
    return {key: value for key, value in record.items() if key not in PHYSICAL}


def assert_equivalent(v2: PredictionWindowReader, v3: PredictionWindowReader) -> None:
    """Every scientific field, including exact membership, is unchanged."""
    assert v2.decision_count == v3.decision_count
    assert v2.header()["record_counts"] == v3.header()["record_counts"]
    for old, new in zip(v2.iterate_decisions(), v3.iterate_decisions(), strict=True):
        assert canonical(logical(new.expanded_record())) == canonical(
            logical(old.to_primitive())
        )


def test_schema_3_reconstructs_schema_2_exactly_with_new_physical_ids(
    window: PredictionWindowResult[Any, Any, Any], tmp_path: Path
) -> None:
    v2 = CompactPredictionWindowResult.from_window(window)
    v3 = normalized(window)
    assert v3.serialize() == normalized(window).serialize()  # Deterministic bytes.
    old = write_compact(tmp_path / "v2.jsonl", v2)
    new = write_compact(tmp_path / "v3.jsonl", v3)
    assert (old.schema_version, new.schema_version) == ("2", "3")
    new.verify_integrity()
    verify(new, window)
    verify(old, window)
    assert_equivalent(old, new)
    # Scientific scope and IDs are shared; physical identities are not.
    assert new.evidence.identity_snapshot == old.evidence.identity_snapshot
    assert new.evidence.evidence_id != old.evidence.evidence_id
    assert new.header()["window_id"] != old.header()["window_id"]
    assert new.header()["schedule_id"] == old.header()["schedule_id"]
    for left, right, original in zip(
        old.iterate_decisions(), new.iterate_decisions(), window.decisions, strict=True
    ):
        a, b = left.to_primitive(), right.to_primitive()
        for field in (
            "context_id",
            "prediction_study_id",
            "generated_signals",
            "status",
        ):
            assert a[field] == b[field] == original.to_primitive()[field]
        assert (
            mapping(a["prediction_study"])["rows"]
            == mapping(b["prediction_study"])["rows"]
        )
        assert a["decision_id"] != b["decision_id"]
        assert b["prediction_study_id"] == original.result.study_id
    header = new.header()
    catalogues = cast(list[dict[str, Any]], header["membership_catalogues"])
    assert len(catalogues) == 4  # 5m, weekly, daily and 4h fixture timeframes.
    assert catalogues == sorted(catalogues, key=lambda item: item["catalogue_id"])


def test_membership_is_physically_stored_once_and_not_reexpanded(
    window: PredictionWindowResult[Any, Any, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2 = CompactPredictionWindowResult.from_window(window).serialize()
    v3 = normalized(window).serialize()
    assert b'"visible_bar_ids"' in v2
    assert b'"visible_bar_ids"' not in v3
    ids = {
        bar
        for decision in window.decisions
        for bar in json.dumps(decision.to_primitive()).split('"')
        if len(bar) == 64
    }
    stored = [
        bar
        for line in v3.splitlines()[2:]
        for segment in cast(
            list[dict[str, Any]], json.loads(line)[CATALOGUE_SEGMENTS_FIELD]
        )
        for bar in segment["bar_ids"]
    ]
    assert len(stored) == len(set(stored))
    assert set(stored) <= ids
    assert all(v3.count(bar.encode()) == 1 for bar in stored)
    assert len(v3) < len(v2)
    path = tmp_path / "v3.jsonl"
    path.write_bytes(v3)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("structural reading must not re-expand membership")

    monkeypatch.setattr(MembershipView, "expand", forbidden)
    reader = PredictionWindowReader.open(path)
    reader.verify_integrity()
    for decision in reader.iterate_decisions():
        record = decision.to_primitive()
        assert decision.schema_version == "3"
        assert '"visible_bar_ids"' not in canonical(prediction_context(record)).decode()


@pytest.mark.parametrize("threshold", ["0", "12.5"])
def test_candidate_rejected_and_no_candidate_decisions_are_equivalent(
    tmp_path: Path, threshold: str
) -> None:
    provider = WindowProvider()
    rule = confluence_rule(
        threshold=Decimal(threshold), backend_id=NATIVE_INDICATOR_BACKEND
    )
    study, _ = confluence_study(rule)
    candidates = run_prediction_window(
        _prediction_dataset(),
        study,
        schedule=schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment={},
    )

    class EmptyRule(WindowRule):
        def generate_with_context(
            self, context: PredictionRuleContext
        ) -> PredictionStrategyOutput:
            return replace(super().generate_with_context(context), signals=())

    empty = run_prediction_window(
        _prediction_dataset(),
        _study(EmptyRule(_requirements())),
        schedule=schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment={},
    )
    for name, item, parameters in (
        ("candidates", candidates, rule.parameters.to_primitive()),
        ("empty", empty, None),
    ):
        old = write_compact(
            tmp_path / f"{name}-v2.jsonl",
            CompactPredictionWindowResult.from_window(item),
        )
        new = write_compact(tmp_path / f"{name}-v3.jsonl", normalized(item))
        assert_equivalent(old, new)
        if parameters is None:
            verify(new, item)
            assert (
                mapping(new.header()["record_counts"])["no_prediction_decisions"] == 4
            )
        else:
            validate_prediction_window_reader(
                new,
                expected_identity=item.identity_snapshot,
                schedule=item.schedule,
                outcome_sessions=tuple(
                    bar.session_date for bar in _prediction_dataset().bars
                ),
                strategy_parameters=parameters,
            )


@pytest.mark.parametrize(
    "change",
    [
        "unknown_header",
        "header_2_evidence_3",
        "header_3_evidence_2",
        "v2_decision_in_v3",
        "v3_decision_in_v2",
    ],
)
def test_versions_are_explicit_and_mixed_forms_fail_closed(
    window: PredictionWindowResult[Any, Any, Any], tmp_path: Path, change: str
) -> None:
    v2, v3 = (
        records(CompactPredictionWindowResult.from_window(window)),
        records(normalized(window)),
    )
    if change == "unknown_header":
        items = v3
        items[0]["schema_version"] = "4"
    elif change == "header_2_evidence_3":
        items = [v2[0], v3[1], *v3[2:]]
    elif change == "header_3_evidence_2":
        items = [v3[0], v2[1], *v2[2:]]
    elif change == "v2_decision_in_v3":
        items = v3
        replacement = dict(v2[2])
        replacement["shared_evidence_id"] = PredictionWindowEvidence.from_primitive(
            v3[1]
        ).evidence_id
        rehash_decision(replacement)
        items[2] = replacement
    else:
        items = v2
        replacement = dict(v3[2])
        replacement["shared_evidence_id"] = PredictionWindowEvidence.from_primitive(
            v2[1]
        ).evidence_id
        rehash_decision(replacement)
        items[2] = replacement
    path = tmp_path / "mixed.jsonl"
    write_records(path, items)
    with pytest.raises(InvalidPredictionOutputError):
        PredictionWindowReader.open(path).verify_integrity()


def test_legacy_embedded_validator_rejects_normalized_inspection_form(
    window: PredictionWindowResult[Any, Any, Any],
) -> None:
    with pytest.raises(InvalidPredictionOutputError):
        validate_prediction_window_snapshot(
            normalized(window).to_primitive(),
            expected_identity=window.identity_snapshot,
            schedule=window.schedule,
            outcome_sessions=(),
            strategy_parameters={},
        )


def rebuilt(
    compact: CompactPredictionWindowResult, items: list[PrimitiveMapping]
) -> CompactPredictionWindowResult:
    """Re-derive physical IDs and header, as an adversary could."""
    decisions: list[CompactPredictionWindowDecision] = []
    for item in items:
        rehash_decision(item)
        decisions.append(
            CompactPredictionWindowDecision(PrimitiveMappingSnapshot.capture(item))
        )
    return CompactPredictionWindowResult(compact.evidence, tuple(decisions))


def source_range(item: PrimitiveMapping, index: int) -> dict[str, Any]:
    source = mapping(prediction_context(item)["source_context"])
    return cast(list[dict[str, Any]], source["timeframes"])[index]["visible_bar_range"]


@pytest.mark.parametrize("change", ["reordered", "shifted", "narrowed"])
def test_rehashed_membership_changes_fail_scientific_context_identity(
    window: PredictionWindowResult[Any, Any, Any], tmp_path: Path, change: str
) -> None:
    compact = normalized(window)
    items = [decision.to_primitive() for decision in compact.decisions]
    if change == "reordered":
        segment = cast(list[dict[str, Any]], items[0][CATALOGUE_SEGMENTS_FIELD])[0]
        segment["bar_ids"][0], segment["bar_ids"][1] = (
            segment["bar_ids"][1],
            segment["bar_ids"][0],
        )
    elif change == "shifted":
        source_range(items[3], 0)["start_index"] += 1
    else:
        source_range(items[3], 0)["stop_index"] -= 1
    changed = rebuilt(compact, items)
    reader = write_compact(tmp_path / "changed.jsonl", changed)
    # Ranges, segments, decision IDs and the header are all self-consistent, but
    # the reconstructed membership no longer matches the original QF-11/QF-20 IDs.
    reader.verify_integrity()
    with pytest.raises(InvalidPredictionOutputError, match=r"(study|context) identity"):
        verify(reader, window)


@pytest.mark.parametrize(
    "change",
    ["segment_bar", "reference", "catalogue_summary", "catalogue_count", "missing"],
)
def test_unrehashed_membership_corruption_fails_structural_integrity(
    window: PredictionWindowResult[Any, Any, Any], tmp_path: Path, change: str
) -> None:
    items = records(normalized(window))
    header = items[0]
    catalogues = cast(list[dict[str, Any]], header["membership_catalogues"])
    if change == "segment_bar":
        segment = cast(list[dict[str, Any]], items[2][CATALOGUE_SEGMENTS_FIELD])[0]
        segment["bar_ids"][0] = "0" * 64
    elif change == "reference":
        source_range(items[5], 0)["start_index"] += 1
    elif change == "catalogue_summary":
        catalogues[0]["catalogue_content_id"] = "0" * 64
    elif change == "catalogue_count":
        catalogues[0]["bar_count"] += 1
    else:
        del header["membership_catalogues"]
    path = tmp_path / "corrupt.jsonl"
    write_records(path, items)
    with pytest.raises(InvalidPredictionOutputError):
        PredictionWindowReader.open(path).verify_integrity()


def execute(path: Path, provider: WindowProvider) -> PredictionWindowReader:
    return run_incremental_prediction_window_in_session(
        prepare_prediction_study_dataset(_prediction_dataset()),
        _study(WindowRule(_requirements())),
        path=path,
        schedule=schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment={"provider": "immutable_fixture", "version": "1"},
        schema_version="3",
    )


@pytest.mark.parametrize("prefix", [0, 1, 3, 4])
def test_incremental_resume_publishes_the_same_normalized_bytes(
    tmp_path: Path, window: PredictionWindowResult[Any, Any, Any], prefix: int
) -> None:
    compact = normalized(window)
    path = tmp_path / "window.jsonl"
    writer = IncrementalPredictionWindowWriter.open(
        path, validator=validator(window), schema_version="3"
    )
    assert writer.checkpoint()["schema_version"] == "3"
    for decision in compact.decisions[:prefix]:
        writer.append(decision)
    resumed = IncrementalPredictionWindowWriter.open(
        path, validator=validator(window), schema_version="3"
    )
    assert resumed.checkpoint() == writer.checkpoint()
    assert resumed.membership is not None
    assert resumed.membership.lengths() == (
        writer.membership.lengths() if writer.membership is not None else None
    )
    for decision in compact.decisions[prefix:]:
        resumed.append(decision)
    reader = resumed.finalize()
    assert path.read_bytes() == compact.serialize()
    verify(reader, window)
    assert IncrementalPredictionWindowWriter.open(
        path, validator=validator(window), schema_version="3"
    ).finalized


@pytest.mark.parametrize("prefix", [0, 1, 3])
def test_execution_interruption_resumes_without_recomputing_prefix(
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
    resumed_calls: list[object] = []

    def resumed(*args: Any, **kwargs: Any) -> Any:
        resumed_calls.append(kwargs["as_of"])
        return original(*args, **kwargs)

    monkeypatch.setattr(provider, "get_context_at", resumed)
    reader = execute(path, provider)
    assert resumed_calls == list(window.schedule.decision_timestamps[prefix:])
    assert path.read_bytes() == normalized(window).serialize()
    verify(reader, window)
    resumed_calls.clear()
    execute(path, provider)
    assert not resumed_calls


@pytest.mark.parametrize("change", ["segment", "reference", "schema_2_writer"])
def test_corrupt_or_incompatible_normalized_prefix_fails_closed(
    tmp_path: Path, window: PredictionWindowResult[Any, Any, Any], change: str
) -> None:
    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "window.jsonl", validator=validator(window), schema_version="3"
    )
    for decision in normalized(window).decisions[:3]:
        writer.append(decision)
    with writer.journal_path.open("ab") as stream:
        stream.write(b'{"uncommitted":')
    before = writer.journal_path.read_bytes()
    if change != "schema_2_writer":
        lines = before.splitlines(keepends=True)
        record = mapping(json.loads(lines[1]))
        if change == "segment":
            segment = cast(list[dict[str, Any]], record[CATALOGUE_SEGMENTS_FIELD])[0]
            segment["bar_ids"] = [*segment["bar_ids"], "f" * 64]
        else:
            source_range(record, 0)["stop_index"] -= 1
        rehash_decision(record)
        lines[1] = canonical(record) + b"\n"
        before = b"".join(lines)
        writer.journal_path.write_bytes(before)
    with pytest.raises(InvalidPredictionOutputError):
        IncrementalPredictionWindowWriter.open(
            writer.path,
            validator=validator(window),
            schema_version="2" if change == "schema_2_writer" else "3",
        )
    assert writer.journal_path.read_bytes() == before  # No truncation or reset.
    assert not writer.path.exists()


def test_failed_append_rolls_back_catalogue_growth(
    tmp_path: Path, window: PredictionWindowResult[Any, Any, Any]
) -> None:
    compact = normalized(window)
    writer = IncrementalPredictionWindowWriter.open(
        tmp_path / "window.jsonl", validator=validator(window), schema_version="3"
    )
    writer.append(compact.decisions[0])
    assert writer.membership is not None
    before = writer.membership.lengths()
    corrupt = compact.decisions[1].to_primitive()
    corrupt["prediction_study_id"] = "0" * 64
    rehash_decision(corrupt)
    with pytest.raises(InvalidPredictionOutputError):
        writer.append(
            CompactPredictionWindowDecision(PrimitiveMappingSnapshot.capture(corrupt))
        )
    assert writer.membership.lengths() == before
    assert writer.completed_count == 1
    for decision in compact.decisions[1:]:
        writer.append(decision)
    assert writer.finalize().path.read_bytes() == compact.serialize()


def test_schema_2_conversion_rejects_catalogue_state(
    window: PredictionWindowResult[Any, Any, Any],
) -> None:
    evidence = PredictionWindowEvidence(window.identity_snapshot)
    with pytest.raises(InvalidPredictionOutputError, match="schema 2"):
        CompactPredictionWindowDecision.from_embedded(
            window.decisions[0].to_primitive(),
            sequence=0,
            evidence=evidence,
            membership=MembershipCatalogues(),
        )
    normalized_evidence = PredictionWindowEvidence(window.identity_snapshot, "3")
    with pytest.raises(InvalidPredictionOutputError, match="catalogue state"):
        CompactPredictionWindowDecision.from_embedded(
            window.decisions[0].to_primitive(), sequence=0, evidence=normalized_evidence
        )
    with pytest.raises(InvalidPredictionOutputError, match="version"):
        PredictionWindowEvidence(window.identity_snapshot, "4")
    assert configuration_identity(evidence.to_primitive()) == evidence.evidence_id


@pytest.mark.parametrize("kind", ["empty", "skipped", "developing_skipped"])
def test_empty_and_skipped_windows_keep_explicit_membership(
    tmp_path: Path, kind: str
) -> None:
    if kind == "empty":
        item = run_window(
            decision_schedule=schedule(
                START + timedelta(days=2), START + timedelta(days=2)
            )
        )
    else:
        requirements = _requirements(failure_policy=PredictionContextFailurePolicy.SKIP)
        provider = WindowProvider()
        if kind == "skipped":
            provider.series = ()
        else:
            requirements = replace(
                requirements,
                contextual=tuple(
                    replace(
                        declared,
                        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
                    )
                    for declared in requirements.contextual
                ),
            )
        item = run_window(provider, requirements=requirements)
    old = write_compact(
        tmp_path / "v2.jsonl", CompactPredictionWindowResult.from_window(item)
    )
    new = write_compact(tmp_path / "v3.jsonl", normalized(item))
    verify(new, item)
    assert_equivalent(old, new)
    assert new.decision_count == len(item.decisions)
    if kind != "empty":
        assert mapping(new.header()["record_counts"])["skipped_decisions"] == 4
    assert all(
        not decision.to_primitive()[CATALOGUE_SEGMENTS_FIELD]
        for decision in new.iterate_decisions()
    )
    assert new.header()["membership_catalogues"] == []
