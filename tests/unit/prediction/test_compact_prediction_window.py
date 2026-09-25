"""Compact representation, strict versions/references and unchanged QF-42 semantics."""

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
from quantforge.indicators import NATIVE_INDICATOR_BACKEND
from quantforge.prediction import (
    PredictionContextFailurePolicy,
    PredictionRuleContext,
    PredictionStrategyOutput,
    PredictionWindowResult,
    run_prediction_window,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_compact import (
    CompactPredictionWindowDecision,
    CompactPredictionWindowResult,
    PredictionWindowEvidence,
)
from quantforge.prediction.window_compact_validation import (
    validate_prediction_window_reader,
)
from quantforge.prediction.window_encoding import (
    StudyIdentity,
    canonical,
    mapping,
    ordered_result_identity,
)
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.prediction.window_validation import validate_prediction_window_snapshot
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
    _provider_with_final_session,  # pyright: ignore[reportPrivateUsage]
    run_window,
    schedule,
)
from tests.unit.prediction.test_technical_confluence import (
    _rule as confluence_rule,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_technical_confluence import (
    _study as confluence_study,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture(scope="module")
def window() -> PredictionWindowResult[Any, Any, Any]:
    return run_window()


def write_compact(
    path: Path, compact: CompactPredictionWindowResult
) -> PredictionWindowReader:
    path.write_bytes(compact.serialize())
    return PredictionWindowReader.open(path)


def verify(
    reader: PredictionWindowReader, window: PredictionWindowResult[Any, Any, Any]
) -> None:
    validate_prediction_window_reader(
        reader,
        expected_identity=window.identity_snapshot,
        schedule=window.schedule,
        outcome_sessions=tuple(bar.session_date for bar in _prediction_dataset().bars),
        strategy_parameters=FixtureParameters().to_primitive(),
    )


def records(compact: CompactPredictionWindowResult) -> list[PrimitiveMapping]:
    return [
        cast(PrimitiveMapping, json.loads(line))
        for line in compact.serialize().splitlines()
    ]


def write_records(path: Path, items: list[PrimitiveMapping]) -> None:
    path.write_bytes(b"".join(canonical(item) + b"\n" for item in items))


def rehash_decision(record: PrimitiveMapping) -> None:
    record["decision_id"] = configuration_identity(
        {key: value for key, value in record.items() if key != "decision_id"}
    )


@pytest.mark.parametrize("version", ["1", "2"])
def test_versions_share_normalized_reader_and_preserve_every_decision(
    window: PredictionWindowResult[Any, Any, Any],
    tmp_path: Path,
    version: str,
) -> None:
    compact = CompactPredictionWindowResult.from_window(window)
    path = tmp_path / "window.json"
    path.write_bytes(window.serialize() if version == "1" else compact.serialize())
    before = path.read_bytes()
    reader = PredictionWindowReader.open(path)
    assert reader.schema_version == version
    assert reader.decision_count == len(window.decisions) == 4
    assert reader.shared_evidence().identity_snapshot == window.identity_snapshot
    reader.verify_integrity()
    verify(reader, window)
    normalized = tuple(reader.iterate_decisions())
    assert normalized == compact.decisions
    assert len({item.decision_id for item in normalized}) == 4
    for index, (local, original) in enumerate(
        zip(normalized, window.decisions, strict=True)
    ):
        record, old = local.to_primitive(), original.to_primitive()
        assert record["decision_timestamp"] == original.decision_timestamp.isoformat()
        assert record["sequence"] == index
        assert record["shared_evidence_id"] == compact.evidence.evidence_id
        for field in (
            "context_id",
            "status",
            "prediction_study_id",
            "generated_signals",
        ):
            assert record[field] == old[field]
        study, old_study = (
            mapping(record["prediction_study"]),
            mapping(old["prediction_study"]),
        )
        assert study["rows"] == old_study["rows"]
        manifest, old_manifest = (
            mapping(study["manifest"]),
            mapping(old_study["manifest"]),
        )
        assert not {"market_data", "configuration", "engine_version"} & manifest.keys()
        assert manifest == {
            key: value
            for key, value in old_manifest.items()
            if key not in {"market_data", "configuration", "engine_version"}
        }
    assert path.read_bytes() == before  # Reading never rewrites persisted v1.


def test_canonical_hashes_and_round_trip(
    window: PredictionWindowResult[Any, Any, Any], tmp_path: Path
) -> None:
    compact = CompactPredictionWindowResult.from_window(window)
    assert compact == CompactPredictionWindowResult.from_window(window)
    assert compact.window_id != window.window_id
    expected: PrimitiveMapping = {
        "window_id": compact.window_id,
        "decisions": [d.to_primitive() for d in compact.decisions],
    }
    assert compact.window_result_id == configuration_identity(expected)
    assert (
        ordered_result_identity(
            compact.window_id, (d.to_primitive() for d in compact.decisions)
        )
        == compact.window_result_id
    )
    identity = window.identity_snapshot.to_primitive()
    hasher = StudyIdentity(identity)
    for decision in window.decisions:
        context = mapping(
            mapping(mapping(decision.to_primitive()["prediction_study"])["manifest"])[
                "prediction_context"
            ]
        )
        assert hasher.for_context(context) == decision.result.study_id
    reader = write_compact(tmp_path / "window.jsonl", compact)
    rebuilt = CompactPredictionWindowResult(
        reader.shared_evidence(), tuple(reader.iterate_decisions())
    )
    assert rebuilt.serialize() == compact.serialize()
    detached = reader.shared_evidence().identity_snapshot.to_primitive()
    detached["market_data"] = None
    assert reader.shared_evidence() == compact.evidence


def test_pretty_legacy_remains_readable(
    window: PredictionWindowResult[Any, Any, Any], tmp_path: Path
) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(window.to_primitive(), indent=2))
    verify(PredictionWindowReader.open(path), window)


def test_shared_scope_comparison_keeps_json_types_distinct(
    window: PredictionWindowResult[Any, Any, Any],
) -> None:
    evidence = PredictionWindowEvidence(window.identity_snapshot)
    decision = window.decisions[0].to_primitive()
    market = mapping(
        mapping(mapping(decision["prediction_study"])["manifest"])["market_data"]
    )
    assert market["corporate_actions_complete"] is True
    market["corporate_actions_complete"] = 1
    with pytest.raises(InvalidPredictionOutputError, match="foreign"):
        CompactPredictionWindowDecision.from_embedded(
            decision, sequence=0, evidence=evidence
        )


@pytest.mark.parametrize("version", [None, 2, True, "", "3", "1"])
def test_compact_unknown_corrupted_or_legacy_version_rejects(
    window: PredictionWindowResult[Any, Any, Any], tmp_path: Path, version: object
) -> None:
    items = records(CompactPredictionWindowResult.from_window(window))
    items[0]["schema_version"] = cast(Any, version)
    path = tmp_path / "window.jsonl"
    write_records(path, items)
    with pytest.raises(InvalidPredictionOutputError):
        PredictionWindowReader.open(path)


@pytest.mark.parametrize("version", [None, 1, True, "2", "3"])
def test_embedded_version_fails_closed_even_with_rehashed_identity(
    window: PredictionWindowResult[Any, Any, Any], tmp_path: Path, version: object
) -> None:
    snapshot = window.to_primitive()
    manifest = mapping(snapshot["manifest"])
    manifest["schema_version"] = cast(Any, version)
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"window_id", "window_result_id", "schedule_id", "record_counts"}
    }
    manifest["window_id"] = configuration_identity(identity)
    manifest["window_result_id"] = configuration_identity(
        {"window_id": manifest["window_id"], "decisions": snapshot["decisions"]}
    )
    path = tmp_path / "window.json"
    path.write_bytes(canonical(snapshot))
    with pytest.raises(InvalidPredictionOutputError):
        PredictionWindowReader.open(path)
    with pytest.raises(InvalidPredictionOutputError, match="schema version"):
        validate_prediction_window_snapshot(
            snapshot,
            expected_identity=PrimitiveMappingSnapshot.capture(identity),
            schedule=window.schedule,
            outcome_sessions=(),
            strategy_parameters={},
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_shared",
        "wrong_shared",
        "foreign_shared",
        "missing_reference",
        "wrong_reference",
        "foreign_reference",
        "reordered",
        "duplicate",
        "truncated",
        "extra",
        "count",
        "result_id",
        "embedded",
        "decision_id",
        "counts",
        "sequence",
    ],
)
def test_invalid_structure_references_or_membership_reject(
    window: PredictionWindowResult[Any, Any, Any], tmp_path: Path, mutation: str
) -> None:
    compact = CompactPredictionWindowResult.from_window(window)
    items = records(compact)
    if mutation == "missing_shared":
        items.pop(1)
    elif mutation in {"wrong_shared", "foreign_shared"}:
        mapping(items[1]["window_identity"])["context_environment"] = {
            "provider": "other-window"
        }
        if mutation == "foreign_shared":
            foreign = PredictionWindowEvidence.from_primitive(items[1])
            items[0]["shared_evidence_id"] = foreign.evidence_id
            items[0]["window_id"] = foreign.window_id
    elif mutation == "missing_reference":
        items[2].pop("shared_evidence_id")
    elif mutation in {"wrong_reference", "foreign_reference"}:
        items[2]["shared_evidence_id"] = "0" * 64
        if mutation == "foreign_reference":
            rehash_decision(items[2])
    elif mutation == "reordered":
        items[2], items[3] = items[3], items[2]
    elif mutation == "duplicate":
        items[3] = items[2]
    elif mutation == "truncated":
        items.pop()
    elif mutation == "extra":
        items.append(items[-1])
    elif mutation == "count":
        items[0]["decision_count"] = 1
    elif mutation == "result_id":
        items[0]["window_result_id"] = "0" * 64
    elif mutation == "embedded":
        mapping(mapping(items[2]["prediction_study"])["manifest"])["market_data"] = (
            mapping(items[1]["window_identity"])["market_data"]
        )
        rehash_decision(items[2])
    elif mutation == "decision_id":
        items[2]["decision_id"] = "0" * 64
    elif mutation == "sequence":
        items[2]["sequence"] = True
        rehash_decision(items[2])
    else:
        mapping(items[0]["record_counts"])["generated_predictions"] = 0
    path = tmp_path / "corrupt.jsonl"
    write_records(path, items)
    with pytest.raises(InvalidPredictionOutputError):
        PredictionWindowReader.open(path).verify_integrity()


@pytest.mark.parametrize(
    "mutation", ["context", "outcome", "study_id", "status", "signals"]
)
def test_rehashed_normalized_corruption_still_fails_semantic_validation(
    window: PredictionWindowResult[Any, Any, Any], tmp_path: Path, mutation: str
) -> None:
    compact = CompactPredictionWindowResult.from_window(window)
    record = compact.decisions[0].to_primitive()
    study = mapping(record["prediction_study"])
    if mutation == "context":
        mapping(mapping(study["manifest"])["prediction_context"])[
            "decision_session"
        ] = "2024-07-10"
    elif mutation == "outcome":
        mapping(mapping(cast(list[Any], study["rows"])[0])["outcome"])["outcome_id"] = (
            "0" * 64
        )
    elif mutation == "study_id":
        record["prediction_study_id"] = "0" * 64
    elif mutation == "status":
        record["status"] = "skipped"
    else:
        record["generated_signals"] = []
    rehash_decision(record)
    changed = replace(
        compact,
        decisions=(
            CompactPredictionWindowDecision(PrimitiveMappingSnapshot.capture(record)),
            *compact.decisions[1:],
        ),
    )
    reader = write_compact(tmp_path / "changed.jsonl", changed)
    reader.verify_integrity()  # Checksums are insufficient to prove research semantics.
    with pytest.raises(InvalidPredictionOutputError):
        verify(reader, window)


def test_metadata_and_count_do_not_read_decisions(
    window: PredictionWindowResult[Any, Any, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compact = CompactPredictionWindowResult.from_window(window)
    path = tmp_path / "header-only.jsonl"
    lines = compact.serialize().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:2]) + b"deliberately unreadable decision")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("metadata/count must not construct a decision or embedded result")

    monkeypatch.setattr(CompactPredictionWindowDecision, "__post_init__", forbidden)
    reader = PredictionWindowReader.open(path)
    assert reader.decision_count == 4
    assert reader.header()["schedule_id"] == window.schedule.schedule_id
    assert reader.shared_evidence() == compact.evidence


def test_reader_detects_changed_evidence_between_passes(
    window: PredictionWindowResult[Any, Any, Any], tmp_path: Path
) -> None:
    compact = CompactPredictionWindowResult.from_window(window)
    path = tmp_path / "window.jsonl"
    reader = write_compact(path, compact)
    items = records(compact)
    mapping(items[1]["window_identity"])["context_environment"] = {"changed": True}
    write_records(path, items)
    with pytest.raises(InvalidPredictionOutputError, match="changed during reading"):
        reader.verify_integrity()


@pytest.mark.parametrize(
    "damage", ["duplicate_key", "nan", "partial_line", "blank_line", "pretty_v2"]
)
def test_noncanonical_json_rejects(
    window: PredictionWindowResult[Any, Any, Any], tmp_path: Path, damage: str
) -> None:
    compact = CompactPredictionWindowResult.from_window(window)
    encoded = compact.serialize()
    if damage == "duplicate_key":
        encoded = encoded.replace(
            b'"schema_version":"2"', b'"schema_version":"2","schema_version":"2"', 1
        )
    elif damage == "nan":
        encoded = encoded.replace(b'"decision_count":4', b'"decision_count":NaN')
    elif damage == "partial_line":
        encoded = encoded[:-1]
    elif damage == "blank_line":
        encoded += b"\n"
    else:
        encoded = json.dumps(compact.to_primitive(), indent=2).encode()
    path = tmp_path / "bad.jsonl"
    path.write_bytes(encoded)
    with pytest.raises(InvalidPredictionOutputError):
        PredictionWindowReader.open(path).verify_integrity()


def test_empty_and_skipped_windows_remain_distinct(tmp_path: Path) -> None:
    empty = run_window(
        decision_schedule=schedule(START + timedelta(days=2), START + timedelta(days=2))
    )
    provider = WindowProvider()
    provider.series = ()
    skipped = run_window(
        provider,
        requirements=_requirements(failure_policy=PredictionContextFailurePolicy.SKIP),
    )
    for name, window in (("empty", empty), ("skipped", skipped)):
        reader = write_compact(
            tmp_path / name, CompactPredictionWindowResult.from_window(window)
        )
        verify(reader, window)
        assert reader.decision_count == len(window.decisions)
    assert empty.decisions == ()
    assert all(
        decision.to_primitive()["status"] == "skipped" for decision in skipped.decisions
    )


def test_material_scope_changes_identity_and_prevents_foreign_reuse(
    window: PredictionWindowResult[Any, Any, Any],
) -> None:
    compact = CompactPredictionWindowResult.from_window(window)
    identity = window.identity_snapshot.to_primitive()
    mapping(identity["market_data"])["bars_fingerprint"] = "changed-source"
    changed = PredictionWindowEvidence(PrimitiveMappingSnapshot.capture(identity))
    assert changed.evidence_id != compact.evidence.evidence_id
    assert changed.window_id != compact.window_id
    with pytest.raises(InvalidPredictionOutputError, match="foreign"):
        CompactPredictionWindowDecision.from_embedded(
            window.decisions[0].to_primitive(), sequence=0, evidence=changed
        )
    with pytest.raises(InvalidPredictionOutputError, match="reference/order"):
        replace(compact, evidence=changed)
    with pytest.raises(InvalidPredictionOutputError, match="reference/order"):
        replace(compact, decisions=tuple(reversed(compact.decisions)))
    assert (
        ordered_result_identity(
            compact.window_id, (d.to_primitive() for d in reversed(compact.decisions))
        )
        != compact.window_result_id
    )
    identity = window.identity_snapshot.to_primitive()
    identity["schedule"] = schedule(START, START).to_primitive()
    assert (
        PredictionWindowEvidence(PrimitiveMappingSnapshot.capture(identity)).window_id
        != compact.window_id
    )

    identity = window.identity_snapshot.to_primitive()
    mapping(mapping(identity["configuration"])["outcome_labeler"])[
        "configuration_id"
    ] = "changed-horizon"
    assert (
        PredictionWindowEvidence(PrimitiveMappingSnapshot.capture(identity)).window_id
        != compact.window_id
    )


def test_unlabeled_and_no_signal_decisions_preserve_distinct_counts(
    tmp_path: Path,
) -> None:
    class EmptyRule(WindowRule):
        def generate_with_context(
            self, context: PredictionRuleContext
        ) -> PredictionStrategyOutput:
            return replace(super().generate_with_context(context), signals=())

    provider = WindowProvider()
    empty = run_prediction_window(
        _prediction_dataset(),
        _study(EmptyRule(_requirements())),
        schedule=schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment={},
    )
    tomorrow = START + timedelta(days=1)
    unlabeled = run_window(
        _provider_with_final_session(), decision_schedule=schedule(tomorrow, tomorrow)
    )
    for name, window in (("no-signals", empty), ("unlabeled", unlabeled)):
        compact = CompactPredictionWindowResult.from_window(window)
        reader = write_compact(tmp_path / name, compact)
        verify(reader, window)
        assert reader.header()["record_counts"] == window.counts_primitive()
    assert empty.counts_primitive()["no_prediction_decisions"] == 4
    assert unlabeled.counts_primitive()["unavailable_outcomes"] == 1


@pytest.mark.parametrize("threshold", [Decimal("0"), Decimal("12.5")])
def test_accepted_and_rejected_candidate_evidence_is_lossless(
    tmp_path: Path, threshold: Decimal
) -> None:
    provider = WindowProvider()
    rule = confluence_rule(threshold=threshold, backend_id=NATIVE_INDICATOR_BACKEND)
    study, _ = confluence_study(rule)
    window = run_prediction_window(
        _prediction_dataset(),
        study,
        schedule=schedule(),
        context_provider=provider,
        dataset_family_fingerprint=provider.family.family_id,
        context_environment={},
    )
    reader = write_compact(
        tmp_path / "candidates.jsonl", CompactPredictionWindowResult.from_window(window)
    )
    validate_prediction_window_reader(
        reader,
        expected_identity=window.identity_snapshot,
        schedule=window.schedule,
        outcome_sessions=tuple(bar.session_date for bar in _prediction_dataset().bars),
        strategy_parameters=rule.parameters.to_primitive(),
    )
    for original, compact in zip(
        window.decisions, reader.iterate_decisions(), strict=True
    ):
        assert (
            compact.to_primitive()["generated_signals"]
            == original.to_primitive()["generated_signals"]
        )
    assert reader.header()["record_counts"] == window.counts_primitive()
