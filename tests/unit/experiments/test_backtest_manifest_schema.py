"""A refreshed export sidecar cannot certify an incomplete QF-5 manifest."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from quantforge.backtesting import (
    export_backtest_result,
    validate_backtest_result_artifact,
)
from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.experiments._json import mapping
from tests.unit.backtesting.test_runner import configured_result
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record


@pytest.fixture
def export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = export_backtest_result(configured_result(), tmp_path / "backtests")
    block_research(monkeypatch)
    return root


def inspect_manifest(
    export: Path, manifest: PrimitiveMapping, *, reject: bool = True
) -> None:
    write_json(export / "manifest.json", manifest)
    integrity = read_record(export / "integrity.json")
    mapping(integrity["files"])["manifest.json"] = sha256(
        (export / "manifest.json").read_bytes()
    ).hexdigest()
    write_json(export / "integrity.json", integrity)
    if "run_id" in manifest:
        assert validate_backtest_result_artifact(export) == export
    before = {item: item.read_bytes() for item in export.iterdir() if item.is_file()}
    for source in (export, export / "manifest.json"):
        if reject:
            with pytest.raises(ManifestError):
                inspect_study(StudyType.BACKTEST, source, artifact_root=export.parent)
        else:
            inspected = inspect_study(
                StudyType.BACKTEST, source, artifact_root=export.parent
            )
            assert verify_artifacts(inspected.index, export.parent).valid
    assert {item: item.read_bytes() for item in before} == before


def section(manifest: PrimitiveMapping, path: tuple[str, ...]) -> PrimitiveMapping:
    result = manifest
    for key in path:
        result = mapping(result[key])
    return result


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("record_counts",),
        ("performance",),
        ("market_data",),
        ("strategy",),
        ("benchmark",),
        ("benchmark", "order"),
        ("benchmark", "fill"),
        ("benchmark", "performance"),
        ("benchmark", "dividend_accounting"),
        ("corporate_action_accounting",),
        ("corporate_action_accounting", "dividends"),
    ],
)
def test_every_saved_manifest_field_is_required(
    export: Path, path: tuple[str, ...]
) -> None:
    original = read_record(export / "manifest.json")
    for field in (*section(original, path), "undeclared"):
        manifest = deepcopy(original)
        target = section(manifest, path)
        if field == "undeclared":
            target[field] = None
        else:
            del target[field]
        inspect_manifest(export, manifest)


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("warnings",), "warning"),
        (("warnings",), [1]),
        (("limitations",), None),
        (("limitations",), [False]),
        (("initiated_at",), "2024-01-02T12:00:00"),
        (("initiated_at",), "invalid"),
        (("record_counts", "signals"), True),
        (("record_counts", "fills"), -1),
        (("record_counts", "orders"), "0"),
        (("record_counts", "positions"), 0.0),
        (("performance", "trade_count"), True),
        (("performance", "winning_trades"), -1),
        (("performance", "ending_equity"), 100),
        (("performance", "total_return"), None),
        (("performance", "cagr"), "NaN"),
        (("performance", "sharpe_ratio"), "Infinity"),
        (("performance", "annualization_factor"), 0),
        (("performance", "win_rate"), "1.1"),
        (("performance", "exposure"), "-0.1"),
        (("performance", "profit_factor"), "-1"),
        (("performance", "gross_profit"), "-1"),
        (("performance", "gross_loss"), "1"),
        (("performance", "maximum_drawdown"), "0.1"),
        (("performance", "volatility_standard_deviation"), "population"),
        (("performance", "maximum_drawdown_convention"), "positive_decimal"),
        (("benchmark", "performance", "gross_profit"), True),
        (("benchmark", "dividend_accounting", "dividend_events_present"), False),
        (("benchmark", "dividend_cashflows"), {}),
        (("benchmark", "dividend_cashflows"), [dict[str, Primitive]()]),
        (("benchmark", "split_adjustments"), [dict[str, Primitive]()]),
        (("benchmark", "order", "side"), "unknown"),
        (("benchmark", "order", "requested_quantity"), True),
        (("benchmark", "fill", "fill_price"), False),
        (("benchmark", "fill"), False),
        (("market_data", "bar_count"), True),
        (("market_data", "retrieved_at"), "yesterday"),
        (("market_data", "corporate_actions_complete"), 1),
        (("market_data", "missing_sessions"), ["not-a-session"]),
        (("strategy", "warm_up_observations"), 0),
        (("corporate_action_accounting", "dividends", "warning"), []),
        (("corporate_action_accounting", "dividends", "return_basis"), "unknown"),
    ],
)
def test_saved_manifest_fields_keep_their_primitive_domains(
    export: Path, path: tuple[str, ...], invalid: Primitive
) -> None:
    manifest = read_record(export / "manifest.json")
    section(manifest, path[:-1])[path[-1]] = invalid
    inspect_manifest(export, manifest)


def test_native_manifest_keeps_nullable_metrics_and_empty_disclosures(
    export: Path,
) -> None:
    manifest = read_record(export / "manifest.json")
    assert manifest["initiated_at"] is None
    mapping(manifest["performance"])["sharpe_ratio"] = None
    manifest["warnings"] = []
    manifest["limitations"] = []
    inspect_manifest(export, manifest, reject=False)
