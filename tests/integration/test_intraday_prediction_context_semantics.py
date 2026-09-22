"""Rehashed feature contexts must satisfy the existing runtime semantics."""

from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.prediction import (
    InvalidPredictionOutputError,
    SignalFeatureDatasetResult,
)
from quantforge.prediction.window_context_validation import (
    validate_window_context_snapshot,
)
from tests.integration.test_intraday_prediction_manifest_integrity import (
    _write_rehashed_feature,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_manifest_integrity import (
    feature_result as feature_result,
)
from tests.integration.test_intraday_prediction_provenance import fixture as fixture
from tests.integration.test_intraday_prediction_request_integrity import (
    _rehash_nested,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.test_intraday_prediction_source_requirements import (
    empty_artifacts as empty_artifacts,
)
from tests.unit.experiments.test_adapters import block_research


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("same", ""),
        ("completion", "context completion policy"),
        ("source_age", "source contextual requirements"),
        ("source_timeframe", "source contextual requirements"),
        ("declared_age", "source contextual requirements"),
        ("aligned_age", "source timeframe requirement"),
        ("selected_bars", "rule selected bar IDs"),
    ],
)
@pytest.mark.parametrize("empty", [False, True], ids=["candidates", "no_candidates"])
@pytest.mark.parametrize("directory", [False, True], ids=["snapshot", "directory"])
def test_rehashed_feature_context_requires_valid_semantics(
    feature_result: SignalFeatureDatasetResult,
    empty_artifacts: tuple[PrimitiveMapping, SignalFeatureDatasetResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    message: str,
    empty: bool,
    directory: bool,
) -> None:
    result = empty_artifacts[1] if empty else feature_result
    configuration = result.configuration
    context = cast(PrimitiveMapping, configuration["prediction_context"])
    source = cast(PrimitiveMapping, context["source_context"])
    requirements = cast(PrimitiveMapping, context["requirements"])
    if change == "completion":
        source["completion_policy"] = "developing_bar_as_of"
    elif change in {"source_age", "source_timeframe"}:
        contextual = cast(list[PrimitiveMapping], source["required_timeframes"])[0]
        if change == "source_age":
            contextual["maximum_age_microseconds"] = 3_600_000_000
        else:
            contextual["timeframe"] = source["primary_timeframe"]
    elif change == "declared_age":
        cast(list[PrimitiveMapping], requirements["contextual"])[0][
            "maximum_age_microseconds"
        ] = 3_600_000_000
    elif change == "aligned_age":
        aligned = cast(list[PrimitiveMapping], source["timeframes"])[1]
        cast(PrimitiveMapping, aligned["requirement"])["maximum_age_microseconds"] = (
            3_600_000_000
        )
    elif change == "selected_bars":
        selected = cast(list[PrimitiveMapping], context["timeframes"])[0]
        cast(list[str], selected["visible_bar_ids"]).pop()
    _rehash_nested(configuration)
    primary = cast(PrimitiveMapping, requirements["primary"])

    def validate_semantics() -> None:
        validate_window_context_snapshot(
            context=context,
            source=source,
            requirements=requirements,
            market_data=result.market_data.to_primitive(),
            primary_timeframe=cast(PrimitiveMapping, primary["timeframe"]),
        )

    block_research(monkeypatch)
    if change == "same":
        validate_semantics()
    else:
        with pytest.raises(InvalidPredictionOutputError, match=message):
            validate_semantics()
    path = _write_rehashed_feature(result, configuration, tmp_path, directory)
    if change == "same":
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    else:
        with pytest.raises(ManifestError, match=message):
            inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
