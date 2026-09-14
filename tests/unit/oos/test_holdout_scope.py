"""Holdout exposure uses the same session labels on either validation axis."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest

from quantforge.oos import (
    HoldoutEvaluation,
    HoldoutLedger,
    OOSIntegrityError,
    load_oos_source,
)
from quantforge.oos._records import mapping
from quantforge.timeframes import resolve_exchange_session
from quantforge.validation import (
    DatasetProvenance,
    ExchangeSessionBoundary,
    IndicatorComponent,
    IndicatorProvenance,
    PurgePolicy,
    TemporalOffset,
    TimestampBoundary,
    ValidationBoundary,
    ValidationInterval,
    ValidationWindow,
)
from quantforge.walk_forward import BacktestEvaluator, WalkForwardStudy
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.helpers import SESSIONS, make_dataset
from tests.unit.walk_forward.fixtures import backtest_fixture

from .conftest import CompletedStudy


def midnight_study(root: Path, *, timestamp_axis: bool) -> CompletedStudy:
    config, original = backtest_fixture(root)
    sessions = tuple(
        SESSIONS[0] + timedelta(days=index) for index in range(len(SESSIONS))
    )
    dataset = make_dataset(
        tuple(str(bar.close) for bar in original.dataset.bars),
        sessions=sessions,
        calendar="24/7",
    )
    evaluator = BacktestEvaluator(dataset, original.factory, original.grid_config)
    provenance = DatasetProvenance.from_market_dataset(dataset)
    timeframe = provenance.standalone_timeframe
    assert timeframe is not None
    indicators = {
        indicator.configuration_id: IndicatorProvenance.capture(
            cast(IndicatorComponent, indicator), timeframe
        )
        for candidate in evaluator.universe.candidates
        for indicator in evaluator.factory.build(
            candidate.parameters.to_primitive()
        ).required_indicators
    }
    environment = replace(
        config.plan.environment,
        dataset=provenance,
        timeframes=(timeframe,),
        indicators=tuple(indicators.values()),
    )

    def window(original_window: ValidationWindow) -> ValidationWindow:
        def boundary(original_boundary: ValidationBoundary) -> ValidationBoundary:
            assert isinstance(original_boundary, ExchangeSessionBoundary)
            session = sessions[SESSIONS.index(original_boundary.session_date)]
            if timestamp_axis:
                return TimestampBoundary(
                    resolve_exchange_session(
                        session, timeframe.session_policy
                    ).close_timestamp
                )
            return ExchangeSessionBoundary(session, timeframe.session_policy)

        return replace(
            original_window,
            interval=ValidationInterval(
                boundary(original_window.interval.start),
                boundary(original_window.interval.end),
            ),
        )

    # One session makes UTC-day and session-label scopes completely disjoint.
    final = config.plan.final_holdout.window
    final = replace(
        final, interval=ValidationInterval(final.interval.end, final.interval.end)
    )
    plan = replace(
        config.plan,
        environment=environment,
        folds=tuple(
            replace(fold, development=window(fold.development), test=window(fold.test))
            for fold in config.plan.folds
        ),
        final_holdout=replace(config.plan.final_holdout, window=window(final)),
        purge_policy=PurgePolicy(
            TemporalOffset.duration(timedelta(0)), TemporalOffset.duration(timedelta(0))
        )
        if timestamp_axis
        else config.plan.purge_policy,
    )
    study = WalkForwardStudy(replace(config, plan=plan), evaluator, root / "study")
    study.run()
    source = load_oos_source(plan, study.study_path)
    assert all(fold.artifact for fold in source.folds)
    return CompletedStudy(study, source, evaluator)


@pytest.mark.parametrize("timestamp_first", [False, True])
def test_same_holdout_cannot_be_reconsumed_on_another_axis(
    tmp_path: Path, timestamp_first: bool
) -> None:
    first = midnight_study(tmp_path / "first", timestamp_axis=timestamp_first)
    other = midnight_study(tmp_path / "other", timestamp_axis=not timestamp_first)
    assert first.source.lineage_id != other.source.lineage_id
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(first.source)
    evaluation = HoldoutEvaluation.prepare(
        first.source, first.evaluator, selection_fold_id=first.source.folds[-1].fold_id
    )
    consumed = ledger.consume(evaluation, run_id="first")
    assert consumed.consumption is not None
    scope = mapping(consumed.consumption.to_primitive()["exposure_scope"])
    assert scope["start"] == scope["end"] == "2024-07-15"
    with pytest.raises(OOSIntegrityError, match="another lineage"):
        ledger.reserve(other.source)
    assert ledger.state(first.source) == consumed


def test_legacy_utc_exposure_cannot_grant_a_pristine_session_scope(
    tmp_path: Path,
) -> None:
    original = midnight_study(tmp_path / "original", timestamp_axis=True)
    other = midnight_study(tmp_path / "other", timestamp_axis=False)
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(original.source)
    ledger.consume(
        HoldoutEvaluation.prepare(
            original.source,
            original.evaluator,
            selection_fold_id=original.source.folds[-1].fold_id,
        ),
        run_id="old-implementation",
    )
    path = ledger.root / "exposures" / f"{original.source.lineage_id}.json"
    marker = read_record(path)
    marker["exposure_scope"] = {
        "symbol": "SPY",
        "start": "2024-07-16",
        "end": "2024-07-16",
    }
    write_record(path, marker)
    with pytest.raises(OOSIntegrityError, match="exposure scope"):
        ledger.reserve(other.source)
