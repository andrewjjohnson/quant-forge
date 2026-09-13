"""Select execution membership before accounting, retaining immutable provenance."""

from dataclasses import replace

from quantforge.backtesting.config import EvaluationInterval
from quantforge.backtesting.corporate_actions import actions_by_session
from quantforge.backtesting.errors import InvalidMarketDataError
from quantforge.data.models import MarketDataset


def select_backtest_datasets(
    dataset: MarketDataset, interval: EvaluationInterval | None
) -> tuple[MarketDataset, MarketDataset]:
    """Return strategy history and evaluation views of an already validated source.

    These ephemeral views retain the complete source metadata/action identities;
    they are not independently valid QF-3 datasets or cacheable source artifacts.
    The caller must validate the original dataset before selecting membership.
    """
    if interval is None:
        return dataset, dataset
    sessions = {bar.session_date for bar in dataset.bars}
    if interval.start_session not in sessions or interval.end_session not in sessions:
        raise InvalidMarketDataError(
            "both evaluation endpoints must be observed source exchange sessions"
        )
    indexed_actions = actions_by_session(dataset)
    history_bars = tuple(
        bar for bar in dataset.bars if bar.session_date <= interval.end_session
    )
    evaluation_bars = tuple(
        bar for bar in history_bars if bar.session_date >= interval.start_session
    )
    history = replace(
        dataset,
        bars=history_bars,
        corporate_actions=tuple(
            action
            for bar in history_bars
            for action in indexed_actions.get(bar.session_date, ())
        ),
    )
    evaluation = replace(
        dataset,
        bars=evaluation_bars,
        corporate_actions=tuple(
            action
            for bar in evaluation_bars
            for action in indexed_actions.get(bar.session_date, ())
        ),
    )
    return history, evaluation
