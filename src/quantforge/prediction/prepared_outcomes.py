"""Execution-local outcome capabilities for explicit observation requests.

Preparation owns only invariant source work. Requests and concrete labelers keep
their original temporal, price-basis, path-completeness and mutation checks.
Nothing in this module is supplied to prediction rules or serialized results.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from hashlib import sha256
from types import MappingProxyType
from typing import cast

from quantforge.configuration import configuration_identity
from quantforge.data import IntradayBar, MarketDataset, TimeframeBarSeries
from quantforge.data.models import BoundedPredictionProvenance, DatasetMetadata
from quantforge.data.prediction_inputs import validate_prediction_source
from quantforge.data.prepared_prediction_views import (
    _immutable,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.models import PredictionMarketData
from quantforge.prediction.outcome_resolution import OutcomeEvaluationRequest
from quantforge.prediction.outcome_temporal import OutcomeTemporalError
from quantforge.prediction.source_sharing import (
    _immutable_source_value,  # pyright: ignore[reportPrivateUsage]
    prediction_source_copy_memo,
)


def _immutable_metadata(metadata: DatasetMetadata) -> DatasetMetadata | None:
    """Normalize QF-52 calendar cutoffs using the existing QF-58 exact rule."""
    provenance = metadata.intraday_provenance
    if type(provenance) is BoundedPredictionProvenance:
        cutoff = _immutable_source_value(provenance.causal_cutoff, {})
        if type(cutoff) is not datetime:
            return None
        if cutoff is not provenance.causal_cutoff:
            metadata = replace(
                metadata, intraday_provenance=replace(provenance, causal_cutoff=cutoff)
            )
    return metadata if _immutable(metadata) else None


def _source_identity(source: TimeframeBarSeries, input_identity: str) -> str:
    """Stream complete bar content; never allocate a second serialized dataset."""
    digest = sha256()
    for bar in source.bars:
        if not isinstance(bar, IntradayBar):
            raise OutcomeTemporalError("prepared outcomes require intraday bars")
        digest.update(bar.serialize())
        digest.update(b"\n")
    evidence = source._developing_source_evidence  # pyright: ignore[reportPrivateUsage]
    return configuration_identity(
        {
            "preparation_version": "1",
            "prediction_input": input_identity,
            "source_reference": source.dataset_reference.to_primitive(
                include_feed_scope=True
            ),
            "family_manifest_id": source.dataset_family_manifest_id,
            "timeframe": source.timeframe.to_primitive(),
            "bar_count": len(source.bars),
            "contents_sha256": digest.hexdigest(),
            "coverage": None
            if evidence is None
            else {
                "start": evidence.request_start_timestamp.isoformat(),
                "end": evidence.request_end_timestamp.isoformat(),
                "expected_intervals": [
                    interval.to_primitive() for interval in evidence.expected_intervals
                ],
            },
        }
    )


@dataclass(frozen=True, slots=True, init=False)
class PreparedOutcomeSource:
    """Validated immutable backing plus operational indexes, with no rule state.

    Obtain this capability through ``PreparedOutcomeSources.prepare``. Its identity
    binds full contents and provenance, not an object address or path. Holding the
    exact admitted immutable backing proves that identity remains valid on requests;
    an arbitrary replacement source must pass preparation again.
    """

    source: TimeframeBarSeries
    compatibility_id: str
    dataset_id: str
    dataset_fingerprint: str
    timestamps: tuple[datetime, ...]
    session_ranges: Mapping[date, tuple[int, int]]

    @classmethod
    def _build(
        cls, dataset: MarketDataset, source: TimeframeBarSeries, identity: str
    ) -> "PreparedOutcomeSource":
        validate_prediction_source(dataset, source)
        timestamps, ranges = _build_indexes(source)
        prepared = object.__new__(cls)
        for name, value in (
            ("source", source),
            ("compatibility_id", identity),
            ("dataset_id", dataset.metadata.dataset_id),
            ("dataset_fingerprint", dataset.metadata.data_sha256),
            ("timestamps", timestamps),
            ("session_ranges", ranges),
        ):
            object.__setattr__(prepared, name, value)
        return prepared

    def validate_request(self, request: OutcomeEvaluationRequest) -> None:
        """Check exact input/source/temporal binding for each requested anchor."""
        if (
            request.dataset_id != self.dataset_id
            or request.dataset_fingerprint != self.dataset_fingerprint
            or request.source_reference != self.source.dataset_reference
            or request.temporal_configuration.observation_timeframe
            != self.source.timeframe
            or request.anchor.decision_timestamp is None
        ):
            raise OutcomeTemporalError("prepared outcome source differs from request")

    def _bounded_bars(
        self, request: OutcomeEvaluationRequest
    ) -> tuple[IntradayBar, ...]:
        """Same-session preceding anchor and permitted future reach, by position."""
        decision = request.anchor.decision_timestamp
        assert decision is not None
        start, stop = self.session_ranges.get(request.anchor.signal_session, (0, 0))
        split = bisect_right(self.timestamps, decision, start, stop)
        end = bisect_right(
            self.timestamps,
            decision + request.temporal_configuration.required_future_duration,
            split,
            stop,
        )
        return cast(
            tuple[IntradayBar, ...], self.source.bars[max(start, split - 1) : end]
        )

    def endpoint_candidates(self, expected: datetime) -> tuple[IntradayBar, ...]:
        """Exact end plus a possible developing interval preceding that end."""
        position = bisect_left(self.timestamps, expected)
        return cast(
            tuple[IntradayBar, ...],
            self.source.bars[max(0, position - 1) : position + 1],
        )


def _build_indexes(
    source: TimeframeBarSeries,
) -> tuple[tuple[datetime, ...], Mapping[date, tuple[int, int]]]:
    timestamps: list[datetime] = []
    ranges: dict[date, tuple[int, int]] = {}
    previous: IntradayBar | None = None
    for position, bar in enumerate(source.bars):
        if not isinstance(bar, IntradayBar):
            raise OutcomeTemporalError(
                "prepared outcomes require canonical intraday bars"
            )
        if bar.timeframe != source.timeframe or (
            previous is not None
            and (
                bar.end_timestamp <= previous.end_timestamp
                or bar.start_timestamp < previous.end_timestamp
                or bar.session_date < previous.session_date
            )
        ):
            raise OutcomeTemporalError("outcome source chronology/timeframe is invalid")
        start, _ = ranges.get(bar.session_date, (position, position))
        ranges[bar.session_date] = (start, position + 1)
        timestamps.append(bar.end_timestamp)
        previous = bar
    return tuple(timestamps), MappingProxyType(ranges)


class PreparedOutcomeSources:
    """One dataset-session registry shared by all configured outcome labelers.

    Only finite sources admitted during this execution are retained. No global cache,
    persisted handles, labels or decisions. Unknown/mutable graphs return ``None``
    and retain the original validation path. Equivalent new immutable handles are
    content-authenticated before reusing an existing scientific compatibility key;
    an original already normalized in this session reuses its retained backing.
    """

    def __init__(self) -> None:
        self._metadata: DatasetMetadata | None = None
        self._input_identity: str | None = None
        self._sources: dict[str, PreparedOutcomeSource] = {}
        self._copy_memos: list[tuple[TimeframeBarSeries, dict[int, object]]] = []

    def copy_memo(
        self,
        source: TimeframeBarSeries | None,
        normalize: Callable[
            [TimeframeBarSeries | None], dict[int, object]
        ] = prediction_source_copy_memo,
    ) -> dict[int, object]:
        """Return a fresh QF-58 copy memo, reusing this session's normalized backing.

        QF-42 normalizes a study's source once per iterator. Without this, every
        later iterator in one session (QF-32 trials 2..k) obtains a new backing,
        misses the exact-backing check below and re-authenticates the complete
        source for every decision. The retained original is a capability, not a
        scientific key: it was recursively verified immutable when first seen.
        """
        if source is None:
            return {}
        for original, memo in self._copy_memos:
            if original is source:
                return dict(memo)
        memo = normalize(source)
        if memo:
            self._copy_memos.append((source, memo))
        return dict(memo)

    def prepare(
        self, dataset: MarketDataset, source: TimeframeBarSeries
    ) -> PreparedOutcomeSource | None:
        metadata = _immutable_metadata(dataset.metadata)
        if metadata is None:
            return None
        if self._metadata is None:
            self._metadata = metadata
            self._input_identity = configuration_identity(
                PredictionMarketData.from_qf3(dataset.metadata).to_primitive()
            )
        elif metadata != self._metadata:
            raise OutcomeTemporalError("prepared outcome prediction input differs")
        # This is a capability binding check, not a scientific cache key. Only an
        # exact, previously recursively verified immutable backing can skip hashing.
        for prepared in self._sources.values():
            if source is prepared.source:
                return prepared
        # An original already normalized in this session maps to its retained
        # backing, so repeated callers (QF-7 outcomes, later QF-42 iterators)
        # reuse the exact admitted object instead of rehashing every bar.
        memo = self.copy_memo(source)
        if not memo:
            return None
        immutable = cast(TimeframeBarSeries, memo[id(source)])
        for prepared in self._sources.values():
            if immutable is prepared.source:
                return prepared
        assert self._input_identity is not None
        identity = _source_identity(immutable, self._input_identity)
        if identity not in self._sources:
            self._sources[identity] = PreparedOutcomeSource._build(  # pyright: ignore[reportPrivateUsage]
                dataset, immutable, identity
            )
        return self._sources[identity]
