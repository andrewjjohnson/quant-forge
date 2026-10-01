"""Execution-local prepared causal context runs and reusable indicator series.

QF-63 prepares invariant historical work once per QF-39 permitted-context scope
(plan, window, role and immutable source family), then serves each scheduled
decision from exact positions. Nothing here enters a scientific identity, result,
checkpoint or artifact; after a restart the scope is rebuilt deterministically.

Causality: a scope's runs end at the window's last permitted cutoff. Each
decision's ``MultiTimeframeContext`` binds only that decision's ``[start, stop)``
positions, and the rule-facing ``PredictionRuleContext`` receives only visible
slices. Indicator series are computed over a run from an exact input start and
exposed only through a prefix slice; admitted formulas are reviewed to depend on
no later input (prefix stability) and each series is verified against the
reference evaluation on first use.
"""

from bisect import bisect_left
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, time
from operator import attrgetter, is_
from types import MappingProxyType
from typing import cast

from quantforge.configuration import PrimitiveMappingSnapshot, configuration_identity
from quantforge.data.exceptions import ValidationError
from quantforge.data.intraday import IntradayBar
from quantforge.data.lineage import AdjustmentBasis, DatasetFamilyReference
from quantforge.data.models import DatasetMetadata, MarketDataset
from quantforge.data.multi_timeframe import (
    ArtifactBar,
    ContextCompletionPolicy,
    ContextTimeframeRequirement,
    MultiTimeframeContext,
    MultiTimeframeContextValidationError,
    TimeframeBarSeries,
)
from quantforge.data.prediction_inputs import (
    _validate_source,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.data.session_aggregation import AggregatedSessionBar
from quantforge.indicators import (
    BollingerBands,
    ExponentialMovingAverage,
    MovingAverageConvergenceDivergence,
    SimpleMovingAverage,
    StochasticOscillator,
    WilderAverageTrueRange,
    WilderDirectionalMovement,
    WilderRelativeStrengthIndex,
)
from quantforge.indicators.backends import (
    IndicatorBackendIdentity,
    StandardIndicatorDefinition,
)
from quantforge.indicators.backends.native import NativeIndicatorBackend
from quantforge.indicators.backends.talib import TalibIndicatorBackend
from quantforge.indicators.exceptions import InvalidIndicatorBackendError
from quantforge.indicators.models import IndicatorFieldOutput
from quantforge.indicators.timeframe import TimeframeIndicatorOutput, bind_indicator
from quantforge.prediction.context import (
    PredictionContextError,
    PredictionIndicatorRequirement,
    PredictionRuleContext,
    PredictionTimeframeInput,
    PreparedOutputEvidence,
)
from quantforge.prediction.source_sharing import (
    _SOURCE_RECORD_TYPES,  # pyright: ignore[reportPrivateUsage]
    _SOURCE_SCALAR_TYPES,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.timeframes import BarCompletion, Timeframe

PREPARED_FEATURE_VERSION = "1"

# Reviewed exact types. Each computes an output position only from input at and
# before it, from a fixed input start (TA-Lib default compatibility and zero
# unstable period are enforced by talib_v1; native_v1 uses fixed Decimal
# contexts). Subclasses, custom indicators and other backends use the reference
# evaluation. tests/unit/prediction/test_prepared_features.py proves prefix
# stability for every pair below.
_PREFIX_STABLE_INDICATORS: frozenset[type] = frozenset(
    {
        SimpleMovingAverage,
        ExponentialMovingAverage,
        WilderRelativeStrengthIndex,
        WilderAverageTrueRange,
        WilderDirectionalMovement,
        BollingerBands,
        MovingAverageConvergenceDivergence,
        StochasticOscillator,
    }
)
_PREFIX_STABLE_BACKENDS: frozenset[type] = frozenset(
    {TalibIndicatorBackend, NativeIndicatorBackend}
)
_BAR_TYPES: frozenset[type] = frozenset({IntradayBar, AggregatedSessionBar})
_UNSUPPORTED = object()


def prefix_stable_indicator(indicator: object) -> bool:
    """Whether an indicator is a reviewed exact prefix-stable type and backend.

    The same admission ``indicator_output`` applies before serving a prepared
    series. Consumers that read prepared prefixes (QF-72 rapid scans) use it to
    refuse anything else instead of approximating it.
    """
    return (
        type(indicator) in _PREFIX_STABLE_INDICATORS
        and type(getattr(indicator, "_backend", None)) in _PREFIX_STABLE_BACKENDS
    )


type _Getter = Callable[[object], tuple[object, ...]]


class PreparedFeatureIntegrityError(RuntimeError):
    """Shared prepared backing changed; the scope refuses all further use."""


def _field_getter(record_type: type) -> _Getter:
    names = tuple(item.name for item in fields(record_type))
    if len(names) == 1:
        name = names[0]
        return lambda record: (getattr(record, name),)
    return cast(_Getter, attrgetter(*names))


def _nested_records(value: object, found: dict[int, object]) -> bool:
    """Collect reviewed immutable records below one bar field; False if unknown."""
    value_type = type(value)
    if value_type in _SOURCE_SCALAR_TYPES or value_type in (datetime, time):
        return True
    if value_type is tuple:
        return all(
            _nested_records(item, found) for item in cast(tuple[object, ...], value)
        )
    if value_type in _SOURCE_RECORD_TYPES and value_type not in _BAR_TYPES:
        if id(value) not in found:
            found[id(value)] = value
            return all(
                _nested_records(getattr(value, item.name), found)
                for item in fields(value_type)
            )
        return True
    return False


@dataclass(frozen=True, slots=True, eq=False)
class _RecordGroup:
    """Nested records reachable from exactly one bar, in owning-bar order."""

    positions: tuple[int, ...]
    records: tuple[object, ...]
    getter: _Getter
    values: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True, eq=False)
class _SharedRecord:
    """A nested record reachable from several bars of one run."""

    record: object
    getter: _Getter
    values: tuple[object, ...]
    positions: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class _BackingIntegrity:
    """Pristine field identities of every admitted bar and nested record.

    Frozen records can change only through a bypass such as ``object.__setattr__``,
    which replaces a field value. Comparing current field objects with these
    pristine ones detects such a change as completely as reserializing the bars,
    at C-level slice/map cost. Leaves are reviewed immutable scalars.
    """

    bar_getter: _Getter
    bar_values: tuple[tuple[object, ...], ...]
    groups: tuple[_RecordGroup, ...]
    shared: tuple[_SharedRecord, ...]

    @classmethod
    def capture(cls, bars: tuple[ArtifactBar, ...]) -> "_BackingIntegrity | None":
        bar_type = type(bars[0])
        bar_fields = fields(bar_type)
        reached: dict[int, list[int]] = {}
        records: dict[int, object] = {}
        for position, bar in enumerate(bars):
            found: dict[int, object] = {}
            if not all(
                _nested_records(getattr(bar, item.name), found) for item in bar_fields
            ):
                return None
            for identity, record in found.items():
                reached.setdefault(identity, []).append(position)
                records[identity] = record
        exclusive: dict[type, list[tuple[int, object]]] = {}
        shared: list[_SharedRecord] = []
        for identity, positions in reached.items():
            record = records[identity]
            if len(positions) == 1:
                exclusive.setdefault(type(record), []).append((positions[0], record))
            else:
                getter = _field_getter(type(record))
                shared.append(
                    _SharedRecord(record, getter, getter(record), tuple(positions))
                )
        groups: list[_RecordGroup] = []
        for record_type, members in exclusive.items():
            members.sort(key=lambda item: item[0])
            getter = _field_getter(record_type)
            groups.append(
                _RecordGroup(
                    tuple(position for position, _ in members),
                    tuple(record for _, record in members),
                    getter,
                    tuple(getter(record) for _, record in members),
                )
            )
        bar_getter = _field_getter(bar_type)
        return cls(
            bar_getter,
            tuple(bar_getter(bar) for bar in bars),
            tuple(groups),
            tuple(shared),
        )

    def window(self, bars: tuple[ArtifactBar, ...], start: int) -> "_IntegrityWindow":
        """Pristine evidence restricted to records reachable from visible bars."""
        stop = start + len(bars)
        groups: list[tuple[tuple[object, ...], _Getter, tuple[object, ...]]] = []
        for group in self.groups:
            low = bisect_left(group.positions, start)
            high = bisect_left(group.positions, stop)
            if high > low:
                groups.append(
                    (group.records[low:high], group.getter, group.values[low:high])
                )
        return _IntegrityWindow(
            bars,
            self.bar_getter,
            self.bar_values[start:stop],
            tuple(groups),
            tuple(
                (item.record, item.getter, item.values)
                for item in self.shared
                if (index := bisect_left(item.positions, start)) < len(item.positions)
                and item.positions[index] < stop
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class _IntegrityWindow:
    bars: tuple[ArtifactBar, ...]
    bar_getter: _Getter
    bar_values: tuple[tuple[object, ...], ...]
    groups: tuple[tuple[tuple[object, ...], _Getter, tuple[object, ...]], ...]
    shared: tuple[tuple[object, _Getter, tuple[object, ...]], ...]

    def intact(self) -> bool:
        return (
            tuple(map(self.bar_getter, self.bars)) == self.bar_values
            and all(
                tuple(map(getter, records)) == values
                for records, getter, values in self.groups
            )
            and all(getter(record) == values for record, getter, values in self.shared)
        )


@dataclass(frozen=True, slots=True, eq=False)
class PreparedTimeframeRun:
    """One contiguous, validated run of an admitted immutable source.

    ``offset`` is the absolute source position of ``bars[0]``. Preparation proves
    once that the run is one reviewed bar type and timeframe, completed-only,
    strictly ordered and non-overlapping with unique identities. Bar IDs are
    hashed once here instead of repeatedly for every decision.
    """

    source: TimeframeBarSeries
    offset: int
    bars: tuple[ArtifactBar, ...]
    end_timestamps: tuple[datetime, ...]
    completion_states: tuple[BarCompletion, ...]
    bar_ids: tuple[str, ...]
    run_id: str
    intraday: bool
    integrity: _BackingIntegrity

    @classmethod
    def prepare(
        cls, source: TimeframeBarSeries, start: int, stop: int
    ) -> "PreparedTimeframeRun | None":
        """Return ``None`` when the run cannot be admitted; callers use reference."""
        if not 0 <= start < stop <= len(source.bars):
            return None
        bars = source.bars[start:stop]
        bar_type = type(bars[0])
        if bar_type not in _BAR_TYPES:
            return None
        previous: ArtifactBar | None = None
        for bar in bars:
            if (
                type(bar) is not bar_type
                or bar.timeframe != source.timeframe
                or bar.completion is BarCompletion.DEVELOPING
                or (
                    previous is not None
                    and (
                        bar.end_timestamp <= previous.end_timestamp
                        or bar.start_timestamp < previous.end_timestamp
                    )
                )
            ):
                return None
            previous = bar
        bar_ids = tuple(bar.bar_id for bar in bars)
        integrity = _BackingIntegrity.capture(bars)
        if len(set(bar_ids)) != len(bar_ids) or integrity is None:
            return None
        run_id = configuration_identity(
            {
                "component": "quantforge_prepared_timeframe_run",
                "preparation_version": PREPARED_FEATURE_VERSION,
                "dataset_reference": source.dataset_reference.to_primitive(
                    include_feed_scope=True
                ),
                "dataset_family_manifest_id": source.dataset_family_manifest_id,
                "timeframe": source.timeframe.to_primitive(),
                "source_start": start,
                "source_stop": stop,
                "bar_ids": list(bar_ids),
            }
        )
        return cls(
            source,
            start,
            bars,
            tuple(bar.end_timestamp for bar in bars),
            tuple(bar.completion for bar in bars),
            bar_ids,
            run_id,
            bar_type is IntradayBar,
            integrity,
        )


type _Selection = tuple[PreparedTimeframeRun, int, int]


def _slice_output(
    output: TimeframeIndicatorOutput, count: int
) -> TimeframeIndicatorOutput:
    """Structural prefix of a normally validated full-run output.

    Slicing a validated output preserves one-to-one alignment, unique IDs,
    chronology and finite values, so the constructor's O(N) checks are not
    repeated. A new object is always returned; no decision shares a mutable shell.
    """
    sliced = object.__new__(TimeframeIndicatorOutput)
    for item in fields(TimeframeIndicatorOutput):
        object.__setattr__(sliced, item.name, getattr(output, item.name))
    object.__setattr__(sliced, "bar_ids", output.bar_ids[:count])
    object.__setattr__(sliced, "bar_end_timestamps", output.bar_end_timestamps[:count])
    object.__setattr__(sliced, "completion_states", output.completion_states[:count])
    object.__setattr__(
        sliced,
        "fields",
        tuple(
            IndicatorFieldOutput(item.name, item.values[:count])
            for item in output.fields
        ),
    )
    return sliced


@contextmanager
def _backend_evaluation(indicator: object) -> Generator[None]:
    """Serve a series only inside the backend's own per-evaluation checks.

    A reused series skips ``compute()``. ``talib_v1``'s ``compute()`` validates
    TA-Lib's process-global compatibility and unstable periods before and
    after evaluating (in ``finally``); the indicator then compares the reported
    backend identity with its captured one. Serving inside
    ``evaluation_state`` repeats both state checks around the cache access, so
    drift fails with the same errors whether or not the series is cached.
    ``native_v1`` has no process-global state; its identity is compared too.
    """
    backend = getattr(indicator, "_backend")
    definition = cast(
        StandardIndicatorDefinition, getattr(indicator, "standard_definition")
    )
    if type(backend) is TalibIndicatorBackend:
        with backend.evaluation_state(definition) as identity:
            _require_backend_identity(indicator, identity)
            yield
    else:
        identity = cast(NativeIndicatorBackend, backend).identity_for(definition)
        _require_backend_identity(indicator, identity)
        yield


def _require_backend_identity(
    indicator: object, identity: IndicatorBackendIdentity
) -> None:
    if identity != getattr(indicator, "backend_identity"):
        raise InvalidIndicatorBackendError(
            "indicator backend result metadata changed during calculation"
        )


def _exact_values(output: TimeframeIndicatorOutput) -> tuple[object, ...]:
    """Field values including Decimal representation, not only numeric equality."""
    return tuple(
        (
            item.name,
            tuple(None if value is None else value.as_tuple() for value in item.values),
        )
        for item in output.fields
    )


class _ScopeState:
    """Counters and the fail-closed flag; deliberately holds no market data.

    Guards reachable from a rule context reference only this object, never the
    scope and its future-bearing runs or series.
    """

    __slots__ = ("compromised", "counts")

    def __init__(self) -> None:
        self.compromised = False
        self.counts: dict[str, int] = dict.fromkeys(
            (
                "contexts",
                "series_built",
                "series_reused",
                "series_verified",
                "series_verification_fallbacks",
                "reference_indicator_fallbacks",
                "prepared_feature_lookups",
                "guard_checks",
                "source_bars_validated",
                "rule_bars_validated",
            ),
            0,
        )


class _SeriesEntry:
    __slots__ = ("output", "rejected", "verified")

    def __init__(self, output: TimeframeIndicatorOutput) -> None:
        self.output = output
        self.verified = False
        self.rejected = False


class PreparedContextScope:
    """Provider-owned runs, per-dataset validation spans and reusable series.

    Owned by exactly one QF-39 permitted-context provider, so every QF-32 trial
    of that fold/role shares it and no fold, window or warm-up scope can mix.
    Series keys are content identities (run, input start, bound indicator
    configuration including backend/version), never object identity.
    """

    def __init__(self, runs: Mapping[str, PreparedTimeframeRun]) -> None:
        self._runs = MappingProxyType(dict(runs))
        self._series: dict[str, _SeriesEntry] = {}
        self._bound: dict[tuple[object, ...], str] = {}
        self._spans: dict[tuple[object, ...], tuple[int, int]] = {}
        self._metadata: list[tuple[DatasetMetadata, str | None]] = []
        self._state = _ScopeState()

    @classmethod
    def capture(
        cls, runs: tuple[tuple[TimeframeBarSeries, int, int], ...]
    ) -> "PreparedContextScope | None":
        """Prepare every source run, or return ``None`` for the reference path."""
        prepared: dict[str, PreparedTimeframeRun] = {}
        for source, start, stop in runs:
            run = PreparedTimeframeRun.prepare(source, start, stop)
            if run is None:
                return None
            prepared[source.timeframe.configuration_id] = run
        return cls(prepared) if prepared else None

    @property
    def runs(self) -> Mapping[str, PreparedTimeframeRun]:
        """Prepared runs by timeframe configuration ID (read-only)."""
        return self._runs

    def count(self, name: str, amount: int = 1) -> None:
        self._state.counts[name] += amount

    def statistics(self) -> dict[str, int]:
        return {
            **self._state.counts,
            "runs": len(self._runs),
            "prepared_bars": sum(len(run.bars) for run in self._runs.values()),
            "unique_series": len(self._series),
        }

    def memory_report(self) -> dict[str, object]:
        """Approximate bytes owned by prepared state (bars are shared backing)."""
        from sys import getsizeof

        def tuple_bytes(values: tuple[object, ...], *, owned: bool) -> int:
            return getsizeof(values) + (
                sum(getsizeof(item) for item in values) if owned else 0
            )

        runs: dict[str, object] = {}
        for identifier, run in self._runs.items():
            integrity = run.integrity
            runs[identifier] = {
                "bars": len(run.bars),
                "bar_ids_bytes": tuple_bytes(run.bar_ids, owned=True),
                "index_bytes": tuple_bytes(run.bars, owned=False)
                + tuple_bytes(run.end_timestamps, owned=False)
                + tuple_bytes(run.completion_states, owned=False),
                "integrity_bytes": tuple_bytes(integrity.bar_values, owned=False)
                + sum(getsizeof(values) for values in integrity.bar_values)
                + sum(
                    tuple_bytes(group.records, owned=False)
                    + sum(getsizeof(values) for values in group.values)
                    for group in integrity.groups
                )
                + len(integrity.shared) * 200,
            }
        series: list[dict[str, object]] = []
        for entry in self._series.values():
            output = entry.output
            series.append(
                {
                    "indicator": output.indicator_name,
                    "timeframe": output.source_timeframe.configuration_id,
                    "points": len(output.bar_ids),
                    "values_bytes": sum(
                        tuple_bytes(item.values, owned=True) for item in output.fields
                    ),
                    "index_bytes": tuple_bytes(output.bar_ids, owned=False)
                    + tuple_bytes(output.bar_end_timestamps, owned=False)
                    + tuple_bytes(output.completion_states, owned=False),
                }
            )
        return {"runs": runs, "series": series}

    def context_at(
        self,
        requirements_primary: Timeframe,
        required_timeframes: tuple[ContextTimeframeRequirement, ...],
        *,
        as_of: datetime,
        positions: tuple[tuple[str, int, int], ...],
    ) -> MultiTimeframeContext:
        """Build the exact completed-only QF-20 context from source positions.

        ``positions`` are absolute ``[start, stop)`` from QF-59, whose stop is the
        inclusive bar-end cutoff. The O(1) boundary checks below prove the slice
        is the complete causal prefix of its validated run.
        """
        self.require_intact()
        selections: dict[str, _Selection] = {}
        slices: dict[str, tuple[object, ...]] = {}
        for timeframe_id, start, stop in positions:
            run = self._runs.get(timeframe_id)
            if run is None:
                raise MultiTimeframeContextValidationError(
                    "prepared context has no run for a requested timeframe"
                )
            first, last = start - run.offset, stop - run.offset
            if (
                not 0 <= first <= last <= len(run.bars)
                or (last > 0 and run.end_timestamps[last - 1] > as_of)
                or (last < len(run.bars) and run.end_timestamps[last] <= as_of)
            ):
                raise MultiTimeframeContextValidationError(
                    "prepared context position is outside its causal run"
                )
            selections[timeframe_id] = (run, first, last)
            slices[timeframe_id] = (
                run.source.dataset_reference,
                run.bars[first:last],
                run.bar_ids[first:last],
                run.source.dataset_family_manifest_id,
            )
        self._state.counts["contexts"] += 1
        return MultiTimeframeContext._from_prepared_runs(  # pyright: ignore[reportPrivateUsage]
            as_of=as_of,
            primary_timeframe=requirements_primary,
            required_timeframes=required_timeframes,
            runs=cast(
                Mapping[
                    str,
                    tuple[object, tuple[ArtifactBar, ...], tuple[str, ...], str | None],
                ],
                slices,
            ),
            prepared_evidence=PreparedDecisionContext(
                self, MappingProxyType(selections)
            ),
        )

    # -- incremental validation of visible bars -------------------------------

    def validate_span(
        self,
        key: tuple[object, ...],
        run: PreparedTimeframeRun,
        first: int,
        last: int,
        check: Callable[[tuple[ArtifactBar, ...]], None],
    ) -> int:
        """Check only bars not yet checked under ``key``; return the count checked.

        Checks are per bar, so a decision fails exactly when the reference would
        (a previously checked bar passed under the same key). Visible slices of
        a run always share a start within one bar, so checked bars form one span.
        """
        span = self._spans.get(key)
        if span is None or last < span[0] or first > span[1]:
            pieces = run.bars[first:last]
            check(pieces)
            if span is None or last - first >= span[1] - span[0]:
                self._spans[key] = (first, last)
            return len(pieces)
        low, high = span
        pieces = (*run.bars[first:low], *run.bars[high:last])
        check(pieces)
        self._spans[key] = (min(first, low), max(last, high))
        return len(pieces)

    def metadata_key(self, metadata: DatasetMetadata) -> tuple[int, str | None]:
        # Capability binding: validation spans stay valid only for the exact
        # retained metadata object; an equal new object is validated afresh.
        for index, (retained, manifest_id) in enumerate(self._metadata):
            if retained is metadata:
                return index, manifest_id
        provenance = metadata.intraday_provenance
        manifest_id = (
            None
            if provenance is None
            else cast(str, provenance.family_manifest.to_primitive()["manifest_id"])
        )
        self._metadata.append((metadata, manifest_id))
        return len(self._metadata) - 1, manifest_id

    # -- series ---------------------------------------------------------------

    def series_output(
        self,
        selection: _Selection,
        requirement: PredictionIndicatorRequirement,
        context: MultiTimeframeContext,
        timeframe: Timeframe,
    ) -> tuple[TimeframeIndicatorOutput, PreparedOutputEvidence] | None:
        indicator = requirement.indicator
        requirement.validate_unchanged()
        # Reuse skips the backend's compute(): serve only inside its own
        # before/after checks, so drift fails exactly as compute() would.
        with _backend_evaluation(indicator):
            return self._serve_series(selection, requirement, context, timeframe)

    def _serve_series(
        self,
        selection: _Selection,
        requirement: PredictionIndicatorRequirement,
        context: MultiTimeframeContext,
        timeframe: Timeframe,
    ) -> tuple[TimeframeIndicatorOutput, PreparedOutputEvidence] | None:
        run, first, last = selection
        indicator = requirement.indicator
        bound_key = (
            requirement.configuration_id,
            type(indicator),
            run.run_id,
        )
        bound_id = self._bound.get(bound_key)
        if bound_id is None:
            bound_id = bind_indicator(
                indicator,
                context,
                timeframe,
                completion_policy=ContextCompletionPolicy.COMPLETED_BARS_ONLY,
            ).configuration_id
            self._bound[bound_key] = bound_id
        key = configuration_identity(
            {
                "component": "quantforge_prepared_indicator_series",
                "preparation_version": PREPARED_FEATURE_VERSION,
                "run_id": run.run_id,
                "input_start": run.offset + first,
                "timeframe_indicator_configuration_id": bound_id,
                "indicator_type": (
                    f"{type(indicator).__module__}.{type(indicator).__qualname__}"
                ),
                "backend_type": (
                    f"{type(getattr(indicator, '_backend')).__module__}."
                    f"{type(getattr(indicator, '_backend')).__qualname__}"
                ),
            }
        )
        entry = self._series.get(key)
        if entry is None:
            entry = _SeriesEntry(self._build_series(run, first, requirement, timeframe))
            if entry.output.configuration_id != bound_id:
                entry.rejected = True
            self._series[key] = entry
            self._state.counts["series_built"] += 1
        else:
            self._state.counts["series_reused"] += 1
        if entry.rejected:
            self._state.counts["reference_indicator_fallbacks"] += 1
            return None
        output = _slice_output(entry.output, last - first)
        if not entry.verified:
            # Equivalence proof at first use: the reference evaluation of this
            # decision's exact context must equal the prepared prefix exactly.
            reference = requirement.evaluate(
                context, timeframe, ContextCompletionPolicy.COMPLETED_BARS_ONLY
            )
            if reference != output or _exact_values(reference) != _exact_values(output):
                entry.rejected = True
                self._state.counts["series_verification_fallbacks"] += 1
                return None
            entry.verified = True
            self._state.counts["series_verified"] += 1
        self._state.counts["prepared_feature_lookups"] += 1
        return output, PreparedOutputEvidence(
            bound_id,
            run.bar_ids[first:last],
            run.end_timestamps[first:last],
            run.completion_states[first:last],
        )

    def _build_series(
        self,
        run: PreparedTimeframeRun,
        first: int,
        requirement: PredictionIndicatorRequirement,
        timeframe: Timeframe,
    ) -> TimeframeIndicatorOutput:
        """Evaluate once over the run from this exact input start.

        The unchanged QF-22 binding/calculation path runs on a one-timeframe
        context holding exactly ``bars[first:]`` of the run. Seeding therefore
        starts at the same bar as every decision using this key.
        """
        bars = run.bars[first:]
        as_of = run.end_timestamps[-1]
        derivation = MultiTimeframeContext._from_prepared_runs(  # pyright: ignore[reportPrivateUsage]
            as_of=as_of,
            primary_timeframe=timeframe,
            required_timeframes=(),
            runs={
                timeframe.configuration_id: (
                    run.source.dataset_reference,
                    bars,
                    run.bar_ids[first:],
                    run.source.dataset_family_manifest_id,
                )
            },
            prepared_evidence=None,
        )
        return bind_indicator(
            requirement.indicator,
            derivation,
            timeframe,
            completion_policy=ContextCompletionPolicy.COMPLETED_BARS_ONLY,
        ).calculate(derivation)

    # -- integrity ------------------------------------------------------------

    def require_intact(self) -> None:
        if self._state.compromised:
            raise PreparedFeatureIntegrityError(
                "prepared context backing changed after preparation"
            )

    def compromise(self) -> None:
        self._state.compromised = True


@dataclass(frozen=True, slots=True, eq=False)
class PreparedDecisionContext:
    """One decision's binding: its scope and exact run positions per timeframe.

    Attached to ``MultiTimeframeContext`` only. It is never copied into the
    rule-facing ``PredictionRuleContext``; everything handed to rules is a slice
    ending at this decision's causal cutoff.
    """

    scope: PreparedContextScope
    selections: Mapping[str, _Selection]

    def _selection(self, timeframe: Timeframe) -> _Selection | None:
        return self.selections.get(timeframe.configuration_id)

    def validate_rule_inputs(
        self,
        context: MultiTimeframeContext,
        *,
        symbol: str,
        adjustment_basis: AdjustmentBasis,
    ) -> None:
        """QF-28 symbol and adjustment checks, run once per newly visible bar.

        Raises exactly when the reference scan of all visible bars would.
        """
        scope = self.scope
        selections: list[_Selection] = []
        for item in context.timeframes:
            selection = self._selection(item.timeframe)
            if selection is None:
                if item.bars:
                    raise PredictionContextError(
                        "prepared context timeframe has no prepared run"
                    )
                continue
            selections.append(selection)

        def symbol_check(bars: tuple[ArtifactBar, ...]) -> None:
            if any(bar.symbol != symbol for bar in bars):
                raise PredictionContextError(
                    "prediction context symbol is incompatible with the prediction "
                    "dataset"
                )

        def basis_check(bars: tuple[ArtifactBar, ...]) -> None:
            if any(
                cast(IntradayBar, bar).provenance.adjustment_basis != adjustment_basis
                for bar in bars
            ):
                raise PredictionContextError(
                    "prediction context adjustment basis is incompatible with the "
                    "prediction dataset"
                )

        for run, first, last in selections:
            scope.count(
                "rule_bars_validated",
                scope.validate_span(
                    ("symbol", run.run_id, symbol), run, first, last, symbol_check
                ),
            )
        intraday_visible = False
        for run, first, last in selections:
            if run.intraday and last > first:
                intraday_visible = True
                scope.count(
                    "rule_bars_validated",
                    scope.validate_span(
                        ("basis", run.run_id, adjustment_basis),
                        run,
                        first,
                        last,
                        basis_check,
                    ),
                )
        if not intraday_visible:
            raise PredictionContextError(
                "prediction context adjustment basis is incompatible with the "
                "prediction dataset"
            )

    def validate_sources(
        self, dataset: MarketDataset, context: MultiTimeframeContext
    ) -> None:
        """QF-52 source binding with per-bar lineage checked once per bar.

        Mirrors ``validate_prediction_context_sources``: the family manifest and
        each reference/session policy are checked every decision; the per-bar
        symbol, basis and lineage checks run only for bars not yet validated
        against this exact metadata object.
        """
        scope = self.scope
        metadata_index, manifest_id = scope.metadata_key(dataset.metadata)
        if (
            dataset.metadata.intraday_provenance is not None
            and context.dataset_family_manifest_id != manifest_id
        ):
            raise ValidationError("prediction context family manifest is incompatible")
        for item in context.timeframes:
            reference = item.dataset_reference
            if reference is None:
                continue
            timeframe = item.requirement.timeframe
            selection = self._selection(timeframe)
            if selection is None:
                _validate_source(dataset, reference, timeframe, item.bars)
                continue
            run, first, last = selection

            def check(
                bars: tuple[ArtifactBar, ...],
                reference: DatasetFamilyReference = reference,
                timeframe: Timeframe = timeframe,
            ) -> None:
                _validate_source(dataset, reference, timeframe, bars)

            scope.count(
                "source_bars_validated",
                scope.validate_span(
                    ("source", metadata_index, run.run_id), run, first, last, check
                ),
            )

    def indicator_output(
        self,
        requirement: PredictionIndicatorRequirement,
        context: MultiTimeframeContext,
        timeframe: Timeframe,
        completion_policy: ContextCompletionPolicy,
    ) -> tuple[TimeframeIndicatorOutput, PreparedOutputEvidence] | None:
        """Prepared prefix for reviewed indicators; ``None`` means reference."""
        selection = self._selection(timeframe)
        indicator = requirement.indicator
        if (
            selection is None
            or selection[2] <= selection[1]
            or completion_policy is not ContextCompletionPolicy.COMPLETED_BARS_ONLY
            or not prefix_stable_indicator(indicator)
        ):
            self.scope.count("reference_indicator_fallbacks")
            return None
        return self.scope.series_output(selection, requirement, context, timeframe)

    def values_guard(self, context: PredictionRuleContext) -> "PreparedValuesGuard":
        windows: list[tuple[PredictionTimeframeInput, _IntegrityWindow]] = []
        for item in context.timeframes:
            selection = self._selection(item.requirement.timeframe)
            if selection is None:
                raise PredictionContextError(
                    "prepared rule timeframe has no prepared run"
                )
            run, first, last = selection
            if len(item.bars) != last - first or not all(
                map(is_, item.bars, run.bars[first:last])
            ):
                raise PredictionContextError(
                    "prepared rule bars differ from their prepared run"
                )
            windows.append(
                (
                    item,
                    run.integrity.window(
                        cast(tuple[ArtifactBar, ...], item.bars), first
                    ),
                )
            )
        return PreparedValuesGuard(
            self.scope._state,  # pyright: ignore[reportPrivateUsage]
            context,
            tuple(windows),
        )


def _record_values(value: object, out: list[tuple[object, tuple[object, ...]]]) -> None:
    """Identity evidence for small frozen records (wrappers, references, keys)."""
    if isinstance(value, tuple):
        for item in cast(tuple[object, ...], value):
            _record_values(item, out)
    elif is_dataclass(value) and not isinstance(value, type):
        values = tuple(getattr(value, item.name) for item in fields(value))
        out.append((value, values))
        for item in values:
            _record_values(item, out)


def _wrapper_evidence(
    context: PredictionRuleContext,
) -> tuple[tuple[object, tuple[object, ...]], ...]:
    """Every per-decision wrapper field, with large immutable tuples by identity.

    Bar tuples are verified through the run integrity window; ID, timestamp,
    completion and value tuples hold only immutable scalars.
    """
    out: list[tuple[object, tuple[object, ...]]] = [
        (
            context,
            (
                context.prediction_dataset_id,
                context.symbol,
                context.adjustment_basis,
                context.requirements,
                context.source_context_snapshot,
                context.timeframes,
                context.dataset_family_manifest_id,
            ),
        )
    ]
    _record_values(context.adjustment_basis, out)
    _record_values(context.source_context_snapshot, out)
    for item in context.timeframes:
        out.append(
            (
                item,
                (
                    item.requirement,
                    item.bars,
                    item.indicators,
                    item._bar_ids,  # pyright: ignore[reportPrivateUsage]
                ),
            )
        )
        for named in item.indicators:
            output = named.output
            out.append((named, (named.alias, output)))
            out.append(
                (output, tuple(getattr(output, entry.name) for entry in fields(output)))
            )
            for field_output in output.fields:
                out.append((field_output, (field_output.name, field_output.values)))
            for record in (
                output.source_timeframe,
                output.dataset_reference,
                output.feed_scope,
                output.backend_identity,
            ):
                _record_values(record, out)
    return tuple(out)


class PreparedValuesGuard:
    """Prove a prepared rule context is unchanged, as its values snapshot would.

    Holds only this decision's visible bars and their pristine field evidence,
    the per-decision wrapper evidence and the requirement snapshot.
    """

    __slots__ = ("_requirements", "_state", "_windows", "_wrappers")

    def __init__(
        self,
        state: _ScopeState,
        context: PredictionRuleContext,
        windows: tuple[tuple[PredictionTimeframeInput, _IntegrityWindow], ...],
    ) -> None:
        self._state = state
        self._windows = windows
        self._wrappers = _wrapper_evidence(context)
        self._requirements = PrimitiveMappingSnapshot.capture(
            context.requirements.to_primitive()
        )

    def verify(self, context: PredictionRuleContext) -> None:
        self._state.counts["guard_checks"] += 1
        backing_intact = all(window.intact() for _, window in self._windows)
        if not backing_intact:
            # Shared backing changed: refuse every later use of this scope.
            self._state.compromised = True
        if (
            not backing_intact
            or _wrapper_evidence(context) != self._wrappers
            or any(item.bars is not window.bars for item, window in self._windows)
            or PrimitiveMappingSnapshot.capture(context.requirements.to_primitive())
            != self._requirements
        ):
            raise PredictionContextError(
                "prediction rule context values changed after construction"
            )


def validate_prepared_context_sources(
    dataset: MarketDataset, context: MultiTimeframeContext
) -> bool:
    """Validate a prepared context's sources; ``False`` means use the reference."""
    evidence = context.prepared_evidence
    if type(evidence) is not PreparedDecisionContext:
        return False
    evidence.validate_sources(dataset, context)
    return True


__all__ = [
    "PREPARED_FEATURE_VERSION",
    "PreparedContextScope",
    "PreparedDecisionContext",
    "PreparedFeatureIntegrityError",
    "PreparedTimeframeRun",
    "PreparedValuesGuard",
    "prefix_stable_indicator",
    "validate_prepared_context_sources",
]
