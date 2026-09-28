"""QF-62 shared membership catalogues and exact visible-bar range references.

Schema "3" historical windows persist every QF-20 source, QF-28 rule-timeframe
and normalized-indicator ``visible_bar_ids`` list either unchanged or as an exact
``[start_index, stop_index)`` reference into an append-only catalogue owned by
the window. A catalogue is bound to one source dataset reference and one
timeframe (including its session policy); its identity hashes only that binding.

Entries are introduced, in first-appearance order, by the decision whose
membership first needs them and are physically stored once. A decision can
therefore never reference an entry first introduced by a later decision. The
encoding is canonical: readers replay it from the preceding catalogue state and
reject any other representation. Scientific identities (QF-20 context, QF-11
study and row IDs) remain hashes of the exact reconstructed expanded lists.

Lists that are not one contiguous catalogue run, contain a developing bar, have
no complete source binding or are empty remain explicit. Nothing is coerced.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    configuration_identity,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_encoding import canonical, mapping

MEMBERSHIP_CATALOGUE_SCHEMA_VERSION = "1"
EXPANDED_MEMBERSHIP_FIELD = "visible_bar_ids"
MEMBERSHIP_RANGE_FIELD = "visible_bar_range"
CATALOGUE_SEGMENTS_FIELD = "catalogue_segments"
_REFERENCE_FIELDS = frozenset({"catalogue_id", "start_index", "stop_index"})
_DATASET_REFERENCE_FIELDS = frozenset(
    {
        "canonical_source_snapshot_id",
        "dataset_id",
        "family_id",
        "timeframe_configuration_id",
    }
)


def membership_binding(
    dataset_reference: Primitive, timeframe: Primitive
) -> PrimitiveMapping | None:
    """Return an exact source/timeframe/session binding, or None if incomplete.

    Indicator references additionally carry their declared feed scope. The feed
    remains in the indicator record; membership belongs to the dataset's bars.
    """
    if not isinstance(dataset_reference, dict) or not isinstance(timeframe, dict):
        return None
    reference = {
        key: value for key, value in dataset_reference.items() if key != "feed_scope"
    }
    configuration = timeframe.get("configuration")
    if (
        frozenset(reference) != _DATASET_REFERENCE_FIELDS
        or any(not isinstance(value, str) or not value for value in reference.values())
        or set(timeframe) != {"configuration", "configuration_id"}
        or not isinstance(configuration, dict)
        or timeframe["configuration_id"] != reference["timeframe_configuration_id"]
        or configuration_identity(configuration) != timeframe["configuration_id"]
    ):
        return None
    return {"dataset_reference": reference, "timeframe": timeframe}


def catalogue_identity(binding: PrimitiveMapping) -> str:
    """Deterministic identity of one catalogue's immutable source binding."""
    return configuration_identity(
        {
            "component": "quantforge_membership_catalogue",
            "schema_version": MEMBERSHIP_CATALOGUE_SCHEMA_VERSION,
            "binding": binding,
        }
    )


@dataclass(frozen=True, slots=True)
class _Slot:
    """One persisted membership location and the binding its evidence claims."""

    owner: PrimitiveMapping
    binding: PrimitiveMapping | None


def _requirement_timeframe(entry: PrimitiveMapping) -> Primitive:
    requirement = entry.get("requirement")
    return requirement.get("timeframe") if isinstance(requirement, dict) else None


def _slots(context: PrimitiveMapping) -> tuple[list[_Slot], frozenset[str]]:
    """Every membership location in canonical order, plus developing bar IDs.

    Source timeframes bind to their own dataset reference. A rule timeframe binds
    to its index-aligned source entry only when both declare the same timeframe;
    QF-28 validation independently requires that alignment. Indicators bind to
    their own reference and source timeframe.
    """
    slots: list[_Slot] = []
    developing: set[str] = set()
    aligned: list[PrimitiveMapping | None] = []
    source = context.get("source_context")
    if isinstance(source, dict) and isinstance(source.get("timeframes"), list):
        for entry in cast(list[Primitive], source["timeframes"]):
            if not isinstance(entry, dict):
                aligned.append(None)
                continue
            aligned.append(entry)
            slots.append(
                _Slot(
                    entry,
                    membership_binding(
                        entry.get("dataset_reference"), _requirement_timeframe(entry)
                    ),
                )
            )
            bar = entry.get("developing_bar")
            if isinstance(bar, dict) and isinstance(bar.get("bar_id"), str):
                developing.add(cast(str, bar["bar_id"]))
    rules = context.get("timeframes")
    if isinstance(rules, list):
        for index, entry in enumerate(cast(list[Primitive], rules)):
            if not isinstance(entry, dict):
                continue
            timeframe = _requirement_timeframe(entry)
            source_entry = aligned[index] if index < len(aligned) else None
            binding = (
                membership_binding(source_entry.get("dataset_reference"), timeframe)
                if source_entry is not None
                and _requirement_timeframe(source_entry) == timeframe
                else None
            )
            slots.append(_Slot(entry, binding))
            indicators = entry.get("indicators")
            if isinstance(indicators, list):
                slots.extend(
                    _Slot(
                        indicator,
                        membership_binding(
                            indicator.get("dataset_reference"),
                            indicator.get("source_timeframe"),
                        ),
                    )
                    for indicator in cast(list[Primitive], indicators)
                    if isinstance(indicator, dict)
                )
    return slots, frozenset(developing)


def _rebuild(
    context: PrimitiveMapping,
    replace: Callable[[_Slot], PrimitiveMapping],
) -> PrimitiveMapping:
    """Copy only containers on membership paths; never mutate the caller's context."""
    slots, _ = _slots(context)
    replacements = {id(slot.owner): replace(slot) for slot in slots}

    def entries(values: Primitive) -> list[Primitive]:
        return [
            replacements.get(id(item), item) if isinstance(item, dict) else item
            for item in cast(list[Primitive], values)
        ]

    result = dict(context)
    source = context.get("source_context")
    if isinstance(source, dict) and isinstance(source.get("timeframes"), list):
        result["source_context"] = {
            **source,
            "timeframes": entries(source["timeframes"]),
        }
    rules = context.get("timeframes")
    if isinstance(rules, list):
        rebuilt: list[Primitive] = []
        for entry in cast(list[Primitive], rules):
            if isinstance(entry, dict):
                replaced: PrimitiveMapping = replacements.get(id(entry), entry)
                indicators = replaced.get("indicators")
                if isinstance(indicators, list):
                    replaced = {**replaced, "indicators": entries(indicators)}
                rebuilt.append(replaced)
            else:
                rebuilt.append(entry)
        result["timeframes"] = rebuilt
    return result


def _walk(value: Primitive) -> Iterator[PrimitiveMapping]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _reference(value: Primitive) -> tuple[str, int, int]:
    if (
        not isinstance(value, dict)
        or frozenset(value) != _REFERENCE_FIELDS
        or not isinstance(value["catalogue_id"], str)
        or type(value["start_index"]) is not int
        or type(value["stop_index"]) is not int
    ):
        raise InvalidPredictionOutputError("invalid membership range reference")
    catalogue_id, start, stop = (
        value["catalogue_id"],
        value["start_index"],
        value["stop_index"],
    )
    if not 0 <= start < stop:
        raise InvalidPredictionOutputError("membership range bounds are invalid")
    return catalogue_id, start, stop


def prediction_context(record: PrimitiveMapping) -> PrimitiveMapping:
    study = mapping(record["prediction_study"])
    return mapping(mapping(study["manifest"])["prediction_context"])


def with_prediction_context(
    record: PrimitiveMapping, context: PrimitiveMapping
) -> PrimitiveMapping:
    """Return the record with a replaced context; the input is not modified."""
    study = mapping(record["prediction_study"])
    manifest = mapping(study["manifest"])
    return {
        **record,
        "prediction_study": {
            **study,
            "manifest": {**manifest, "prediction_context": context},
        },
    }


@dataclass(slots=True)
class _Catalogue:
    binding: PrimitiveMapping
    bar_ids: list[str] = field(default_factory=list[str])
    positions: dict[str, int] = field(default_factory=dict[str, int])

    def append(self, bar_ids: list[str]) -> None:
        for bar_id in bar_ids:
            self.positions[bar_id] = len(self.bar_ids)
            self.bar_ids.append(bar_id)

    def truncate(self, length: int) -> None:
        for bar_id in self.bar_ids[length:]:
            del self.positions[bar_id]
        del self.bar_ids[length:]


@dataclass(frozen=True, slots=True)
class MembershipView:
    """Catalogue state visible to one accepted decision, expanded only on demand.

    Catalogues are append-only, so the lists may later grow without changing any
    range this decision was validated against.
    """

    catalogues: dict[str, list[str]]

    def expand(self, context: PrimitiveMapping) -> PrimitiveMapping:
        """Reconstruct the exact expanded QF-20/QF-28 context without mutation."""

        def replace(slot: _Slot) -> PrimitiveMapping:
            if MEMBERSHIP_RANGE_FIELD not in slot.owner:
                return slot.owner
            catalogue_id, start, stop = _reference(slot.owner[MEMBERSHIP_RANGE_FIELD])
            bar_ids = self.catalogues.get(catalogue_id)
            if (
                bar_ids is None
                or stop > len(bar_ids)
                or EXPANDED_MEMBERSHIP_FIELD in slot.owner
            ):
                raise InvalidPredictionOutputError(
                    "membership range does not resolve in its catalogue"
                )
            expanded = {
                key: value
                for key, value in slot.owner.items()
                if key != MEMBERSHIP_RANGE_FIELD
            }
            expanded[EXPANDED_MEMBERSHIP_FIELD] = cast(
                list[Primitive], bar_ids[start:stop]
            )
            return expanded

        return _rebuild(context, replace)

    def expand_record(self, record: PrimitiveMapping) -> PrimitiveMapping:
        """Logical record: exact expanded membership and no physical segments."""
        expanded = with_prediction_context(
            record, self.expand(prediction_context(record))
        )
        expanded.pop(CATALOGUE_SEGMENTS_FIELD, None)
        return expanded


class MembershipCatalogues:
    """Append-only catalogues for one ordered window traversal.

    ``normalize`` derives the canonical representation without changing state.
    ``accept`` authenticates one persisted record against the preceding state and
    commits its segments. ``restore`` rolls back to earlier lengths.
    """

    def __init__(self) -> None:
        self._catalogues: dict[str, _Catalogue] = {}

    def lengths(self) -> dict[str, int]:
        return {key: len(value.bar_ids) for key, value in self._catalogues.items()}

    def restore(self, lengths: dict[str, int]) -> None:
        for catalogue_id in list(self._catalogues):
            if catalogue_id in lengths:
                self._catalogues[catalogue_id].truncate(lengths[catalogue_id])
            else:
                del self._catalogues[catalogue_id]

    def view(self) -> MembershipView:
        return MembershipView(
            {key: value.bar_ids for key, value in self._catalogues.items()}
        )

    def summaries(self) -> list[Primitive]:
        """Window-level identities of the complete ordered catalogue contents."""
        return [
            {
                "catalogue_id": catalogue_id,
                "bar_count": len(catalogue.bar_ids),
                "catalogue_content_id": configuration_identity(
                    {
                        "catalogue_id": catalogue_id,
                        "bar_ids": cast(list[Primitive], catalogue.bar_ids),
                    }
                ),
            }
            for catalogue_id, catalogue in sorted(self._catalogues.items())
        ]

    def _encode(
        self,
        members: Primitive,
        binding: PrimitiveMapping | None,
        developing: frozenset[str],
    ) -> PrimitiveMapping | None:
        """Canonical range for one list, extending its catalogue if required."""
        if binding is None or not isinstance(members, list) or not members:
            return None
        if any(
            not isinstance(item, str) or not item or item in developing
            for item in members
        ) or len(set(cast(list[str], members))) != len(members):
            return None
        bar_ids = cast(list[str], members)
        catalogue_id = catalogue_identity(binding)
        catalogue = self._catalogues.get(catalogue_id)
        if catalogue is None:
            catalogue = self._catalogues[catalogue_id] = _Catalogue(binding)
        start = catalogue.positions.get(bar_ids[0])
        if start is None:
            start, overlap = len(catalogue.bar_ids), 0
        else:
            overlap = min(len(bar_ids), len(catalogue.bar_ids) - start)
            if catalogue.bar_ids[start : start + overlap] != bar_ids[:overlap]:
                return None
        if any(item in catalogue.positions for item in bar_ids[overlap:]):
            return None
        catalogue.append(bar_ids[overlap:])
        return {
            "catalogue_id": catalogue_id,
            "start_index": start,
            "stop_index": start + len(bar_ids),
        }

    def _visit(
        self,
        context: PrimitiveMapping,
        members: Callable[[PrimitiveMapping], Primitive],
    ) -> tuple[PrimitiveMapping, list[Primitive]]:
        """Canonically encode every slot, mutating state; callers own rollback."""
        before = self.lengths()
        _, developing = _slots(context)

        def replace(slot: _Slot) -> PrimitiveMapping:
            value = members(slot.owner)
            reference = self._encode(value, slot.binding, developing)
            owner = {
                key: item
                for key, item in slot.owner.items()
                if key not in (EXPANDED_MEMBERSHIP_FIELD, MEMBERSHIP_RANGE_FIELD)
            }
            if reference is None:
                if value is not None:
                    owner[EXPANDED_MEMBERSHIP_FIELD] = value
            else:
                owner[MEMBERSHIP_RANGE_FIELD] = reference
            return owner

        normalized = _rebuild(context, replace)
        segments: list[Primitive] = []
        for catalogue_id, catalogue in sorted(self._catalogues.items()):
            start = before.get(catalogue_id, 0)
            if len(catalogue.bar_ids) == start:
                continue
            segment: PrimitiveMapping = {
                "catalogue_id": catalogue_id,
                "start_index": start,
                "bar_ids": cast(list[Primitive], catalogue.bar_ids[start:]),
            }
            if start == 0:
                segment["binding"] = catalogue.binding
            segments.append(segment)
        return normalized, segments

    def normalize(
        self, context: PrimitiveMapping
    ) -> tuple[PrimitiveMapping, list[Primitive]]:
        """Return the canonical context and new segments; state is unchanged."""
        if any(MEMBERSHIP_RANGE_FIELD in item for item in _walk(context)):
            raise InvalidPredictionOutputError(
                "logical context already contains membership references"
            )
        lengths = self.lengths()
        try:
            return self._visit(
                context, lambda owner: owner.get(EXPANDED_MEMBERSHIP_FIELD)
            )
        finally:
            self.restore(lengths)

    def _post_catalogues(self, segments: Primitive) -> dict[str, list[str]]:
        """Validate segment shape/order/novelty; return post-decision lists."""
        if not isinstance(segments, list):
            raise InvalidPredictionOutputError("catalogue segments are missing")
        catalogues = {key: value.bar_ids for key, value in self._catalogues.items()}
        previous = ""
        for item in cast(list[Primitive], segments):
            segment = mapping(item)
            catalogue_id = segment.get("catalogue_id")
            start = segment.get("start_index")
            bar_ids = segment.get("bar_ids")
            existing = self._catalogues.get(cast(str, catalogue_id))
            expected: set[str] = {"catalogue_id", "start_index", "bar_ids"} | (
                {"binding"} if start == 0 else set()
            )
            if (
                set(segment) != expected
                or not isinstance(catalogue_id, str)
                or catalogue_id <= previous
                or type(start) is not int
                or start != (0 if existing is None else len(existing.bar_ids))
                or not isinstance(bar_ids, list)
                or not bar_ids
                or any(not isinstance(bar_id, str) or not bar_id for bar_id in bar_ids)
                or len(set(cast(list[str], bar_ids))) != len(bar_ids)
                or (
                    existing is not None
                    and any(bar_id in existing.positions for bar_id in bar_ids)
                )
            ):
                raise InvalidPredictionOutputError(
                    "invalid, duplicated, reordered or non-contiguous catalogue segment"
                )
            if start == 0:
                binding = mapping(segment["binding"])
                if (
                    membership_binding(
                        binding.get("dataset_reference"), binding.get("timeframe")
                    )
                    != binding
                    or catalogue_identity(binding) != catalogue_id
                ):
                    raise InvalidPredictionOutputError(
                        "catalogue identity differs from its source binding"
                    )
            previous = catalogue_id
            catalogues[catalogue_id] = [
                *catalogues.get(catalogue_id, []),
                *cast(list[str], bar_ids),
            ]
        return catalogues

    def accept(self, record: PrimitiveMapping) -> MembershipView:
        """Authenticate one normalized record's segments/ranges and commit them.

        The persisted form must equal the canonical normalization of the exact
        membership it references, replayed from the preceding catalogue state.
        Foreign, rebound, shifted, widened, future and unneeded references and
        segments therefore fail closed. Scientific identities are checked by the
        caller against the expanded view.
        """
        try:
            context = prediction_context(record)
            slots, _ = _slots(context)
            owners = {id(slot.owner) for slot in slots}
            if any(
                MEMBERSHIP_RANGE_FIELD in item and id(item) not in owners
                for item in _walk(context)
            ):
                raise InvalidPredictionOutputError(
                    "membership reference is outside a recognized membership slot"
                )
            segments = record.get(CATALOGUE_SEGMENTS_FIELD)
            post = self._post_catalogues(segments)
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidPredictionOutputError(
                f"invalid normalized membership: {error}"
            ) from error

        def members(owner: PrimitiveMapping) -> Primitive:
            if MEMBERSHIP_RANGE_FIELD not in owner:
                return owner.get(EXPANDED_MEMBERSHIP_FIELD)
            if EXPANDED_MEMBERSHIP_FIELD in owner:
                raise InvalidPredictionOutputError(
                    "membership is both explicit and range-referenced"
                )
            catalogue_id, start, stop = _reference(owner[MEMBERSHIP_RANGE_FIELD])
            bar_ids = post.get(catalogue_id)
            if bar_ids is None or stop > len(bar_ids):
                raise InvalidPredictionOutputError(
                    "membership range exceeds its catalogue at this decision"
                )
            return cast(list[Primitive], bar_ids[start:stop])

        lengths = self.lengths()
        try:
            normalized, derived = self._visit(context, members)
            if canonical(normalized) != canonical(context) or canonical(
                {"segments": derived}
            ) != canonical({"segments": cast(list[Primitive], segments)}):
                raise InvalidPredictionOutputError(
                    "membership encoding is not the exact canonical normalization"
                )
        except (KeyError, TypeError, ValueError) as error:
            self.restore(lengths)
            raise InvalidPredictionOutputError(
                f"invalid normalized membership: {error}"
            ) from error
        except BaseException:
            self.restore(lengths)
            raise
        return self.view()


__all__ = [
    "CATALOGUE_SEGMENTS_FIELD",
    "EXPANDED_MEMBERSHIP_FIELD",
    "MEMBERSHIP_CATALOGUE_SCHEMA_VERSION",
    "MEMBERSHIP_RANGE_FIELD",
    "MembershipCatalogues",
    "MembershipView",
    "catalogue_identity",
    "membership_binding",
]
