"""Frozen event-model envelopes: freeze once, validate offline (QF-68).

Freezing persists the complete fitted model and its selection evidence before
any out-of-sample use::

    <output-root>/<model-id>/
        model.json   canonical {"payload": ..., "fingerprint": sha256} line

The payload is inspectable JSON: the bound model configuration (dataset,
population, feature schema and order, target, fold chronology, runtime
versions), the exact fitting observations and exclusions, the preprocessing
state, the estimator's intercept and coefficients, the training-prevalence
baseline, and the selection prediction set's identity, summary and metrics.
Reading never deserializes a pickle, joblib or any library object.

``read_event_model`` is the only constructor of ``FrozenEventModel``. Every
out-of-sample or holdout call re-reads and re-validates the envelope at its
path, so the persisted, validated contract (not a caller flag) is what
authorizes inference.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.ml.modeling.configuration import runtime_platform
from quantforge.ml.modeling.errors import (
    ModelArtifactIntegrityError,
    ModelFreezeError,
)
from quantforge.ml.modeling.predictions import (
    publish_directory,
    safe_output_root,
    score_partition,
)
from quantforge.ml.modeling.training import (
    INTERPRETATION,
    EventModelState,
    FittedEventModel,
    load_event_dataset,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_encoding import canonical, decode
from quantforge.validation import PartitionRole

EVENT_MODEL_COMPONENT = "quantforge_event_model"
EVENT_MODEL_SCHEMA_VERSION = "1"
MODEL_FILE = "model.json"
_PAYLOAD_FIELDS = frozenset(
    {
        "component",
        "schema_version",
        "model_id",
        "lifecycle",
        "model",
        "selection",
        "provenance",
        "interpretation",
    }
)
_LIFECYCLE: PrimitiveMapping = {
    "state": "frozen",
    "frozen_before": ["walk_forward_test_inference", "final_holdout_inference"],
    "authority": "validated_persisted_envelope_reread_at_every_inference",
    "after_freeze": "no_refit_no_selection_any_change_is_a_new_model",
}
_VERIFIED = object()


@dataclass(frozen=True, slots=True)
class FrozenEventModel:
    """A validated frozen model read from its persisted envelope."""

    path: Path
    state: EventModelState
    selection: PrimitiveMappingSnapshot
    fingerprint: str
    _token: object = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self._token is not _VERIFIED:
            raise ModelFreezeError(
                "frozen models are created only by read_event_model from a "
                "persisted envelope"
            )

    @property
    def model_id(self) -> str:
        return self.state.model_id

    @property
    def selection_prediction_set_id(self) -> str:
        return cast(str, self.selection.to_primitive()["prediction_set_id"])


def _payload(model: FittedEventModel, selection: PrimitiveMapping) -> PrimitiveMapping:
    state = model.state
    return {
        "component": EVENT_MODEL_COMPONENT,
        "schema_version": EVENT_MODEL_SCHEMA_VERSION,
        "model_id": state.model_id,
        "lifecycle": _LIFECYCLE,
        "model": state.to_primitive(),
        "selection": selection,
        "provenance": {
            "platform": runtime_platform(),
            "note": "operational only; outside model and configuration identity",
        },
        "interpretation": INTERPRETATION,
    }


def freeze_event_model(
    model: FittedEventModel, *, dataset: Path, output_root: Path
) -> Path:
    """Persist the fitted model and its selection evidence; return its path.

    The selection prediction set is regenerated here from ``dataset`` (the
    model's exact training dataset), so frozen selection evidence cannot be
    supplied by the caller. The envelope is written and validated in a private
    staging directory and then atomically renamed to ``<output-root>/<model-id>``.
    """
    if type(cast(object, model)) is not FittedEventModel:
        raise ModelFreezeError("only a FittedEventModel from fit_event_model freezes")
    root = safe_output_root(output_root, "model output")
    loaded = load_event_dataset(dataset)
    selection = score_partition(model.state, loaded, PartitionRole.SELECTION)
    record: PrimitiveMapping = {
        "prediction_set_id": selection.prediction_set_id,
        "summary": selection.summary(),
        "metrics": selection.metrics(),
        "baseline_metrics": selection.baseline_metrics(),
        "note": "selection evidence frozen with the model; no parameter was selected",
    }
    payload = _payload(model, record)
    content = (
        canonical({"payload": payload, "fingerprint": configuration_identity(payload)})
        + b"\n"
    )

    def verify(path: Path) -> None:
        frozen = _read(path, model.model_id)
        if frozen.state != model.state or frozen.selection.to_primitive() != record:
            raise ModelArtifactIntegrityError("existing model directory differs")

    return publish_directory(root, model.model_id, {MODEL_FILE: content}, verify)


def _read(path: Path, name: str) -> FrozenEventModel:
    try:
        envelope = decode((path / MODEL_FILE).read_bytes(), canonical_line=True)
    except (OSError, InvalidPredictionOutputError) as error:
        raise ModelArtifactIntegrityError(
            "model envelope is missing or invalid"
        ) from error
    payload = envelope.get("payload")
    if (
        set(envelope) != {"payload", "fingerprint"}
        or not isinstance(payload, dict)
        or envelope["fingerprint"] != configuration_identity(payload)
    ):
        raise ModelArtifactIntegrityError("model envelope fingerprint is inconsistent")
    payload = cast(PrimitiveMapping, payload)
    if (
        frozenset(payload) != _PAYLOAD_FIELDS
        or payload["component"] != EVENT_MODEL_COMPONENT
        or payload["schema_version"] != EVENT_MODEL_SCHEMA_VERSION
        or payload["lifecycle"] != _LIFECYCLE
        or payload["interpretation"] != INTERPRETATION
    ):
        raise ModelArtifactIntegrityError("unsupported model envelope")
    state = EventModelState.from_primitive(payload["model"])
    selection = payload["selection"]
    if (
        payload["model_id"] != state.model_id
        or name != state.model_id
        or not isinstance(selection, dict)
        or set(selection)
        != {"prediction_set_id", "summary", "metrics", "baseline_metrics", "note"}
        or not isinstance(selection["prediction_set_id"], str)
    ):
        raise ModelArtifactIntegrityError(
            "model identity differs from its content or directory"
        )
    return FrozenEventModel(
        path.resolve(),
        state,
        PrimitiveMappingSnapshot.capture(cast(PrimitiveMapping, selection)),
        cast(str, envelope["fingerprint"]),
        _VERIFIED,
    )


def read_event_model(path: Path) -> FrozenEventModel:
    """Validate a frozen model directory offline and return it."""
    path = Path(path)
    return _read(path, path.name)


def require_frozen(model: object) -> FrozenEventModel:
    """Re-read the persisted envelope; refuse anything that is not frozen."""
    if type(model) is FittedEventModel:
        raise ModelFreezeError(
            "out-of-sample and holdout inference require a frozen model: call "
            "freeze_event_model and read_event_model first"
        )
    if type(model) is not FrozenEventModel:
        raise ModelFreezeError("a FrozenEventModel from read_event_model is required")
    try:
        current = read_event_model(model.path)
    except ModelArtifactIntegrityError as error:
        raise ModelFreezeError(
            f"the frozen model envelope is missing or no longer valid: {error}"
        ) from error
    if current != model or current.fingerprint != model.fingerprint:
        raise ModelFreezeError("the frozen model differs from its persisted envelope")
    return current


__all__ = [
    "EVENT_MODEL_COMPONENT",
    "MODEL_FILE",
    "FrozenEventModel",
    "freeze_event_model",
    "read_event_model",
    "require_frozen",
]
