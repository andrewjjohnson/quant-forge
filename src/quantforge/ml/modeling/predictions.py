"""Selection, out-of-sample and holdout prediction sets (QF-68).

A ``PredictionSet`` holds one partition's predictions from one model, kept
separate from fitted training state. Each record binds the exact dataset row
(``row_index``, ``source_observation_id``, decision timestamp, role and fold),
the score status, the positive-class probability, the optional class from the
frozen threshold and the target where evaluation is permitted. Records follow
QF-67 dataset row order. Every row of the partition is present: a row whose
null feature cannot be scored under ``exclude_observation`` is recorded as
``unscored_null_feature``; a scored row with an unavailable label is reported
separately from the labeled rows used for metrics.

Artifact layout (one immutable directory named by ``prediction_set_id``)::

    <output-root>/<prediction-set-id>/
        manifest.json      canonical {"payload": ..., "fingerprint": sha256} line
        predictions.jsonl  one canonical JSON record per line

Probabilities are stored as JSON binary64 (shortest round-trip text), so a
read reproduces every stored float exactly. Regenerated probabilities must
match within ``PROBABILITY_TOLERANCE`` (``exp`` may differ by one unit in the
last place across platforms); on one platform they are bitwise identical.
"""

import hashlib
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.ml.dataset import EventDataset
from quantforge.ml.modeling.chronology import (
    DatasetBinding,
    parse_utc_instant,
    require_fold_membership,
)
from quantforge.ml.modeling.errors import (
    ModelArtifactIntegrityError,
    ModelBindingError,
    PredictionIntegrityError,
)
from quantforge.ml.modeling.metrics import describe_probabilities
from quantforge.ml.modeling.preprocessing import finite
from quantforge.ml.modeling.training import EventModelState, numeric_rows
from quantforge.ml.sources import WORKSPACE_HOLDOUT_LEDGER, reject_non_authoritative
from quantforge.ml.targets import TARGET_STATUSES
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_encoding import canonical, decode
from quantforge.validation import PartitionRole

PREDICTION_SET_COMPONENT = "quantforge_event_model_predictions"
PREDICTION_SET_SCHEMA_VERSION = "1"
MANIFEST_FILE = "manifest.json"
PREDICTIONS_FILE = "predictions.jsonl"
# Absolute probability tolerance for regenerated predictions (cross-platform).
PROBABILITY_TOLERANCE = 1e-12
SCORED_ROLES = (
    PartitionRole.SELECTION,
    PartitionRole.WALK_FORWARD_TEST,
    PartitionRole.FINAL_HOLDOUT,
)
_PAYLOAD_FIELDS = frozenset(
    {
        "component",
        "schema_version",
        "prediction_set_id",
        "bindings",
        "authorization",
        "evaluation",
        "records",
        "summary",
        "metrics",
        "baseline",
        "numerical_contract",
        "interpretation",
        "files",
    }
)


class ScoreStatus(StrEnum):
    """Whether an observation received a probability."""

    SCORED = "scored"
    UNSCORED_NULL_FEATURE = "unscored_null_feature"


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One observation's prediction; ``target`` only where evaluation is permitted."""

    row_index: int
    source_observation_id: str
    decision_timestamp: datetime
    partition_role: PartitionRole
    fold_id: str | None
    score_status: ScoreStatus
    probability: float | None
    predicted_class: bool | None
    imputed_features: int
    target: bool | None
    target_status: str

    @property
    def membership_key(self) -> tuple[int, str, datetime, PartitionRole, str | None]:
        return (
            self.row_index,
            self.source_observation_id,
            self.decision_timestamp,
            self.partition_role,
            self.fold_id,
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "row_index": self.row_index,
            "source_observation_id": self.source_observation_id,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "partition_role": self.partition_role.value,
            "fold_id": self.fold_id,
            "score_status": self.score_status.value,
            "probability": self.probability,
            "predicted_class": self.predicted_class,
            "imputed_features": self.imputed_features,
            "target": self.target,
            "target_status": self.target_status,
        }

    @classmethod
    def from_primitive(cls, value: Primitive) -> "PredictionRecord":
        if not isinstance(value, dict):
            raise PredictionIntegrityError("prediction record must be a mapping")
        try:
            probability = value["probability"]
            record = cls(
                row_index=cast(int, value["row_index"]),
                source_observation_id=cast(str, value["source_observation_id"]),
                decision_timestamp=parse_utc_instant(
                    value["decision_timestamp"], "prediction timestamp"
                ),
                partition_role=PartitionRole(cast(str, value["partition_role"])),
                fold_id=cast(str | None, value["fold_id"]),
                score_status=ScoreStatus(cast(str, value["score_status"])),
                probability=None
                if probability is None
                else finite(probability, "probability"),
                predicted_class=cast(bool | None, value["predicted_class"]),
                imputed_features=cast(int, value["imputed_features"]),
                target=cast(bool | None, value["target"]),
                target_status=cast(str, value["target_status"]),
            )
        except (KeyError, ValueError) as error:
            if isinstance(error, ModelArtifactIntegrityError):
                raise
            raise PredictionIntegrityError("prediction record is malformed") from error
        scored = record.score_status is ScoreStatus.SCORED
        # Exact JSON types: equality alone accepts 1/0 for true/false.
        exact = (
            ("row_index", (int,)),
            ("imputed_features", (int,)),
            ("source_observation_id", (str,)),
            ("fold_id", (str, type(None))),
            ("predicted_class", (bool, type(None))),
            ("target", (bool, type(None))),
        )
        if (
            record.to_primitive() != value
            or any(type(value[name]) not in kinds for name, kinds in exact)
            or record.row_index < 0
            or record.imputed_features < 0
            or (scored and not 0.0 <= cast(float, record.probability) <= 1.0)
            or (not scored and record.probability is not None)
            or (not scored and record.predicted_class is not None)
            or record.target_status not in TARGET_STATUSES
            or (record.target is None) is (record.target_status == "available")
        ):
            raise PredictionIntegrityError("prediction record is inconsistent")
        return record


@dataclass(frozen=True, slots=True)
class PredictionSet:
    """One model's predictions for one partition of one dataset."""

    bindings: PrimitiveMappingSnapshot
    authorization: PrimitiveMappingSnapshot | None
    threshold: float | None
    calibration_bins: int
    minimum_evaluation_observations: int
    baseline_probability: float
    records: tuple[PredictionRecord, ...]

    @property
    def role(self) -> PartitionRole:
        return PartitionRole(cast(str, self.bindings.to_primitive()["partition_role"]))

    @property
    def model_id(self) -> str:
        return cast(str, self.bindings.to_primitive()["model_id"])

    @property
    def dataset_id(self) -> str:
        dataset = cast(PrimitiveMapping, self.bindings.to_primitive()["dataset"])
        return cast(str, dataset["dataset_id"])

    def _labeled(self) -> tuple[list[float], list[bool]]:
        pairs = [
            (cast(float, record.probability), record.target)
            for record in self.records
            if record.score_status is ScoreStatus.SCORED and record.target is not None
        ]
        return [p for p, _ in pairs], [y for _, y in pairs]

    def summary(self) -> PrimitiveMapping:
        scored = [r for r in self.records if r.score_status is ScoreStatus.SCORED]
        return {
            "rows": len(self.records),
            "scored": len(scored),
            "unscored_by_reason": dict(
                sorted(
                    Counter(
                        r.score_status.value
                        for r in self.records
                        if r.score_status is not ScoreStatus.SCORED
                    ).items()
                )
            ),
            "labeled_scored": sum(r.target is not None for r in scored),
            "unlabeled_scored_by_status": dict(
                sorted(
                    Counter(r.target_status for r in scored if r.target is None).items()
                )
            ),
            "imputed_feature_values": sum(r.imputed_features for r in self.records),
        }

    def metrics(self) -> PrimitiveMapping:
        probabilities, labels = self._labeled()
        return describe_probabilities(
            probabilities,
            labels,
            threshold=self.threshold,
            bins=self.calibration_bins,
            minimum=self.minimum_evaluation_observations,
        )

    def baseline_metrics(self) -> PrimitiveMapping:
        """The constant training prevalence on exactly the same observations."""
        _, labels = self._labeled()
        return describe_probabilities(
            [self.baseline_probability] * len(labels),
            labels,
            threshold=self.threshold,
            bins=self.calibration_bins,
            minimum=self.minimum_evaluation_observations,
        )

    def records_bytes(self) -> bytes:
        return b"".join(canonical(r.to_primitive()) + b"\n" for r in self.records)

    def _scientific(self) -> PrimitiveMapping:
        return {
            "component": PREDICTION_SET_COMPONENT,
            "schema_version": PREDICTION_SET_SCHEMA_VERSION,
            "bindings": self.bindings.to_primitive(),
            "authorization": None
            if self.authorization is None
            else self.authorization.to_primitive(),
            "evaluation": {
                "class_threshold": self.threshold,
                "calibration_bins": self.calibration_bins,
                "minimum_evaluation_observations": self.minimum_evaluation_observations,
                "target_policy": "labels_recorded_where_evaluation_is_permitted",
                "metric_rows": "scored_rows_with_available_labels",
            },
            "records": {
                "count": len(self.records),
                "ordering": "qf67_dataset_row_index",
                "sha256": hashlib.sha256(self.records_bytes()).hexdigest(),
            },
            "summary": self.summary(),
            "metrics": self.metrics(),
            "baseline": {
                "kind": "constant_training_prevalence",
                "probability": self.baseline_probability,
                "metrics": self.baseline_metrics(),
                "interpretation": "reference only; models need not beat it",
            },
            "numerical_contract": {
                "probability": "binary64_pure_python_from_persisted_state",
                "storage": "json_shortest_round_trip_binary64",
                "reproduction_absolute_tolerance": PROBABILITY_TOLERANCE,
            },
            "interpretation": (
                "Descriptive classification evidence conditional on the dataset "
                "population; a raw-return target without costs. Not evidence of "
                "profitable trading."
            ),
        }

    @property
    def prediction_set_id(self) -> str:
        return configuration_identity(self._scientific())

    def manifest(self) -> PrimitiveMapping:
        scientific = self._scientific()
        return {**scientific, "prediction_set_id": configuration_identity(scientific)}


def score_partition(
    state: EventModelState,
    dataset: EventDataset,
    role: PartitionRole,
    *,
    authorization: PrimitiveMapping | None = None,
) -> PredictionSet:
    """Score every row of one partition with the given fitted state.

    Selection and test rows require the model's exact training dataset. The
    final holdout accepts a compatible dataset (identical rows outside the
    holdout) and requires ``authorization`` from the consumed-ledger check.
    Nothing is refitted and nothing is selected from these rows.
    """
    if role not in SCORED_ROLES:
        raise ModelBindingError("training rows are never scored as predictions")
    if (role is PartitionRole.FINAL_HOLDOUT) is (authorization is None):
        raise ModelBindingError("holdout scoring requires explicit authorization")
    bound, chronology = state.bound, state.chronology
    binding = DatasetBinding.capture(dataset)
    if role is PartitionRole.FINAL_HOLDOUT:
        bound.dataset.require_compatible(binding)
    elif binding.dataset_id != bound.dataset.dataset_id:
        raise ModelBindingError("dataset differs from the model's training dataset")
    require_fold_membership(dataset, chronology, binding)
    fitting = {identity: stamp for identity, stamp in state.membership.fitting}
    for index in dataset.rows_for(
        PartitionRole.DEVELOPMENT, fold_id=chronology.fold_id
    ):
        row = dataset.rows[index]
        if row.source_observation_id in fitting and (
            fitting.pop(row.source_observation_id) != row.decision_timestamp
            or row.label.value is None
        ):
            raise ModelBindingError("fitting observation differs in the dataset")
    if fitting:
        raise ModelBindingError("fitting observations are missing from the dataset")
    numeric = numeric_rows(dataset)
    threshold = bound.configuration.threshold
    indices = (
        dataset.rows_for(role)
        if role is PartitionRole.FINAL_HOLDOUT
        else dataset.rows_for(role, fold_id=chronology.fold_id)
    )
    records: list[PredictionRecord] = []
    for index in indices:
        row = dataset.rows[index]
        transformed = state.preprocessing.transform(
            numeric[index], f"{role.value} row {row.source_observation_id}"
        )
        probability = (
            None
            if transformed is None
            else state.estimator.probability(
                transformed.values, f"{role.value} row {row.source_observation_id}"
            )
        )
        records.append(
            PredictionRecord(
                row_index=index,
                source_observation_id=row.source_observation_id,
                decision_timestamp=row.decision_timestamp,
                partition_role=row.partition_role,
                fold_id=row.fold_id,
                score_status=ScoreStatus.UNSCORED_NULL_FEATURE
                if probability is None
                else ScoreStatus.SCORED,
                probability=probability,
                predicted_class=None
                if probability is None or threshold is None
                else probability >= threshold,
                imputed_features=0
                if transformed is None
                else transformed.imputed_features,
                target=row.label.value,
                target_status=row.label.status,
            )
        )
    bindings: PrimitiveMapping = {
        "model_id": state.model_id,
        "model_configuration_id": bound.model_configuration_id,
        "fitted_state_id": state.fitted_state_id,
        "model_dataset_id": bound.dataset.dataset_id,
        "dataset": binding.to_primitive(),
        "plan_id": chronology.plan_id,
        "fold_id": chronology.fold_id,
        "fold_index": chronology.fold_index,
        "partition_role": role.value,
        "feature_columns": list(bound.dataset.feature_columns),
    }
    return PredictionSet(
        PrimitiveMappingSnapshot.capture(bindings),
        None
        if authorization is None
        else PrimitiveMappingSnapshot.capture(authorization),
        threshold,
        bound.configuration.calibration_bins,
        bound.configuration.minimum_evaluation_observations,
        state.baseline_probability,
        tuple(records),
    )


def _write(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def safe_output_root(output_root: Path, label: str) -> Path:
    """Refuse exploratory exports and any output inside the holdout ledger."""
    reject_non_authoritative(output_root, label=label)
    root = Path(output_root).resolve()
    if any(part.lower() == WORKSPACE_HOLDOUT_LEDGER.name for part in root.parts):
        raise ModelBindingError(f"{label} cannot be inside the holdout ledger")
    return root


def publish_directory(
    root: Path, name: str, files: dict[str, bytes], verify: Callable[[Path], None]
) -> Path:
    """Stage, fsync, verify and atomically rename one immutable directory.

    Nothing unverifiable ever appears under ``name``; an existing directory is
    verified and reused, and a conflicting one is an error.
    """
    final = root / name
    if final.exists():
        verify(final)
        return final
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=root))
    try:
        for file_name, content in files.items():
            _write(staging / file_name, content)
        _sync_directory(staging)
        verify(staging)
        try:
            staging.rename(final)
        except OSError as error:
            if not final.exists():
                raise ModelArtifactIntegrityError("cannot publish artifact") from error
            verify(final)
        _sync_directory(root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final


def export_prediction_set(predictions: PredictionSet, output_root: Path) -> Path:
    """Publish one immutable prediction directory; return its path."""
    if type(cast(object, predictions)) is not PredictionSet:
        raise PredictionIntegrityError("only scored prediction sets can be exported")
    root = safe_output_root(output_root, "prediction output")
    records = predictions.records_bytes()
    payload: PrimitiveMapping = {
        **predictions.manifest(),
        "files": {
            PREDICTIONS_FILE: {
                "sha256": hashlib.sha256(records).hexdigest(),
                "bytes": len(records),
            }
        },
    }
    manifest = (
        canonical({"payload": payload, "fingerprint": configuration_identity(payload)})
        + b"\n"
    )
    identity = predictions.prediction_set_id

    def verify(path: Path) -> None:
        if _read(path, identity) != predictions:
            raise PredictionIntegrityError("prediction directory differs")

    return publish_directory(
        root, identity, {PREDICTIONS_FILE: records, MANIFEST_FILE: manifest}, verify
    )


def _envelope(path: Path) -> PrimitiveMapping:
    try:
        envelope = decode((path / MANIFEST_FILE).read_bytes(), canonical_line=True)
    except (OSError, InvalidPredictionOutputError) as error:
        raise PredictionIntegrityError(
            "prediction manifest is missing or invalid"
        ) from error
    payload = envelope.get("payload")
    if (
        set(envelope) != {"payload", "fingerprint"}
        or not isinstance(payload, dict)
        or envelope["fingerprint"] != configuration_identity(payload)
    ):
        raise PredictionIntegrityError(
            "prediction manifest fingerprint is inconsistent"
        )
    return cast(PrimitiveMapping, payload)


def _read(path: Path, name: str) -> PredictionSet:
    payload = _envelope(path)
    if frozenset(payload) != _PAYLOAD_FIELDS or (
        payload["component"],
        payload["schema_version"],
    ) != (PREDICTION_SET_COMPONENT, PREDICTION_SET_SCHEMA_VERSION):
        raise PredictionIntegrityError("unsupported prediction manifest")
    files = payload["files"]
    try:
        content = (path / PREDICTIONS_FILE).read_bytes()
    except OSError as error:
        raise PredictionIntegrityError("predictions file is missing") from error
    # Files are read by exact name (Finder metadata elsewhere is irrelevant).
    if files != {
        PREDICTIONS_FILE: {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    }:
        raise PredictionIntegrityError("prediction files differ from their manifest")
    records: list[PredictionRecord] = []
    for line in content.splitlines(keepends=True):
        try:
            records.append(
                PredictionRecord.from_primitive(decode(line, canonical_line=True))
            )
        except InvalidPredictionOutputError as error:
            raise PredictionIntegrityError(
                "prediction line is not canonical"
            ) from error
    try:
        bindings = cast(PrimitiveMapping, payload["bindings"])
        evaluation = cast(PrimitiveMapping, payload["evaluation"])
        baseline = cast(PrimitiveMapping, payload["baseline"])
        authorization = payload["authorization"]
        threshold = evaluation["class_threshold"]
        predictions = PredictionSet(
            PrimitiveMappingSnapshot.capture(bindings),
            None
            if authorization is None
            else PrimitiveMappingSnapshot.capture(
                cast(PrimitiveMapping, authorization)
            ),
            None if threshold is None else finite(threshold, "class threshold"),
            cast(int, evaluation["calibration_bins"]),
            cast(int, evaluation["minimum_evaluation_observations"]),
            finite(baseline["probability"], "baseline probability"),
            tuple(records),
        )
        role = predictions.role
        fold = bindings["fold_id"]
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ModelArtifactIntegrityError):
            raise
        raise PredictionIntegrityError("prediction bindings are invalid") from error
    identities = [record.source_observation_id for record in records]
    if len(set(identities)) != len(identities):
        raise PredictionIntegrityError("duplicate prediction observations")
    if any(left.row_index >= right.row_index for left, right in pairwise(records)):
        raise PredictionIntegrityError("predictions are not in dataset row order")
    expected_fold = None if role is PartitionRole.FINAL_HOLDOUT else fold
    if role not in SCORED_ROLES or any(
        record.partition_role is not role or record.fold_id != expected_fold
        for record in records
    ):
        raise PredictionIntegrityError("prediction roles differ from their set")
    if (role is PartitionRole.FINAL_HOLDOUT) is (predictions.authorization is None):
        raise PredictionIntegrityError("holdout predictions lack authorization")
    threshold = predictions.threshold
    if any(
        record.predicted_class
        != (
            None
            if threshold is None or record.probability is None
            else record.probability >= threshold
        )
        for record in records
    ):
        raise PredictionIntegrityError("predicted classes differ from the threshold")
    manifest = {key: value for key, value in payload.items() if key != "files"}
    if manifest != predictions.manifest() or name != predictions.prediction_set_id:
        raise PredictionIntegrityError(
            "prediction manifest differs from its records or identity"
        )
    return predictions


def read_prediction_set(path: Path) -> PredictionSet:
    """Validate a prediction directory offline (structure, metrics, identity).

    Membership against a dataset and agreement with a model are checked by
    ``reproduce_prediction_set``.
    """
    path = Path(path)
    return _read(path, path.name)


def compare_prediction_sets(stored: PredictionSet, regenerated: PredictionSet) -> float:
    """Exact membership and bindings; probabilities within the tolerance.

    Returns the maximum absolute probability difference.
    """
    if stored.bindings != regenerated.bindings:
        raise PredictionIntegrityError(
            "prediction set is bound to another model, dataset, fold or role"
        )
    if stored.authorization != regenerated.authorization:
        raise PredictionIntegrityError("prediction authorization differs")
    evaluation = (
        "threshold",
        "calibration_bins",
        "minimum_evaluation_observations",
        "baseline_probability",
    )
    if any(getattr(stored, name) != getattr(regenerated, name) for name in evaluation):
        raise PredictionIntegrityError("prediction evaluation settings differ")
    stored_keys = [record.membership_key for record in stored.records]
    expected_keys = [record.membership_key for record in regenerated.records]
    if stored_keys != expected_keys:
        missing = len(set(expected_keys) - set(stored_keys))
        extra = len(set(stored_keys) - set(expected_keys))
        raise PredictionIntegrityError(
            f"prediction membership differs: {missing} missing, {extra} unexpected"
        )
    difference = 0.0
    for left, right in zip(stored.records, regenerated.records, strict=True):
        if (
            left.score_status,
            left.predicted_class,
            left.imputed_features,
            left.target,
            left.target_status,
        ) != (
            right.score_status,
            right.predicted_class,
            right.imputed_features,
            right.target,
            right.target_status,
        ):
            raise PredictionIntegrityError(
                f"prediction for {left.source_observation_id} differs"
            )
        if left.probability is not None and right.probability is not None:
            difference = max(difference, abs(left.probability - right.probability))
    if difference > PROBABILITY_TOLERANCE:
        raise PredictionIntegrityError(
            f"probabilities differ by {difference!r} > {PROBABILITY_TOLERANCE!r}"
        )
    return difference


__all__ = [
    "MANIFEST_FILE",
    "PREDICTIONS_FILE",
    "PREDICTION_SET_COMPONENT",
    "PROBABILITY_TOLERANCE",
    "PredictionRecord",
    "PredictionSet",
    "ScoreStatus",
    "compare_prediction_sets",
    "export_prediction_set",
    "publish_directory",
    "read_prediction_set",
    "safe_output_root",
    "score_partition",
]
