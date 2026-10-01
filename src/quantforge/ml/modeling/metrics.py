"""Descriptive probability metrics with explicit undefined cases (QF-68).

Every metric is computed in pure Python over labeled, scored observations of
one partition. Unavailable labels never enter (they are not negatives).
Numerical conventions:

- **Log loss**: mean of ``-(y ln p + (1 - y) ln(1 - p))`` with ``p`` clipped
  to ``[1e-15, 1 - 1e-15]``; natural logarithm; ``math.fsum`` sums.
- **Brier score**: mean of ``(p - y)^2``; no clipping.
- **ROC AUC**: Mann-Whitney statistic with tied scores counted as one half,
  computed from exact integer doubled ranks. Undefined unless both classes
  are present.
- **Calibration**: equal-width bins on ``[0, 1]`` (bin ``min(floor(p * B),
  B - 1)``) with count, mean probability and observed positive rate (null for
  an empty bin), plus mean probability minus observed prevalence.
- **Precision/recall** only with a frozen threshold (``p >= threshold`` is the
  positive class). Precision is undefined with no predicted positives and
  recall with no actual positives.

Fewer labeled observations than the configured minimum makes every metric
undefined (``insufficient_labeled_observations``; zero is
``no_labeled_observations``). Undefined metrics stay explicitly undefined and
are never replaced by a number.
"""

import math
from collections.abc import Sequence

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.ml.modeling.configuration import LOG_LOSS_EPSILON


def _defined(value: float) -> PrimitiveMapping:
    return {"status": "defined", "value": value}


def _undefined(reason: str) -> PrimitiveMapping:
    return {"status": "undefined", "reason": reason}


def log_loss(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    terms: list[float] = []
    for probability, label in zip(probabilities, labels, strict=True):
        clipped = min(max(probability, LOG_LOSS_EPSILON), 1.0 - LOG_LOSS_EPSILON)
        terms.append(-math.log(clipped if label else 1.0 - clipped))
    return math.fsum(terms) / len(terms)


def brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    return math.fsum(
        (probability - float(label)) ** 2
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(probabilities)


def roc_auc(probabilities: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Mann-Whitney AUC with ties as one half; ``None`` without both classes."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(probabilities, labels, strict=True), key=lambda item: item[0])
    doubled_rank_sum = 0  # sum over positives of 2 * average rank (1-based)
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[start][0]:
            end += 1
        doubled_rank = (start + 1) + (end + 1)  # 2 * average rank of the tie group
        doubled_rank_sum += doubled_rank * sum(
            label for _, label in ordered[start : end + 1]
        )
        start = end + 1
    numerator = doubled_rank_sum - positives * (positives + 1)
    return numerator / (2 * positives * negatives)


def calibration(
    probabilities: Sequence[float], labels: Sequence[bool], bins: int
) -> PrimitiveMapping:
    counts = [0] * bins
    totals: list[list[float]] = [[] for _ in range(bins)]
    hits = [0] * bins
    for probability, label in zip(probabilities, labels, strict=True):
        index = min(math.floor(probability * bins), bins - 1)
        counts[index] += 1
        totals[index].append(probability)
        hits[index] += int(label)
    rows: list[Primitive] = []
    for index in range(bins):
        rows.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": counts[index],
                "mean_probability": math.fsum(totals[index]) / counts[index]
                if counts[index]
                else None,
                "observed_positive_rate": hits[index] / counts[index]
                if counts[index]
                else None,
            }
        )
    mean = math.fsum(probabilities) / len(probabilities)
    prevalence = sum(labels) / len(labels)
    return {
        "status": "defined",
        "bins": rows,
        "mean_probability_minus_observed_prevalence": mean - prevalence,
    }


def classification(
    probabilities: Sequence[float], labels: Sequence[bool], threshold: float
) -> PrimitiveMapping:
    predicted = [probability >= threshold for probability in probabilities]
    true_positive = sum(p and y for p, y in zip(predicted, labels, strict=True))
    false_positive = sum(p and not y for p, y in zip(predicted, labels, strict=True))
    false_negative = sum(not p and y for p, y in zip(predicted, labels, strict=True))
    true_negative = len(labels) - true_positive - false_positive - false_negative
    return {
        "status": "defined",
        "threshold": threshold,
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "precision": _defined(true_positive / (true_positive + false_positive))
        if true_positive + false_positive
        else _undefined("no_predicted_positives"),
        "recall": _defined(true_positive / (true_positive + false_negative))
        if true_positive + false_negative
        else _undefined("no_actual_positives"),
    }


def describe_probabilities(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    *,
    threshold: float | None,
    bins: int,
    minimum: int,
) -> PrimitiveMapping:
    """All descriptive metrics of one labeled, scored partition."""
    count = len(probabilities)
    positives = sum(labels)
    base: PrimitiveMapping = {
        "labeled_observations": count,
        "positives": positives,
        "negatives": count - positives,
    }
    names = (
        "observed_prevalence",
        "mean_probability",
        "log_loss",
        "brier_score",
        "roc_auc",
        "calibration",
    )
    if count < minimum or count == 0:
        reason = (
            "no_labeled_observations"
            if count == 0
            else "insufficient_labeled_observations"
        )
        return {
            **base,
            **{name: _undefined(reason) for name in names},
            "classification": _undefined(reason)
            if threshold is not None
            else _undefined("no_frozen_threshold"),
        }
    auc = roc_auc(probabilities, labels)
    return {
        **base,
        "observed_prevalence": _defined(positives / count),
        "mean_probability": _defined(math.fsum(probabilities) / count),
        "log_loss": _defined(log_loss(probabilities, labels)),
        "brier_score": _defined(brier_score(probabilities, labels)),
        "roc_auc": _undefined("single_class_labels") if auc is None else _defined(auc),
        "calibration": calibration(probabilities, labels, bins),
        "classification": _undefined("no_frozen_threshold")
        if threshold is None
        else classification(probabilities, labels, threshold),
    }


__all__ = [
    "brier_score",
    "calibration",
    "classification",
    "describe_probabilities",
    "log_loss",
    "roc_auc",
]
