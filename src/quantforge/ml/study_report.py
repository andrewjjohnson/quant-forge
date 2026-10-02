"""Descriptive QF-69 study statistics over persisted QF-67/QF-68 artifacts.

Everything here is computed offline from a validated event dataset and
validated prediction sets. Nothing is fitted, selected or tuned, and no
statistic feeds back into a study choice. Conventions (fixed before fitting):

- **Raw returns** are the persisted exact QF-49 30-minute ``raw_return``
  decimals (``target_source_value``); only available labels contribute.
  Arithmetic uses a 34-digit decimal context. ``positive`` means strictly
  greater than zero (zero is not positive), matching the target.
- **Score buckets** are ``B`` equal-width bins on ``[0, 1]``; a probability
  ``p`` falls in bin ``min(floor(p * B), B - 1)`` (QF-68's calibration
  convention), so the last bin is closed at 1.
- **Score quantiles** use the nearest-rank rule on sorted probabilities
  (``sorted[ceil(q * n) - 1]``), with no interpolation.
- Undefined statistics are explicit ``{"status": "undefined", "reason": ...}``
  values, never numbers.

Intraday labels can overlap, so observations are not independent; no
standard error, confidence interval or significance test is reported.
"""

import math
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal, localcontext

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.ml.dataset import EventDataset
from quantforge.ml.modeling.predictions import PredictionSet, ScoreStatus
from quantforge.validation import PartitionRole

DECIMAL_PRECISION = 34
QUANTILES = (("p25", 0.25), ("median", 0.5), ("p75", 0.75))


def undefined(reason: str) -> PrimitiveMapping:
    return {"status": "undefined", "reason": reason}


def _text(value: Decimal) -> str:
    return format(value.normalize(), "f") if value else "0"


def return_summary(values: Sequence[Decimal]) -> PrimitiveMapping:
    """Count, mean, median, extremes and sign counts of exact raw returns."""
    count = len(values)
    signs: PrimitiveMapping = {
        "count": count,
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
    }
    if not count:
        reason = "no_available_returns"
        return {
            **signs,
            **{
                name: undefined(reason)
                for name in (
                    "mean",
                    "median",
                    "minimum",
                    "maximum",
                    "positive_fraction",
                )
            },
        }
    ordered = sorted(values)
    with localcontext() as arithmetic:
        arithmetic.prec = DECIMAL_PRECISION
        mean = sum(ordered, Decimal(0)) / count
        middle = count // 2
        median = (
            ordered[middle]
            if count % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        fraction = Decimal(cast_int(signs["positive"])) / count
    return {
        **signs,
        "mean": {"status": "defined", "value": _text(mean)},
        "median": {"status": "defined", "value": _text(median)},
        "minimum": {"status": "defined", "value": _text(ordered[0])},
        "maximum": {"status": "defined", "value": _text(ordered[-1])},
        "positive_fraction": {"status": "defined", "value": _text(fraction)},
    }


def cast_int(value: Primitive) -> int:
    if type(value) is not int:
        raise TypeError("expected an integer count")
    return value


def score_distribution(probabilities: Sequence[float]) -> PrimitiveMapping:
    """Nearest-rank quantiles, extremes and mean of scored probabilities."""
    count = len(probabilities)
    names = ("minimum", *(name for name, _ in QUANTILES), "maximum", "mean")
    if not count:
        return {"count": 0, **{name: undefined("no_scored_rows") for name in names}}
    ordered = sorted(probabilities)

    def rank(quantile: float) -> float:
        return ordered[max(math.ceil(quantile * count), 1) - 1]

    return {
        "count": count,
        "minimum": {"status": "defined", "value": ordered[0]},
        **{
            name: {"status": "defined", "value": rank(quantile)}
            for name, quantile in QUANTILES
        },
        "maximum": {"status": "defined", "value": ordered[-1]},
        "mean": {"status": "defined", "value": math.fsum(ordered) / count},
    }


def bucket_index(probability: float, buckets: int) -> int:
    """QF-68 equal-width binning; the last bucket is closed at 1."""
    if buckets < 1 or not 0.0 <= probability <= 1.0:
        raise ValueError("probability buckets need p in [0, 1] and B >= 1")
    return min(math.floor(probability * buckets), buckets - 1)


def _returns(dataset: EventDataset, indices: Sequence[int]) -> list[Decimal]:
    values: list[Decimal] = []
    for index in indices:
        label = dataset.rows[index].label
        if label.value is not None:
            if label.source_value is None:
                raise ValueError("available label without its persisted return")
            values.append(Decimal(label.source_value))
    return values


def label_counts(dataset: EventDataset, indices: Sequence[int]) -> PrimitiveMapping:
    """Positive/negative/unavailable labels and prevalence of some rows."""
    rows = [dataset.rows[index] for index in indices]
    positives = sum(row.label.value is True for row in rows)
    negatives = sum(row.label.value is False for row in rows)
    labeled = positives + negatives
    return {
        "rows": len(rows),
        "positive": positives,
        "negative": negatives,
        "unavailable_by_status": dict(
            sorted(
                Counter(r.label.status for r in rows if r.label.value is None).items()
            )
        ),
        "prevalence": {"status": "defined", "value": positives / labeled}
        if labeled
        else undefined("no_available_labels"),
    }


def base_strategy(
    dataset: EventDataset, role: PartitionRole, fold_id: str
) -> PrimitiveMapping:
    """Unfiltered outcomes of every trigger in one partition."""
    indices = dataset.rows_for(role, fold_id=fold_id)
    return {
        "partition_role": role.value,
        "fold_id": fold_id,
        "labels": label_counts(dataset, indices),
        "raw_return_30m": return_summary(_returns(dataset, indices)),
        "scope": "all triggers of the partition, without any model filter",
    }


def bucket_table(
    predictions: PredictionSet, dataset: EventDataset, buckets: int
) -> list[Primitive]:
    """Counts and raw-return summaries inside predeclared score buckets."""
    members: list[list[int]] = [[] for _ in range(buckets)]
    probabilities: list[list[float]] = [[] for _ in range(buckets)]
    for record in predictions.records:
        if record.score_status is not ScoreStatus.SCORED:
            continue
        probability = record.probability
        assert probability is not None
        index = bucket_index(probability, buckets)
        members[index].append(record.row_index)
        probabilities[index].append(probability)
    table: list[Primitive] = []
    for index in range(buckets):
        labels = label_counts(dataset, members[index])
        table.append(
            {
                "bucket": index,
                "lower": index / buckets,
                "upper": (index + 1) / buckets,
                "closed": "[lower, upper]"
                if index == buckets - 1
                else "[lower, upper)",
                "scored": len(members[index]),
                "mean_probability": {
                    "status": "defined",
                    "value": math.fsum(probabilities[index])
                    / len(probabilities[index]),
                }
                if probabilities[index]
                else undefined("empty_bucket"),
                "labels": labels,
                "raw_return_30m": return_summary(_returns(dataset, members[index])),
            }
        )
    return table


def prediction_partition(
    predictions: PredictionSet, dataset: EventDataset, buckets: int
) -> PrimitiveMapping:
    """Model versus baseline on the same rows, scores and bucketed outcomes."""
    scored = [
        record
        for record in predictions.records
        if record.score_status is ScoreStatus.SCORED
    ]
    metric_rows = [record for record in scored if record.target is not None]
    return {
        "prediction_set_id": predictions.prediction_set_id,
        "partition_role": predictions.role.value,
        "summary": predictions.summary(),
        "coverage": {
            "partition_rows": len(predictions.records),
            "metric_rows": len(metric_rows),
            "excluded_unscored": len(predictions.records) - len(scored),
            "excluded_unlabeled": len(scored) - len(metric_rows),
            "rule": "metrics use scored rows with available labels; the model and "
            "the baseline are evaluated on exactly these rows",
        },
        "model_metrics": predictions.metrics(),
        "baseline": {
            "kind": "constant_training_prevalence",
            "probability": predictions.baseline_probability,
            "metrics": predictions.baseline_metrics(),
        },
        "score_distribution": score_distribution(
            [record.probability for record in scored if record.probability is not None]
        ),
        "score_buckets": bucket_table(predictions, dataset, buckets),
    }


def source_coverage(dataset: EventDataset) -> list[Primitive]:
    """Scheduled decisions, statuses, signals and rows of every source window."""
    plan = dataset.scientific.to_primitive()["partition_plan"]
    assert isinstance(plan, dict)
    sources = plan["sources"]
    assert isinstance(sources, list)
    keys = (
        "role",
        "fold_id",
        "scheduled_decisions",
        "decision_statuses",
        "generated_signals",
        "excluded_by_disposition",
        "rows",
        "first_scheduled_decision",
        "last_scheduled_decision",
    )
    return [
        {key: source.get(key) for key in keys}
        for source in sources
        if isinstance(source, dict)
    ]


def _shown(value: Primitive, digits: int = 6) -> str:
    """Readable text for a metric value; exact values stay in the JSON report."""
    if isinstance(value, dict):
        if value.get("status") == "undefined":
            return f"undefined ({value.get('reason')})"
        if "value" in value:
            return _shown(value["value"], digits)
    if isinstance(value, bool) or value is None:
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    if isinstance(value, str):
        try:
            number = Decimal(value)
        except ArithmeticError:
            return value
        return f"{number:.{digits}g}" if number.is_finite() else value
    return str(value)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(" --- " for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _mapping(value: Primitive) -> PrimitiveMapping:
    return value if isinstance(value, dict) else {}


def _items(value: Primitive) -> list[Primitive]:
    return value if isinstance(value, list) else []


def _returns_row(label: str, summary: PrimitiveMapping) -> list[str]:
    labels = _mapping(summary.get("labels"))
    returns = _mapping(summary.get("raw_return_30m"))
    unavailable = _mapping(labels.get("unavailable_by_status"))
    return [
        label,
        str(labels.get("rows")),
        str(labels.get("positive")),
        str(labels.get("negative")),
        str(sum(cast_int(v) for v in unavailable.values()))
        + (
            f" ({', '.join(f'{k}: {v}' for k, v in unavailable.items())})"
            if unavailable
            else ""
        ),
        _shown(labels.get("prevalence"), 4),
        _shown(returns.get("mean")),
        _shown(returns.get("median")),
        _shown(returns.get("minimum")),
        _shown(returns.get("maximum")),
    ]


def render_markdown(report: PrimitiveMapping, gate: PrimitiveMapping | None) -> str:
    """A human-readable view of a study report; the JSON report is authoritative."""
    specification = _mapping(report.get("specification"))
    design = _mapping(specification.get("design"))
    windows = _mapping(design.get("windows"))
    model = _mapping(specification.get("model_configuration"))
    lines: list[str] = [
        f"# Conditional ML smoke study: {specification.get('name')} "
        f"v{specification.get('version')}",
        "",
        "> Descriptive classification evidence conditional on one frozen strategy "
        "population and a 30-minute raw-return target without costs. Not evidence "
        "of an edge, robustness or profitability. Overlapping intraday labels are "
        "not independent; no significance is claimed.",
        "",
        "## Frozen specification",
        "",
        f"- Specification ID: `{report.get('specification_id')}`",
        f"- Plan: `{specification.get('plan_id')}`; final holdout "
        f"`{specification.get('final_holdout_id')}`",
        f"- QF-39 study: `{specification.get('qf39_study_id')}`; fold "
        f"`{specification.get('fold_id')}`",
        f"- Feature schema `{specification.get('feature_schema_id')}`; target "
        f"`{specification.get('target_id')}`",
        f"- Model configuration `{specification.get('model_configuration_id')}` "
        f"({model.get('name')} v{model.get('version')})",
        *(
            f"- {key.replace('_', ' ').capitalize()}: {value}"
            for key, value in _mapping(specification.get("minimums")).items()
        ),
        "",
    ]
    rows = [
        [name, " .. ".join(str(item) for item in _items(windows.get(name)))]
        for name in (
            "warm_up_context",
            "development",
            "selection",
            "walk_forward_test",
            "final_holdout",
        )
        if name in windows
    ]
    if rows:
        lines += [*_table(("Window", "Inclusive sessions"), rows), ""]
    artifacts = _mapping(report.get("artifacts"))
    lines += [
        "## Artifacts",
        "",
        *_table(
            ("Artifact", "Identity"),
            [[key, f"`{value}`"] for key, value in artifacts.items()],
        ),
    ]
    events = _mapping(report.get("events"))
    lines += [
        "",
        "## Event counts",
        "",
        *_table(
            ("Role", "Scheduled", "Decision statuses", "Signals", "Rows"),
            [
                [
                    str(item.get("role")),
                    str(item.get("scheduled_decisions")),
                    ", ".join(
                        f"{k}: {v}"
                        for k, v in _mapping(item.get("decision_statuses")).items()
                    ),
                    str(item.get("generated_signals")),
                    str(item.get("rows")),
                ]
                for item in map(_mapping, _items(events.get("sources")))
            ],
        ),
    ]
    excluded = _items(events.get("excluded_sources"))
    lines += ["", f"Excluded sources: {excluded if excluded else 'none'}", ""]
    lines += [
        "## Base strategy (unfiltered)",
        "",
        *_table(
            (
                "Partition",
                "Rows",
                "Positive",
                "Negative",
                "Unavailable",
                "Prevalence",
                "Mean 30m",
                "Median 30m",
                "Min",
                "Max",
            ),
            [
                _returns_row(name, _mapping(events.get(name)))
                for name in ("development", "selection", "walk_forward_test")
                if name in events
            ],
        ),
        "",
    ]
    training = _mapping(report.get("training"))
    membership = _mapping(training.get("membership"))
    lines += [
        "## Training",
        "",
        f"- Status: **{training.get('status')}** ({training.get('detail')})",
        f"- Development rows {membership.get('training_rows')}, fitting "
        f"{membership.get('fitting_observations')} "
        f"({membership.get('fitting_positives')} positive / "
        f"{membership.get('fitting_negatives')} negative); exclusions "
        f"{membership.get('excluded_by_reason')}",
        "",
    ]
    estimator = _mapping(_mapping(report.get("model")).get("estimator"))
    if estimator:
        lines += [
            *_table(
                ("Coefficient (standardized)", "Value"),
                [["intercept", _shown(estimator.get("intercept"))]]
                + [
                    [str(item.get("feature")), _shown(item.get("coefficient"))]
                    for item in map(_mapping, _items(estimator.get("coefficients")))
                ],
            ),
            "",
        ]
    for name, title in (
        ("selection", "Selection (in-sample evidence; nothing selected)"),
        ("out_of_sample", "Walk-forward test (OOS, frozen model)"),
    ):
        section = _mapping(report.get(name))
        lines += [f"## {title}", ""]
        if "model_metrics" not in section:
            lines += [f"Not evaluated: {section.get('reason')}", ""]
            continue
        coverage = _mapping(section.get("coverage"))
        metrics = _mapping(section.get("model_metrics"))
        baseline = _mapping(_mapping(section.get("baseline")).get("metrics"))
        lines += [
            f"Rows {coverage.get('partition_rows')}; metric rows "
            f"{coverage.get('metric_rows')}; unscored "
            f"{coverage.get('excluded_unscored')}; unlabeled "
            f"{coverage.get('excluded_unlabeled')}. Baseline probability "
            f"{_shown(_mapping(section.get('baseline')).get('probability'), 4)}.",
            "",
            *_table(
                ("Metric", "Model", "Training-prevalence baseline"),
                [
                    [key, _shown(metrics.get(key)), _shown(baseline.get(key))]
                    for key in (
                        "observed_prevalence",
                        "mean_probability",
                        "log_loss",
                        "brier_score",
                        "roc_auc",
                    )
                ],
            ),
            "",
        ]
        distribution = _mapping(section.get("score_distribution"))
        lines += [
            "Score distribution: "
            + ", ".join(
                f"{key} {_shown(distribution.get(key), 4)}"
                for key in ("count", "minimum", "p25", "median", "p75", "maximum")
            ),
            "",
            *_table(
                (
                    "Score bucket",
                    "Scored",
                    "Labeled +/-",
                    "Observed rate",
                    "Mean prob.",
                    "Mean 30m",
                    "Median 30m",
                ),
                [
                    [
                        f"{_shown(item.get('lower'), 2)}"
                        f"-{_shown(item.get('upper'), 2)}",
                        str(item.get("scored")),
                        f"{_mapping(item.get('labels')).get('positive')}/"
                        f"{_mapping(item.get('labels')).get('negative')}",
                        _shown(_mapping(item.get("labels")).get("prevalence"), 4),
                        _shown(item.get("mean_probability"), 4),
                        _shown(_mapping(item.get("raw_return_30m")).get("mean")),
                        _shown(_mapping(item.get("raw_return_30m")).get("median")),
                    ]
                    for item in map(_mapping, _items(section.get("score_buckets")))
                ],
            ),
            "",
        ]
    stages = _mapping(report.get("stages"))
    reproduction = _mapping(stages.get("reproduction"))
    if reproduction:
        lines += [
            "## Reproduction",
            "",
            *_table(
                ("Check", "Result"),
                [
                    [key, _shown(value)]
                    for key, value in _mapping(reproduction.get("checks")).items()
                ],
            ),
            "",
        ]
    if gate is not None:
        lines += [
            "## Pre-holdout gate",
            "",
            f"Outcome: **{gate.get('outcome')}**; holdout "
            f"**{_mapping(gate.get('holdout')).get('state')}**.",
            "",
            *_table(
                ("Item", "Status"),
                [
                    [key, str(_mapping(value).get("status"))]
                    for key, value in _mapping(gate.get("items")).items()
                ],
            ),
            "",
        ]
    limitations = _items(report.get("limitations"))
    if limitations:
        lines += ["## Limitations", "", *(f"- {item}" for item in limitations), ""]
    return "\n".join(lines)


__all__ = [
    "DECIMAL_PRECISION",
    "base_strategy",
    "bucket_index",
    "bucket_table",
    "label_counts",
    "prediction_partition",
    "render_markdown",
    "return_summary",
    "score_distribution",
    "source_coverage",
    "undefined",
]
