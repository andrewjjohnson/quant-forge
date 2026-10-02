"""QF-69 descriptive statistics: exact returns, buckets, explicit undefined values."""

from decimal import Decimal

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.ml.study_report import (
    bucket_index,
    render_markdown,
    return_summary,
    score_distribution,
)


def test_return_summary_is_exact_and_undefined_when_empty() -> None:
    empty = return_summary([])
    assert empty["count"] == 0
    assert empty["mean"] == {"status": "undefined", "reason": "no_available_returns"}
    assert empty["positive_fraction"] == {
        "status": "undefined",
        "reason": "no_available_returns",
    }
    values = [Decimal("0.001"), Decimal(0), Decimal("-0.003"), Decimal("0.004")]
    summary = return_summary(values)
    assert (summary["positive"], summary["zero"], summary["negative"]) == (2, 1, 1)
    assert summary["mean"] == {"status": "defined", "value": "0.0005"}
    assert summary["median"] == {"status": "defined", "value": "0.0005"}
    assert summary["minimum"] == {"status": "defined", "value": "-0.003"}
    assert summary["maximum"] == {"status": "defined", "value": "0.004"}
    # Zero is not positive (matches the strictly-greater target).
    assert summary["positive_fraction"] == {"status": "defined", "value": "0.5"}
    odd = return_summary([Decimal("0.3"), Decimal("0.1"), Decimal("0.2")])
    assert odd["median"] == {"status": "defined", "value": "0.2"}


def test_score_distribution_uses_nearest_rank_quantiles() -> None:
    assert score_distribution([])["median"] == {
        "status": "undefined",
        "reason": "no_scored_rows",
    }
    distribution = score_distribution([0.4, 0.1, 0.3, 0.2])
    assert distribution["minimum"] == {"status": "defined", "value": 0.1}
    assert distribution["p25"] == {"status": "defined", "value": 0.1}
    assert distribution["median"] == {"status": "defined", "value": 0.2}
    assert distribution["p75"] == {"status": "defined", "value": 0.3}
    assert distribution["maximum"] == {"status": "defined", "value": 0.4}


@pytest.mark.parametrize(
    ("probability", "bucket"),
    [(0.0, 0), (0.19999999, 0), (0.2, 1), (0.5, 2), (0.8, 4), (1.0, 4)],
)
def test_bucket_index_is_equal_width_and_closed_at_one(
    probability: float, bucket: int
) -> None:
    assert bucket_index(probability, 5) == bucket


@pytest.mark.parametrize(("probability", "buckets"), [(-0.1, 5), (1.1, 5), (0.5, 0)])
def test_bucket_index_rejects_invalid_inputs(probability: float, buckets: int) -> None:
    with pytest.raises(ValueError, match="probability buckets"):
        bucket_index(probability, buckets)


def test_markdown_renders_undefined_metrics_and_gate() -> None:
    undefined: PrimitiveMapping = {
        "status": "undefined",
        "reason": "single_class_labels",
    }
    report: PrimitiveMapping = {
        "specification_id": "s" * 64,
        "specification": {"name": "fixture", "version": "1", "design": {}},
        "artifacts": {"dataset_id": "d" * 64},
        "events": {"sources": [], "excluded_sources": []},
        "training": {"status": "fitted", "detail": "ok", "membership": {}},
        "out_of_sample": {
            "coverage": {"partition_rows": 1, "metric_rows": 1},
            "model_metrics": {"roc_auc": undefined},
            "baseline": {"probability": 0.5, "metrics": {"roc_auc": undefined}},
            "score_distribution": {"count": 1},
            "score_buckets": [],
        },
        "limitations": ["fixture limitation"],
    }
    gate: PrimitiveMapping = {
        "outcome": "PRE_HOLDOUT_COMPLETE",
        "holdout": {"state": "reserved_unconsumed"},
        "items": {"holdout_state": {"status": "passed"}},
    }
    text = render_markdown(report, gate)
    assert "undefined (single_class_labels)" in text
    assert "**reserved_unconsumed**" in text
    assert "Not evaluated" in text  # The selection section is absent.
    assert "- fixture limitation" in text
