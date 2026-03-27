"""Unit tests for RQ2 metrics, paired statistics, and audit helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.evaluation.rq2_metrics import (
    _ordinal_error,
    aggregate_condition_metrics,
    build_paired_results,
    build_question_level_results,
    compute_rq2_paired_statistics,
    majority_baseline,
    normalize_label,
    token_f1,
)
from rag.types import RQ2SampleResult


def _make_result(
    condition="with_esrs_context",
    question_id="Q1",
    company="TestCorp",
    gold="High",
    predicted="High",
    success=True,
):
    return RQ2SampleResult(
        condition=condition,
        question_id=question_id,
        question="Test question",
        esrs_standard="ESRS E1",
        esrs_disclosure_id="E1-1",
        company=company,
        gold_fulfillment_level=gold,
        predicted_fulfillment_level=predicted,
        direct_answer_token_f1=0.5,
        rationale_token_f1=0.4,
        evidence_support_ratio=0.9,
        exact_match=(gold == predicted) if success else False,
        ordinal_error=_ordinal_error(gold, predicted) if success else 2,
        success=success,
    )


def _make_paired():
    rows = []
    for company in ["A", "B", "C"]:
        for condition in ("with_esrs_context", "without_esrs_context"):
            gold = "High"
            pred = "High" if company != "C" else ("Medium" if condition == "with_esrs_context" else "High")
            rows.append(_make_result(condition=condition, question_id="Q1", company=company, gold=gold, predicted=pred))
    return rows


# --- normalize_label ---

def test_normalize_label_canonical():
    assert normalize_label("Low") == "Low"
    assert normalize_label("Medium") == "Medium"
    assert normalize_label("High") == "High"

def test_normalize_label_case():
    assert normalize_label("low") == "Low"
    assert normalize_label("HIGH") == "High"
    assert normalize_label("  medium  ") == "Medium"

def test_normalize_label_invalid():
    assert normalize_label("") is None
    assert normalize_label("None") is None


# --- ordinal_error ---

def test_ordinal_error_same():
    assert _ordinal_error("High", "High") == 0

def test_ordinal_error_adjacent():
    assert _ordinal_error("Low", "Medium") == 1
    assert _ordinal_error("Medium", "High") == 1

def test_ordinal_error_extreme():
    assert _ordinal_error("Low", "High") == 2

def test_ordinal_error_invalid():
    assert _ordinal_error("Low", "") == 2


# --- token_f1 ---

def test_token_f1_identical():
    assert token_f1("hello world", "hello world") == 1.0

def test_token_f1_no_overlap():
    assert token_f1("hello world", "foo bar") == 0.0

def test_token_f1_empty_both():
    assert token_f1("", "") == 1.0

def test_token_f1_partial():
    f1 = token_f1("the quick brown fox", "the quick red fox")
    assert 0.5 < f1 < 1.0


# --- aggregate_condition_metrics ---

def test_aggregate_all_correct():
    rows = [_make_result(gold="High", predicted="High") for _ in range(5)]
    agg = aggregate_condition_metrics(rows)
    assert agg["n_scored"] == 5
    assert agg["exact_accuracy"] == 1.0
    assert agg["ordinal_error_mean"] == 0.0

def test_aggregate_with_failures():
    rows = [
        _make_result(gold="High", predicted="High"),
        _make_result(gold="High", predicted="", success=False),
    ]
    agg = aggregate_condition_metrics(rows)
    assert agg["n_scored"] == 1
    assert agg["n_failures"] == 1

def test_aggregate_per_label():
    rows = [
        _make_result(gold="High", predicted="High"),
        _make_result(gold="Low", predicted="Low"),
    ]
    agg = aggregate_condition_metrics(rows)
    assert "per_label_metrics" in agg
    assert "High" in agg["per_label_metrics"]

def test_aggregate_empty():
    agg = aggregate_condition_metrics([])
    assert agg["n_scored"] == 0
    assert math.isnan(agg["macro_f1"])


# --- majority_baseline ---

def test_majority_baseline():
    rows = [_make_result(gold="High")] * 8 + [_make_result(gold="Low")] * 2
    mb = majority_baseline(rows)
    assert mb["majority_label"] == "High"
    assert mb["exact_accuracy"] == 0.8


# --- build_paired_results ---

def test_paired_correct_count():
    paired = build_paired_results(_make_paired())
    assert len(paired) == 3

def test_paired_single_condition_skipped():
    rows = [_make_result(condition="with_esrs_context")]
    assert len(build_paired_results(rows)) == 0


# --- build_question_level_results ---

def test_question_level():
    qlr = build_question_level_results(_make_paired())
    assert len(qlr) == 1
    assert qlr[0]["question_id"] == "Q1"
    assert "macro_f1_delta" in qlr[0]


# --- compute_rq2_paired_statistics ---

def test_paired_stats_keys():
    rows = _make_paired()
    paired = build_paired_results(rows)
    qlr = build_question_level_results(rows)
    stats = compute_rq2_paired_statistics(rows, paired, qlr)
    assert "cluster_bootstrap" in stats
    assert "question_level_tests" in stats
    assert "exact_match_test" in stats
    assert "macro_f1" in stats["cluster_bootstrap"]

def test_sign_test_p_value_in_range():
    rows = _make_paired()
    paired = build_paired_results(rows)
    qlr = build_question_level_results(rows)
    stats = compute_rq2_paired_statistics(rows, paired, qlr)
    p = stats["question_level_tests"]["macro_f1"]["p_value"]
    assert 0.0 <= p <= 1.0


# --- _cohens_d ---

from rag.evaluation.rq2_metrics import _cohens_d

def test_cohens_d_zero_deltas():
    result = _cohens_d([0.0, 0.0, 0.0])
    assert result["cohens_d"] == 0.0
    assert result["effect_magnitude"] == "negligible"

def test_cohens_d_known_value():
    # mean=1.0, stdev=1.0 → d=1.0 → large
    result = _cohens_d([0.0, 1.0, 2.0])
    assert abs(result["cohens_d"] - 1.0) < 0.01
    assert result["effect_magnitude"] == "large"

def test_cohens_d_single_element():
    result = _cohens_d([5.0])
    assert result["cohens_d"] == 0.0
    assert result["effect_magnitude"] == "negligible"


# --- _post_hoc_power ---

from rag.evaluation.rq2_metrics import _post_hoc_power

def test_post_hoc_power_low_n_small_d():
    result = _post_hoc_power(12, 0.1)
    # With n=12 and small d, power should be very low
    assert result["achieved_power"] < 0.20
    # MDES at 80% should be relatively large
    assert result["mdes_80pct"] > 0.5

def test_post_hoc_power_large_effect():
    result = _post_hoc_power(12, 2.0)
    # With a very large effect, power should be high
    assert result["achieved_power"] > 0.50


# --- _direction_summary ---

from rag.evaluation.rq2_metrics import _direction_summary

def test_direction_summary_mixed():
    qrows = [
        {"question_id": "Q1", "macro_f1_delta": 0.05},
        {"question_id": "Q2", "macro_f1_delta": -0.03},
        {"question_id": "Q3", "macro_f1_delta": 0.0},
    ]
    result = _direction_summary(qrows, metric="macro_f1")
    assert result["n_improved"] == 1
    assert result["n_degraded"] == 1
    assert result["n_unchanged"] == 1

def test_direction_summary_all_improved():
    qrows = [{"question_id": f"Q{i}", "macro_f1_delta": 0.1} for i in range(5)]
    result = _direction_summary(qrows, metric="macro_f1")
    assert result["n_improved"] == 5
    assert result["n_degraded"] == 0


# --- _prediction_bias_diagnosis ---

from rag.evaluation.rq2_metrics import _prediction_bias_diagnosis

def test_prediction_bias_no_shift():
    rows = []
    for company in ["A", "B", "C"]:
        for condition in ("with_esrs_context", "without_esrs_context"):
            rows.append(_make_result(condition=condition, company=company, gold="High", predicted="High"))
    result = _prediction_bias_diagnosis(rows)
    assert result["ordinal_shift"] == 0.0
    assert result["interpretation"] == "no_systematic_bias"

def test_prediction_bias_upward():
    rows = []
    for company in ["A", "B", "C"]:
        rows.append(_make_result(condition="with_esrs_context", company=company, gold="Medium", predicted="High"))
        rows.append(_make_result(condition="without_esrs_context", company=company, gold="Medium", predicted="Medium"))
    result = _prediction_bias_diagnosis(rows)
    assert result["ordinal_shift"] > 0
    assert result["interpretation"] == "context_shifts_predictions_upward"


# --- cluster bootstrap new fields ---

def test_cluster_bootstrap_has_cohens_d():
    rows = _make_paired()
    paired = build_paired_results(rows)
    qlr = build_question_level_results(rows)
    stats = compute_rq2_paired_statistics(rows, paired, qlr)
    cb_f1 = stats["cluster_bootstrap"]["macro_f1"]
    assert "cohens_d" in cb_f1
    assert "effect_magnitude" in cb_f1
    assert "n_resamples" in cb_f1
    assert cb_f1["n_resamples"] == 10_000
    # New top-level keys
    assert "power_analysis" in stats
    assert "prediction_bias" in stats
    assert "direction_summary" in stats

