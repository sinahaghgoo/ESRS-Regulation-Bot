"""Metrics, aggregation, paired statistics, and audit helpers for RQ2."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, stdev

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from rag.config.settings import ESRS_JSON_DIR, RQ2_RESULTS_VERSION
from rag.data.text_utils import normalize_text, sentence_split
from rag.types import (
    ESRSContextResult,
    GeneratedAnswer,
    GenerationSample,
    RQ2SampleResult,
)

_TOKEN_RE = re.compile(r"\w+")
_LABELS = ["Low", "Medium", "High"]
_LABEL_ORD = {"Low": 0, "Medium": 1, "High": 2}


# ---------------------------------------------------------------------------
# Core text helpers
# ---------------------------------------------------------------------------

def normalize_label(value: str) -> str | None:
    """Normalize fulfillment labels to the canonical Low/Medium/High set."""
    mapping = {"low": "Low", "medium": "Medium", "high": "High"}
    return mapping.get(str(value).strip().lower())


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def token_f1(predicted: str, gold: str) -> float:
    """Token-level F1 score with multiplicity-aware overlap."""
    pred_tokens = _tokenize(predicted)
    gold_tokens = _tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = sum(min(pred_counts.get(token, 0), gold_counts.get(token, 0)) for token in gold_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _token_overlap_ratio(gold_text: str, evidence_text: str) -> float:
    gold_tokens = _tokenize(gold_text)
    evidence_tokens = _tokenize(evidence_text)
    if not gold_tokens or not evidence_tokens:
        return 0.0
    evidence_counts: dict[str, int] = {}
    for token in evidence_tokens:
        evidence_counts[token] = evidence_counts.get(token, 0) + 1
    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    matched = sum(min(count, evidence_counts.get(token, 0)) for token, count in gold_counts.items())
    return matched / len(gold_tokens)


def evidence_support_ratio(text: str, evidence_texts: list[str], threshold: float = 0.5) -> float:
    """Fraction of generated sentences supported by retrieved evidence."""
    sentences = [sent.strip() for sent in sentence_split(text) if len(sent.strip()) >= 20]
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        if any(_token_overlap_ratio(sentence, ev) >= threshold for ev in evidence_texts):
            supported += 1
    return supported / len(sentences)


def _ordinal_error(gold_label: str, pred_label: str) -> int:
    """Absolute ordinal distance between two labels."""
    g = _LABEL_ORD.get(gold_label)
    p = _LABEL_ORD.get(pred_label)
    if g is None or p is None:
        return 2  # max penalty for unparseable
    return abs(g - p)


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def build_sample_result(
    *,
    condition: str,
    sample: GenerationSample,
    answer: GeneratedAnswer,
    evidence_texts: list[str],
    retrieval_pages: list[str],
) -> RQ2SampleResult:
    """Evaluate one generated answer against its gold sample."""
    predicted = normalize_label(answer.compliance_level) or ""
    gold_norm = normalize_label(sample.fulfillment_level) or ""
    combined_text = " ".join(
        part for part in [answer.direct_answer, answer.compliance_rationale] if part
    ).strip()
    is_success = bool(predicted) and answer.parse_error is None
    return RQ2SampleResult(
        condition=condition,
        question_id=sample.question_id,
        question=sample.question,
        esrs_standard=sample.esrs_standard,
        esrs_disclosure_id=sample.esrs_disclosure_id,
        company=sample.company,
        gold_fulfillment_level=sample.fulfillment_level,
        predicted_fulfillment_level=predicted,
        direct_answer_token_f1=token_f1(answer.direct_answer, sample.ground_truth_answer),
        rationale_token_f1=token_f1(answer.compliance_rationale, sample.fulfillment_rationale),
        evidence_support_ratio=evidence_support_ratio(combined_text, evidence_texts),
        exact_match=(predicted == gold_norm) if is_success else False,
        ordinal_error=_ordinal_error(gold_norm, predicted) if is_success else 2,
        cited_pages=list(answer.cited_pages),
        retrieval_pages=retrieval_pages,
        success=is_success,
        error=answer.parse_error,
    )


# ---------------------------------------------------------------------------
# Majority baseline
# ---------------------------------------------------------------------------

def majority_baseline(results: list[RQ2SampleResult]) -> dict[str, object]:
    """Return the majority-class baseline on the evaluated gold labels."""
    scored = [row for row in results if normalize_label(row.gold_fulfillment_level)]
    if not scored:
        return {}
    gold = [normalize_label(row.gold_fulfillment_level) for row in scored]
    majority_label = max(_LABELS, key=gold.count)
    pred = [majority_label] * len(gold)
    return {
        "majority_label": majority_label,
        "macro_f1": float(f1_score(gold, pred, labels=_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(_balanced_accuracy(gold, pred)),
        "exact_accuracy": float(accuracy_score(gold, pred)),
        "weighted_kappa": float(_weighted_kappa(gold, pred)),
    }


# ---------------------------------------------------------------------------
# Aggregate classification metrics
# ---------------------------------------------------------------------------

def _balanced_accuracy(gold: list[str], pred: list[str]) -> float:
    recalls = []
    for label in sorted(set(gold)):
        positives = sum(1 for v in gold if v == label)
        hits = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        recalls.append(hits / positives if positives else 0.0)
    return float(mean(recalls)) if recalls else math.nan


def _weighted_kappa(gold: list[str], pred: list[str]) -> float:
    if len(set(gold)) <= 1:
        return 1.0 if gold == pred else 0.0
    return float(cohen_kappa_score(gold, pred, labels=_LABELS, weights="linear"))


def _per_label_metrics(gold: list[str], pred: list[str]) -> dict[str, dict[str, float]]:
    """Per-label precision, recall, F1, and support."""
    prec, rec, f1s, sup = precision_recall_fscore_support(
        gold, pred, labels=_LABELS, zero_division=0,
    )
    out: dict[str, dict[str, float]] = {}
    for i, label in enumerate(_LABELS):
        out[label] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1s[i]),
            "support": int(sup[i]),
        }
    return out


def aggregate_condition_metrics(results: list[RQ2SampleResult]) -> dict[str, object]:
    """Aggregate classification and descriptive metrics for one condition."""
    scored = [row for row in results if row.success and normalize_label(row.predicted_fulfillment_level)]
    gold = [normalize_label(row.gold_fulfillment_level) for row in scored]
    pred = [normalize_label(row.predicted_fulfillment_level) for row in scored]

    if not scored:
        return {
            "n_total": len(results),
            "n_scored": 0,
            "n_failures": len(results),
            "macro_f1": math.nan,
            "macro_f1_label_basis": "full_label_set",
            "balanced_accuracy": math.nan,
            "exact_accuracy": math.nan,
            "weighted_kappa": math.nan,
            "ordinal_error_mean": math.nan,
            "direct_answer_token_f1": math.nan,
            "rationale_token_f1": math.nan,
            "evidence_support_ratio": math.nan,
            "gold_label_distribution": {},
            "per_label_metrics": {},
            "confusion_matrix": [[0, 0, 0] for _ in _LABELS],
            "labels": list(_LABELS),
            "scored_labels": [],
        }

    gold_dist = dict(Counter(gold))
    scored_labels = sorted(set(gold))
    return {
        "n_total": len(results),
        "n_scored": len(scored),
        "n_failures": len(results) - len(scored),
        "macro_f1": float(f1_score(gold, pred, labels=_LABELS, average="macro", zero_division=0)),
        "macro_f1_label_basis": "full_label_set",
        "balanced_accuracy": float(_balanced_accuracy(gold, pred)),
        "exact_accuracy": float(accuracy_score(gold, pred)),
        "weighted_kappa": float(_weighted_kappa(gold, pred)),
        "ordinal_error_mean": float(mean(row.ordinal_error for row in scored)),
        "direct_answer_token_f1": float(mean(row.direct_answer_token_f1 for row in scored)),
        "rationale_token_f1": float(mean(row.rationale_token_f1 for row in scored)),
        "evidence_support_ratio": float(mean(row.evidence_support_ratio for row in scored)),
        "gold_label_distribution": gold_dist,
        "per_label_metrics": _per_label_metrics(gold, pred),
        "confusion_matrix": confusion_matrix(gold, pred, labels=_LABELS).tolist(),
        "labels": list(_LABELS),
        "scored_labels": scored_labels,
    }


# ---------------------------------------------------------------------------
# Paired results and question-level aggregation
# ---------------------------------------------------------------------------

def build_paired_results(rows: list[RQ2SampleResult]) -> list[dict[str, object]]:
    """Create paired with/without rows for each (question, company) sample.

    Returns one row per sample with metrics from both conditions and deltas.
    Samples where either condition failed are included but flagged.
    """
    by_key: dict[tuple[str, str], dict[str, RQ2SampleResult]] = {}
    for row in rows:
        key = (row.question_id, row.company)
        by_key.setdefault(key, {})[row.condition] = row

    paired: list[dict[str, object]] = []
    for (qid, company), cond_map in sorted(by_key.items()):
        w = cond_map.get("with_esrs_context")
        wo = cond_map.get("without_esrs_context")
        if w is None or wo is None:
            continue

        both_scored = w.success and wo.success
        metrics = [
            "direct_answer_token_f1",
            "rationale_token_f1",
            "evidence_support_ratio",
        ]
        row_dict: dict[str, object] = {
            "question_id": qid,
            "esrs_standard": w.esrs_standard,
            "esrs_disclosure_id": w.esrs_disclosure_id,
            "company": company,
            "gold_fulfillment_level": w.gold_fulfillment_level,
            "predicted_with": w.predicted_fulfillment_level,
            "predicted_without": wo.predicted_fulfillment_level,
            "exact_match_with": w.exact_match,
            "exact_match_without": wo.exact_match,
            "ordinal_error_with": w.ordinal_error,
            "ordinal_error_without": wo.ordinal_error,
            "both_scored": both_scored,
        }
        for m in metrics:
            val_w = getattr(w, m)
            val_wo = getattr(wo, m)
            row_dict[f"{m}__with"] = val_w
            row_dict[f"{m}__without"] = val_wo
            row_dict[f"{m}_delta"] = val_w - val_wo if both_scored else math.nan

        row_dict["exact_match_delta"] = (
            int(w.exact_match) - int(wo.exact_match) if both_scored else math.nan
        )
        row_dict["ordinal_error_delta"] = (
            w.ordinal_error - wo.ordinal_error if both_scored else math.nan
        )
        row_dict["error_with"] = w.error
        row_dict["error_without"] = wo.error
        paired.append(row_dict)

    return paired


def build_question_level_results(
    rows: list[RQ2SampleResult],
) -> list[dict[str, object]]:
    """Aggregate per-question classification metrics for both conditions."""
    questions: dict[str, dict[str, list[RQ2SampleResult]]] = {}
    for row in rows:
        questions.setdefault(row.question_id, {}).setdefault(row.condition, []).append(row)

    out: list[dict[str, object]] = []
    for qid in sorted(questions):
        cond_map = questions[qid]
        first = next(iter(next(iter(cond_map.values()))))
        row_dict: dict[str, object] = {
            "question_id": qid,
            "esrs_standard": first.esrs_standard,
            "esrs_disclosure_id": first.esrs_disclosure_id,
        }

        agg: dict[str, dict[str, object]] = {}
        for condition in ("with_esrs_context", "without_esrs_context"):
            cond_rows = cond_map.get(condition, [])
            agg[condition] = aggregate_condition_metrics(cond_rows)

        classification_metrics = [
            "macro_f1", "balanced_accuracy", "exact_accuracy",
            "weighted_kappa", "ordinal_error_mean",
        ]
        text_metrics = [
            "direct_answer_token_f1", "rationale_token_f1", "evidence_support_ratio",
        ]
        for m in classification_metrics + text_metrics:
            val_w = agg["with_esrs_context"].get(m, math.nan)
            val_wo = agg["without_esrs_context"].get(m, math.nan)
            row_dict[f"{m}__with_esrs_context"] = val_w
            row_dict[f"{m}__without_esrs_context"] = val_wo
            try:
                row_dict[f"{m}_delta"] = float(val_w) - float(val_wo)
            except (TypeError, ValueError):
                row_dict[f"{m}_delta"] = math.nan

        out.append(row_dict)
    return out


# ---------------------------------------------------------------------------
# Effect size, power analysis, and diagnostic helpers
# ---------------------------------------------------------------------------

def _cohens_d(deltas: list[float]) -> dict[str, object]:
    """Compute Cohen's d for paired question-level deltas.

    Cohen's d = mean(deltas) / stdev(deltas).
    Magnitude thresholds follow Cohen (1988):
      |d| < 0.2  → negligible
      0.2-0.5    → small
      0.5-0.8    → medium
      > 0.8      → large
    """
    if len(deltas) < 2:
        return {"cohens_d": 0.0, "effect_magnitude": "negligible"}
    sd = stdev(deltas)
    if sd == 0:
        return {"cohens_d": 0.0, "effect_magnitude": "negligible"}
    d = mean(deltas) / sd
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    return {"cohens_d": d, "effect_magnitude": magnitude}


def _post_hoc_power(
    n: int,
    effect_size: float,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Post-hoc power analysis using normal approximation to the sign test.

    For the sign test with n discordant pairs under the alternative
    that P(positive) = Phi(d/sqrt(2)), the power is computed via the
    normal approximation to the binomial.

    Also computes the minimum detectable effect size (MDES) at 80% power.
    """
    # Normal CDF helper (no scipy dependency).
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # Normal inverse CDF approximation (Abramowitz & Stegun 26.2.23).
    def _norm_ppf(p: float) -> float:
        if p <= 0:
            return -math.inf
        if p >= 1:
            return math.inf
        if p < 0.5:
            return -_norm_ppf(1 - p)
        # Rational approximation for 0.5 <= p < 1
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)

    if n < 2:
        return {"achieved_power": math.nan, "mdes_80pct": math.nan}

    z_alpha = _norm_ppf(1.0 - alpha / 2.0)

    # Under H_a, P(positive delta) via sign test ≈ Phi(d / sqrt(2))
    abs_d = abs(effect_size)
    p_alt = _norm_cdf(abs_d / math.sqrt(2.0))
    # Power = P(reject H0 | H_a) using normal approximation
    # Z = (X - n*0.5) / sqrt(n*0.25) where X ~ Bin(n, p_alt)
    mu_alt = n * p_alt
    sigma = math.sqrt(n * 0.25)
    if sigma == 0:
        achieved_power = 0.0
    else:
        # Critical value under H0
        crit = 0.5 * n + z_alpha * sigma
        achieved_power = 1.0 - _norm_cdf((crit - mu_alt) / math.sqrt(n * p_alt * (1 - p_alt))) if p_alt < 1 else 1.0
    achieved_power = max(0.0, min(1.0, achieved_power))

    # MDES: find d such that power = 0.80 via binary search
    z_beta = _norm_ppf(0.80)
    lo_d, hi_d = 0.0, 5.0
    for _ in range(100):
        mid_d = (lo_d + hi_d) / 2.0
        p_mid = _norm_cdf(mid_d / math.sqrt(2.0))
        if p_mid >= 1.0:
            hi_d = mid_d
            continue
        mu_mid = n * p_mid
        sigma_mid = math.sqrt(n * p_mid * (1 - p_mid))
        if sigma_mid == 0:
            hi_d = mid_d
            continue
        power_mid = 1.0 - _norm_cdf((crit - mu_mid) / sigma_mid)
        if power_mid < 0.80:
            lo_d = mid_d
        else:
            hi_d = mid_d
    mdes = (lo_d + hi_d) / 2.0

    return {
        "achieved_power": round(achieved_power, 4),
        "mdes_80pct": round(mdes, 4),
    }


def _prediction_bias_diagnosis(
    rows: list["RQ2SampleResult"],
) -> dict[str, object]:
    """Diagnose whether ESRS context systematically shifts predicted labels.

    Computes the mean ordinal prediction for each condition and reports
    the difference.  A positive shift means the treatment condition
    predicts *higher* fulfillment on average.
    """
    with_ords: list[int] = []
    without_ords: list[int] = []
    for row in rows:
        if not row.success:
            continue
        pred_ord = _LABEL_ORD.get(normalize_label(row.predicted_fulfillment_level))
        if pred_ord is None:
            continue
        if row.condition == "with_esrs_context":
            with_ords.append(pred_ord)
        elif row.condition == "without_esrs_context":
            without_ords.append(pred_ord)

    if not with_ords or not without_ords:
        return {"mean_ordinal_with": math.nan, "mean_ordinal_without": math.nan,
                "ordinal_shift": math.nan, "interpretation": "insufficient_data"}

    mean_with = mean(with_ords)
    mean_without = mean(without_ords)
    shift = mean_with - mean_without
    if abs(shift) < 0.05:
        interpretation = "no_systematic_bias"
    elif shift > 0:
        interpretation = "context_shifts_predictions_upward"
    else:
        interpretation = "context_shifts_predictions_downward"

    return {
        "mean_ordinal_with": round(mean_with, 4),
        "mean_ordinal_without": round(mean_without, 4),
        "ordinal_shift": round(shift, 4),
        "interpretation": interpretation,
    }


def _direction_summary(
    question_rows: list[dict[str, object]],
    metric: str = "macro_f1",
) -> dict[str, object]:
    """Count how many questions improved, degraded, or were unchanged."""
    improved = 0
    degraded = 0
    unchanged = 0
    for qr in question_rows:
        delta = qr.get(f"{metric}_delta")
        if delta is None or (isinstance(delta, float) and math.isnan(delta)):
            continue
        d = float(delta)
        if d > 0:
            improved += 1
        elif d < 0:
            degraded += 1
        else:
            unchanged += 1
    return {
        "n_improved": improved,
        "n_degraded": degraded,
        "n_unchanged": unchanged,
        "metric": metric,
    }


# ---------------------------------------------------------------------------
# Paired statistical tests
# ---------------------------------------------------------------------------

def _cluster_bootstrap_ci(
    question_deltas: dict[str, float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, object]:
    """Bootstrap CI treating each question as a cluster.

    Resamples question-level deltas with replacement, computes the mean
    of each resample, and returns percentile CIs.  Also reports Cohen's d
    and effect magnitude for the observed deltas.
    """
    deltas = list(question_deltas.values())
    n = len(deltas)
    if n == 0:
        return {"mean_delta": math.nan, "ci_lower": math.nan, "ci_upper": math.nan,
                "approx_two_sided_p": math.nan, "n_question_clusters": 0,
                "cohens_d": math.nan, "effect_magnitude": "n/a",
                "n_resamples": n_resamples}

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_resamples):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()

    alpha = 1.0 - confidence
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    obs_mean = mean(deltas)

    # Approximate two-sided p: fraction of resamples on opposite side of zero.
    if obs_mean >= 0:
        p_approx = sum(1 for m in boot_means if m <= 0) / n_resamples
    else:
        p_approx = sum(1 for m in boot_means if m >= 0) / n_resamples
    p_approx = min(1.0, 2.0 * p_approx)  # two-sided

    # Cohen's d for the observed deltas.
    cd = _cohens_d(deltas)

    return {
        "mean_delta": obs_mean,
        "ci_lower": boot_means[lo_idx],
        "ci_upper": boot_means[hi_idx],
        "approx_two_sided_p": p_approx,
        "n_question_clusters": n,
        "cohens_d": cd["cohens_d"],
        "effect_magnitude": cd["effect_magnitude"],
        "n_resamples": n_resamples,
    }


def _question_level_sign_test(
    question_deltas: dict[str, float],
    *,
    direction: str = "higher_is_better",
    endpoint_role: str = "confirmatory_primary",
) -> dict[str, object]:
    """Exact sign test across question-level deltas.

    For n ≤ 25 (we have 12), the sign test is the most appropriate
    non-parametric paired test.
    """
    deltas = list(question_deltas.values())
    n = len(deltas)
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    ties = n - pos - neg
    n_disc = pos + neg

    if n_disc == 0:
        p_value = 1.0
    else:
        # Two-sided exact sign test: P(X ≥ max(pos,neg)) under Binomial(n_disc, 0.5)
        from math import comb
        k = max(pos, neg)
        p_tail = sum(comb(n_disc, i) for i in range(k, n_disc + 1)) / (2 ** n_disc)
        p_value = min(1.0, 2.0 * p_tail)

    obs_mean = mean(deltas) if deltas else math.nan
    obs_median = median(deltas) if deltas else math.nan
    d_std = stdev(deltas) if len(deltas) > 1 else 0.0
    effect_size = obs_mean / d_std if d_std > 0 else 0.0

    return {
        "mean_with": math.nan,  # filled by caller
        "mean_without": math.nan,
        "mean_delta": obs_mean,
        "median_delta": obs_median,
        "p_value": p_value,
        "effect_size": effect_size,
        "direction": direction,
        "endpoint_role": endpoint_role,
        "n_questions": n,
        "positive_deltas": pos,
        "negative_deltas": neg,
        "ties": ties,
    }


def _exact_match_discordant_test(
    paired_rows: list[dict[str, object]],
) -> dict[str, object]:
    """McNemar-style exact sign test on discordant exact-match pairs."""
    scored = [r for r in paired_rows if r.get("both_scored")]
    n = len(scored)
    wins = sum(1 for r in scored if r["exact_match_with"] and not r["exact_match_without"])
    losses = sum(1 for r in scored if not r["exact_match_with"] and r["exact_match_without"])
    ties = n - wins - losses
    disc = wins + losses

    with_rate = sum(1 for r in scored if r["exact_match_with"]) / n if n else math.nan
    without_rate = sum(1 for r in scored if r["exact_match_without"]) / n if n else math.nan

    if disc == 0:
        sign_p = 1.0
    else:
        from math import comb
        k = max(wins, losses)
        p_tail = sum(comb(disc, i) for i in range(k, disc + 1)) / (2 ** disc)
        sign_p = min(1.0, 2.0 * p_tail)

    effect = (wins - losses) / n if n else 0.0
    return {
        "with_rate": with_rate,
        "without_rate": without_rate,
        "delta": with_rate - without_rate if n else math.nan,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "discordant_pairs": disc,
        "sign_test_p_value": sign_p,
        "mcnemar_exact_p_value": sign_p,
        "effect_size": effect,
        "n_pairs": n,
        "endpoint_role": "exploratory_unclustered",
    }


def compute_rq2_paired_statistics(
    rows: list[RQ2SampleResult],
    paired_rows: list[dict[str, object]],
    question_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Compute all paired inference for the RQ2 ablation."""
    # Build question-level delta dicts for each metric
    classification_metrics = {
        "macro_f1": ("higher_is_better", "confirmatory_primary"),
        "balanced_accuracy": ("higher_is_better", "supportive_classification"),
        "exact_accuracy": ("higher_is_better", "supportive_classification"),
        "weighted_kappa": ("higher_is_better", "supportive_classification"),
        "ordinal_error_mean": ("lower_is_better", "supportive_classification"),
    }
    text_metrics = {
        "direct_answer_token_f1": ("higher_is_better", "exploratory_text"),
        "rationale_token_f1": ("higher_is_better", "exploratory_text"),
        "evidence_support_ratio": ("higher_is_better", "exploratory_text"),
    }

    cluster_bootstrap: dict[str, dict[str, object]] = {}
    question_tests: dict[str, dict[str, object]] = {}

    all_metrics = {**classification_metrics, **text_metrics}
    for metric, (direction, role) in all_metrics.items():
        q_deltas: dict[str, float] = {}
        q_with: dict[str, float] = {}
        q_without: dict[str, float] = {}
        for qr in question_rows:
            qid = str(qr["question_id"])
            delta = qr.get(f"{metric}_delta")
            if delta is not None and not (isinstance(delta, float) and math.isnan(delta)):
                q_deltas[qid] = float(delta)
                q_with[qid] = float(qr.get(f"{metric}__with_esrs_context", math.nan))
                q_without[qid] = float(qr.get(f"{metric}__without_esrs_context", math.nan))

        cb = _cluster_bootstrap_ci(q_deltas)
        cluster_bootstrap[metric] = cb

        st = _question_level_sign_test(q_deltas, direction=direction, endpoint_role=role)
        if q_with:
            st["mean_with"] = mean(q_with.values())
            st["mean_without"] = mean(q_without.values())
        question_tests[metric] = st

    exact_match_test = _exact_match_discordant_test(paired_rows)

    # --- New diagnostics (thesis-level) ---
    # Power analysis for the primary endpoint (macro_f1)
    primary_cb = cluster_bootstrap.get("macro_f1", {})
    primary_d = primary_cb.get("cohens_d", 0.0)
    n_clusters = primary_cb.get("n_question_clusters", 0)
    power_analysis = _post_hoc_power(n_clusters, primary_d)

    # Prediction bias diagnosis
    prediction_bias = _prediction_bias_diagnosis(rows)

    # Direction summary for primary metric
    direction_summary = _direction_summary(question_rows, metric="macro_f1")

    scored = [r for r in paired_rows if r.get("both_scored")]
    return {
        "n_pairs": len(scored),
        "n_question_clusters": len({str(qr["question_id"]) for qr in question_rows}),
        "cluster_bootstrap": cluster_bootstrap,
        "question_level_tests": question_tests,
        "exact_match_test": exact_match_test,
        "power_analysis": power_analysis,
        "prediction_bias": prediction_bias,
        "direction_summary": direction_summary,
        "notes": {
            "primary_endpoint": "macro_f1",
            "bootstrap_unit": "question_cluster",
            "primary_inference": (
                "Cluster bootstrap CI and exact sign test across 12 question-level "
                "deltas. Alpha=0.05, two-sided. If CI includes zero, the effect is "
                "not statistically defensible."
            ),
            "secondary_tests_multiplicity": (
                "Secondary classification and text metrics are supportive/exploratory. "
                "No multiplicity correction is applied; interpret with caution."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Full summary builder
# ---------------------------------------------------------------------------

def build_rq2_summary(
    rows: list[RQ2SampleResult],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, dict[str, object]]],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    """Build per-condition aggregates, paired results, question-level
    results, and paired statistical tests.

    Returns
    -------
    aggregate_metrics, per_standard_metrics, majority, paired_rows,
    question_rows, paired_statistics
    """
    aggregate: dict[str, dict[str, object]] = {}
    per_standard: dict[str, dict[str, dict[str, object]]] = {}
    conditions = sorted({row.condition for row in rows})

    for condition in conditions:
        cond_rows = [row for row in rows if row.condition == condition]
        aggregate[condition] = aggregate_condition_metrics(cond_rows)

        standard_metrics: dict[str, dict[str, object]] = {}
        for standard in sorted({row.esrs_standard for row in cond_rows}):
            standard_rows = [row for row in cond_rows if row.esrs_standard == standard]
            standard_metrics[standard] = aggregate_condition_metrics(standard_rows)
        per_standard[condition] = standard_metrics

    majority = majority_baseline(rows)
    paired_rows = build_paired_results(rows)
    question_rows = build_question_level_results(rows)
    paired_statistics = compute_rq2_paired_statistics(rows, paired_rows, question_rows)

    return aggregate, per_standard, majority, paired_rows, question_rows, paired_statistics


# ---------------------------------------------------------------------------
# Audit and validation helpers
# ---------------------------------------------------------------------------

def build_context_summary(
    context_results: dict[str, ESRSContextResult],
) -> list[dict[str, object]]:
    """Summarize ESRS context sizes per question for audit display."""
    out: list[dict[str, object]] = []
    for qid, ctx in sorted(context_results.items()):
        text = ctx.context_text or ""
        lines = [line for line in text.split("\n") if line.strip()]
        # Rough token estimate: character count / 4
        est_tokens = len(text) // 4
        included = len(ctx.selected_requirements)
        # Count sections excluded (heuristic: look for bracketed section headers)
        total_sections = text.count("[") if text else 0
        out.append({
            "question_id": qid,
            "esrs_standard": ctx.esrs_standard,
            "esrs_disclosure_id": ctx.esrs_disclosure_id,
            "included_section_count": included,
            "excluded_section_count": max(0, total_sections - included),
            "estimated_tokens": est_tokens,
            "character_count": len(text),
        })
    return out


def build_question_manifest(
    samples: list[GenerationSample],
) -> list[dict[str, object]]:
    """Build an audit manifest mapping each question to its ESRS standard.

    Checks that every listed disclosure ID can be resolved to a section
    title in the corresponding ESRS JSON file.
    """
    seen: dict[str, GenerationSample] = {}
    for s in samples:
        if s.question_id not in seen:
            seen[s.question_id] = s

    out: list[dict[str, object]] = []
    for qid in sorted(seen):
        s = seen[qid]
        # Try to resolve disclosure IDs in the ESRS JSON
        disclosure_ids = [d.strip() for d in s.esrs_disclosure_id.split(",") if d.strip()]
        resolved_titles: list[str] = []
        missing_ids: list[str] = []

        json_path = ESRS_JSON_DIR / f"ESRS_{s.esrs_standard.replace('ESRS ', '').replace(' ', '_')}.json"
        if json_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            flat_text = json.dumps(payload)
            for did in disclosure_ids:
                # Look for the disclosure requirement title containing this ID
                found = False
                for key in ("disclosure_requirements", "disclosures"):
                    section = payload.get(key, [])
                    if isinstance(section, list):
                        for item in section:
                            title = str(item.get("title", "")) if isinstance(item, dict) else ""
                            if did in title:
                                resolved_titles.append(title)
                                found = True
                                break
                    if found:
                        break
                if not found:
                    # Fallback: search entire JSON text
                    if did in flat_text:
                        resolved_titles.append(f"(found in standard text: {did})")
                        found = True
                if not found:
                    missing_ids.append(did)
        else:
            missing_ids = list(disclosure_ids)

        out.append({
            "question_id": qid,
            "esrs_standard": s.esrs_standard,
            "esrs_disclosure_id": s.esrs_disclosure_id,
            "resolved_titles": resolved_titles,
            "missing_disclosure_ids": missing_ids,
            "ok": len(missing_ids) == 0,
        })
    return out


def validate_rq2_artifacts(
    *,
    summary_payload: dict,
    sample_rows: list[dict],
    paired_rows: list[dict],
    question_rows: list[dict],
    expected_audit_manifest: dict | None = None,
    expected_question_manifest: list[dict] | None = None,
) -> None:
    """Validate saved RQ2 artifacts for internal consistency.

    Raises ValueError if any check fails.
    """
    errors: list[str] = []

    # Check sample count
    n_with = sum(1 for r in sample_rows if r.get("condition") == "with_esrs_context")
    n_without = sum(1 for r in sample_rows if r.get("condition") == "without_esrs_context")
    if n_with == 0:
        errors.append("No with_esrs_context samples found")
    if n_without == 0:
        errors.append("No without_esrs_context samples found")
    if n_with != n_without:
        errors.append(f"Unequal sample counts: with={n_with}, without={n_without}")

    # Check paired rows cover expected count
    expected_pairs = n_with  # one pair per (question, company)
    if len(paired_rows) != expected_pairs:
        errors.append(f"Expected {expected_pairs} paired rows, got {len(paired_rows)}")

    # Check question rows
    question_ids_in_samples = sorted({r.get("question_id") for r in sample_rows})
    question_ids_in_qrows = sorted({r.get("question_id") for r in question_rows})
    if question_ids_in_samples != question_ids_in_qrows:
        errors.append(
            f"Question IDs mismatch: samples have {question_ids_in_samples}, "
            f"question_rows have {question_ids_in_qrows}"
        )

    if errors:
        raise ValueError("RQ2 artifact validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def sample_results_to_rows(results: list[RQ2SampleResult]) -> list[dict[str, object]]:
    """Serialize sample-result dataclasses for CSV/JSON persistence."""
    return [asdict(result) for result in results]
