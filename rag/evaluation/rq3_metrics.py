"""Deterministic metrics and citation page analysis for RQ3.

RQ3 is a single-condition characterisation study.  This module provides:

  1. Compliance classification metrics (macro-F1, balanced accuracy, kappa, etc.)
  2. Citation page overlap (precision / recall / F1 as set overlap of page numbers)
  3. Page number normalisation to handle format variations between gold and generated
  4. Bootstrap confidence interval computation (95% CI, 10 000 resamples)
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from statistics import mean

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LABELS = ["Low", "Medium", "High"]
_LABEL_ORD = {"Low": 0, "Medium": 1, "High": 2}

# Regex patterns for extracting page numbers from various formats
_PAGE_NUM_RE = re.compile(r"\b(?:p\.?\s*)?(\d+)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def normalize_label(value: str) -> str | None:
    """Normalize fulfillment labels to canonical Low/Medium/High."""
    mapping = {"low": "Low", "medium": "Medium", "high": "High"}
    return mapping.get(str(value).strip().lower())


def ordinal_error(gold_label: str, pred_label: str) -> int:
    """Absolute ordinal distance between two labels (0-2)."""
    g = _LABEL_ORD.get(gold_label)
    p = _LABEL_ORD.get(pred_label)
    if g is None or p is None:
        return 2
    return abs(g - p)


# ---------------------------------------------------------------------------
# Citation page normalisation and matching
# ---------------------------------------------------------------------------

def normalize_page_numbers(raw_pages: list[str]) -> set[int]:
    """Extract integer page numbers from various formats.

    Handles formats like:
      - "42"           → {42}
      - "43-44"        → {43, 44}
      - "p.12"         → {12}
      - "Evidence 3 (... p.150)" → {150}
      - "Sustainability Statement (p.135)" → {135}
      - "150"          → {150}
    """
    pages: set[int] = set()
    for raw in raw_pages:
        raw_str = str(raw).strip()
        # Handle explicit page ranges like "43-44"
        range_match = re.findall(r"(\d+)\s*[-–]\s*(\d+)", raw_str)
        for start_s, end_s in range_match:
            start, end = int(start_s), int(end_s)
            if end - start < 50:  # sanity: skip implausible ranges
                pages.update(range(start, end + 1))

        # Extract all standalone page numbers
        for m in _PAGE_NUM_RE.finditer(raw_str):
            num = int(m.group(1))
            # Filter out implausibly small numbers that are likely evidence indices
            # and implausibly large numbers. Only keep if the raw string
            # also contains "page", "p.", or if the number is standalone.
            if num >= 1:
                pages.add(num)
    return pages


def citation_page_metrics(
    gold_pages: set[int],
    predicted_pages: set[int],
) -> dict[str, float]:
    """Compute citation page precision, recall, and F1 as set overlap."""
    if not gold_pages and not predicted_pages:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gold_pages or not predicted_pages:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = gold_pages & predicted_pages
    precision = len(overlap) / len(predicted_pages)
    recall = len(overlap) / len(gold_pages)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Range-overlap citation metrics (avoids page-range expansion bias)
# ---------------------------------------------------------------------------

def normalize_page_ranges(raw_pages: list[str]) -> list[tuple[int, int]]:
    """Parse raw page strings into (start, end) range tuples.

    Each raw string becomes one range:
      - "42"       → (42, 42)
      - "43-44"    → (43, 44)
      - "125-129"  → (125, 129)
      - "p.12"     → (12, 12)

    This avoids the expansion bias of ``normalize_page_numbers`` where
    "125-129" becomes five individual pages.
    """
    ranges: list[tuple[int, int]] = []
    for raw in raw_pages:
        raw_str = str(raw).strip()
        # Try explicit range first (e.g. "43-44", "125–129")
        range_match = re.findall(r"(\d+)\s*[-–]\s*(\d+)", raw_str)
        if range_match:
            for s, e in range_match:
                start, end = int(s), int(e)
                if end - start < 50:  # sanity check
                    ranges.append((start, end))
        else:
            # Single page number
            nums = _PAGE_NUM_RE.findall(raw_str)
            for n in nums:
                v = int(n)
                if v >= 5:  # filter evidence indices / section numbers
                    ranges.append((v, v))
    return ranges


def citation_range_overlap_metrics(
    gold_ranges: list[tuple[int, int]],
    pred_ranges: list[tuple[int, int]],
    tolerance: int = 1,
) -> dict[str, float]:
    """Citation precision/recall/F1 using range overlap with tolerance.

    Each citation unit is a page range (start, end).  A predicted range
    counts as a true-positive if it overlaps (within ±*tolerance* pages)
    with any gold range, and vice-versa for recall.

    Parameters
    ----------
    tolerance : int
        Pages of slack added to each range boundary before checking
        overlap.  ``tolerance=1`` means "125-129" matches gold "130".
    """
    if not gold_ranges and not pred_ranges:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gold_ranges or not pred_ranges:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def _overlaps(r1: tuple[int, int], r2: tuple[int, int]) -> bool:
        return (r1[0] - tolerance) <= r2[1] and (r2[0] - tolerance) <= r1[1]

    tp_pred = sum(
        1 for p in pred_ranges if any(_overlaps(p, g) for g in gold_ranges)
    )
    tp_gold = sum(
        1 for g in gold_ranges if any(_overlaps(p, g) for p in pred_ranges)
    )
    precision = tp_pred / len(pred_ranges)
    recall = tp_gold / len(gold_ranges)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Aggregate compliance classification metrics
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


def compliance_metrics(
    gold: list[str],
    pred: list[str],
) -> dict[str, object]:
    """Compute all deterministic compliance classification metrics for RQ3."""
    if not gold or not pred:
        return {}

    n = len(gold)
    cm = confusion_matrix(gold, pred, labels=_LABELS)

    # Ordinal errors
    ords = [abs(_LABEL_ORD.get(g, 1) - _LABEL_ORD.get(p, 1)) for g, p in zip(gold, pred)]
    obo = sum(1 for o in ords if o <= 1) / n

    # Per-label metrics
    prec, rec, f1s, sup = precision_recall_fscore_support(
        gold, pred, labels=_LABELS, zero_division=0,
    )
    per_label = {}
    for i, lbl in enumerate(_LABELS):
        per_label[lbl] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1s[i]),
            "support": int(sup[i]),
        }

    return {
        "n": n,
        "macro_f1": float(f1_score(gold, pred, labels=_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(_balanced_accuracy(gold, pred)),
        "exact_accuracy": float(accuracy_score(gold, pred)),
        "weighted_kappa": float(_weighted_kappa(gold, pred)),
        "off_by_one_accuracy": obo,
        "mean_ordinal_error": float(mean(ords)),
        "confusion_matrix": cm.tolist(),
        "labels": list(_LABELS),
        "gold_distribution": dict(Counter(gold)),
        "per_label": per_label,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for the mean of values.

    Returns dict with keys: mean, ci_lower, ci_upper, std.
    """
    if not values:
        return {"mean": math.nan, "ci_lower": math.nan, "ci_upper": math.nan, "std": math.nan}

    rng = random.Random(seed)
    n = len(values)
    boot_means = []
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(mean(sample))

    boot_means.sort()
    alpha = 1.0 - ci
    lo_idx = int(math.floor(alpha / 2 * n_resamples))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_resamples)) - 1

    sample_mean = mean(values)
    return {
        "mean": sample_mean,
        "ci_lower": boot_means[lo_idx],
        "ci_upper": boot_means[hi_idx],
        "std": (sum((x - sample_mean) ** 2 for x in values) / max(n - 1, 1)) ** 0.5,
    }


def bootstrap_metric_ci(
    gold: list[str],
    pred: list[str],
    metric_fn,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for a classification metric function.

    metric_fn(gold_list, pred_list) -> float
    """
    if not gold:
        return {"mean": math.nan, "ci_lower": math.nan, "ci_upper": math.nan}

    rng = random.Random(seed)
    n = len(gold)
    boot_vals = []
    for _ in range(n_resamples):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        g_boot = [gold[i] for i in idx]
        p_boot = [pred[i] for i in idx]
        boot_vals.append(metric_fn(g_boot, p_boot))

    boot_vals.sort()
    alpha = 0.05
    lo_idx = int(math.floor(alpha / 2 * n_resamples))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_resamples)) - 1

    return {
        "mean": metric_fn(gold, pred),
        "ci_lower": boot_vals[lo_idx],
        "ci_upper": boot_vals[hi_idx],
    }
