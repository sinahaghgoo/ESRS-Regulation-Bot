#!/usr/bin/env python3
"""Compare RQ3 metrics before vs after the citation precision overhaul.

Reads:
  * the ``rq3_summary.json`` produced by the notebook (current state)
  * the ``rq3_sample_results.csv`` (current per-sample data)
  * if available, a snapshot of the old summary saved as
    ``rq3_summary.before_v2.json`` so the old precision/recall is shown
    side-by-side with the new figures.

Prints a thesis-ready table summarising the change and flags whether the
key acceptance criteria have been met:

  * Citation precision (set-overlap, ±1) >= 0.70
  * Mean predicted page count <= 5
  * Recall >= 0.85
  * |Spearman correlation with judge citation_usefulness| >= 0.40
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent
RQ3_DIR = PROJECT_ROOT / "data/artifacts/evaluation_results/rq3"
SUMMARY_PATH = RQ3_DIR / "rq3_summary.json"
SAMPLE_PATH = RQ3_DIR / "rq3_sample_results.csv"
OLD_SNAPSHOT = RQ3_DIR / "rq3_summary.before_v2.json"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    if not SUMMARY_PATH.exists():
        print(f"[ERROR] summary not found: {SUMMARY_PATH}")
        sys.exit(1)

    cur = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    rows = load_csv(SAMPLE_PATH) if SAMPLE_PATH.exists() else []

    cm = cur.get("citation_metrics", {})
    cp = cm.get("precision", {})
    cr = cm.get("recall", {})
    cf = cm.get("f1", {})
    cma = cur.get("cross_metric_alignment", {})

    print("=" * 76)
    print("RQ3 RESULTS — citation precision overhaul (v2)")
    print("=" * 76)
    print()
    print(f"Method                : {cm.get('method')}, tolerance ±{cm.get('tolerance_pages')}")
    print(f"Mean #pred pages      : {cm.get('mean_pred_pages', 'n/a'):.2f}"
          if isinstance(cm.get('mean_pred_pages'), (int, float)) else
          f"Mean #pred pages      : {cm.get('mean_pred_pages', 'n/a')}")
    print(f"Mean #gold pages      : {cm.get('mean_gold_pages', 'n/a'):.2f}"
          if isinstance(cm.get('mean_gold_pages'), (int, float)) else
          f"Mean #gold pages      : {cm.get('mean_gold_pages', 'n/a')}")
    print()
    print(f"{'Metric':<22s} {'Mean':>8s}  {'95% CI':>22s}")
    print("─" * 56)
    if isinstance(cp, dict):
        print(f"{'Precision':<22s} {cp.get('mean', float('nan')):>8.4f}  "
              f"[{cp.get('ci_lower', 0):.4f}, {cp.get('ci_upper', 0):.4f}]")
        print(f"{'Recall':<22s} {cr.get('mean', float('nan')):>8.4f}  "
              f"[{cr.get('ci_lower', 0):.4f}, {cr.get('ci_upper', 0):.4f}]")
        print(f"{'F1':<22s} {cf.get('mean', float('nan')):>8.4f}  "
              f"[{cf.get('ci_lower', 0):.4f}, {cf.get('ci_upper', 0):.4f}]")

    base = cm.get("baseline_range_overlap", {})
    if base:
        print()
        print("Baseline (range-overlap on raw entries, tol ±1):")
        bp = base.get("precision", {})
        br = base.get("recall", {})
        bf = base.get("f1", {})
        if isinstance(bp, dict):
            print(f"  P={bp.get('mean', float('nan')):.4f}  R={br.get('mean', float('nan')):.4f}  "
                  f"F1={bf.get('mean', float('nan')):.4f}")

    print()
    print("CROSS-METRIC ALIGNMENT")
    print("─" * 56)
    if cma:
        print(f"  Pearson  r (cite_p vs judge_cu)   : {cma.get('pearson_cite_p_vs_judge_cu', 0):+.4f}  "
              f"(p={cma.get('pearson_p_value', 'n/a')})")
        print(f"  Spearman ρ (cite_p vs judge_cu)   : {cma.get('spearman_cite_p_vs_judge_cu', 0):+.4f}  "
              f"(p={cma.get('spearman_p_value', 'n/a')})")
        print(f"  Pearson r vs judge_mean_score      : {cma.get('pearson_cite_p_vs_judge_mean', 0):+.4f}")
        print(f"  Spearman ρ vs judge_mean_score     : {cma.get('spearman_cite_p_vs_judge_mean', 0):+.4f}")
        print(f"  Range-overlap baseline pearson r   : {cma.get('pearson_range_baseline_vs_cu', 0):+.4f}")
        print(f"  Mean abs gap (judge_cu_norm − cite_p): {cma.get('mean_abs_gap', 0):.4f}")

    if rows:
        # Diagnostic: distribution of prediction page counts
        n_pred = [int(r.get("n_pred_pages", 0)) for r in rows if r.get("n_pred_pages")]
        if n_pred:
            print()
            print(f"Mean #pred per sample : {mean(n_pred):.2f}  (range {min(n_pred)}-{max(n_pred)})")

    if OLD_SNAPSHOT.exists():
        old = json.loads(OLD_SNAPSHOT.read_text(encoding="utf-8"))
        print()
        print("DELTA vs PREVIOUS RUN (rq3_summary.before_v2.json)")
        print("─" * 56)
        old_cm = old.get("citation_metrics", {})
        for label, k in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]:
            cur_v = cm.get(k, {}).get("mean", float("nan"))
            old_v = old_cm.get(k, {}).get("mean", float("nan"))
            delta = cur_v - old_v
            print(f"  {label:<10s} {old_v:>7.4f} → {cur_v:>7.4f}  ({delta:+.4f})")

    # Acceptance criteria for the master thesis
    print()
    print("ACCEPTANCE CRITERIA FOR MASTER THESIS")
    print("─" * 56)
    p_mean = cp.get("mean", 0) if isinstance(cp, dict) else 0
    r_mean = cr.get("mean", 0) if isinstance(cr, dict) else 0
    spearman = abs(cma.get("spearman_cite_p_vs_judge_cu", 0))
    pred_pages = cm.get("mean_pred_pages", 99) if isinstance(cm.get("mean_pred_pages"), (int, float)) else 99

    checks = [
        ("Precision >= 0.70 (range-overlap, ±1)", p_mean >= 0.70, f"got {p_mean:.4f}"),
        ("Recall >= 0.85", r_mean >= 0.85, f"got {r_mean:.4f}"),
        ("Mean #pred pages <= 5", pred_pages <= 5, f"got {pred_pages:.2f}"),
        ("|Spearman ρ| >= 0.40 (vs judge_cu)", spearman >= 0.40, f"got {spearman:.4f}"),
    ]
    for label, ok, info in checks:
        sym = "PASS" if ok else "FAIL"
        print(f"  [{sym}] {label}  ({info})")
    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n{n_pass}/{len(checks)} criteria met.")


if __name__ == "__main__":
    main()
