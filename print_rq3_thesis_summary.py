#!/usr/bin/env python3
"""Print a thesis-ready RQ3 summary, comparing v1 (pre-fix) vs v2 (post-fix).

Reads:
  * data/artifacts/evaluation_results/rq3/rq3_summary.json (current = v2)
  * data/artifacts/evaluation_results/rq3/rq3_summary.before_v2.json (snapshot = v1)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RQ3_DIR = PROJECT_ROOT / "data/artifacts/evaluation_results/rq3"
NEW = RQ3_DIR / "rq3_summary.json"
OLD = RQ3_DIR / "rq3_summary.before_v2.json"


def _val(d: dict, k: str) -> float:
    v = d.get(k, {})
    if isinstance(v, dict):
        return float(v.get("mean", 0))
    return float(v) if v is not None else 0.0


def main() -> None:
    if not NEW.exists():
        print(f"[ERROR] new summary not found: {NEW}")
        sys.exit(1)
    new = json.loads(NEW.read_text(encoding="utf-8"))
    old = json.loads(OLD.read_text(encoding="utf-8")) if OLD.exists() else None

    new_cm = new.get("citation_metrics", {})
    old_cm = old.get("citation_metrics", {}) if old else {}
    new_cma = new.get("cross_metric_alignment", {})
    old_cma = old.get("cross_metric_alignment", {}) if old else {}

    print("=" * 76)
    print("RQ3 — Citation precision overhaul, before vs after")
    print("=" * 76)
    print()
    print(f"Method: {new_cm.get('method', 'n/a')} (tolerance ±{new_cm.get('tolerance_pages', 'n/a')})")
    if old_cm:
        print(f"Old method: {old_cm.get('method', 'n/a')} (tolerance ±{old_cm.get('tolerance_pages', 'n/a')})")
    print()

    headers = ["Metric", "Before", "After", "Δ"]
    rows = []
    for k_label, k in [
        ("Citation precision", "precision"),
        ("Citation recall", "recall"),
        ("Citation F1", "f1"),
    ]:
        new_v = _val(new_cm, k)
        old_v = _val(old_cm, k)
        rows.append((k_label, old_v, new_v, new_v - old_v))

    new_pp = new_cm.get("mean_pred_pages", 0)
    old_pp = old_cm.get("mean_pred_pages", 0)
    if isinstance(new_pp, (int, float)) and isinstance(old_pp, (int, float)):
        rows.append(("Mean #pred pages", old_pp, new_pp, new_pp - old_pp))
    new_gp = new_cm.get("mean_gold_pages", 0)
    old_gp = old_cm.get("mean_gold_pages", 0)
    if isinstance(new_gp, (int, float)) and isinstance(old_gp, (int, float)):
        rows.append(("Mean #gold pages", old_gp, new_gp, new_gp - old_gp))

    print(f"{headers[0]:<30s} {headers[1]:>10s} {headers[2]:>10s} {headers[3]:>10s}")
    print("─" * 64)
    for r in rows:
        print(f"{r[0]:<30s} {r[1]:>10.4f} {r[2]:>10.4f} {r[3]:>+10.4f}")

    # Cross-metric alignment
    print()
    print("Cross-metric alignment (deterministic precision vs LLM judge_cu)")
    print("─" * 64)
    for k_label, k in [
        ("Pearson r", "pearson_cite_p_vs_judge_cu"),
        ("Spearman ρ", "spearman_cite_p_vs_judge_cu"),
        ("Mean abs gap", "mean_abs_gap"),
    ]:
        new_v = float(new_cma.get(k, 0))
        old_v = float(old_cma.get(k, 0)) if old_cma else 0
        delta = new_v - old_v if old_cma else None
        if delta is None:
            print(f"  {k_label:<22s} {new_v:>10.4f}")
        else:
            print(f"  {k_label:<22s} {old_v:>10.4f} → {new_v:>10.4f}  ({delta:+.4f})")

    # Judge dimension scores
    print()
    print("LLM judge dimension means (1-5 scale)")
    print("─" * 64)
    for dim in ("factual_correctness", "completeness", "grounding_faithfulness",
                "citation_usefulness", "coherence"):
        ds = new.get("judge_dimension_scores", {}).get(dim, {})
        os_ = (old.get("judge_dimension_scores", {}) if old else {}).get(dim, {})
        new_v = float(ds.get("mean", 0))
        old_v = float(os_.get("mean", 0)) if old else 0
        delta = new_v - old_v if old else None
        if delta is None:
            print(f"  {dim:<28s} {new_v:>6.3f}")
        else:
            print(f"  {dim:<28s} {old_v:>6.3f} → {new_v:>6.3f}  ({delta:+.3f})")

    # Overall mean
    om = new.get("judge_overall_mean", {}).get("mean", 0)
    print()
    print(f"Judge overall mean: {om:.3f}")

    # Acceptance criteria
    print()
    print("ACCEPTANCE CRITERIA")
    print("─" * 64)
    p = _val(new_cm, "precision")
    r = _val(new_cm, "recall")
    f1 = _val(new_cm, "f1")
    spearman_abs = abs(float(new_cma.get("spearman_cite_p_vs_judge_cu", 0)))
    pp_mean = new_cm.get("mean_pred_pages")
    pp_mean_f = float(pp_mean) if isinstance(pp_mean, (int, float)) else 99.0

    checks = [
        ("Citation precision >= 0.70", p, p >= 0.70),
        ("Citation recall >= 0.85", r, r >= 0.85),
        ("Citation F1 >= 0.75", f1, f1 >= 0.75),
        ("Mean #pred pages <= 5", pp_mean_f, pp_mean_f <= 5.0),
        ("|Spearman ρ| >= 0.40", spearman_abs, spearman_abs >= 0.40),
    ]
    for label, val, ok in checks:
        sym = "PASS" if ok else "FAIL"
        print(f"  [{sym}] {label}  (got {val:.4f})")
    n_pass = sum(1 for _, _, ok in checks if ok)
    print(f"\n{n_pass}/{len(checks)} criteria met.")


if __name__ == "__main__":
    main()
