#!/usr/bin/env python3
"""Recompute rq2_summary.json from saved CSVs — zero LLM calls.

Run this after upgrading rq2_metrics.py to regenerate the summary JSON
with new statistical fields (Cohen's d, power analysis, prediction bias,
direction summary) without re-running the full pipeline.

Usage:
    python recompute_rq2_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.config.settings import (
    RQ2_RESULTS_DIR,
    RQ2_RESULTS_VERSION,
    build_rq2_audit_manifest,
)
from rag.evaluation.rq2_judge import get_judge_version
from rag.evaluation.rq2_metrics import (
    aggregate_condition_metrics,
    build_paired_results,
    build_question_level_results,
    compute_rq2_paired_statistics,
    majority_baseline,
)
from rag.types import RQ2SampleResult


def _load_results_from_csv(results_dir: Path) -> list[RQ2SampleResult]:
    """Reconstruct RQ2SampleResult objects from the sample results CSV."""
    import csv

    csv_path = results_dir / "rq2_sample_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return []

    rows: list[RQ2SampleResult] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            success = row.get("success", "True").lower() == "true"
            rows.append(
                RQ2SampleResult(
                    condition=row["condition"],
                    question_id=row["question_id"],
                    question=row.get("question", ""),
                    esrs_standard=row.get("esrs_standard", ""),
                    esrs_disclosure_id=row.get("esrs_disclosure_id", ""),
                    company=row.get("company", ""),
                    gold_fulfillment_level=row.get("gold_fulfillment_level", ""),
                    predicted_fulfillment_level=row.get("predicted_fulfillment_level", ""),
                    direct_answer_token_f1=float(row.get("direct_answer_token_f1", 0)),
                    rationale_token_f1=float(row.get("rationale_token_f1", 0)),
                    exact_match=row.get("exact_match", "False").lower() == "true",
                    ordinal_error=int(float(row.get("ordinal_error", 2))),
                    success=success,
                )
            )
    return rows


def main() -> None:
    results_dir = RQ2_RESULTS_DIR
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    summary_path = results_dir / "rq2_summary.json"

    # Load existing summary to preserve non-statistical fields
    existing: dict = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # Reconstruct results from CSVs
    rows = _load_results_from_csv(results_dir)
    if not rows:
        print("ERROR: No results found in CSVs.")
        sys.exit(1)

    with_rows = [r for r in rows if r.condition == "with_esrs_context"]
    without_rows = [r for r in rows if r.condition == "without_esrs_context"]

    print(f"Loaded {len(with_rows)} with-context + {len(without_rows)} without-context rows")

    # Recompute all metrics
    with_agg = aggregate_condition_metrics(with_rows)
    without_agg = aggregate_condition_metrics(without_rows)
    paired = build_paired_results(rows)
    question_rows = build_question_level_results(rows)
    paired_stats = compute_rq2_paired_statistics(rows, paired, question_rows)
    baseline = majority_baseline(rows)

    # Per-standard metrics
    per_standard: dict[str, dict] = {}
    for condition_name, condition_rows in [("with_esrs_context", with_rows), ("without_esrs_context", without_rows)]:
        by_std: dict[str, list] = {}
        for r in condition_rows:
            by_std.setdefault(r.esrs_standard, []).append(r)
        per_standard[condition_name] = {
            std: aggregate_condition_metrics(std_rows)
            for std, std_rows in sorted(by_std.items())
        }

    # Build updated summary preserving non-statistical fields from existing
    summary = {
        "context_mode": existing.get("context_mode", "substantive_full_standard"),
        "results_version": RQ2_RESULTS_VERSION,
        "judge_version_used": get_judge_version(),
        "model": existing.get("model", "n/a"),
        "temperature": existing.get("temperature", "n/a"),
        "condition_parallelism": existing.get("condition_parallelism", "n/a"),
        "n_questions": existing.get("n_questions", len(question_rows)),
        "n_companies": existing.get("n_companies", "n/a"),
        "condition_metrics": {
            "with_esrs_context": with_agg,
            "without_esrs_context": without_agg,
        },
        "paired_statistics": paired_stats,
        "question_level_results": question_rows,
        "majority_baseline": baseline,
        "per_standard_metrics": per_standard,
        "audit_manifest": build_rq2_audit_manifest(),
    }

    # Write updated summary
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✅ Recomputed summary written to: {summary_path}")
    print(f"   Pairs: {paired_stats['n_pairs']}")
    print(f"   Question clusters: {paired_stats['n_question_clusters']}")
    wt = paired_stats['question_level_wilcoxon_tests']['macro_f1']
    print(f"   Wilcoxon p (macro_f1): {wt['p_value']:.4f}")
    print(f"   Cohen's d (macro_f1): {paired_stats['cluster_bootstrap']['macro_f1'].get('cohens_d', 'n/a'):.4f}")
    print(f"   Effect magnitude: {paired_stats['cluster_bootstrap']['macro_f1'].get('effect_magnitude', 'n/a')}")
    print(f"   Achieved power: {paired_stats['power_analysis'].get('achieved_power', 'n/a')}")
    print(f"   MDES at 80%%: {paired_stats['power_analysis'].get('mdes_80pct', 'n/a')}")
    print(f"   Prediction bias: {paired_stats['prediction_bias'].get('interpretation', 'n/a')}")
    ds = paired_stats['direction_summary']
    print(f"   Direction summary: {ds['n_improved']}↑ {ds['n_degraded']}↓ {ds['n_unchanged']}=")


if __name__ == "__main__":
    main()
