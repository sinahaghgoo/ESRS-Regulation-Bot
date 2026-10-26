"""Check progress of RQ2 regeneration and compute interim citation metrics.

Compares newly regenerated nb_cache files against the backup (pre-regen)
to estimate how many samples have been processed so far and whether the
new prompt is reducing over-citation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.evaluation.rq3_metrics import (  # noqa: E402
    citation_set_metrics, citation_range_overlap_metrics, normalize_page_ranges,
)

CACHE_DIR = PROJECT_ROOT / "data/artifacts/evaluation_results/rq2/nb_cache/no_esrs_context"
BACKUP_DIR = PROJECT_ROOT / "data/artifacts/evaluation_results/rq2/nb_cache/no_esrs_context_backup_v1"
GS_PATH = PROJECT_ROOT / "golden_standard/golden_standard.json"


def cache_key(question: str, company: str, condition: str, ctx_len: int) -> str:
    blob = f"{question}|{company}|{condition}|{ctx_len}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def main() -> None:
    if not BACKUP_DIR.exists():
        print(f"Backup dir not found: {BACKUP_DIR}")
        return

    gs = json.loads(GS_PATH.read_text(encoding="utf-8"))
    samples = []
    for q in gs["questions"]:
        for c in q["per_company"]:
            samples.append({
                "qid": q["id"], "company": c["company"], "question": q["question"],
                "gold_pages": [str(ev.get("page_number", "")).strip() for ev in c.get("evidence", []) if ev.get("page_number")],
            })

    n_total = len(samples)
    n_changed = 0
    cp_count_old = []
    cp_count_new = []
    metrics_old = []  # (precision, recall, f1)
    metrics_new = []
    metrics_old_range = []
    metrics_new_range = []

    for s in samples:
        ck = cache_key(s["question"], s["company"], "no_esrs_context", 0)
        new_fp = CACHE_DIR / f"{ck}.json"
        old_fp = BACKUP_DIR / f"{ck}.json"
        if not new_fp.exists() or not old_fp.exists():
            continue

        old_data = json.loads(old_fp.read_text(encoding="utf-8"))
        new_data = json.loads(new_fp.read_text(encoding="utf-8"))
        old_cp = old_data.get("cited_pages") or []
        new_cp = new_data.get("cited_pages") or []

        # Has this sample been regenerated?  Compare cited_pages.
        if old_cp != new_cp:
            n_changed += 1
            cp_count_old.append(len(old_cp))
            cp_count_new.append(len(new_cp))

            # Set-overlap metric
            m_old = citation_set_metrics(s["gold_pages"], old_cp, tolerance=1)
            m_new = citation_set_metrics(s["gold_pages"], new_cp, tolerance=1)
            metrics_old.append((m_old["precision"], m_old["recall"], m_old["f1"]))
            metrics_new.append((m_new["precision"], m_new["recall"], m_new["f1"]))

            # Range-overlap metric (baseline)
            g_r = normalize_page_ranges(s["gold_pages"])
            m_old_r = citation_range_overlap_metrics(g_r, normalize_page_ranges(old_cp), tolerance=1)
            m_new_r = citation_range_overlap_metrics(g_r, normalize_page_ranges(new_cp), tolerance=1)
            metrics_old_range.append((m_old_r["precision"], m_old_r["recall"], m_old_r["f1"]))
            metrics_new_range.append((m_new_r["precision"], m_new_r["recall"], m_new_r["f1"]))

    print(f"Samples regenerated so far: {n_changed} / {n_total}  ({100 * n_changed / n_total:.1f}%)")
    if not n_changed:
        return

    print(f"\nMean cited_pages count:  OLD = {mean(cp_count_old):.2f}  →  NEW = {mean(cp_count_new):.2f}  "
          f"(Δ = {mean(cp_count_new) - mean(cp_count_old):+.2f})")

    def _agg(metrics):
        ps = [m[0] for m in metrics]
        rs = [m[1] for m in metrics]
        f1s = [m[2] for m in metrics]
        return mean(ps), mean(rs), mean(f1s)

    print(f"\nSet-overlap metric (PRIMARY, on regenerated subset):")
    p_o, r_o, f_o = _agg(metrics_old)
    p_n, r_n, f_n = _agg(metrics_new)
    print(f"  OLD: P={p_o:.4f}  R={r_o:.4f}  F1={f_o:.4f}")
    print(f"  NEW: P={p_n:.4f}  R={r_n:.4f}  F1={f_n:.4f}")
    print(f"  Δ:   P{p_n - p_o:+.4f}  R{r_n - r_o:+.4f}  F1{f_n - f_o:+.4f}")

    print(f"\nRange-overlap baseline (on regenerated subset):")
    p_o, r_o, f_o = _agg(metrics_old_range)
    p_n, r_n, f_n = _agg(metrics_new_range)
    print(f"  OLD: P={p_o:.4f}  R={r_o:.4f}  F1={f_o:.4f}")
    print(f"  NEW: P={p_n:.4f}  R={r_n:.4f}  F1={f_n:.4f}")
    print(f"  Δ:   P{p_n - p_o:+.4f}  R{r_n - r_o:+.4f}  F1{f_n - f_o:+.4f}")


if __name__ == "__main__":
    main()
