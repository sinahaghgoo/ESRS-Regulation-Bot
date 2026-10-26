#!/usr/bin/env python3
"""Restore Round 1 (selective-citation) generation state from the judge cache.

Background
----------
Round 1 of the citation overhaul produced focused citations (mean 3.4
pages per sample) and a complete RQ3 judge cache (204 entries).  Round
2 (more inclusive prompt) overwrote the nb_cache and tried to re-judge
but ran into API credit exhaustion mid-run, leaving 30 samples
unjudged.

This script reconstructs the Round 1 nb_cache files by reading the
``generated_*`` fields out of the Round 1 judge cache (which embeds the
full ``judge_input`` payload).  After restoration, every nb_cache entry
matches a cached judge result, so the notebook can be re-run with no
new API calls.

Usage
-----
    python restore_round1.py        # restore (writes new nb_cache files)
    python restore_round1.py --dry  # preview only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Round 1 was active from ~15:05 to ~15:50 on 2026-04-30 (CEST)
ROUND1_START = datetime(2026, 4, 30, 15, 5, 0).timestamp()
ROUND1_END = datetime(2026, 4, 30, 15, 50, 0).timestamp()

NB_CACHE_DIR = PROJECT_ROOT / "data/artifacts/evaluation_results/rq2/nb_cache/no_esrs_context"
JUDGE_CACHE_DIR = PROJECT_ROOT / "data/artifacts/evaluation_results/rq3/judge_cache"
GS_PATH = PROJECT_ROOT / "golden_standard/golden_standard.json"


def cache_key(question: str, company: str, condition: str, ctx_len: int) -> str:
    blob = f"{question}|{company}|{condition}|{ctx_len}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_round1_judge_inputs() -> dict[tuple[str, str], dict[str, Any]]:
    """Map (question, company) → judge_input from Round 1 cache files."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for fp in JUDGE_CACHE_DIR.glob("*.json"):
        mtime = fp.stat().st_mtime
        if not (ROUND1_START <= mtime <= ROUND1_END):
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        inp = data.get("judge_input", {})
        q = inp.get("question", "")
        co = inp.get("company", "")
        if q and co:
            out[(q, co)] = inp
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true",
                        help="Preview only — do not write files.")
    args = parser.parse_args()

    print(f"Loading Round 1 judge inputs from {JUDGE_CACHE_DIR}...")
    round1 = load_round1_judge_inputs()
    print(f"Found {len(round1)} Round 1 judge entries.")

    gs = json.loads(GS_PATH.read_text(encoding="utf-8"))
    n_total = 0
    n_restored = 0
    n_missing = 0
    cp_counts: list[int] = []

    for q in gs["questions"]:
        for c in q["per_company"]:
            n_total += 1
            inp = round1.get((q["question"], c["company"]))
            if inp is None:
                n_missing += 1
                continue

            ck = cache_key(q["question"], c["company"], "no_esrs_context", 0)
            cache_path = NB_CACHE_DIR / f"{ck}.json"

            cited_pages = inp.get("generated_cited_pages", []) or []
            cp_counts.append(len(cited_pages))

            payload = {
                "compliance_level": "",  # the judge doesn't see this; left blank
                "answer": inp.get("generated_direct_answer", ""),
                "rationale": inp.get("generated_compliance_rationale", ""),
                "key_findings": inp.get("generated_key_findings", []) or [],
                "disclosure_gaps": inp.get("generated_disclosure_gaps", []) or [],
                "cited_pages": cited_pages,
                "raw_response": "",  # not preserved through judge cache
                "parse_error": None,
                # Provenance marker so future runs know this came from a restore
                "_restored_from_round1_judge": True,
            }

            # Recover compliance_level from the existing nb_cache file (the
            # judge cache deliberately strips it).  Do NOT preserve the
            # existing ``raw_response`` — it came from a later regeneration
            # whose cited_pages disagree with the Round 1 list above, and the
            # notebook's ``_extract_gen`` prefers raw_response over the
            # top-level fields.  Synthesise a small JSON envelope so the
            # notebook's parser sees a self-consistent payload.
            if cache_path.exists():
                try:
                    existing = json.loads(cache_path.read_text(encoding="utf-8"))
                    payload["compliance_level"] = existing.get("compliance_level", "") or ""
                except Exception:
                    pass
            payload["raw_response"] = json.dumps({
                "direct_answer": payload["answer"],
                "compliance_level": payload["compliance_level"],
                "compliance_rationale": payload["rationale"],
                "key_findings": payload["key_findings"],
                "disclosure_gaps": payload["disclosure_gaps"],
                "cited_pages": payload["cited_pages"],
            })

            if not args.dry:
                cache_path.write_text(json.dumps(payload, indent=2),
                                      encoding="utf-8")
            n_restored += 1

    print()
    print(f"Total samples in golden standard:    {n_total}")
    print(f"Restored from Round 1 judge cache:   {n_restored}")
    print(f"Missing (no Round 1 judge entry):    {n_missing}")
    if cp_counts:
        from statistics import mean
        print(f"Mean cited_pages count (restored):    {mean(cp_counts):.2f}")
    if args.dry:
        print("(--dry mode; no files were written.)")


if __name__ == "__main__":
    main()
