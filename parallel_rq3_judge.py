#!/usr/bin/env python3
"""Parallel RQ3 judge runner.

Builds the same RQ3JudgeInput list the notebook builds, then runs the
judge on each one in parallel (multiple Claude API calls at once).
Populates the judge cache so the notebook can read it back without any
API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env first
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            if _k:
                os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))

from rag.config.settings import (  # noqa: E402
    GOLDEN_STANDARD_PATH, RQ2_RESULTS_DIR,
    load_esrs_context_decision,
)
from rag.evaluation.rq3_judge import (  # noqa: E402
    RQ3JudgeInput, judge_sample, get_judge_version, JUDGE_CACHE_DIR,
)


ESRS_CTX_LENGTHS = {
    "Q1": 14309, "Q2": 2517, "Q3": 10595, "Q4": 9590,
    "Q5": 4009, "Q6": 25453, "Q7": 9480, "Q8": 13080,
    "Q9": 19566, "Q10": 18847, "Q11": 8089, "Q12": 9763,
}


def cache_key(question: str, company: str, condition: str, ctx_len: int) -> str:
    blob = f"{question}|{company}|{condition}|{ctx_len}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel API calls (default 8).")
    parser.add_argument("--max", type=int, default=None,
                        help="Cap on samples (debug).")
    args = parser.parse_args()

    api_key = os.getenv("CLAUDE_API_KEY", "")
    if not api_key:
        raise SystemExit("CLAUDE_API_KEY not set in .env")

    helps = load_esrs_context_decision()
    condition = "full_system" if helps else "no_esrs_context"
    print(f"Condition: {condition}")
    print(f"Judge version: {get_judge_version()}")

    raw_golden = json.loads(GOLDEN_STANDARD_PATH.read_text(encoding="utf-8"))
    samples = []
    for q in raw_golden["questions"]:
        for c in q["per_company"]:
            samples.append({
                "qid": q["id"],
                "question": q["question"],
                "esrs_standard": q.get("esrs_standard", ""),
                "esrs_disclosure_id": q.get("esrs_disclosure_id", ""),
                "company": c["company"],
                "ground_truth_answer": c.get("ground_truth_answer", ""),
                "fulfillment_level": c.get("fulfillment_level", ""),
                "fulfillment_rationale": c.get("fulfillment_rationale", ""),
            })

    # Build judge inputs from cache
    nb_cache_dir = RQ2_RESULTS_DIR / "nb_cache" / condition
    inputs: list[RQ3JudgeInput] = []
    for s in samples:
        ctx_len = ESRS_CTX_LENGTHS.get(s["qid"], 0) if condition == "full_system" else 0
        ck = cache_key(s["question"], s["company"], condition, ctx_len)
        fp = nb_cache_dir / f"{ck}.json"
        if not fp.exists():
            print(f"  WARN: no cache for {s['qid']}/{s['company']}")
            continue
        cache_data = json.loads(fp.read_text("utf-8"))
        # Extract from raw_response if available (canonical names)
        raw = cache_data.get("raw_response", "")
        gen = {
            "compliance_level": cache_data.get("compliance_level", ""),
            "direct_answer": cache_data.get("answer", ""),
            "compliance_rationale": cache_data.get("rationale", ""),
            "key_findings": cache_data.get("key_findings", []),
            "disclosure_gaps": cache_data.get("disclosure_gaps", []),
            "cited_pages": cache_data.get("cited_pages", []),
        }
        if raw:
            try:
                parsed = json.loads(raw)
                for k in gen:
                    if k in parsed:
                        gen[k] = parsed[k]
            except (json.JSONDecodeError, TypeError):
                pass

        inputs.append(RQ3JudgeInput(
            question=s["question"],
            esrs_standard=s["esrs_standard"],
            esrs_disclosure_id=s["esrs_disclosure_id"],
            company=s["company"],
            gold_fulfillment_level=s["fulfillment_level"],
            gold_fulfillment_rationale=s["fulfillment_rationale"],
            gold_ground_truth_answer=s["ground_truth_answer"],
            generated_direct_answer=gen["direct_answer"],
            generated_compliance_rationale=gen["compliance_rationale"],
            generated_key_findings=gen["key_findings"] or [],
            generated_disclosure_gaps=gen["disclosure_gaps"] or [],
            generated_cited_pages=gen["cited_pages"] or [],
        ))

    if args.max:
        inputs = inputs[: args.max]
    print(f"Total inputs: {len(inputs)}")

    # Skip ones that are already cached
    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo: list[tuple[int, RQ3JudgeInput]] = []
    n_cached = 0
    for i, inp in enumerate(inputs):
        from rag.evaluation.rq3_judge import _cache_key, _load_cached
        key = _cache_key(inp, get_judge_version())
        if _load_cached(key) is not None:
            n_cached += 1
            continue
        todo.append((i, inp))

    print(f"Already cached: {n_cached}")
    print(f"To judge:       {len(todo)}")
    print(f"Workers:        {args.workers}")

    if not todo:
        print("Nothing to do.")
        return

    print_lock = Lock()
    n_done = 0

    def _worker(i: int, inp: RQ3JudgeInput) -> None:
        nonlocal n_done
        try:
            out, _ = judge_sample(inp, api_key=api_key, use_cache=False)
            with print_lock:
                n_done += 1
                print(f"  [{n_done}/{len(todo)}] {inp.company} (mean={out.mean_score:.2f})", flush=True)
        except Exception as exc:
            with print_lock:
                print(f"  FAILED {inp.company}: {exc}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker, i, inp) for i, inp in todo]
        for f in as_completed(futures):
            f.result()
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed / max(n_done, 1):.1f}s/sample)")


if __name__ == "__main__":
    main()
