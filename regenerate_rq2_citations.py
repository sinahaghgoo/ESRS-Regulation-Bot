"""Re-run RQ2 generation with the updated cited_pages prompt.

Reads existing retrieved_chunks from
``data/artifacts/evaluation_results/rq2/responses/without_esrs_context/``
and regenerates each answer with the new prompt that instructs the model
to cite only 3-6 directly-supporting pages (instead of dumping every
retrieved page label).  Writes new cache files into
``data/artifacts/evaluation_results/rq2/nb_cache/no_esrs_context/``
using the same hash-key format the RQ2 / RQ3 notebooks expect.

Usage
-----
    python regenerate_rq2_citations.py --pilot          # 5 samples
    python regenerate_rq2_citations.py --max 10         # arbitrary cap
    python regenerate_rq2_citations.py                  # full 204
    python regenerate_rq2_citations.py --keep-existing  # skip non-empty cache files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env BEFORE importing anything that needs the API key
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            if _k:
                os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))

from rag.config.settings import RQ2_RESULTS_DIR  # noqa: E402
from rag.generation import AnswerGenerator  # noqa: E402
from rag.types import GenerationSample, RetrievedEvidenceChunk  # noqa: E402

RESPONSES_DIR = RQ2_RESULTS_DIR / "responses" / "without_esrs_context"
NB_CACHE_DIR = RQ2_RESULTS_DIR / "nb_cache" / "no_esrs_context"
NB_CACHE_BACKUP_DIR = RQ2_RESULTS_DIR / "nb_cache" / "no_esrs_context_backup_v1"


def _cache_key(question: str, company: str, condition: str, ctx_len: int) -> str:
    blob = f"{question}|{company}|{condition}|{ctx_len}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_response_payload(fp: Path) -> dict:
    return json.loads(fp.read_text(encoding="utf-8"))


def reconstruct_sample(payload: dict) -> tuple[GenerationSample, list[RetrievedEvidenceChunk]]:
    sample = GenerationSample(
        question_id=payload["question_id"],
        question=payload["question"],
        esrs_standard=payload.get("esrs_standard", ""),
        esrs_disclosure_id=payload.get("esrs_disclosure_id", ""),
        company=payload["company"],
        ground_truth_answer="",
        fulfillment_level="",
        fulfillment_rationale="",
        evidences=[],
    )
    chunks = []
    for c in payload.get("retrieved_chunks", []):
        chunks.append(RetrievedEvidenceChunk(
            chunk_id=c["chunk_id"],
            text=c["text"],
            score=float(c["score"]),
            rank=int(c["rank"]),
            page_start=c.get("page_start"),
            page_end=c.get("page_end"),
            header_hierarchy=list(c.get("header_hierarchy", [])),
        ))
    return sample, chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true",
                        help="Run only 5 samples (quick sanity check).")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of samples to (re)generate.")
    parser.add_argument("--keep-existing", action="store_true",
                        help="Skip samples whose nb_cache file already exists "
                             "and is non-empty (resume mode).")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip moving the existing nb_cache to backup.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel API calls (default 4).")
    args = parser.parse_args()

    if not RESPONSES_DIR.exists():
        raise SystemExit(f"Responses dir missing: {RESPONSES_DIR}")
    response_files = sorted(RESPONSES_DIR.glob("*.json"))
    if not response_files:
        raise SystemExit(f"No response files found in {RESPONSES_DIR}")

    cap = 5 if args.pilot else args.max
    if cap is not None:
        response_files = response_files[:cap]

    # Backup the existing nb_cache directory once (before any writes) unless
    # we are resuming or the user opted out.
    NB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if (
        not args.no_backup
        and not args.keep_existing
        and not NB_CACHE_BACKUP_DIR.exists()
        and any(NB_CACHE_DIR.glob("*.json"))
    ):
        print(f"Backing up old nb_cache -> {NB_CACHE_BACKUP_DIR}")
        shutil.copytree(NB_CACHE_DIR, NB_CACHE_BACKUP_DIR)

    print(f"Files to process: {len(response_files)}")
    print(f"Parallelism: {args.workers}")

    # Filter out responses that should be skipped before submitting jobs.
    work_items = []
    n_skipped = 0
    citation_counts_before_map: dict[str, int] = {}
    for fp in response_files:
        payload = load_response_payload(fp)
        if payload.get("condition") not in {"without_esrs_context", "no_esrs_context"}:
            n_skipped += 1
            continue
        sample = GenerationSample(
            question_id=payload["question_id"], question=payload["question"],
            esrs_standard=payload.get("esrs_standard", ""),
            esrs_disclosure_id=payload.get("esrs_disclosure_id", ""),
            company=payload["company"],
            ground_truth_answer="", fulfillment_level="",
            fulfillment_rationale="", evidences=[],
        )
        cache_path = NB_CACHE_DIR / f"{_cache_key(sample.question, sample.company, 'no_esrs_context', 0)}.json"
        if cache_path.exists():
            try:
                old = json.loads(cache_path.read_text("utf-8"))
                citation_counts_before_map[cache_path.stem] = len(old.get("cited_pages") or [])
                if args.keep_existing and old.get("compliance_level"):
                    n_skipped += 1
                    continue
            except Exception:
                pass
        work_items.append((fp, payload, sample, cache_path))

    print(f"After filtering: {len(work_items)} to (re)generate, {n_skipped} skipped")

    # Each thread needs its own AnswerGenerator (and OpenAI client) to be safe.
    citation_counts_before: list[int] = []
    citation_counts_after: list[int] = []
    print_lock = Lock()
    n_done = 0

    def _worker(i: int, item: tuple) -> str:
        nonlocal n_done
        fp, payload, sample, cache_path = item
        _, chunks = reconstruct_sample(payload)
        gen = AnswerGenerator(require_openai=True)
        prev_cp_count = citation_counts_before_map.get(cache_path.stem)
        try:
            answer = gen.generate(
                sample=sample,
                retrieved_chunks=chunks,
                esrs_context="",
                condition="no_esrs_context",
            )
        except Exception as exc:
            with print_lock:
                print(f"  [{i:3d}] FAILED {sample.question_id} / {sample.company}: {exc}", flush=True)
            return "fail"

        new_cp = list(answer.cited_pages or [])
        result = {
            "compliance_level": answer.compliance_level,
            "answer": answer.direct_answer,
            "rationale": answer.compliance_rationale,
            "key_findings": answer.key_findings,
            "disclosure_gaps": answer.disclosure_gaps,
            "cited_pages": answer.cited_pages,
            "raw_response": answer.raw_response,
            "parse_error": answer.parse_error,
        }
        cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        with print_lock:
            citation_counts_after.append(len(new_cp))
            if prev_cp_count is not None:
                citation_counts_before.append(prev_cp_count)
            n_done += 1
            prev_str = f"prev={prev_cp_count}" if prev_cp_count is not None else "prev=?"
            print(f"  [{n_done:3d}/{len(work_items)}] {sample.question_id} / {sample.company}: "
                  f"{prev_str} -> new={len(new_cp)}", flush=True)
        return "ok"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker, i + 1, item) for i, item in enumerate(work_items)]
        for fut in as_completed(futures):
            fut.result()
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed / max(n_done, 1):.1f}s/sample)")
    print(f"Done: {n_done} regenerated, {n_skipped} skipped")
    if citation_counts_before:
        print(f"Citation count BEFORE (mean): {mean(citation_counts_before):.2f}")
    if citation_counts_after:
        print(f"Citation count AFTER  (mean): {mean(citation_counts_after):.2f}")
    if citation_counts_before and citation_counts_after:
        # Only compare matched samples for fairness
        n = min(len(citation_counts_before), len(citation_counts_after))
        delta = mean(citation_counts_after[:n]) - mean(citation_counts_before[:n])
        print(f"Mean change: {delta:+.2f}")


if __name__ == "__main__":
    main()
